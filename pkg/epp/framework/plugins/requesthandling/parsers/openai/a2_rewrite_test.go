/*
Copyright 2026 The llm-d Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package openai

import (
	"bytes"
	"context"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const (
	a2ClientModel = "A2-client-model"
	a2TargetModel = "a2-backend/model-v2"
	mediaOneMiB   = 1 << 20
)

// videoBodyPart is a part read back from a forwarded body.
type videoBodyPart struct {
	name     string
	fileName string
	value    []byte
}

func readVideoBodyParts(t *testing.T, body []byte, contentTypeValue string) []videoBodyPart {
	t.Helper()
	_, params, err := mime.ParseMediaType(contentTypeValue)
	if err != nil {
		t.Fatalf("ParseMediaType(%q) error = %v", contentTypeValue, err)
	}
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	parts := []videoBodyPart{}
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			return parts
		}
		if err != nil {
			t.Fatalf("NextPart() error = %v", err)
		}
		value, err := io.ReadAll(part)
		if err != nil {
			t.Fatalf("reading part %q: %v", part.FormName(), err)
		}
		parts = append(parts, videoBodyPart{name: part.FormName(), fileName: part.FileName(), value: value})
	}
}

// compareVideoParts asserts that out holds the same parts as in, in the same order
// and under the same names and filenames. The caller compares the values.
func compareVideoParts(t *testing.T, in, out []videoBodyPart) {
	t.Helper()
	if len(in) != len(out) {
		t.Fatalf("rewritten body has %d parts, want %d", len(out), len(in))
	}
	for i := range in {
		if in[i].name != out[i].name || in[i].fileName != out[i].fileName {
			t.Errorf("part %d = %q/%q, want %q/%q", i, out[i].name, out[i].fileName, in[i].name, in[i].fileName)
		}
	}
}

// mediaBlob returns deterministic media bytes that exercise what a reassembling
// rewrite damages first: NUL, high bytes, CRLF, and lines that open like a delimiter.
func mediaBlob(size int) string {
	pattern := []byte{0x00, 0x01, 0xff, 0xfe, '\r', '\n', '-', '-', '\r', '\n', 0x89, 'P', 'N', 'G'}
	blob := make([]byte, size)
	for i := range blob {
		blob[i] = pattern[i%len(pattern)]
	}
	return string(blob)
}

// A2-U06b: rewriting the model of a multipart video request replaces the value bytes
// of the model part and nothing else. The media part, the boundary, the part order,
// and the header text are compared against what the client sent.
func TestA2U06b_RewriteModelNameLeavesEveryOtherByte(t *testing.T) {
	parser := NewOpenAIParser()
	media := mediaBlob(mediaOneMiB)
	videoRef := `{"video_url":"https://example.com/input.mp4"}`
	body, ct := buildVideoMultipart(t, []videoPart{
		{name: "prompt", value: "a cinematic tracking shot"},
		{name: "model", value: a2ClientModel},
		{name: "input_reference", value: media, fileName: "reference.mp4", contentType: "video/mp4"},
		{name: "width", value: "832"},
		{name: "fps", value: "16"},
		{name: "video_reference", value: videoRef},
	})
	headers := map[string]string{":path": videosPath, contentType: ct}
	got, err := parser.ParseRequest(context.Background(), body, headers)
	if err != nil {
		t.Fatalf("ParseRequest() error = %v", err)
	}
	if got.Body.Model != a2ClientModel {
		t.Fatalf("Model = %q, want %q", got.Body.Model, a2ClientModel)
	}
	if got.Body.Mutated {
		t.Error("Mutated = true after parsing: an unchanged body must forward the bytes the client sent")
	}

	rewritten, err := parser.RewriteModelName(got.Body.Payload.(fwkrh.MarshalablePayload), a2TargetModel)
	if err != nil {
		t.Fatalf("RewriteModelName() error = %v", err)
	}
	out, err := rewritten.Marshal()
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	// Stated as offsets: the bytes that differ are exactly the model value, and the
	// value sits where an independent search for it in the received body says it does.
	anchor := []byte("\r\n\r\n" + a2ClientModel + "\r\n--")
	at := bytes.Index(body, anchor)
	if at < 0 {
		t.Fatalf("the received body does not contain the model part at %q", anchor)
	}
	valueStart := at + len("\r\n\r\n")
	want := make([]byte, 0, len(body)+len(a2TargetModel)-len(a2ClientModel))
	want = append(want, body[:valueStart]...)
	want = append(want, a2TargetModel...)
	want = append(want, body[valueStart+len(a2ClientModel):]...)
	if !bytes.Equal(out, want) {
		t.Error("rewritten body is not the received body with the model value replaced")
	}
	if prefix, suffix := commonPrefixLen(body, out), commonSuffixLen(body, out); prefix != valueStart ||
		suffix != len(body)-valueStart-len(a2ClientModel) {
		t.Errorf("diff spans [%d:%d] of the received body, want the model value at [%d:%d]",
			prefix, len(body)-suffix, valueStart, valueStart+len(a2ClientModel))
	}

	// Stated as parts: same parts in the same order, one value changed.
	inParts := readVideoBodyParts(t, body, ct)
	outParts := readVideoBodyParts(t, out, ct)
	compareVideoParts(t, inParts, outParts)
	for i := range inParts {
		switch outParts[i].name {
		case "model":
			if string(inParts[i].value) != a2ClientModel || string(outParts[i].value) != a2TargetModel {
				t.Errorf("model part = %q, want %q replaced by %q", outParts[i].value, a2ClientModel, a2TargetModel)
			}
		default:
			if !bytes.Equal(inParts[i].value, outParts[i].value) {
				t.Errorf("part %q changed: %d bytes became %d bytes", inParts[i].name, len(inParts[i].value), len(outParts[i].value))
			}
		}
	}
	mediaPart := outParts[2]
	if mediaPart.name != "input_reference" || inParts[2].name != "input_reference" || string(mediaPart.value) != media {
		t.Error("the 1 MiB media part is not byte-identical to the bytes the client sent")
	}

	// The forwarded body still parses, and reads back under the rewritten name.
	reread, err := parser.ParseRequest(context.Background(), out, headers)
	if err != nil {
		t.Fatalf("ParseRequest() on the rewritten body error = %v", err)
	}
	if reread.Body.Model != a2TargetModel {
		t.Errorf("Model = %q after the rewrite, want %q", reread.Body.Model, a2TargetModel)
	}
	if reread.Body.Videos.Prompt != got.Body.Videos.Prompt || *reread.Body.Videos.Width != *got.Body.Videos.Width ||
		*reread.Body.Videos.FPS != *got.Body.Videos.FPS {
		t.Error("rewriting the model changed another parsed field")
	}
}

// A2-U06b: a multipart body that carries two model parts has to be rewritten where
// the backend reads, which is the last one.
func TestA2U06b_RewriteModelNameDuplicateModelTakesLast(t *testing.T) {
	parser := NewOpenAIParser()
	body, ct := buildVideoMultipart(t, []videoPart{
		{name: "model", value: "first-client-model"},
		{name: "prompt", value: "continue this motion"},
		{name: "model", value: "second-client-model"},
		{name: "input_reference", value: mediaBlob(4096), fileName: "reference.mp4", contentType: "video/mp4"},
	})
	headers := map[string]string{":path": videosPath, contentType: ct}
	got, err := parser.ParseRequest(context.Background(), body, headers)
	if err != nil {
		t.Fatalf("ParseRequest() error = %v", err)
	}
	if got.Body.Model != "second-client-model" {
		t.Fatalf("Model = %q, want the last declared value", got.Body.Model)
	}
	rewritten, err := parser.RewriteModelName(got.Body.Payload.(fwkrh.MarshalablePayload), a2TargetModel)
	if err != nil {
		t.Fatalf("RewriteModelName() error = %v", err)
	}
	out, err := rewritten.Marshal()
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	models := []string{}
	for _, part := range readVideoBodyParts(t, out, ct) {
		if part.name == "model" {
			models = append(models, string(part.value))
		}
	}
	if len(models) != 2 || models[0] != "first-client-model" || models[1] != a2TargetModel {
		t.Errorf("model parts = %q, want [first-client-model %s]", models, a2TargetModel)
	}
	// The untouched prefix proves the rewrite landed on the second part.
	prefix := commonPrefixLen(body, out)
	if !bytes.HasPrefix(body[prefix:], []byte("second-client-model")) {
		t.Errorf("the rewrite started at offset %d, which is not the start of the last model value", prefix)
	}
	if reread, err := parser.ParseRequest(context.Background(), out, headers); err != nil {
		t.Fatalf("ParseRequest() on the rewritten body error = %v", err)
	} else if reread.Body.Model != a2TargetModel {
		t.Errorf("Model = %q after the rewrite, want %q", reread.Body.Model, a2TargetModel)
	}
}

// A2-U06b: a body with no model form field is forwarded byte for byte. A file part
// named model is a media upload, not the field.
func TestA2U06b_RewriteModelNameWithoutModelField(t *testing.T) {
	parser := NewOpenAIParser()
	for _, parts := range [][]videoPart{
		{
			{name: "prompt", value: "a cat surfing"},
			{name: "input_reference", value: mediaBlob(2048), fileName: "reference.mp4", contentType: "video/mp4"},
		},
		{
			{name: "prompt", value: "a cat surfing"},
			{name: "model", value: a2ClientModel, fileName: "model.mp4", contentType: "video/mp4"},
		},
	} {
		body, ct := buildVideoMultipart(t, parts)
		headers := map[string]string{":path": videosPath, contentType: ct}
		got, err := parser.ParseRequest(context.Background(), body, headers)
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		if got.Body.Model != "" {
			t.Fatalf("Model = %q, want empty: the body declares no model form field", got.Body.Model)
		}
		rewritten, err := parser.RewriteModelName(got.Body.Payload.(fwkrh.MarshalablePayload), a2TargetModel)
		if err != nil {
			t.Fatalf("RewriteModelName() error = %v", err)
		}
		out, err := rewritten.Marshal()
		if err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		if !bytes.Equal(out, body) {
			t.Error("a body with no model form field was not forwarded byte for byte")
		}
	}
}

// A2-U06b: Marshal must not allocate in proportion to the media, which a per-part
// buffer or a decode/re-encode pass would.
func TestA2U06b_RewriteModelNameAllocationsDoNotScaleWithMedia(t *testing.T) {
	parser := NewOpenAIParser()
	allocs := func(mediaSize int) float64 {
		body, ct := buildVideoMultipart(t, []videoPart{
			{name: "model", value: a2ClientModel},
			{name: "prompt", value: "a cat surfing"},
			{name: "input_reference", value: mediaBlob(mediaSize), fileName: "reference.mp4", contentType: "video/mp4"},
		})
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		rewritten, err := parser.RewriteModelName(got.Body.Payload.(fwkrh.MarshalablePayload), a2TargetModel)
		if err != nil {
			t.Fatalf("RewriteModelName() error = %v", err)
		}
		if _, err := rewritten.Marshal(); err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		return testing.AllocsPerRun(20, func() {
			_, _ = rewritten.Marshal()
		})
	}
	small, large := allocs(1<<10), allocs(mediaOneMiB)
	t.Logf("Marshal allocations: %.0f for a 1 KiB media part, %.0f for a 1 MiB media part", small, large)
	if small != large {
		t.Errorf("Marshal allocations scale with the media part: %.0f for 1 KiB, %.0f for 1 MiB", small, large)
	}
}

// A2-U06b: the backend coerces form values to the declared type, so an integer field
// accepts an integral float spelling and surrounding whitespace, and rejects a
// fractional one. The router must agree, or it refuses a request the backend serves.
func TestA2U06b_IntegerFormCoercionMatchesBackend(t *testing.T) {
	parser := NewOpenAIParser()
	accepted := []struct {
		field string
		value string
		want  int64
	}{
		{field: "num_frames", value: "9.0", want: 9},
		{field: "num_frames", value: " 9", want: 9},
		{field: "num_frames", value: "9 ", want: 9},
		{field: "num_inference_steps", value: "4.0", want: 4},
		{field: "width", value: "832.0", want: 832},
	}
	for _, tt := range accepted {
		t.Run(tt.field+"="+tt.value+" is accepted", func(t *testing.T) {
			body, ct := buildVideoMultipart(t, []videoPart{
				{name: "prompt", value: "a cat surfing"},
				{name: tt.field, value: tt.value},
			})
			got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
			if err != nil {
				t.Fatalf("ParseRequest() rejected %s=%q, which the backend accepts: %v", tt.field, tt.value, err)
			}
			var got64 int64
			switch tt.field {
			case "num_frames":
				got64 = *got.Body.Videos.NumFrames
			case "num_inference_steps":
				got64 = *got.Body.Videos.NumInferenceSteps
			case "width":
				got64 = *got.Body.Videos.Width
			}
			if got64 != tt.want {
				t.Errorf("%s = %d, want %d", tt.field, got64, tt.want)
			}
		})
	}
	rejected := []struct {
		field string
		value string
	}{
		{field: "num_frames", value: "9.5"},
		{field: "num_inference_steps", value: "40.5"},
		{field: "num_frames", value: "9.0.0"},
		{field: "num_frames", value: "1e-1"},
	}
	for _, tt := range rejected {
		t.Run(tt.field+"="+tt.value+" is rejected", func(t *testing.T) {
			body, ct := buildVideoMultipart(t, []videoPart{
				{name: "prompt", value: "a cat surfing"},
				{name: tt.field, value: tt.value},
			})
			if _, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct}); err == nil {
				t.Fatalf("ParseRequest() accepted %s=%q, which the backend rejects", tt.field, tt.value)
			}
		})
	}
}

func commonPrefixLen(a, b []byte) int {
	limit := min(len(a), len(b))
	i := 0
	for i < limit && a[i] == b[i] {
		i++
	}
	return i
}

func commonSuffixLen(a, b []byte) int {
	limit := min(len(a), len(b))
	i := 0
	for i < limit && a[len(a)-1-i] == b[len(b)-1-i] {
		i++
	}
	return i
}
