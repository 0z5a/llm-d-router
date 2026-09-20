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
	"mime/multipart"
	"net/textproto"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/utils/ptr"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

const (
	videosPath     = "/v1/videos"
	videosSyncPath = "/v1/videos/sync"
)

// videoPart is one multipart part. An empty filename writes a plain form field;
// a non-empty filename writes a file part with the given bytes.
type videoPart struct {
	name        string
	value       string
	fileName    string
	contentType string
}

func buildVideoMultipart(t *testing.T, parts []videoPart) ([]byte, string) {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, p := range parts {
		if p.fileName == "" {
			if err := w.WriteField(p.name, p.value); err != nil {
				t.Fatalf("WriteField(%q) error = %v", p.name, err)
			}
			continue
		}
		h := make(textproto.MIMEHeader)
		h.Set("Content-Disposition", `form-data; name="`+p.name+`"; filename="`+p.fileName+`"`)
		ct := p.contentType
		if ct == "" {
			ct = "image/png"
		}
		h.Set("Content-Type", ct)
		fw, err := w.CreatePart(h)
		if err != nil {
			t.Fatalf("CreatePart(%q) error = %v", p.name, err)
		}
		if _, err := fw.Write([]byte(p.value)); err != nil {
			t.Fatalf("writing part %q: %v", p.name, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	return buf.Bytes(), w.FormDataContentType()
}
func fieldsParts(fields ...string) []videoPart {
	parts := make([]videoPart, 0, len(fields)/2)
	for i := 0; i+1 < len(fields); i += 2 {
		parts = append(parts, videoPart{name: fields[i], value: fields[i+1]})
	}
	return parts
}

// marshalPayload returns the bytes a parsed request forwards, so a test can assert on
// the wire form regardless of which payload type carries the body.
func marshalPayload(t *testing.T, body *fwkrh.InferenceRequestBody) []byte {
	t.Helper()
	marshaler, ok := body.Payload.(fwkrh.Marshaler)
	if !ok {
		t.Fatalf("Payload = %T, want a fwkrh.Marshaler", body.Payload)
	}
	raw, err := marshaler.Marshal()
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	return raw
}

// A2-U01 continued: both video endpoints declare the same multipart content
// type, so the sync endpoint must be parsed with the same field contract as the
// async one. Path-to-parser selection is covered in the handlers package.
func TestA2U01_VideoEndpointsParseMultipart(t *testing.T) {
	parser := NewOpenAIParser()
	for _, path := range []string{videosPath, videosSyncPath} {
		t.Run(path, func(t *testing.T) {
			body, ct := buildVideoMultipart(t, fieldsParts(
				"prompt", "a cinematic tracking shot",
				"model", "wan-t2v",
				"width", "1280",
				"height", "720",
				"num_frames", "80",
				"num_inference_steps", "40",
			))
			headers := map[string]string{":path": path, contentType: ct}
			got, err := parser.ParseRequest(context.Background(), body, headers)
			if err != nil {
				t.Fatalf("ParseRequest() error = %v", err)
			}
			if got.Body.Videos == nil {
				t.Fatalf("ParseRequest() body.Videos = nil, want parsed video request")
			}
			if got.Body.Model != "wan-t2v" {
				t.Errorf("Model = %q, want %q", got.Body.Model, "wan-t2v")
			}
			if got.Body.Stream {
				t.Error("Stream = true, want false: the video form has no stream field")
			}
			want := &fwkrh.VideoGenerationRequest{
				Prompt:              "a cinematic tracking shot",
				Width:               ptr.To[int64](1280),
				Height:              ptr.To[int64](720),
				NumFrames:           ptr.To[int64](80),
				NumInferenceSteps:   ptr.To[int64](40),
				NumOutputsPerPrompt: ptr.To[int64](1),
			}
			if diff := cmp.Diff(want, got.Body.Videos); diff != "" {
				t.Errorf("Videos mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(body, marshalPayload(t, got.Body)); diff != "" {
				t.Errorf("Payload must forward the raw multipart body (-want +got):\n%s", diff)
			}
		})
	}
}

// A2-U02: field order must not matter, optional fields must stay absent rather
// than default to zero, and non-ASCII prompts must survive.
func TestA2U02_FieldOrderMissingAndUnicode(t *testing.T) {
	parser := NewOpenAIParser()
	t.Run("reordered fields", func(t *testing.T) {
		body, ct := buildVideoMultipart(t, fieldsParts(
			"num_inference_steps", "28",
			"model", "wan-i2v",
			"prompt", "animate this still frame",
			"fps", "16",
			"size", "832x480",
			"num_frames", "64",
			"width", "832",
			"height", "480",
		))
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		want := &fwkrh.VideoGenerationRequest{
			Prompt:              "animate this still frame",
			Size:                "832x480",
			Width:               ptr.To[int64](832),
			Height:              ptr.To[int64](480),
			NumFrames:           ptr.To[int64](64),
			FPS:                 ptr.To[float64](16),
			NumInferenceSteps:   ptr.To[int64](28),
			NumOutputsPerPrompt: ptr.To[int64](1),
		}
		if diff := cmp.Diff(want, got.Body.Videos); diff != "" {
			t.Errorf("Videos mismatch (-want +got):\n%s", diff)
		}
		if got.Body.Model != "wan-i2v" {
			t.Errorf("Model = %q, want %q", got.Body.Model, "wan-i2v")
		}
	})
	t.Run("prompt only leaves optional fields nil", func(t *testing.T) {
		body, ct := buildVideoMultipart(t, fieldsParts("prompt", "a paper boat on a puddle"))
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		// num_outputs_per_prompt is the one field the backend always fills in,
		// so recording its default of 1 is not a fabricated value.
		want := &fwkrh.VideoGenerationRequest{
			Prompt:              "a paper boat on a puddle",
			NumOutputsPerPrompt: ptr.To[int64](1),
		}
		if diff := cmp.Diff(want, got.Body.Videos); diff != "" {
			t.Errorf("Videos mismatch (-want +got):\n%s", diff)
		}
		if got.Body.Model != "" {
			t.Errorf("Model = %q, want empty when the form omits it", got.Body.Model)
		}
	})
	t.Run("unicode prompt and negative prompt", func(t *testing.T) {
		prompt := "一只纸船漂过水坑 — 4K, cinematic"
		body, ct := buildVideoMultipart(t, fieldsParts(
			"prompt", prompt,
			"negative_prompt", "模糊, 抖动",
		))
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		if got.Body.Videos.Prompt != prompt {
			t.Errorf("Prompt = %q, want %q", got.Body.Videos.Prompt, prompt)
		}
		if got.Body.Videos.NegativePrompt != "模糊, 抖动" {
			t.Errorf("NegativePrompt = %q, want %q", got.Body.Videos.NegativePrompt, "模糊, 抖动")
		}
	})
	t.Run("seconds and fps stay declared values", func(t *testing.T) {
		body, ct := buildVideoMultipart(t, fieldsParts(
			"prompt", "timelapse",
			"seconds", "4",
			"fps", "24",
		))
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		if got.Body.Videos.Seconds != "4" {
			t.Errorf("Seconds = %q, want %q", got.Body.Videos.Seconds, "4")
		}
		if got.Body.Videos.FPS == nil || *got.Body.Videos.FPS != 24 {
			t.Errorf("FPS = %v, want 24", got.Body.Videos.FPS)
		}
		// 4s x 24fps must not be folded into a derived num_frames.
		if got.Body.Videos.NumFrames != nil {
			t.Errorf("NumFrames = %v, want nil: the router must not derive frames from seconds x fps",
				*got.Body.Videos.NumFrames)
		}
	})
}

// A2-U03: duplicate model parts, invalid numbers, and overflow must fail
// bounded - no panic, no silent zero, and never a wrong model.
func TestA2U03_DuplicateInvalidAndOverflow(t *testing.T) {
	parser := NewOpenAIParser()
	t.Run("duplicate model takes the last part", func(t *testing.T) {
		body, ct := buildVideoMultipart(t, []videoPart{
			{name: "prompt", value: "a cat surfing"},
			{name: "model", value: "first-model"},
			{name: "model", value: "second-model"},
		})
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		// The backend validates the last model part, so a router that read the first
		// would route on a name the backend does not act on.
		if got.Body.Model != "second-model" {
			t.Errorf("Model = %q, want %q", got.Body.Model, "second-model")
		}
	})
	// Every rejected value below is rejected by the pinned backend too: either it
	// cannot be coerced to the declared type, or it violates the bound the
	// backend declares on that field (ge=1 for dimensions and frames, le=200 for
	// num_inference_steps, le=10 for num_outputs_per_prompt, ge=1 for fps).
	numeric := []struct {
		name  string
		field string
		value string
	}{
		{name: "width not a number", field: "width", value: "wide"},
		{name: "width empty", field: "width", value: ""},
		{name: "height negative", field: "height", value: "-1"},
		{name: "num_frames not a number", field: "num_frames", value: "eighty"},
		{name: "num_frames float", field: "num_frames", value: "80.5"},
		{name: "num_inference_steps zero", field: "num_inference_steps", value: "0"},
		{name: "num_inference_steps above backend cap", field: "num_inference_steps", value: "201"},
		{name: "num_outputs_per_prompt above backend cap", field: "num_outputs_per_prompt", value: "11"},
		{name: "fps zero", field: "fps", value: "0"},
		{name: "fps not a number", field: "fps", value: "fast"},
		{name: "width overflows int64", field: "width", value: "9223372036854775808"},
		{name: "num_frames overflows int64", field: "num_frames", value: "99999999999999999999"},
		{name: "num_inference_steps overflows int64", field: "num_inference_steps", value: "99999999999999999999"},
		{name: "seed overflows int64", field: "seed", value: "9223372036854775808"},
	}
	for _, tt := range numeric {
		t.Run(tt.name, func(t *testing.T) {
			body, ct := buildVideoMultipart(t, []videoPart{
				{name: "prompt", value: "a cat surfing"},
				{name: tt.field, value: tt.value},
			})
			got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
			if err == nil {
				t.Fatalf("ParseRequest() accepted %s=%q, want a bounded error (got %+v)", tt.field, tt.value, got.Body.Videos)
			}
		})
	}
	t.Run("int64 boundary value is accepted, not treated as overflow", func(t *testing.T) {
		body, ct := buildVideoMultipart(t, []videoPart{
			{name: "prompt", value: "a cat surfing"},
			{name: "seed", value: "9223372036854775807"},
		})
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		if got.Body.Videos.Seed == nil || *got.Body.Videos.Seed != 9223372036854775807 {
			t.Errorf("Seed = %v, want 9223372036854775807", got.Body.Videos.Seed)
		}
	})
}

// A2-U04: media bytes and unknown fields must be forwarded untouched, and the
// reference fields the backend expects as JSON text must not be re-encoded.
func TestA2U04_MediaBytesAndUnknownFieldsPreserved(t *testing.T) {
	parser := NewOpenAIParser()
	media := []byte{0x00, 0x01, 0xff, 0xfe, '\r', '\n', '-', '-', 0x89, 'P', 'N', 'G'}
	videoRef := `{"video_url":"https://example.com/input.mp4"}`
	extraParams := `{"preencode_mp4":true}`
	body, ct := buildVideoMultipart(t, []videoPart{
		{name: "prompt", value: "continue this motion"},
		{name: "input_reference", value: string(media), fileName: "ref.mp4", contentType: "video/mp4"},
		{name: "video_reference", value: videoRef},
		{name: "extra_params", value: extraParams},
		{name: "some_future_field", value: "ignored-by-backend"},
	})
	headers := map[string]string{":path": videosPath, contentType: ct}
	got, err := parser.ParseRequest(context.Background(), body, headers)
	if err != nil {
		t.Fatalf("ParseRequest() error = %v", err)
	}
	// The payload must marshal back to the original bytes, so media is byte-identical.
	raw := marshalPayload(t, got.Body)
	if !bytes.Equal(raw, body) {
		t.Error("Payload is not byte-identical to the received multipart body")
	}
	if !bytes.Contains(raw, media) {
		t.Error("media part bytes are not present verbatim in the forwarded payload")
	}
	if !bytes.Contains(raw, []byte(videoRef)) {
		t.Error("video_reference JSON text was re-encoded instead of forwarded verbatim")
	}
	if !bytes.Contains(raw, []byte(extraParams)) {
		t.Error("extra_params JSON text was re-encoded instead of forwarded verbatim")
	}
	// Unknown scalar fields are ignored, exactly as the backend ignores them.
	if got.Body.Videos.Prompt != "continue this motion" {
		t.Errorf("Prompt = %q, want %q", got.Body.Videos.Prompt, "continue this motion")
	}
}

// A2-U05: malformed multipart and oversized input must fail within bounds.
func TestA2U05_BoundedFailureOnMalformedMultipart(t *testing.T) {
	parser := NewOpenAIParser()
	t.Run("missing boundary", func(t *testing.T) {
		body, _ := buildVideoMultipart(t, fieldsParts("prompt", "a cat"))
		headers := map[string]string{":path": videosPath, contentType: "multipart/form-data"}
		if _, err := parser.ParseRequest(context.Background(), body, headers); err == nil {
			t.Error("ParseRequest() accepted a multipart body with no boundary in the content-type")
		}
	})
	t.Run("json content-type is rejected", func(t *testing.T) {
		headers := map[string]string{":path": videosPath, contentType: "application/json"}
		if _, err := parser.ParseRequest(context.Background(), []byte(`{"prompt":"a cat"}`), headers); err == nil {
			t.Error("ParseRequest() accepted a JSON body on a multipart-only endpoint")
		}
	})
	t.Run("truncated body", func(t *testing.T) {
		body, ct := buildVideoMultipart(t, fieldsParts("prompt", "a cat", "width", "640"))
		truncated := body[:len(body)-40]
		headers := map[string]string{":path": videosPath, contentType: ct}
		if _, err := parser.ParseRequest(context.Background(), truncated, headers); err == nil {
			t.Error("ParseRequest() accepted a truncated multipart body")
		}
	})
	t.Run("missing prompt", func(t *testing.T) {
		body, ct := buildVideoMultipart(t, fieldsParts("width", "640"))
		headers := map[string]string{":path": videosPath, contentType: ct}
		if _, err := parser.ParseRequest(context.Background(), body, headers); err == nil {
			t.Error("ParseRequest() accepted a video request with no prompt")
		}
	})
	t.Run("empty body", func(t *testing.T) {
		body, ct := buildVideoMultipart(t, nil)
		headers := map[string]string{":path": videosPath, contentType: ct}
		if _, err := parser.ParseRequest(context.Background(), body, headers); err == nil {
			t.Error("ParseRequest() accepted a multipart body with no parts")
		}
	})
	t.Run("many parts stays bounded", func(t *testing.T) {
		parts := []videoPart{{name: "prompt", value: "a cat"}}
		for i := 0; i < 512; i++ {
			parts = append(parts, videoPart{name: "unknown_field", value: strings.Repeat("x", 64)})
		}
		body, ct := buildVideoMultipart(t, parts)
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		if got.Body.Videos.Prompt != "a cat" {
			t.Errorf("Prompt = %q, want %q", got.Body.Videos.Prompt, "a cat")
		}
	})
}

// A2-U06: adding video support must not change how the existing text and image
// payloads are parsed.
func TestA2U06_NoRegressionForTextAndImage(t *testing.T) {
	parser := NewOpenAIParser()
	t.Run("chat completions still parses", func(t *testing.T) {
		body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":7,"stream":true}`)
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": "/v1/chat/completions"})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		if got.Body.ChatCompletions == nil {
			t.Fatal("ChatCompletions = nil")
		}
		if got.Body.Model != "m" {
			t.Errorf("Model = %q, want %q", got.Body.Model, "m")
		}
		if !got.Body.Stream {
			t.Error("Stream = false, want true")
		}
		if got.Body.MaxOutputTokens == nil || *got.Body.MaxOutputTokens != 7 {
			t.Errorf("MaxOutputTokens = %v, want 7", got.Body.MaxOutputTokens)
		}
	})
	t.Run("images generations still parses", func(t *testing.T) {
		body := []byte(`{"model":"img","prompt":"a cat","n":2,"size":"1024x1024","num_inference_steps":30}`)
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": "/v1/images/generations"})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		want := &fwkrh.ImagesGenerationsRequest{
			Prompt:            "a cat",
			N:                 ptr.To[int64](2),
			Size:              "1024x1024",
			NumInferenceSteps: ptr.To[int64](30),
		}
		if diff := cmp.Diff(want, got.Body.Images); diff != "" {
			t.Errorf("Images mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("images edits multipart still parses", func(t *testing.T) {
		body, ct := buildVideoMultipart(t, []videoPart{
			{name: "prompt", value: "add a hat"},
			{name: "n", value: "2"},
			{name: "image", value: "fake png bytes", fileName: "input.png"},
		})
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": "/v1/images/edits", contentType: ct})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		want := &fwkrh.ImagesGenerationsRequest{Prompt: "add a hat", N: ptr.To[int64](2)}
		if diff := cmp.Diff(want, got.Body.Images); diff != "" {
			t.Errorf("Images mismatch (-want +got):\n%s", diff)
		}
		if got.Body.Videos != nil {
			t.Errorf("Videos = %+v, want nil on the images/edits path", got.Body.Videos)
		}
	})
	t.Run("videos path never yields an images or chat body", func(t *testing.T) {
		body, ct := buildVideoMultipart(t, fieldsParts("prompt", "a cat"))
		got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
		if err != nil {
			t.Fatalf("ParseRequest() error = %v", err)
		}
		if got.Body.Images != nil || got.Body.ChatCompletions != nil || got.Body.Completions != nil {
			t.Errorf("videos path produced a non-video body: %+v", got.Body)
		}
	})
	t.Run("claims list both video suffixes exactly once", func(t *testing.T) {
		paths := NewOpenAIParser().Claims().Paths
		for _, want := range []string{"videos", "videos/sync"} {
			count := 0
			for _, got := range paths {
				if got == want {
					count++
				}
			}
			if count != 1 {
				t.Errorf("Claims().Paths contains %q %d times, want exactly 1 (paths=%v)", want, count, paths)
			}
		}
	})
}

// The video parse result must keep response processing enabled and must not
// claim to have parsed a body it only forwarded.
func TestVideoParseResultShape(t *testing.T) {
	parser := NewOpenAIParser()
	body, ct := buildVideoMultipart(t, fieldsParts("prompt", "a cat"))
	got, err := parser.ParseRequest(context.Background(), body, map[string]string{":path": videosPath, contentType: ct})
	if err != nil {
		t.Fatalf("ParseRequest() error = %v", err)
	}
	if got.SkipResponseProcessing {
		t.Error("SkipResponseProcessing = true, want false so response processing still runs")
	}
	if got.Body.Payload.IsParsed() {
		t.Error("Payload.IsParsed() = true, want false: media is forwarded, not interpreted")
	}
}

// ParseRequest must not panic on any video input, including inputs the
// multipart reader rejects.
func TestVideoParseRequestNeverPanics(t *testing.T) {
	parser := NewOpenAIParser()
	bodies := [][]byte{
		nil,
		{},
		[]byte("--boundary\r\n"),
		[]byte("\r\n\r\n"),
		[]byte("garbage"),
		bytes.Repeat([]byte("\xff"), 4096),
	}
	for i, body := range bodies {
		t.Run(string(rune('a'+i)), func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("ParseRequest panicked: %v", r)
				}
			}()
			_, err := parser.ParseRequest(context.Background(), body,
				map[string]string{":path": videosPath, contentType: "multipart/form-data; boundary=boundary"})
			if err != nil && errors.Is(err, context.Canceled) {
				t.Errorf("unexpected context error: %v", err)
			}
		})
	}
}

// A2-U07: fuzzing the video parser must not panic or allocate without bound.
// The seeds cover the shapes the unit tests already exercise plus raw garbage.
func FuzzVideoParseRequest(f *testing.F) {
	seeded, ct := buildVideoMultipartF(f, []videoPart{
		{name: "prompt", value: "a cat surfing"},
		{name: "model", value: "wan-t2v"},
		{name: "width", value: "1280"},
		{name: "num_inference_steps", value: "40"},
	})
	f.Add(seeded, ct)
	f.Add([]byte(""), "multipart/form-data; boundary=b")
	f.Add([]byte("--b\r\n"), "multipart/form-data; boundary=b")
	f.Add([]byte("garbage"), "application/json")
	f.Add([]byte("\r\n\r\n"), "multipart/form-data")
	parser := NewOpenAIParser()
	f.Fuzz(func(t *testing.T, body []byte, contentType string) {
		_, _ = parser.ParseRequest(context.Background(), body,
			map[string]string{":path": videosPath, contentType: contentType})
	})
}

// buildVideoMultipartF is the testing.F twin of buildVideoMultipart: fuzz seeds
// are registered before any *testing.T exists.
func buildVideoMultipartF(f *testing.F, parts []videoPart) ([]byte, string) {
	f.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, p := range parts {
		if p.fileName == "" {
			if err := w.WriteField(p.name, p.value); err != nil {
				f.Fatalf("WriteField(%q) error = %v", p.name, err)
			}
			continue
		}
		h := make(textproto.MIMEHeader)
		h.Set("Content-Disposition", `form-data; name="`+p.name+`"; filename="`+p.fileName+`"`)
		ct := p.contentType
		if ct == "" {
			ct = "image/png"
		}
		h.Set("Content-Type", ct)
		fw, err := w.CreatePart(h)
		if err != nil {
			f.Fatalf("CreatePart(%q) error = %v", p.name, err)
		}
		if _, err := fw.Write([]byte(p.value)); err != nil {
			f.Fatalf("writing part %q: %v", p.name, err)
		}
	}
	if err := w.Close(); err != nil {
		f.Fatalf("Close() error = %v", err)
	}
	return buf.Bytes(), w.FormDataContentType()
}

// A2-U07: allocation baseline for the video path, next to the images/edits
// multipart path it shares its parsing strategy with. Both pay one full copy of
// the body for io.ReadAll of each part; neither retains the media.
func BenchmarkVideoParseRequest(b *testing.B) {
	parser := NewOpenAIParser()
	videoBody, videoCT := buildVideoMultipartB(b, []videoPart{
		{name: "prompt", value: strings.Repeat("a cinematic shot ", 16)},
		{name: "model", value: "wan-t2v"},
		{name: "width", value: "1280"},
		{name: "height", value: "720"},
		{name: "num_frames", value: "80"},
		{name: "fps", value: "16"},
		{name: "num_inference_steps", value: "40"},
		{name: "input_reference", value: strings.Repeat("m", 1<<20), fileName: "ref.mp4", contentType: "video/mp4"},
	})
	editsBody, editsCT := buildVideoMultipartB(b, []videoPart{
		{name: "prompt", value: strings.Repeat("a cinematic shot ", 16)},
		{name: "n", value: "2"},
		{name: "image", value: strings.Repeat("m", 1<<20), fileName: "in.png"},
	})
	cases := []struct {
		name    string
		path    string
		body    []byte
		headers map[string]string
	}{
		{name: "videos/1MiB-media", path: videosPath, body: videoBody,
			headers: map[string]string{":path": videosPath, contentType: videoCT}},
		{name: "images-edits/1MiB-media", path: "/v1/images/edits", body: editsBody,
			headers: map[string]string{":path": "/v1/images/edits", contentType: editsCT}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(tc.body)))
			for b.Loop() {
				if _, err := parser.ParseRequest(context.Background(), tc.body, tc.headers); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// buildVideoMultipartB is the testing.B twin of buildVideoMultipart.
func buildVideoMultipartB(b *testing.B, parts []videoPart) ([]byte, string) {
	b.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, p := range parts {
		if p.fileName == "" {
			if err := w.WriteField(p.name, p.value); err != nil {
				b.Fatalf("WriteField(%q) error = %v", p.name, err)
			}
			continue
		}
		h := make(textproto.MIMEHeader)
		h.Set("Content-Disposition", `form-data; name="`+p.name+`"; filename="`+p.fileName+`"`)
		ct := p.contentType
		if ct == "" {
			ct = "image/png"
		}
		h.Set("Content-Type", ct)
		fw, err := w.CreatePart(h)
		if err != nil {
			b.Fatalf("CreatePart(%q) error = %v", p.name, err)
		}
		if _, err := fw.Write([]byte(p.value)); err != nil {
			b.Fatalf("writing part %q: %v", p.name, err)
		}
	}
	if err := w.Close(); err != nil {
		b.Fatalf("Close() error = %v", err)
	}
	return buf.Bytes(), w.FormDataContentType()
}
