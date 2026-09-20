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

package requesthandling

import (
	"bytes"
	"errors"
	"io"
	"mime/multipart"
	"net/textproto"
	"testing"
)

const testBoundary = "A2TestBoundary000000000000000000"

// testPart is one part of a generated multipart body. A non-empty fileName makes a
// file upload instead of a form field.
type testPart struct {
	name     string
	value    string
	fileName string
}

func buildTestMultipart(t *testing.T, parts []testPart) []byte {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	if err := w.SetBoundary(testBoundary); err != nil {
		t.Fatalf("SetBoundary() error = %v", err)
	}
	for _, p := range parts {
		if p.fileName == "" {
			if err := w.WriteField(p.name, p.value); err != nil {
				t.Fatalf("WriteField(%q) error = %v", p.name, err)
			}
			continue
		}
		h := make(textproto.MIMEHeader)
		h.Set("Content-Disposition", `form-data; name="`+p.name+`"; filename="`+p.fileName+`"`)
		h.Set("Content-Type", "video/mp4")
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
	return buf.Bytes()
}

// readFieldValue returns the value mime/multipart reads for the last form field named
// field, which is the value a rewrite must land on.
func readFieldValue(t *testing.T, body []byte, field string) (string, bool) {
	t.Helper()
	reader := multipart.NewReader(bytes.NewReader(body), testBoundary)
	value, found := "", false
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			return value, found
		}
		if err != nil {
			t.Fatalf("NextPart() error = %v", err)
		}
		if part.FileName() != "" {
			continue
		}
		data, err := io.ReadAll(part)
		if err != nil {
			t.Fatalf("reading part %q: %v", part.FormName(), err)
		}
		if part.FormName() == field {
			value, found = string(data), true
		}
	}
}

// The located range must hold exactly the bytes mime/multipart reads as the field
// value, for the shapes a video request takes.
func TestLastFormFieldValueMatchesMultipartReader(t *testing.T) {
	cases := []struct {
		name  string
		parts []testPart
	}{
		{"only field", []testPart{{name: "model", value: "wan-t2v"}}},
		{"first of several", []testPart{
			{name: "model", value: "wan-t2v"},
			{name: "prompt", value: "a cat"},
			{name: "width", value: "832"},
		}},
		{"last of several", []testPart{
			{name: "prompt", value: "a cat"},
			{name: "width", value: "832"},
			{name: "model", value: "wan-t2v"},
		}},
		{"between a media part and a field", []testPart{
			{name: "input_reference", value: "media-bytes", fileName: "ref.mp4"},
			{name: "model", value: "wan-t2v"},
			{name: "fps", value: "16"},
		}},
		{"duplicate takes the last", []testPart{
			{name: "model", value: "first"},
			{name: "model", value: "second"},
		}},
		{"empty value", []testPart{
			{name: "model", value: ""},
			{name: "prompt", value: "a cat"},
		}},
		{"value containing a blank line", []testPart{{name: "model", value: "a\r\n\r\nb"}}},
		{"value ending with CRLF", []testPart{{name: "model", value: "a\r\n"}}},
		{"value that is CRLF", []testPart{{name: "model", value: "\r\n"}}},
		{"file part named model is not the field", []testPart{
			{name: "model", value: "media-bytes", fileName: "model.bin"},
			{name: "prompt", value: "a cat"},
		}},
		{"model field after a file part named model", []testPart{
			{name: "model", value: "media-bytes", fileName: "model.bin"},
			{name: "model", value: "wan-t2v"},
		}},
		{"media carries CRLF and boundary-like lines", []testPart{
			{name: "input_reference", value: "\x00\xff\r\n--x\r\n\x00", fileName: "ref.mp4"},
			{name: "model", value: "wan-t2v"},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := buildTestMultipart(t, tc.parts)
			want, wantFound := readFieldValue(t, body, "model")
			start, end, found := lastFormFieldValue(body, testBoundary, "model")
			if found != wantFound {
				t.Fatalf("found = %v, want %v", found, wantFound)
			}
			if !found {
				return
			}
			if got := string(body[start:end]); got != want {
				t.Errorf("located value = %q, want %q, the value mime/multipart reads", got, want)
			}
		})
	}
}

// Framing this cannot address must report no field, because a replacement spliced at
// a guessed offset would land inside a media part.
func TestLastFormFieldValueReportsNoFieldOnUnaddressableFraming(t *testing.T) {
	canonical := buildTestMultipart(t, []testPart{
		{name: "prompt", value: "a cat"},
		{name: "model", value: "wan-t2v"},
	})
	cases := []struct {
		name string
		body []byte
	}{
		{"LF line endings", bytes.ReplaceAll(canonical, []byte("\r\n"), []byte("\n"))},
		{"preamble before the first boundary", append([]byte("preamble\r\n"), canonical...)},
		{"media carries the boundary prefix", buildTestMultipart(t, []testPart{
			{name: "input_reference", value: "\r\n--" + testBoundary + "XY", fileName: "ref.mp4"},
			{name: "model", value: "wan-t2v"},
		})},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if start, end, found := lastFormFieldValue(tc.body, testBoundary, "model"); found {
				t.Errorf("found = true with value %q, want no field", tc.body[start:end])
			}
		})
	}
}

func TestMultipartPayloadMarshal(t *testing.T) {
	const boundary = testBoundary
	parts := []testPart{
		{name: "prompt", value: "a cat"},
		{name: "model", value: "client-model"},
		{name: "input_reference", value: "media-bytes", fileName: "ref.mp4"},
		{name: "model", value: "client-model-last"},
	}
	body := buildTestMultipart(t, parts)
	payload := NewMultipartPayload(body, boundary)

	t.Run("unrewritten payload marshals to the received body", func(t *testing.T) {
		got, err := payload.Marshal()
		if err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		if !bytes.Equal(got, body) {
			t.Error("Marshal() did not return the received body")
		}
	})

	t.Run("rewrite replaces the last model value and nothing else", func(t *testing.T) {
		got, err := payload.WithModel("backend-model").Marshal()
		if err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		if len(got)-len(body) != len("backend-model")-len("client-model-last") {
			t.Errorf("body grew by %d bytes, want %d", len(got)-len(body), len("backend-model")-len("client-model-last"))
		}
		if bytes.Equal(got, body) {
			t.Fatal("Marshal() did not rewrite the model value")
		}
		prefix := commonPrefixLen(body, got)
		suffix := commonSuffixLen(body, got)
		if removed := string(body[prefix : len(body)-suffix]); removed != "client-model-last" {
			t.Errorf("bytes removed = %q, want the last model value", removed)
		}
		if added := string(got[prefix : len(got)-suffix]); added != "backend-model" {
			t.Errorf("bytes added = %q, want the replacement model", added)
		}
		if value, _ := readFieldValue(t, got, "model"); value != "backend-model" {
			t.Errorf("mime/multipart reads model = %q, want %q", value, "backend-model")
		}
		if !bytes.Contains(got, []byte("client-model\r\n")) {
			t.Error("the earlier duplicate model part was not left verbatim")
		}
	})

	t.Run("body without a model field is forwarded unchanged", func(t *testing.T) {
		without := NewMultipartPayload(buildTestMultipart(t, []testPart{
			{name: "prompt", value: "a cat"},
			{name: "input_reference", value: "media-bytes", fileName: "ref.mp4"},
		}), boundary)
		got, err := without.WithModel("backend-model").Marshal()
		if err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		if !bytes.Equal(got, without.body) {
			t.Error("Marshal() changed a body that has no model field")
		}
	})

	t.Run("WithModel leaves the receiver alone", func(t *testing.T) {
		_ = payload.WithModel("backend-model")
		got, err := payload.Marshal()
		if err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		if !bytes.Equal(got, body) {
			t.Error("WithModel mutated the payload it was called on")
		}
	})
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
