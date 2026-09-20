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
	"bufio"
	"bytes"
	"mime"
	"net/textproto"
)

// MultipartPayload is a multipart/form-data request body retained as received.
// Rewriting a form field replaces the value bytes of the single part that carries it,
// so the parts the router does not interpret, media uploads in particular, keep the
// bytes, the headers, and the boundary lines the client sent. A payload with nothing
// to rewrite marshals to the received body itself.
type MultipartPayload struct {
	body     []byte
	boundary string
	model    string
	rewrite  bool
}

// NewMultipartPayload wraps a received multipart/form-data body together with the
// boundary its content type declared.
func NewMultipartPayload(body []byte, boundary string) MultipartPayload {
	return MultipartPayload{body: body, boundary: boundary}
}

func (MultipartPayload) isRequestPayload()         {}
func (MultipartPayload) IsParsed() bool            { return false }
func (MultipartPayload) AsMap() (PayloadMap, bool) { return nil, false }

// WithModel returns the payload with the value of its model part replaced. The part
// replaced is the last form field named model, which is the one the backend reads.
func (p MultipartPayload) WithModel(model string) MultipartPayload {
	p.model = model
	p.rewrite = true
	return p
}

// Marshal returns the body with the rewritten value spliced in. The received body is
// returned unchanged when there is nothing to rewrite, when it carries no model form
// field, or when its framing is not the canonical CRLF form this can address.
func (p MultipartPayload) Marshal() ([]byte, error) {
	if !p.rewrite {
		return p.body, nil
	}
	start, end, ok := lastFormFieldValue(p.body, p.boundary, "model")
	if !ok {
		return p.body, nil
	}
	out := make([]byte, 0, len(p.body)+len(p.model)-(end-start))
	out = append(out, p.body[:start]...)
	out = append(out, p.model...)
	return append(out, p.body[end:]...), nil
}

// lastFormFieldValue returns the byte range holding the value of the last form field
// named field, and whether the body has one. Offsets index the received body, so a
// replacement can be spliced into it without re-encoding any part. A body whose
// framing is not the canonical CRLF form reports no field, because a replacement
// spliced into a part located by a guess would land in a media part instead.
func lastFormFieldValue(body []byte, boundary, field string) (int, int, bool) {
	open := []byte("--" + boundary + "\r\n")
	separator := []byte("\r\n--" + boundary)
	if !bytes.HasPrefix(body, open) {
		return 0, 0, false
	}
	start, end, found := 0, 0, false
	for pos := len(open); ; {
		header := bytes.Index(body[pos:], []byte("\r\n\r\n"))
		if header < 0 {
			return 0, 0, false
		}
		valueStart := pos + header + 4
		next := bytes.Index(body[valueStart:], separator)
		if next < 0 {
			return 0, 0, false
		}
		if formFieldName(body[pos:pos+header+4]) == field {
			found, start, end = true, valueStart, valueStart+next
		}
		pos = valueStart + next + len(separator)
		switch {
		case bytes.HasPrefix(body[pos:], []byte("--")):
			return start, end, found // the closing delimiter ends the body
		case bytes.HasPrefix(body[pos:], []byte("\r\n")):
			pos += 2
		default:
			return 0, 0, false // the boundary string occurs inside a part value
		}
	}
}

// formFieldName returns the name of a form field part, and "" for a file upload or a
// part that is not form-data. The disposition rules match mime/multipart, so the part
// matched here is the part the parser reads a field from. The header block is passed
// with the blank line that terminates it, which is where header reading stops.
func formFieldName(header []byte) string {
	h, err := textproto.NewReader(bufio.NewReader(bytes.NewReader(header))).ReadMIMEHeader()
	if err != nil {
		return ""
	}
	disposition, params, err := mime.ParseMediaType(h.Get("Content-Disposition"))
	if err != nil || disposition != "form-data" || params["filename"] != "" {
		return ""
	}
	return params["name"]
}
