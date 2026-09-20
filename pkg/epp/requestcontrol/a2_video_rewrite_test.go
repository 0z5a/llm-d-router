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

package requestcontrol

import (
	"bytes"
	"context"
	"errors"
	"io"
	"mime/multipart"
	"net/textproto"
	"testing"

	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
)

// readPartValue returns the bytes a forwarded body carries for a named part.
func readPartValue(t *testing.T, body []byte, boundary, name string) []byte {
	t.Helper()
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			t.Fatalf("part %q is not in the forwarded body", name)
		}
		require.NoError(t, err)
		if part.FormName() != name {
			continue
		}
		value, err := io.ReadAll(part)
		require.NoError(t, err)
		return value
	}
}

// A2-U06b: the multipart model rewrite has to reach the bytes the director forwards.
// repackage is the step that replaces the handler body, so asserting on
// reqCtx.Request.RawBody is what shows the rewrite lands on the wire.
func TestA2U06b_RepackageForwardsRewrittenMultipartBody(t *testing.T) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	require.NoError(t, w.WriteField("prompt", "a cinematic tracking shot"))
	require.NoError(t, w.WriteField("model", "A2-client-model"))
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", `form-data; name="input_reference"; filename="reference.mp4"`)
	header.Set("Content-Type", "video/mp4")
	mediaPart, err := w.CreatePart(header)
	require.NoError(t, err)
	media := bytes.Repeat([]byte{0x00, 0xff, '\r', '\n', '-', '-'}, 1<<16)
	_, err = mediaPart.Write(media)
	require.NoError(t, err)
	require.NoError(t, w.Close())
	received := buf.Bytes()

	parser := openai.NewOpenAIParser()
	parsed, err := parser.ParseRequest(context.Background(), received, map[string]string{
		":path":        "/v1/videos",
		"content-type": w.FormDataContentType(),
	})
	require.NoError(t, err)
	require.Equal(t, "A2-client-model", parsed.Body.Model)

	parsed.Body.Payload, err = parser.RewriteModelName(parsed.Body.Payload.(fwkrh.MarshalablePayload), "a2-backend/model")
	require.NoError(t, err)
	parsed.Body.Mutated = true

	reqCtx := &handlers.RequestContext{Request: &handlers.Request{RawBody: received}}
	dir := &Director{}
	require.NoError(t, dir.repackage(context.Background(), reqCtx, parsed.Body))

	forwarded := reqCtx.Request.RawBody
	require.Equal(t, len(received)+len("a2-backend/model")-len("A2-client-model"), len(forwarded))
	require.Equal(t, len(forwarded), reqCtx.RequestSize)
	require.Contains(t, string(forwarded), "name=\"model\"\r\n\r\na2-backend/model")
	require.Equal(t, media, readPartValue(t, forwarded, w.Boundary(), "input_reference"))

	// A body the router did not change is forwarded as it arrived.
	unchanged := &handlers.RequestContext{Request: &handlers.Request{RawBody: received}}
	parsed.Body.Mutated = false
	require.NoError(t, dir.repackage(context.Background(), unchanged, parsed.Body))
	require.Equal(t, received, unchanged.Request.RawBody)
	require.Equal(t, len(received), unchanged.RequestSize)
}
