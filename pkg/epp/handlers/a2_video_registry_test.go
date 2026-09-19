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

package handlers

import (
	"testing"

	"github.com/go-logr/logr"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/passthrough"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmhttp"
)

// defaultRegistry mirrors the parser list the EPP installs when the
// configuration names none (loader.ensureParsers): openai, anthropic,
// vllmhttp, then passthrough last.
func defaultRegistry() *ParserRegistry {
	return NewParserRegistry([]fwkrh.Parser{
		openai.NewOpenAIParser(),
		anthropic.NewAnthropicParser(),
		vllmhttp.NewVllmHTTPParser(),
		passthrough.NewPassthroughParser(),
	}, logr.Discard())
}

// A2-U01: the video endpoints must resolve to the OpenAI parser rather than
// falling through to the passthrough fallback, and /v1/videos/sync must not be
// shadowed by the shorter /v1/videos claim.
func TestA2U01RegistryResolvesVideoPaths(t *testing.T) {
	registry := defaultRegistry()

	tests := []struct {
		path     string
		wantType string
	}{
		{path: "/v1/videos", wantType: openai.OpenAIParserType},
		{path: "/v1/videos/sync", wantType: openai.OpenAIParserType},
		{path: "/openai/v1/videos", wantType: openai.OpenAIParserType},
		{path: "/v1/videos/", wantType: openai.OpenAIParserType},
		// Neighbouring routes must keep resolving to the OpenAI parser.
		{path: "/v1/images/edits", wantType: openai.OpenAIParserType},
		{path: "/v1/images/generations", wantType: openai.OpenAIParserType},
		{path: "/v1/chat/completions", wantType: openai.OpenAIParserType},
		// Non-OpenAI surfaces keep their own parsers.
		{path: "/v1/messages", wantType: anthropic.AnthropicParserType},
		{path: "/inference/v1/generate", wantType: vllmhttp.VllmHTTPParserType},
		// Unclaimed traffic still reaches the passthrough fallback. The job
		// read/download routes are GET-only and carry no body to parse, so they
		// deliberately stay on passthrough rather than being claimed here.
		{path: "/v1/unknown/thing", wantType: passthrough.PassthroughParserType},
		{path: "/v1/videos/video-123", wantType: passthrough.PassthroughParserType},
		{path: "/v1/videos/video-123/content", wantType: passthrough.PassthroughParserType},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			parser, err := registry.Resolve(tt.path)
			if err != nil {
				t.Fatalf("Resolve(%q) error = %v", tt.path, err)
			}
			if got := parser.TypedName().Type; got != tt.wantType {
				t.Errorf("Resolve(%q) type = %q, want %q", tt.path, got, tt.wantType)
			}
		})
	}
}

// Longest-suffix matching means the /v1/videos claim must not steal a path that
// a more specific existing claim owns, even when that path sits under /v1/videos.
func TestA2U01VideoClaimDoesNotShadowLongerSuffixes(t *testing.T) {
	registry := defaultRegistry()

	// /inference/v1/generate is a longer suffix than videos and must keep winning,
	// while plain /v1/videos stays with the OpenAI parser.
	tests := []struct {
		path     string
		wantType string
	}{
		{path: "/inference/v1/generate", wantType: vllmhttp.VllmHTTPParserType},
		{path: "/v1/videos/sync", wantType: openai.OpenAIParserType},
		{path: "/v1/videos", wantType: openai.OpenAIParserType},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			parser, err := registry.Resolve(tt.path)
			if err != nil {
				t.Fatalf("Resolve(%q) error = %v", tt.path, err)
			}
			if got := parser.TypedName().Type; got != tt.wantType {
				t.Errorf("Resolve(%q) type = %q, want %q", tt.path, got, tt.wantType)
			}
		})
	}
}

// A2-U06: the video support must not cause a second OpenAI parser to be
// registered, and the OpenAI parser must be registered exactly once.
func TestA2U06OpenAIParserRegisteredOnce(t *testing.T) {
	registry := defaultRegistry()

	seen := 0
	for _, p := range registry.Parsers() {
		if p.TypedName().Type == openai.OpenAIParserType {
			seen++
		}
	}
	if seen != 1 {
		t.Errorf("openai parser registered %d times, want exactly 1", seen)
	}
}
