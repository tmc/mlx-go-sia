package traindata_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/tmc/mlx-go-experiments/renderer"
	"github.com/tmc/mlx-go-sia/traindata"
)

func TestMessagesFromTrajectory(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		want    []renderer.Message
		wantErr error // sentinel to errors.Is against, or nil
		anyErr  bool  // non-sentinel error expected
	}{
		{
			name: "string content",
			raw:  `[{"role":"user","content":"hi"}]`,
			want: []renderer.Message{{Role: "user", Content: "hi"}},
		},
		{
			name: "parts content text and thinking",
			raw:  `[{"role":"assistant","content":[{"type":"text","text":"a"},{"type":"thinking","thinking":"b"}]}]`,
			want: []renderer.Message{{Role: "assistant", Parts: []renderer.ContentPart{
				{Type: "text", Text: "a"},
				{Type: "thinking", Thinking: "b"},
			}}},
		},
		{
			name: "image part ignored shape preserved",
			raw:  `[{"role":"user","content":[{"type":"text","text":"see"},{"type":"image"}]}]`,
			want: []renderer.Message{{Role: "user", Parts: []renderer.ContentPart{
				{Type: "text", Text: "see"},
				{Type: "image"},
			}}},
		},
		{
			name: "tool call string args round-trip verbatim",
			raw:  `[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]}]`,
			want: []renderer.Message{{Role: "assistant", ToolCalls: []renderer.ToolCall{
				{ID: "c1", Type: "function", Function: renderer.ToolCallFunction{Name: "f", Arguments: `{"x":1}`}},
			}}},
		},
		{
			name: "tool call object args decoded to map",
			raw:  `[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"x":1}}}]}]`,
			want: []renderer.Message{{Role: "assistant", ToolCalls: []renderer.ToolCall{
				{Function: renderer.ToolCallFunction{Name: "f", Arguments: map[string]any{"x": float64(1)}}},
			}}},
		},
		{
			name: "tool role with id and name",
			raw:  `[{"role":"tool","content":"42","tool_call_id":"c1","name":"f"}]`,
			want: []renderer.Message{{Role: "tool", Content: "42", ToolCallID: "c1", Name: "f"}},
		},
		{
			name: "reasoning content",
			raw:  `[{"role":"assistant","content":"x","reasoning_content":"because"}]`,
			want: []renderer.Message{{Role: "assistant", Content: "x", ReasoningContent: "because"}},
		},
		{
			name:    "error object is not a trajectory",
			raw:     `{"error":"boom"}`,
			wantErr: traindata.ErrNotTrajectory,
		},
		{
			name:   "malformed json wrapped error",
			raw:    `[{"role":"user"`,
			anyErr: true,
		},
		{
			name: "empty array yields nil",
			raw:  `[]`,
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := traindata.MessagesFromTrajectory(json.RawMessage(tt.raw))
			switch {
			case tt.wantErr != nil:
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("err = %v, want errors.Is %v", err, tt.wantErr)
				}
				return
			case tt.anyErr:
				if err == nil {
					t.Fatal("expected an error, got nil")
				}
				return
			default:
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
			if !messagesEqual(got, tt.want) {
				t.Errorf("got  %#v\nwant %#v", got, tt.want)
			}
		})
	}
}

// messagesEqual compares messages by their meaningful fields (Arguments is an
// any, so compare via its rendered shape with a value-equality fallback).
func messagesEqual(a, b []renderer.Message) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Role != b[i].Role || a[i].Content != b[i].Content ||
			a[i].ReasoningContent != b[i].ReasoningContent ||
			a[i].ToolCallID != b[i].ToolCallID || a[i].Name != b[i].Name {
			return false
		}
		if len(a[i].Parts) != len(b[i].Parts) {
			return false
		}
		for j := range a[i].Parts {
			if a[i].Parts[j] != b[i].Parts[j] {
				return false
			}
		}
		if len(a[i].ToolCalls) != len(b[i].ToolCalls) {
			return false
		}
		for j := range a[i].ToolCalls {
			ta, tb := a[i].ToolCalls[j], b[i].ToolCalls[j]
			if ta.ID != tb.ID || ta.Type != tb.Type || ta.Function.Name != tb.Function.Name {
				return false
			}
			if !argsEqual(ta.Function.Arguments, tb.Function.Arguments) {
				return false
			}
		}
	}
	return true
}

func argsEqual(a, b any) bool {
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	return string(ja) == string(jb)
}
