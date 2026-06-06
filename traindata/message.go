package traindata

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/tmc/mlx-go-experiments/renderer"
)

// ErrNotTrajectory reports that a trajectory is not a JSON array of messages —
// typically an error object recorded for a failed sample. Callers skip these,
// mirroring the reference treating non-list trajectories as failed.
var ErrNotTrajectory = errors.New("traindata: trajectory is not a message list")

// MessagesFromTrajectory converts one SIA trajectory into renderer messages. A
// trajectory is a JSON array of OpenAI-style messages
// ([{"role","content",...}, ...]); MessagesFromTrajectory tolerates the shapes
// SIA records: string or structured-list content, assistant tool calls (with
// string or object arguments), reasoning content, and tool-role replies.
//
// It returns [ErrNotTrajectory] when raw is not a JSON array (an error object
// or other non-list), so the caller can skip it cleanly, and a wrapped error
// for malformed JSON. A well-formed empty array returns (nil, nil).
func MessagesFromTrajectory(raw json.RawMessage) ([]renderer.Message, error) {
	if !isList(raw) {
		return nil, ErrNotTrajectory
	}
	var entries []trajMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("traindata: decode trajectory: %w", err)
	}
	msgs := make([]renderer.Message, 0, len(entries))
	for i, e := range entries {
		m, err := e.toMessage()
		if err != nil {
			return nil, fmt.Errorf("traindata: message %d: %w", i, err)
		}
		msgs = append(msgs, m)
	}
	if len(msgs) == 0 {
		return nil, nil
	}
	return msgs, nil
}

// trajMessage is the on-the-wire SIA/OpenAI trajectory message. It is a subset
// of [renderer.Message]; unmodeled keys are ignored. Content may be a string or
// a list of parts, so it is held as raw JSON and resolved by [trajMessage.toMessage].
type trajMessage struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	ReasoningContent string          `json:"reasoning_content"`
	ToolCalls        []trajToolCall  `json:"tool_calls"`
	ToolCallID       string          `json:"tool_call_id"`
	Name             string          `json:"name"`
}

// trajToolCall mirrors the OpenAI tool-call shape. Arguments stays raw so a
// JSON-string form (the canonical OpenAI encoding) round-trips verbatim while
// an object form is preserved as a decoded map.
type trajToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

// trajPart is one element of a structured content list.
type trajPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Thinking string `json:"thinking"`
}

func (e trajMessage) toMessage() (renderer.Message, error) {
	m := renderer.Message{
		Role:             e.Role,
		ReasoningContent: e.ReasoningContent,
		ToolCallID:       e.ToolCallID,
		Name:             e.Name,
	}
	if err := resolveContent(e.Content, &m); err != nil {
		return renderer.Message{}, err
	}
	for _, tc := range e.ToolCalls {
		args, err := decodeArguments(tc.Function.Arguments)
		if err != nil {
			return renderer.Message{}, fmt.Errorf("tool call %q arguments: %w", tc.Function.Name, err)
		}
		m.ToolCalls = append(m.ToolCalls, renderer.ToolCall{
			ID:   tc.ID,
			Type: tc.Type,
			Function: renderer.ToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: args,
			},
		})
	}
	return m, nil
}

// resolveContent sets Content or Parts from the raw content value, which may be
// a JSON string, a list of parts, or null/absent (an assistant turn that only
// makes tool calls).
func resolveContent(raw json.RawMessage, m *renderer.Message) error {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		m.Content = s
		return nil
	}
	var parts []trajPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return fmt.Errorf("content is neither a string nor a parts list: %w", err)
	}
	for _, p := range parts {
		m.Parts = append(m.Parts, renderer.ContentPart{
			Type:     p.Type,
			Text:     p.Text,
			Thinking: p.Thinking,
		})
	}
	return nil
}

// decodeArguments returns the tool-call arguments in the form the renderer
// expects: the verbatim string for a JSON-string encoding (OpenAI canonical),
// or a decoded map for an inline object. A nil/empty value yields nil.
func decodeArguments(raw json.RawMessage) (any, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, nil
	}
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, err
	}
	return obj, nil
}

// isList reports whether raw is a JSON array. It mirrors sia.isList without
// exporting that predicate from the faithful core.
func isList(raw json.RawMessage) bool {
	var arr []json.RawMessage
	return json.Unmarshal(raw, &arr) == nil
}
