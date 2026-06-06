package traindata

import "github.com/tmc/mlx-go-experiments/renderer"

// RendererForModel builds an auto-resolved renderer for tok's model family. The
// family decision is delegated to the renderer package via the tokenizer's
// ModelName (see [renderer.Create] with [renderer.AutoConfig]); a tokenizer
// wired with a known id (e.g. "Qwen/Qwen3-8B") resolves to that family's
// hand-coded renderer, and an unknown id falls back to the default renderer
// driven by chatTemplate.
//
// chatTemplate is consulted only by the default renderer; pass nil to require a
// hand-coded family (the default renderer then errors at render time when the
// model is unknown). SIA target ids that the renderer registry does not list
// exactly (e.g. "Qwen/Qwen3-Next-80B-...", "moonshotai/Kimi-K2.6") resolve to
// the default renderer, so supply a chat template for those.
func RendererForModel(tok renderer.Tokenizer, chatTemplate renderer.ChatTemplate) (renderer.Renderer, error) {
	return renderer.Create(tok, renderer.AutoConfig{}, chatTemplate)
}
