# NLA -> VLM and Multi-Layer Token Study

Date: 2026-06-17

Repo studied: `kitft/natural_language_autoencoders`, GitHub `main` at commit `1b7f13d9d8a37075cd2e5d1604eca57820216ed5`.

Note: I checked `seqam02:/common/users/lc1279/Projects`; `natural_language_autoencoders` was not present there, so I kept seqam02 untouched and studied a local clone under `work/`.

## Executive Summary

1. NLA can be migrated to VLM activations, but released text-only checkpoints should not be expected to work zero-shot on VLM activations. The architecture transfers; the data generation and training target must be VLM-native.

2. LLaVA 1.5 is the easiest first VLM target. Its HF implementation computes image features, projects them to the LLM hidden size, overwrites `<image>` placeholder embeddings, then runs the language model. That matches NLA's existing "activation vector in text-model hidden space" assumption.

3. Qwen3-VL is feasible but more invasive. Its text model is explicitly not pure text-only because DeepStack adds visual features into early decoder hidden states. Extraction must run through the full Qwen3-VL multimodal forward path, not only the unwrapped text model, if we want real VLM activations.

4. Multiple AV injection tokens and multiple AR readout tokens are feasible. Current code assumes one vector per sample, but the training transport can naturally carry `[B, M, d_model]`; the hook/loss/schema need to be generalized.

5. For "optimize activations from multiple layers simultaneously", the recommended design is a multi-layer tuple dataset: one sample stores vectors from layers `[K1, K2, ...]` at the same source position. AV gets one injection token per layer; AR gets one readout token per layer; reward/loss averages or weights per-layer normalized MSE.

## Current Single-Vector Assumptions

AV path:

- `nla/datagen/stage3_build.py` writes one `activation_vector` column with shape `[d_model]`.
- `nla/data_source.py` reads one vector per row and stores it in `sample.metadata["activation_vector"]`.
- `nla/rollout/sft_actor.py` stashes one tensor as `{"nla_activation": tensor[1, d_model]}`.
- `nla/train_actor.py` normalizes that tensor to `self._nla_vectors` and the embedding forward hook injects it.
- `nla/injection.py` requires `vectors.ndim == 2` and consumes one vector per valid marker-token match.
- `nla/rollout/nla_generate.py` builds one prompt, normalizes one vector, injects one token embedding, and sends `input_embeds` to SGLang.

AR path:

- There is no AR injection marker in the current design. AR is suffix-anchored: the critic prompt ends with a stable suffix and loss reads `tokens[-1]`.
- `nla/models.py` returns a value vector at every position, shape `[B, T, d_model]`.
- `nla/loss.py` only indexes the final token per packed sample and compares one predicted vector against one gold vector.
- `nla/reward.py` reconstructs one vector from one generated explanation and computes one MSE reward.

This means "multiple AR special tokens" should be interpreted as "multiple AR readout positions/tokens"; it is a new design, not just increasing a current count.

## VLM Migration

### What to transfer

The lowest-risk target is text-decoder residual activations after multimodal fusion:

- For text tokens in an image-conditioned prompt.
- For visual placeholder tokens after their embeddings have been replaced by projected image features.
- At one or more decoder layers.

This keeps the vector width equal to the VLM text hidden size, so AV/AR can still be causal-LM-based.

### LLaVA 1.5

Feasibility: high.

Why:

- HF `LlavaModel` contains `vision_tower`, `multi_modal_projector`, and `language_model`.
- The forward path embeds text, projects image features, `masked_scatter`s them into image-token positions, and then calls `language_model(inputs_embeds=...)`.
- Therefore, activations inside `language_model` are already `[seq, d_model]` residual-stream vectors, compatible with NLA.

Required changes:

- Add a `LlavaExtractor` using `AutoProcessor` / `LlavaForConditionalGeneration` or `AutoModelForImageTextToText`, not the current raw-text `AutoModelForCausalLM` extractor.
- Extend `arch_adapters.resolve_text_model` to handle nested wrappers like `model.model.language_model` and to preserve/load the top-level `lm_head`.
- Extend dataset rows to carry image path/bytes, prompt text, target token type, and target position metadata.
- Replace stage2's text-only API explanation provider with a multimodal provider or a two-step pipeline: caption/image-region summary plus source prompt/context.

Recommended first experiment:

- LLaVA 1.5 7B, one layer around two-thirds depth, text-token activations in image-question prompts.
- 5k to 20k vectors for a smoke run.
- Compare against a text-only NLA trained on the same base LLM to estimate multimodal distribution shift.

### Qwen3-VL

Feasibility: medium/high, but more engineering risk.

Why:

- HF `Qwen3VLModel` has `visual` and `language_model`.
- It replaces image/video placeholder embeddings with visual embeddings, but also passes `visual_pos_masks` and `deepstack_visual_embeds` into the language model.
- `Qwen3VLTextModel` adds DeepStack visual features into hidden states of early layers. Its docstring says it is not pure text-only.
- Qwen3-VL also uses multimodal RoPE and needs processor-produced `mm_token_type_ids` / grid metadata for correct positions.

Required changes:

- Use the full Qwen3-VL model path for activation extraction. Do not unwrap text-only before extraction.
- Add a Qwen3-VL extractor that calls the processor and passes `input_ids`, `pixel_values`, `image_grid_thw`, `mm_token_type_ids`, and optional video fields.
- Capture decoder hidden states after the multimodal fusion/DeepStack updates.
- For AV/AR training, use either a text-side wrapper around the Qwen3-VL language model or the full conditional-generation model with the vision path disabled. This needs a careful `lm_head`/save-pretrained adapter.
- Validate the serving backend. Current NLA RL depends on SGLang `input_embeds`; Qwen3-VL text-side serving may need HF/vLLM fallback or an SGLang compatibility check.

Recommended first experiment:

- Get extraction correct before training anything: run a tiny batch, capture target-layer hidden states, verify text-only extraction is different from full multimodal extraction on image-token and image-conditioned text-token positions.
- Then train a small AR-only model first; AR FVE is the fastest signal that the target distribution is learnable.

## Multiple Special Tokens / Multiple Layers

### Data shape

Change from:

```text
activation_vector: [d_model]
```

to:

```text
activation_vectors: [num_targets, d_model]
activation_layers: [K1, K2, ...]
```

Keep the existing `activation_vector` path for backward compatibility when `num_targets == 1`.

Stage0 should register hooks on all requested layers in one forward pass and write one tuple per sampled source position. For a small number of layers, multiple hooks are cheaper than `output_hidden_states=True`.

### AV: multiple injection tokens

Prompt template example:

```text
<concept>
<layer id="8">{inj_0}</layer>
<layer id="16">{inj_1}</layer>
<layer id="24">{inj_2}</layer>
</concept>
```

Implementation:

- Store token metadata as a list of injection sites in the sidecar.
- Prefer distinct single-token marker chars per layer. Repeating the same char can work by count/order, but distinct chars make failures easier to diagnose.
- Normalize per target/layer, not with one global `injection_scale`.
- Generalize `inject_at_marked_positions` to accept `[B, M, d]` and `M` token specs, then flatten in microbatch/sample order.
- Preserve `cp_size == 1`; context parallel still breaks neighbor checks.

Minimal hook change:

- Current flattened assumption is `[B, d]`.
- New assumption becomes `[B, M, d]`; expected marker count is `B * M`.
- Marker order in the prompt must match `activation_layers`.

### AR: multiple readout tokens

Current AR reads only the last token. For multi-layer AR, add multiple suffix/readout tokens:

```text
Summary of the following text: <text>{explanation}</text>
<reconstruct><L8>{ro_0}</L8><L16>{ro_1}</L16><L24>{ro_2}</L24></reconstruct>
```

Two viable model variants:

1. Simpler baseline: keep one truncated backbone up to `Kmax + 1`, use multiple readout positions at the final hidden state, and attach either one shared `Linear(d,d)` head plus layer-id tokens or one head per target layer.

2. More faithful multi-layer AR: keep hooks at each target layer inside the AR backbone and predict layer `Ki` from the hidden state at readout token `i` after decoder block `Ki`. This better matches the current single-layer design, but needs a custom forward path and more memory bookkeeping.

Recommended first implementation: option 1 with per-layer heads. It is easier to train, easier to checkpoint, and enough to answer whether multi-readout AR works. If it underperforms separate per-layer AR models, try option 2.

### Loss and reward

Use per-layer normalized MSE:

```text
loss_i = mse(normalize(pred_i, mse_scale_i), normalize(gold_i, mse_scale_i))
loss = mean_i(weight_i * loss_i)
reward = -loss
```

Important details:

- Compute FVE baselines per layer.
- Log per-layer MSE/FVE, not only the mean; otherwise one easy layer can hide a failed layer.
- Consider inverse-variance or equal-FVE weighting if late layers dominate.
- Keep the explanation length budget larger for multi-layer training; the text must carry more information.

## Implementation Plan

1. Add `NLAConfig.activation_layers`, `injection_sites`, `readout_sites`, and per-layer scales in sidecar schema.
2. Add multi-layer datagen in `stage0_extract`: `--layer-indices 8,16,24`, multi-hook capture, nested fixed-size Arrow column.
3. Update `NLADataSource` to read `[M,d]` matrices efficiently and store tensors as `[1,M,d]`.
4. Generalize AV injection for `[B,M,d]`.
5. Update `nla_generate` and SFT actor rollout to pass multi-vector metadata.
6. Implement `MultiLayerCriticModel` and `nla_multilayer_critic_loss`.
7. Update reward to compute multi-layer MSE from generated explanation.
8. Run ablations:
   - single-layer NLA vs multi-layer NLA,
   - shared vs per-layer AR heads,
   - 2 vs 3 vs 4 injected/readout tokens,
   - text-only vs VLM-native target activations.

## Key Risks

- Zero-shot transfer from released text-only checkpoints to VLM activations will likely be weak.
- Qwen3-VL extraction can be wrong if DeepStack or M-RoPE metadata is skipped.
- Multiple layers increase the amount of information the AV explanation must encode; too many layers may reduce interpretability or make AR solve only the easiest targets.
- Distinct single-token markers must be verified against each tokenizer. Adding new special tokens is possible but changes the embedding matrix and checkpoint compatibility.
- SGLang `input_embeds` compatibility must be verified for any new text-side Qwen3-VL wrapper.

## Sources

- NLA repo: https://github.com/kitft/natural_language_autoencoders
- NLA README and design docs in local clone: `work/natural_language_autoencoders/README.md`, `docs/design.md`
- LLaVA HF implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
- LLaVA config: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/configuration_llava.py
- Qwen3-VL HF implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
- Qwen3-VL config: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py

