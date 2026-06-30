# VLM-NLA Hypothesis Validation Pilot Results

Date: 2026-06-30

This note records the first pilot runs launched from the hypothesis-validation protocol. These runs are useful for debugging the pipeline and checking early signals, but they are not yet the final confirmatory experiments because they use one split/seed and a 128-train / 64-test pilot scale.

## Setup

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Environment: `seqam02` / `seqamgpu02`, RTX A6000 GPUs
- Main data: MSCOCO val2017 object annotations
- Split: category-balanced image-disjoint manifest, 128 train / 32 val / 64 test
- Label: short category + coarse region only, with full COCO caption removed
- AV: LoRA + activation adapter, 8 injected visual placeholder tokens unless noted
- Metrics:
  - `sensitivity_delta`: `NLL(shuffled activation, correct text) - NLL(matched activation, correct text)`. Higher means the AV depends more on the matched activation.
  - `nll_topK`: candidate-ranking accuracy by matched activation NLL.
  - `gain_topK`: candidate-ranking accuracy after subtracting reference-activation NLL.

## H1: Local Image-Token Groups vs Single Token

Pilot runs:

| Run | Target | Sensitivity Delta | NLL Top1 | NLL Top5 | Gain Top1 | Gain Top5 |
|---|---|---:|---:|---:|---:|---:|
| `H1_PILOT_Q3VL_COCO_L15_CENTER_N128_SEED4201` | single center token | 0.2178 | 0.3281 | 0.5312 | 0.4688 | 0.6250 |
| `H1_PILOT_Q3VL_COCO_L15_BBOX4_N128_SEED4201` | bbox mean, max 4 tokens | 0.2339 | 0.5156 | 0.6875 | 0.6250 | 0.7812 |
| `H1_PILOT_Q3VL_COCO_L15_BBOX8_N128_SEED4201` | bbox mean, max 8 tokens | 0.3410 | 0.4062 | 0.7188 | 0.5781 | 0.7969 |
| `H1_PILOT_Q3VL_COCO_L15_BBOX16_N128_SEED4201` | bbox mean, max 16 tokens | 0.2884 | 0.3906 | 0.7500 | 0.6562 | 0.8750 |

Pilot interpretation:

- H1 is supported at pilot scale.
- The single center token is consistently weakest.
- Local groups improve both activation sensitivity and candidate retrieval.
- `bbox8` has the highest sensitivity delta, while `bbox16` has the best top5 retrieval. This suggests there may be a tradeoff: slightly larger regions improve recall, but may dilute precise local semantics.
- Next confirmatory step: repeat with 1024/256/512 and at least 3 seeds, plus random same-area region and segmentation-mask controls.

## H2: Layer Semantics and Capacity Control

Pilot runs:

| Run | Layer / Setting | Sensitivity Delta | NLL Top1 | NLL Top5 | Gain Top1 | Gain Top5 |
|---|---|---:|---:|---:|---:|---:|
| `H2_PILOT_Q3VL_COCO_L06_BBOX8_N128_SEED4301` | L6, 8 tokens | 0.3076 | 0.5469 | 0.7969 | 0.6719 | 0.8438 |
| `H2_PILOT_Q3VL_COCO_L10_BBOX8_N128_SEED4301` | L10, 8 tokens | 0.3283 | 0.5781 | 0.7812 | 0.6875 | 0.8281 |
| `H2_PILOT_Q3VL_COCO_L15_BBOX8_N128_SEED4301` | L15, 8 tokens | 0.3472 | 0.5938 | 0.7656 | 0.6562 | 0.8594 |
| `H2_PILOT_Q3VL_COCO_L20_BBOX8_N128_SEED4301` | L20, 8 tokens | 0.2848 | 0.4219 | 0.6406 | 0.5938 | 0.8281 |
| `H2_PILOT_Q3VL_COCO_L28_BBOX8_N128_SEED4301` | L28, 8 tokens | 0.1764 | 0.3750 | 0.5938 | 0.5312 | 0.7344 |
| `H2_PILOT_Q3VL_COCO_L34_BBOX8_N128_SEED4301` | L34, 8 tokens | -0.0000 | 0.0156 | 0.0781 | 0.0156 | 0.1094 |
| `H2_PILOT_Q3VL_COCO_L15_BBOX8_12TOK_N128_SEED4301` | L15, 12 tokens | 0.3512 | 0.4219 | 0.6875 | 0.6250 | 0.8281 |

Pilot interpretation:

- H2 is supported at pilot scale for single-layer differences.
- Middle layers are much more interpretable for local object tokens than very late layer L34.
- L15 gives the best sensitivity/top5 balance in this pilot; L10 is also strong.
- The 12-token L15 capacity control does not beat the 8-token L15 baseline on topK retrieval, even though its sensitivity delta is slightly higher. This suggests extra injected tokens alone are not a guaranteed improvement.
- Multi-layer AV itself is not yet implemented in this pilot. The next H2 step is to train true multi-layer activation tuples such as L10+L15+L20 and compare against this L15 12-token capacity control.

## H3: AR Closed-Loop Pilot

AR baseline: hashed word n-gram features from explanation text + dual ridge regression to reconstruct the original activation.

### L15 bbox8

| Text Source | FVE vs Mean | Cosine | Retrieval Top1 | Retrieval Top5 |
|---|---:|---:|---:|---:|
| mean activation baseline | 0.0000 | 0.8858 | 0.0156 | 0.0781 |
| matched gold text | 0.0741 | 0.8948 | 0.0312 | 0.2031 |
| shuffled gold text | -0.0354 | 0.8816 | 0.0156 | 0.0625 |
| AV top1 text | 0.0539 | 0.8922 | 0.0312 | 0.1719 |

### L15 bbox16

| Text Source | FVE vs Mean | Cosine | Retrieval Top1 | Retrieval Top5 |
|---|---:|---:|---:|---:|
| mean activation baseline | 0.0000 | 0.9073 | 0.0156 | 0.0781 |
| matched gold text | 0.0927 | 0.9164 | 0.0625 | 0.2031 |
| shuffled gold text | -0.0384 | 0.9037 | 0.0156 | 0.0625 |
| AV top1 text | 0.0514 | 0.9125 | 0.0469 | 0.1250 |

Pilot interpretation:

- H3 is weakly supported at pilot scale.
- Matched text reconstructs activation better than shuffled text, so the explanation contains reconstructable activation information.
- AV top1 text also reconstructs better than the mean baseline, but weaker than gold text.
- Retrieval remains low, so this ridge AR should be treated as a lower-bound diagnostic, not a full NLA AR.
- Next H3 step: train a neural AR with LoRA and evaluate round-trip `activation -> AV text -> AR activation`.

## Current Status of H4 and H5

H4 and H5 have not yet been validated.

- H4 needs a hallucination-token pipeline: POPE/CHAIR-style hallucination labeling, high-impact image-token discovery by ablation or patching, and then VLM-NLA scoring of those tokens.
- H5 needs counterfactual visual pairs: synthetic shape/diagram/COCO-edit pairs where the visual evidence changes while the prompt stays fixed, followed by explanation-change scoring.

These should not be inferred from H1-H3. They require separate causal/counterfactual experiments.

## Immediate Next Steps

1. Scale H1 to 1024/256/512 with 3 seeds.
2. Implement random same-area region and segmentation-mask controls for H1.
3. Implement true multi-layer AV for H2.
4. Upgrade H3 from ridge AR to neural AR.
5. Implement H4 hallucination-token intervention.
6. Implement H5 counterfactual shortcut evaluation.
