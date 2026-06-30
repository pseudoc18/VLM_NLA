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
| `H2_PILOT_Q3VL_COCO_L10_L15_L20_BBOX8_CONCAT8_N128_SEED4801` | L10+L15+L20 concat, 8 tokens | 0.1909 | 0.3438 | 0.6406 | 0.4219 | 0.7656 |

Pilot interpretation:

- H2 is supported at pilot scale for single-layer differences.
- Middle layers are much more interpretable for local object tokens than very late layer L34.
- L15 gives the best sensitivity/top5 balance in this pilot; L10 is also strong.
- The 12-token L15 capacity control does not beat the 8-token L15 baseline on topK retrieval, even though its sensitivity delta is slightly higher. This suggests extra injected tokens alone are not a guaranteed improvement.
- A first true multi-layer concat AV is now implemented. It concatenates L10, L15, and L20 object-region activations into a 12288-dimensional vector, then learns a projection into the same 8 AV injection tokens.
- This naive concat run does **not** outperform single-layer L15 or L10. It has lower sensitivity and lower ranking accuracy than the best single-layer conditions.
- This is evidence against the simple version of "more layers automatically help." The next H2 design should use per-layer token blocks, gated/low-rank fusion, or layer-specific losses instead of one large undifferentiated concat projection.

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

## H4: Hallucination / Shortcut Token Pilot

H4 asks whether local image-token explanations can help diagnose hallucinated object claims. Two forced-choice absent-object probes were run with the trained L15 bbox8 AV.

Method:

- For each COCO image, choose categories absent from the ground-truth annotations.
- Ask Qwen3-VL a forced-choice question: whether that category exists in the image.
- A false-positive "Yes" preference is treated as an absent-object hallucination probe.
- For hallucinated probes, mask each large visible object region and measure whether the false-positive yes-margin drops.
- For the region with the largest margin drop, use the trained AV to compare two explanations: the true visible object label vs the hallucinated absent label.

Pilot runs:

| Run | Absent Sampling | Images | Probes | False-Positive Probes | Rate | Mean Top-Region Margin Drop | Positive Drop Fraction | AV Prefers Hallucinated |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `H4_PILOT_HALLUCINATION_SHORTCUT_Q3VL_L15_BBOX8_N16_SEED4701` | random absent categories | 16 | 48 | 1 | 0.0208 | 0.0046 | 1.0000 | 0.0000 |
| `H4_PILOT_HARDNEG_SHORTCUT_Q3VL_L15_BBOX8_N32_SEED4702` | COCO co-occurrence hard negatives | 32 | 160 | 7 | 0.0438 | 0.0141 | 0.7143 | 0.1429 |

Metric interpretation:

- `yes_margin = NLL(No) - NLL(Yes)`. Positive means the model prefers "Yes".
- `false-positive probe` means the category is absent in COCO annotations but the model prefers "Yes".
- `top-region margin drop` means how much the false-positive yes-margin decreases after masking the most influential visible object region. Positive is evidence that the visible region supports the hallucinated claim.
- `AV prefers hallucinated` means the AV assigns lower NLL to the hallucinated object explanation than to the true visible object explanation for that region.

Pilot interpretation:

- Random absent categories are too easy; Qwen3-VL rarely says "Yes".
- Co-occurrence hard negatives produce more false positives, but the margins are still small. This is a useful pilot setup, not yet a strong hallucination benchmark.
- In 5/7 hard-negative false positives, masking the top visible region reduces the false-positive margin, so there is some shortcut-like visual support.
- The AV usually still prefers the true visible object label over the hallucinated label. That is encouraging for faithfulness: the AV is not merely copying the model's final hallucinated answer.
- H4 remains only weakly supported. The next version should use POPE/CHAIR-style caption hallucination, larger hard-negative pools, and object-level image-token attribution beyond the few largest COCO boxes.

## H5: Counterfactual Mask Pilot

H5 asks whether the AV explanation changes when the visual evidence for the target local token group is removed.

Method:

- Use the trained H1 L15 bbox8 AV.
- For each test row, mask the COCO target object's bbox with the image mean color.
- Re-extract the same target-region activation from the edited image.
- Score whether the original object explanation is still preferred by the AV.

Pilot run:

| Run | Rows | Candidates | Original Mean Rank | Edited Mean Rank | Mean Rank Delta | Original Top1 | Edited Same-Label Top1 | Original Top5 | Edited Same-Label Top5 | Mean Correct NLL Increase | Correct NLL Increased | Top Response Changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `H5_PILOT_COUNTERFACTUAL_MASK_Q3VL_L15_BBOX8_N32_SEED4601` | 32 | 64 | 6.8438 | 21.0312 | 14.1875 | 0.4062 | 0.1875 | 0.7500 | 0.3125 | 0.1402 | 0.8438 | 0.6562 |

Metric interpretation:

- Lower rank is better. Rank 1 means the original correct explanation has the lowest NLL among candidates.
- `mean_rank_delta = edited_rank - original_rank`; positive means masking made the original explanation less preferred.
- `correct_nll_increase` is the NLL increase for the original correct explanation after masking; positive means the original explanation became harder for the AV to support.
- `top_response_changed` means the best-scoring explanation changed after the visual edit.

Pilot interpretation:

- H5 is supported at pilot scale.
- Masking the target object strongly hurts the original explanation: mean rank worsens by 14.19 places, top5 drops from 75.0% to 31.25%, and 84.38% of rows have increased correct NLL.
- This suggests the AV is grounded in local visual-token activation, not only in the response template.
- This is still a coarse counterfactual: mean-color masking can introduce artifacts, and it tests removal rather than clean object replacement. The next step is to use synthetic controlled edits and real image-edit pairs.

## Immediate Next Steps

1. Scale H1/H2/H5 to 1024/256/512 with at least 3 seeds.
2. Add random same-area region and segmentation-mask controls for H1.
3. Replace naive H2 concat with per-layer AV token blocks, gated/low-rank fusion, and capacity-matched controls.
4. Upgrade H3 from ridge AR to neural AR and evaluate the full round trip `activation -> AV text -> AR activation`.
5. Upgrade H4 to POPE/CHAIR-style caption hallucination and larger co-occurrence hard-negative probes.
6. Upgrade H5 from mean-color masking to controlled synthetic edits and real counterfactual image pairs.
