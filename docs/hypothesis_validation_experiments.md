# VLM-NLA Hypothesis Validation Experiment Plan

This document designs the next-stage experiments for validating the current VLM-NLA hypotheses. It is written as a protocol: each experiment should be reproducible from the recorded configuration, command logs, data split, metrics, and qualitative artifacts.

## How to Use This Protocol

Treat this document as a pre-registration-style plan for the next stage of the project.

- **Smoke runs** check whether extraction, training, and evaluation scripts work. They can use small splits such as `128/64/64`, but they should not be used for final claims.
- **Exploratory runs** are allowed to search for useful layers, token group sizes, prompts, or losses. They should be labeled as exploratory in `run_record.md`.
- **Confirmatory runs** test a fixed hypothesis with fixed data splits, seeds, controls, metrics, and decision rules. Only these runs should be used for strong conclusions.

Initialize every run directory before launching extraction or training:

```bash
python3 tools/init_experiment_run.py \
  --run-id H1_A4_Q3VL_COCO_L15_BBOX8_SEED4201 \
  --hypothesis H1 \
  --study A \
  --status planned
```

Then append exact shell commands to:

```text
experiments/runs/{run_id}/command_log.txt
```

The helper records `environment.json`, `git_diff.patch`, `config.yaml`, `run_record.md`, and empty result JSON placeholders. Large parquets, adapters, and image dumps should stay outside Git under `outputs/{run_id}/`, with paths and checksums recorded in the run record.

## 0. Core Hypotheses

The current exploratory runs suggest five hypotheses:

| ID | Hypothesis | Plain-language version | Main evidence needed |
|---|---|---|---|
| H1 | Object semantics are distributed over local image-token groups. | A real object is usually represented by a small neighborhood of image tokens, not by one perfectly isolated token. | Local bbox/mask token groups outperform single center tokens under the same model, layer, data, and training budget. |
| H2 | Different layers encode different visual/semantic information. | Earlier layers may carry lower-level visual detail; middle layers may carry object/category/region; later layers may mix in answer/language priors. | Per-layer probes/AV/AR show different strengths, and multi-layer targets improve or clarify explanations. |
| H3 | Full NLA faithfulness requires an AR closed loop. | AV text is more trustworthy if the text can reconstruct the original activation, not just sound plausible. | AR reconstruction beats mean/shuffle baselines, and AV->AR round-trip score correlates with semantic correctness. |
| H4 | Hallucinations are often linked to high-impact image tokens. | Some image tokens may push the model toward objects that are not in the image. | Tokens with high hallucination intervention effect are verbalized by VLM-NLA as hallucinated/nearby concepts more often than random controls. |
| H5 | Shortcuts show up as explanations that fail to follow visual counterfactuals. | If the picture changes but the internal explanation stays the same, the model may be using language/background priors. | Counterfactual image edits change correct visual evidence; shortcut runs show low explanation-change despite changed evidence. |

## 1. Shared Experimental Standards

### 1.1 Model and Code Versioning

Every run must record:

- Git commit of this repo.
- Exact script path and command line.
- Hostname, GPU model, CUDA version, Python version.
- `torch`, `transformers`, `peft`, `pyarrow`, `Pillow` versions.
- Model ID and local/HF revision, e.g. `Qwen/Qwen3-VL-8B-Instruct`.
- Tokenizer special-token IDs, especially `<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>`.
- Random seeds for data sampling, train shuffle, and evaluation candidate sampling.

Use `experiments/templates/run_record_template.md` for this.

For confirmatory runs, start from a clean committed tree. A dirty tree is acceptable for smoke debugging, but then the run must be labeled as smoke or exploratory.

### 1.2 Data Split Rules

All primary comparisons must use image-disjoint splits:

| Split | Purpose | Recommended size for serious run |
|---|---|---:|
| train | LoRA/adapter training | 1,024-4,096 images |
| validation | model selection / early diagnostics | 256-512 images |
| test | final reported metrics | 512-1,024 images |

For small GPU smoke tests, use `128/64/64`, but label them clearly as smoke and do not use them for final conclusions.

Sampling should be category-stratified where possible. Avoid a split dominated by `person`, `car`, or other frequent COCO categories.

For serious train/validation/test experiments, build an explicit split manifest first; do not assume that three independent seeds are image-disjoint.

```bash
python3 scripts/qwen3vl/build_coco_object_split_manifest.py \
  --coco-root data/coco2017 \
  --out outputs/{run_id}/split_manifest.json \
  --train-size 1024 \
  --val-size 256 \
  --test-size 512 \
  --seed 4200 \
  --min-area-frac 0.015
```

Then pass that manifest to extraction with `--image-ids-json ... --image-ids-key train|val|test`. If the extractor is used without a manifest, label the run as smoke unless a leakage check proves that image IDs are disjoint.

### 1.3 Label Policy

For local image-token interpretation, the default target response should be local and short:

```text
<explanation>
These image tokens represent a {category} in the {region} region of the image.
</explanation>
```

Do not include full COCO captions in the primary local-token benchmark. Full image captions are allowed only in a separate "caption stress test" because they mix local object semantics with whole-image context.

Optional richer labels:

- object category
- coarse region
- visible attributes, if available
- local relation, e.g. "person riding bicycle"
- OCR text region, if using OCR datasets

The existing `extract_qwen3vl_coco_object_tokens.py` script writes a full COCO caption into the original response. For primary local-token experiments, immediately rewrite that parquet with:

```bash
python3 scripts/qwen3vl/make_qwen3vl_coco_short_label_parquet.py \
  --src outputs/{run_id}/extract/qwen3vl_coco_L15_object_bbox_mean_av_sft.parquet \
  --out-dir outputs/{run_id}/short_labels
```

Runs that keep the full caption should be reported as caption stress tests, not as clean local-token interpretation runs.

### 1.4 Standard AV Training Configuration

Use this as the default unless an ablation changes one field:

```text
model_id: Qwen/Qwen3-VL-8B-Instruct
layer_index: 15 baseline, multi-layer variants in H2
target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0
lr: 3e-4
activation_adapter_lr: 1e-4
epochs: 2 for >=512 rows, 3-5 for <=256 rows
grad_accum: 8
num_injection_tokens: 8 baseline
loss: SFT + activation-shuffle contrastive + response-contrastive
contrastive_margin: 0.02
response_contrastive_margin: 0.02
```

For fair ablations, keep the same training rows, target labels, layer, optimizer settings, and total trainable token budget unless the ablation explicitly changes them.

### 1.5 Required Metrics

Report these metrics for every AV run:

| Metric | Definition | Direction |
|---|---|---|
| `matched_mean_nll` | NLL of correct response under matched activation | lower is better |
| `shuffled_mean_nll` | NLL of correct response under randomly mismatched activation | higher than matched is better |
| `sensitivity_delta` | `shuffled_mean_nll - matched_mean_nll` | higher is better |
| `matched_better_fraction` | fraction of samples where matched NLL < shuffled NLL | higher is better |
| `raw_candidate_mean_rank` | rank of correct response among candidate pool by NLL | lower is better |
| `raw_top1/top3/top5` | correct response in top K candidates | higher is better |
| `activation_gain_rank` | candidate ranking after subtracting reference-activation NLL | diagnostic |
| `unique_label_top1/top5` | candidate ranking after merging duplicate semantic labels | higher is better |
| `semantic_object_acc` | object category match | higher is better |
| `semantic_region_acc` | coarse region match | higher is better |

For AR:

| Metric | Definition | Direction |
|---|---|---|
| normalized MSE | MSE after norm normalization | lower is better |
| cosine similarity | cosine(pred, gold) | higher is better |
| FVE vs mean | fraction of variance explained relative to mean vector baseline | higher is better |
| FVE vs shuffle | FVE relative to shuffled target baseline | higher is better |
| retrieval topK | nearest gold activation by reconstructed vector | higher is better |

### 1.6 Statistical Reporting

For final test metrics:

- Use at least 3 data/training seeds for primary comparisons.
- Report mean and 95% bootstrap confidence intervals.
- Use paired bootstrap or paired permutation tests when comparing targets on the same images.
- Correct for multiple comparisons when making many layer/token claims.
- Always include random-region and shuffled-label controls.

Each comparison should report the paired per-sample differences whenever possible. For example, compare `object_center` and `object_bbox_mean_max8` on the same image IDs and object annotations, then bootstrap the paired difference in `sensitivity_delta`, semantic object accuracy, and unique-label topK.

### 1.7 Candidate Pools and Hard Negatives

Candidate ranking should be evaluated under at least two pools:

| Pool | Contents | Purpose |
|---|---|---|
| random all | candidates sampled from the full test split | easy broad retrieval |
| same-region | candidates from the same coarse region but different object | tests object identity |
| same-supercategory | candidates from related COCO categories | tests fine semantic discrimination |
| duplicate-label merged | one item per object/region label | avoids penalizing semantically identical duplicates |

The primary reported ranking should include unique-label topK. Raw sample-id topK is still useful, but it can be pessimistic when many images share the same label such as "person in the center region".

### 1.8 Current Code Coverage

The table below separates experiments that current scripts can run from experiments that need implementation before main runs.

| Need | Current support | Notes |
|---|---|---|
| Qwen3-VL single-layer extraction | supported | `extract_qwen3vl_coco_object_tokens.py --layer-index ...` |
| object center token | supported | `--target-token object_center` |
| bbox mean with max 4/8/16 tokens | supported | `--target-token object_bbox_mean --max-bbox-tokens K` |
| short local labels | supported as rewrite | `make_qwen3vl_coco_short_label_parquet.py` |
| LoRA AV with 1/4/8/12 injection tokens | supported | `train_qwen3vl_av_lora_tiny.py --num-injection-tokens K` |
| sensitivity and candidate ranking | supported | current Qwen3-VL eval scripts |
| explicit image-disjoint split manifests | supported | `build_coco_object_split_manifest.py` plus extractor `--image-ids-json` |
| center 2x2 region | requires extraction update | H1 A2 |
| segmentation mask token mean | requires extraction update | H1 A6 |
| random same-area region | requires extraction update | H1 A7 |
| multi-layer activation tuples in one AV | requires extraction/trainer update | H2 B2-B4 |
| AR closed loop | requires Qwen3-VL AR script | H3 |
| hallucination token intervention | requires new pipeline | H4 |
| counterfactual shortcut dataset | requires new data generator/evaluator | H5 |

### 1.9 Required Artifacts

Each completed run should write:

```text
experiments/runs/{run_id}/
  run_record.md
  config.yaml
  command_log.txt
  environment.json
  train_summary.json
  sensitivity.json
  ranking.json
  semantic_eval.json
  qualitative_panels/
  failure_cases.json
  git_diff.patch
```

Large files such as parquet datasets and adapters should stay outside Git, but their paths and checksums must be recorded.

## 2. Study A: Token Granularity and Object Semantics

### Goal

Validate H1: object semantics are better represented by local image-token groups than by a single image token.

### Dataset

Primary: MSCOCO val2017 object annotations.

Recommended split:

```text
train: 1024 images, seed 4201
val:   256 images, seed 4202
test:  512 images, seed 4203
```

Each sample selects one prominent non-crowd object. Record bbox, category, coarse region, image size, Qwen image-token grid, and selected token indices.

### Experimental Conditions

All conditions use Qwen3-VL layer 15 and the same short-label response.

| Condition ID | Target activation | Purpose |
|---|---|---|
| A1 | `object_center` single token | Current single-token baseline |
| A2 | `center_2x2_mean` | Does minimal neighborhood help? |
| A3 | `object_bbox_mean_max4` | Small local object group |
| A4 | `object_bbox_mean_max8` | Current best local group baseline |
| A5 | `object_bbox_mean_max16` | Test whether too many tokens dilute semantics |
| A6 | `segmentation_mask_mean` if mask available | More precise object region than bbox |
| A7 | `random_same_area_region_mean` | Spatial random control |
| A8 | `image_mean` | Global upper-bound / non-local baseline |

### Training

Train one AV per condition using the standard AV config. Keep:

- same train images,
- same labels,
- same candidate pool,
- same number of injection tokens,
- same optimizer and LoRA settings.

### Current Executable Baseline Commands

The current scripts can run A1 (`object_center`) and A3/A4/A5 (`object_bbox_mean` with different `--max-bbox-tokens`). The example below is for A4. Use the same structure for A1, A3, and A5.

```bash
RUN_ID=H1_A4_Q3VL_COCO_L15_BBOX8_SEED4201
RUN_DIR=experiments/runs/${RUN_ID}
OUT_DIR=outputs/${RUN_ID}

python3 tools/init_experiment_run.py \
  --run-id ${RUN_ID} \
  --hypothesis H1 \
  --study A \
  --status planned

python3 scripts/qwen3vl/build_coco_object_split_manifest.py \
  --coco-root data/coco2017 \
  --out ${OUT_DIR}/split_manifest.json \
  --train-size 1024 \
  --val-size 256 \
  --test-size 512 \
  --seed 4200 \
  --min-area-frac 0.015

CUDA_VISIBLE_DEVICES=0 python3 scripts/qwen3vl/extract_qwen3vl_coco_object_tokens.py \
  --out-dir ${OUT_DIR}/train_extract \
  --num-samples 1024 \
  --seed 4201 \
  --batch-size 2 \
  --layer-index 15 \
  --target-token object_bbox_mean \
  --max-bbox-tokens 8 \
  --image-ids-json ${OUT_DIR}/split_manifest.json \
  --image-ids-key train

python3 scripts/qwen3vl/make_qwen3vl_coco_short_label_parquet.py \
  --src ${OUT_DIR}/train_extract/qwen3vl_coco_L15_object_bbox_mean_av_sft.parquet \
  --out-dir ${OUT_DIR}/train_short_labels

CUDA_VISIBLE_DEVICES=0 python3 scripts/qwen3vl/extract_qwen3vl_coco_object_tokens.py \
  --out-dir ${OUT_DIR}/test_extract \
  --num-samples 512 \
  --seed 4203 \
  --batch-size 2 \
  --layer-index 15 \
  --target-token object_bbox_mean \
  --max-bbox-tokens 8 \
  --image-ids-json ${OUT_DIR}/split_manifest.json \
  --image-ids-key test

python3 scripts/qwen3vl/make_qwen3vl_coco_short_label_parquet.py \
  --src ${OUT_DIR}/test_extract/qwen3vl_coco_L15_object_bbox_mean_av_sft.parquet \
  --out-dir ${OUT_DIR}/test_short_labels

CUDA_VISIBLE_DEVICES=0 python3 scripts/qwen3vl/train_qwen3vl_av_lora_tiny.py \
  --av-parquet ${OUT_DIR}/train_short_labels/qwen3vl_coco_L15_object_bbox_mean_short_av_sft.parquet \
  --out-dir ${OUT_DIR}/adapter \
  --max-rows 1024 \
  --epochs 2 \
  --grad-accum 8 \
  --lr 3e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --injection-scale 57.75 \
  --num-injection-tokens 8 \
  --train-activation-adapter \
  --activation-adapter-lr 1e-4 \
  --contrastive-shuffle-weight 1.0 \
  --contrastive-margin 0.02 \
  --response-contrastive-weight 1.0 \
  --response-contrastive-margin 0.02 \
  --seed 4201

CUDA_VISIBLE_DEVICES=0 python3 scripts/qwen3vl/eval_qwen3vl_av_activation_sensitivity.py \
  --av-parquet ${OUT_DIR}/test_short_labels/qwen3vl_coco_L15_object_bbox_mean_short_av_sft.parquet \
  --adapter ${OUT_DIR}/adapter/adapter \
  --activation-adapter ${OUT_DIR}/adapter/activation_adapter.pt \
  --out ${RUN_DIR}/sensitivity.json \
  --max-rows 512 \
  --batch-size 8 \
  --injection-scale 57.75 \
  --num-injection-tokens 8 \
  --seed 4203

CUDA_VISIBLE_DEVICES=0 python3 scripts/qwen3vl/eval_qwen3vl_av_candidate_ranking.py \
  --av-parquet ${OUT_DIR}/test_short_labels/qwen3vl_coco_L15_object_bbox_mean_short_av_sft.parquet \
  --adapter ${OUT_DIR}/adapter/adapter \
  --activation-adapter ${OUT_DIR}/adapter/activation_adapter.pt \
  --out ${RUN_DIR}/ranking.json \
  --max-rows 512 \
  --max-queries 128 \
  --candidate-mode all \
  --score-mode nll \
  --batch-size 8 \
  --injection-scale 57.75 \
  --num-injection-tokens 8 \
  --seed 4203
```

For confirmatory A-main, train, validation, and test parquet paths must come from the same explicit image-disjoint manifest. The command above shows the train/test path; add validation extraction when using validation for model selection or early diagnostics.

### Evaluation

Primary:

- test `sensitivity_delta`
- test unique-label top1/top5
- semantic object accuracy
- semantic region accuracy

Secondary:

- raw candidate topK
- activation-gain topK
- qualitative panels for 30 fixed test images

### Decision Rule

H1 is supported if:

1. A3/A4/A6 significantly outperform A1 on test semantic object accuracy and sensitivity delta.
2. A7 performs near random or clearly below object-aligned targets.
3. A8 performs well globally but fails to localize object/region better than local targets.
4. The conclusion holds across at least 3 seeds.

### Expected Failure Modes

- Duplicate labels make raw sample-id ranking pessimistic.
- Small objects may map to too few image tokens.
- Large bbox may include background, making max16 worse than max8.
- COCO bbox may not match visible object exactly.

## 3. Study B: Layer Semantics and Multi-Layer AV

### Goal

Validate H2: different layers encode different kinds of visual/semantic information, and multi-layer activation tuples may improve explanations.

### Dataset

Use the same COCO splits as Study A, preferably with `object_bbox_mean_max8` as the target region.

### Single-Layer Conditions

Extract and train separate AVs for:

```text
layers: 6, 10, 15, 20, 28, 34
target: object_bbox_mean_max8
num_injection_tokens: 8
```

For each layer, also train lightweight probes for:

- object category
- region
- coarse attributes if labels exist
- hallucination-prone category groups, if using hallucination data

### Multi-Layer Conditions

Use the same total data split.

| Condition ID | Layers | Token allocation | Adapter |
|---|---|---|---|
| B1 | L15 only | 8 tokens | baseline |
| B2 | L10 + L15 + L20 | 8 total tokens | concat adapter |
| B3 | L10 + L15 + L20 | 4 tokens per layer, 12 total | separate adapters |
| B4 | L6 + L15 + L28 | 4 tokens per layer, 12 total | separate adapters |
| B5 | L15 only | 12 tokens | capacity-matched control for B3/B4 |

Capacity-matched controls are important. If B3 beats B1, it may be because B3 has 12 tokens instead of 8. B5 tests that.

### Evaluation

Primary:

- per-layer and multi-layer sensitivity delta
- unique-label topK
- semantic object/region/attribute accuracy
- layer-ablation test: zero or replace one layer block at a time in multi-layer AV

Secondary:

- explanation length and specificity
- failure categories by object size and category frequency

### Decision Rule

H2 is supported if:

1. Layer metrics differ in interpretable ways, e.g. earlier layers better for region/visual detail, middle layers better for object identity, later layers more language-prior-sensitive.
2. Multi-layer B2/B3 improves over B1 and capacity-matched B5 on at least one semantic metric without degrading sensitivity.
3. Ablating a layer block changes the explanation in a layer-specific way.

## 4. Study C: AR Closed Loop and Faithfulness

### Goal

Validate H3: AV explanations are more trustworthy if they can reconstruct the original activation.

### Dataset

Use Study A's best local target, likely `object_bbox_mean_max8`, and run on:

```text
single layer: L15
multi-layer: L10 + L15 + L20, if Study B supports it
```

### AR Variants

| Condition ID | AR model | Purpose |
|---|---|---|
| C1 | ridge regression from frozen LM final hidden state | simple lower-bound baseline |
| C2 | LoRA AR with one reconstruction head | trainable text->activation model |
| C3 | multi-layer AR with one head per target layer | reconstruct activation tuple |
| C4 | shuffled text control | ensure AR is not just predicting mean category vectors |

### Inputs

AR input should be the explanation text only, plus a stable reconstruction prompt. Do not include image pixels.

Example:

```text
Reconstruct the LVLM activation described by:
<explanation>
These image tokens represent a train in the bottom region of the image.
</explanation>
<reconstruct>
```

### Evaluation

Primary:

- normalized MSE
- cosine similarity
- FVE vs mean baseline
- FVE vs shuffled baseline
- nearest-neighbor retrieval top1/top5 in activation space

Round-trip:

```text
original activation -> AV explanation -> AR reconstructed activation
```

Report whether high semantic accuracy also yields high AR reconstruction. Cases where AV text sounds correct but AR fails are important.

### Decision Rule

H3 is supported if:

1. C2/C3 beat C1 and mean/shuffle controls on held-out test.
2. Round-trip score correlates with candidate-ranking correctness and semantic accuracy.
3. Wrong or generic explanations reconstruct worse than matched explanations.

## 5. Study D: Hallucinatory Image Tokens

### Goal

Validate H4: tokens that causally affect hallucinated outputs are verbalized differently from random/background tokens.

### Dataset

Use COCO images with object annotations plus a hallucination evaluation protocol:

- POPE-style yes/no object existence questions, or
- model-generated captions with CHAIR-style object hallucination labels, or
- a Hall-COCO/MME-style subset if available locally.

For each image/question:

1. Generate the LVLM answer.
2. Label whether an object hallucination occurred using COCO categories and/or human review.
3. Identify the hallucinated object string.

### Token Selection

For each hallucinated example, select image tokens by three methods:

| Token set | How selected | Purpose |
|---|---|---|
| D1 high-impact tokens | EAZY-style zero-out/patch: tokens whose ablation most reduces hallucinated object likelihood | candidate hallucination tokens |
| D2 high-attention tokens | image tokens attended to when generating hallucinated object noun | cheaper diagnostic |
| D3 true-object tokens | bbox tokens for objects actually present | positive grounded control |
| D4 random background tokens | random non-object or low-impact tokens | negative control |

### VLM-NLA Task

Train or evaluate AV on token groups:

```text
target: selected token group mean, or individual high-impact token
label: object/category/region if grounded; hallucinated object label for hallucination analysis
```

Important: do not train the AV to always say the hallucinated object. Instead, use the same local explanation task as Study A, then analyze whether high-impact hallucination tokens naturally rank hallucinated-object descriptions higher than controls.

### Evaluation

Primary:

- hallucinated-object mention rate in topK candidate explanations
- NLA hallucination score: NLL(true absent object explanation) vs NLL(grounded object/background explanation)
- correlation between intervention effect size and hallucinated-object explanation score
- change in model answer after zeroing top-k NLA-flagged tokens

Secondary:

- qualitative panels showing image, selected tokens, original answer, patched answer, AV top candidates

### Decision Rule

H4 is supported if:

1. High-impact tokens rank hallucinated-object explanations higher than random/background controls.
2. The NLA hallucination score correlates with causal intervention effect.
3. Removing top NLA-flagged tokens reduces hallucinated answer probability more than removing random tokens.

## 6. Study E: Shortcut and Counterfactual Visual Evidence

### Goal

Validate H5: shortcut behavior appears when explanations do not change appropriately under visual counterfactual edits.

### Dataset

Use controlled paired images where language prior is held stable but visual evidence changes.

Recommended sources:

1. Synthetic shape scenes:
   - object color/shape swaps
   - position swaps
   - count changes
2. Diagram/spatial-relation tasks:
   - arrow direction flips
   - left/right swaps
   - connection changes
3. COCO-derived counterfactuals:
   - object crop replacement
   - bbox blur/removal
   - inpainting where feasible

Each pair should have:

```text
image_A, image_B
same prompt
changed visual truth
expected answer_A, expected answer_B
target region tokens for the changed object/relation
```

### Activation Targets

Compare:

- changed object-region image tokens
- unrelated object-region image tokens
- last prompt token
- answer-token hidden state, if answer generation is instrumented

### Evaluation

Define explanation-change score:

```text
semantic_change = distance(explanation_A, explanation_B)
visual_change_expected = whether object/relation label changed
shortcut_score = answer_invariant_or_wrong AND explanation_change_low
```

Metrics:

- answer flip accuracy under counterfactual edit
- object-token explanation flip accuracy
- explanation-change score
- correlation between explanation flip and answer correctness
- NLL preference for old-prior explanation vs new-visual-evidence explanation

### Decision Rule

H5 is supported if:

1. Correct model behavior is accompanied by explanation changes in the edited region.
2. Shortcut failures show low explanation-change despite changed visual evidence.
3. Region-token explanations are more sensitive to visual edits than last-prompt activations when the model is grounded.

## 7. Experiment Schedule

Run in this order:

1. **A-smoke:** Token granularity on 128/64/64 split to ensure scripts work.
2. **A-main:** Token granularity on 1024/256/512 with 3 seeds.
3. **B-probe:** Per-layer probes without AV training, cheap layer scan.
4. **B-AV:** Single-layer and multi-layer AV training.
5. **C-AR:** AR closed-loop on the best A/B targets.
6. **D-hallucination-smoke:** Small POPE/COCO hallucination token pipeline.
7. **E-counterfactual-smoke:** Synthetic visual shortcut pairs.
8. **D/E-main:** Larger hallucination and shortcut studies only after smoke metrics are meaningful.

## 8. Minimal Run IDs

Use stable run IDs:

```text
H1_A1_Q3VL_COCO_L15_CENTER_SEED4201
H1_A4_Q3VL_COCO_L15_BBOX8_SEED4201
H2_B1_Q3VL_COCO_L15_BBOX8_SEED4301
H2_B3_Q3VL_COCO_L10L15L20_BBOX8_4TOKPERLAYER_SEED4301
H3_C2_Q3VL_AR_L15_BBOX8_SEED4401
H4_D1_Q3VL_POPE_HIGHIMPACT_SEED4501
H5_E1_Q3VL_COUNTERFACTUAL_SHAPES_SEED4601
```

## 9. What Would Change Our Mind?

The hypotheses should be considered weakened or rejected if:

- single-token targets match or beat local token groups after better evaluation and multiple seeds;
- multi-layer targets only improve because they have more tokens/parameters, and capacity-matched controls erase the gain;
- AR reconstructs matched and shuffled explanations equally well;
- hallucination high-impact tokens do not differ from random/background tokens in NLA explanations;
- counterfactual visual edits change explanations no more than random noise.

Negative results are valuable. Record them with the same rigor as positive results.
