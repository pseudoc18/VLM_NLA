# VLM-NLA: Natural Language Autoencoders for Vision-Language Model Activations

This repository contains an exploratory implementation of **Natural Language Autoencoder (NLA)-style activation verbalization for VLMs**, with experiments on **LLaVA-1.5** and **Qwen3-VL**.

The core question is:

> Can we turn an internal VLM activation, especially a visual-token activation, into a natural-language explanation that faithfully reflects what the model internally represents?

The short current answer is:

- **Mechanism:** yes. Both LLaVA-1.5 and Qwen3-VL can accept NLA-style `inputs_embeds` injection.
- **Best current model:** Qwen3-VL is much stronger than LLaVA-1.5 for this direction.
- **Best current target:** pooled local visual-token groups are more stable than a single image token, and more specific than global `image_mean`.
- **Current status:** this is a research prototype, not a polished library or released checkpoint.

## Background

[Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) explain LLM residual-stream activations with two learned components:

| Component | Direction | Role |
|---|---|---|
| **AV, activation verbalizer** | activation vector -> text | Inject an activation as one or more token embeddings and generate an explanation. |
| **AR, activation reconstructor** | text -> activation vector | Reconstruct the original activation from the explanation. |

The original NLA work targets text-only LLM activations. This project asks whether the same idea can be transferred to **vision-language models**, where visual information enters the language model through image placeholder tokens, visual embeddings, and multimodal fusion.

This matters because VLM failures such as hallucination, visual shortcut use, and object misgrounding are often not visible from output text alone. If we can explain specific internal visual tokens, we can ask more precise questions:

- What does this **single image token** represent?
- What does this **local group of image tokens** inside an object bounding box represent?
- Which internal visual tokens drive a hallucinated answer?
- Does the model reason from the image, or does it use a language/background-knowledge shortcut?

## Project Status

The project has three experiment tracks.

### 1. LLaVA-1.5 Layer-15 Feasibility

LLaVA-1.5 is a natural first target because its Hugging Face forward path projects image features into the language-model hidden size and replaces `<image>` token embeddings before entering the language model. That is very close to NLA's activation-as-input-embedding mechanism.

Implementation:

- Model: `llava-hf/llava-1.5-7b-hf`
- Target layer: language-model layer 15
- Injection marker: repeated `<image>` tokens
- AV adapter: `Linear(4096, K * 4096)` for `K` injected tokens
- Best run: 8 injected tokens, LoRA AV, activation adapter, dual contrastive loss

Best LLaVA result:

| Run | Target | AV tokens | Sensitivity delta | Mean rank | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|---:|
| LLaVA-1.5 best | L15 `last_prompt` | 8 | +0.0790 | 27.13 / 128 | 3.1% | 15.6% |

Interpretation: the mechanism works and multiple AV tokens help, but the result is not close to ordinary NLA-level performance. The AV decoder still has strong language-prior failure modes.

### 2. Qwen3-VL Layer-15 Synthetic Visual Concepts

Qwen3-VL uses visual placeholder tokens such as `<|image_pad|>`. The first successful Qwen3-VL run injects 8 AV tokens:

```text
<|vision_start|><|image_pad|> x 8 <|vision_end|>
```

and maps one layer-15 activation into all 8 token embeddings:

```text
4096-d activation -> 8 x 4096-d injected embeddings
```

The strongest synthetic target so far is `image_mean`, the mean of all layer-15 image-token activations. This is global rather than local, so it is a good feasibility target but not the final interpretability target.

Best Qwen3-VL synthetic result:

| Split | Target | AV tokens | Sensitivity delta | Mean rank | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|---:|
| Train pool | L15 `image_mean` | 8 | +1.1089 | 1.02 / 128 | 98.4% | 100.0% |
| Held-out seed | L15 `image_mean` | 8 | +1.1497 | 1.06 / 128 | 93.8% | 100.0% |

Interpretation: NLA-style AV is clearly feasible on Qwen3-VL. Qwen3-VL is the better model family for continuing this line.

### 3. Qwen3-VL on Real COCO Object Tokens

The user preference was not to stop at whole-image `image_mean`, but to explain **specific image tokens** or **small groups of image tokens**. The COCO experiment does that.

Dataset:

- Source: MSCOCO val2017
- Train-like split: 128 images, seed 2027
- Held-out split: 64 images, seed 3031
- Each image selects one prominent COCO object annotation
- The COCO bounding box is mapped to the Qwen3-VL image-token grid

Targets:

| Target | Meaning |
|---|---|
| `object_center` | The single image token closest to the COCO object bbox center. |
| `object_bbox_mean` | Mean of the 4-8 local image tokens inside the COCO object bbox. |

Short-label held-out result:

| Target | Sensitivity delta | Matched better | Mean rank | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|
| Single object-center token | +0.156 | 90.6% | 11.31 / 64 | 25.0% | 53.1% |
| Local bbox token group | +0.412 | 98.4% | 5.19 / 64 | 34.4% | 65.6% |

Interpretation: a single visual token contains usable semantic signal, but it is noisy. A local object-token group is much more stable. This suggests Qwen3-VL represents object semantics across small spatial neighborhoods, not as one perfectly isolated token.

## Metrics

The main diagnostics are:

- **Sensitivity delta:** `NLL(shuffled activation, correct text) - NLL(matched activation, correct text)`. Higher is better. A positive value means the explanation text is easier to predict from the correct activation than from a mismatched activation.
- **Matched better fraction:** fraction of examples where matched activation beats shuffled activation.
- **Candidate ranking:** given one activation and many candidate explanations, rank candidates by teacher-forced NLL. Top-1 and Top-5 measure whether the correct explanation is selected.
- **Activation-gain ranking:** a diagnostic that subtracts a reference-activation score to reduce language-prior confounds.

Greedy generation is useful for qualitative inspection, but it is not the primary faithfulness metric here. Ranking and matched-vs-shuffled sensitivity are more stable for small research runs.

## Repository Layout

```text
scripts/
  llava15/        LLaVA extraction, AV training, AR-lite probes, and evaluation
  qwen3vl/        Qwen3-VL extraction, AV training, COCO object-token extraction, and evaluation

results/
  llava15/        lightweight JSON metrics for the best LLaVA baseline
  qwen3vl/
    synthetic/   Qwen3-VL synthetic image_mean metrics
    coco_object_tokens/  Qwen3-VL COCO object-token metrics

assets/figures/
  llava15/        selected LLaVA metric/sample visualizations
  qwen3vl_synthetic/ selected Qwen3-VL synthetic visualizations
  qwen3vl_coco/  selected COCO object-token visualizations

reports/
  *.md            experiment notes copied from the exploratory run
  vlm_nla_research_proposal_preliminary_results.html
                  bilingual proposal + preliminary-results report with all figures embedded

docs/
  nla_to_vlm_multilayer_design.md
                  design notes for migrating NLA to VLMs and multi-layer/multi-token targets
```

Large artifacts are intentionally excluded from Git:

- model checkpoints and LoRA adapters
- `activation_adapter.pt`
- parquet activation datasets
- full synthetic or COCO image dumps
- downloaded COCO archives

The JSON summaries and figures in this repo are enough to understand the current result without downloading the models.

## Reproduction Sketch

The scripts were developed on a GPU server under:

```text
/common/users/lc1279/Projects/nla_llava15_experiment
```

Install a basic environment:

```bash
pip install -r requirements.txt
```

For Qwen3-VL, use a `transformers` version that exposes `Qwen3VLForConditionalGeneration`. The exploratory run used a development build with Qwen3-VL support.

### Qwen3-VL synthetic image_mean

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/qwen3vl/extract_qwen3vl_layer15_dataset.py \
  --out-dir data/qwen3vl_L15_image_mean_n512 \
  --num-samples 512 \
  --batch-size 4 \
  --layer-index 15 \
  --target-token image_mean

CUDA_VISIBLE_DEVICES=0 python scripts/qwen3vl/train_qwen3vl_av_lora_tiny.py \
  --av-parquet data/qwen3vl_L15_image_mean_n512/qwen3vl_L15_image_mean_av_sft.parquet \
  --out-dir outputs/qwen3vl_av_lora_tiny_L15_image_mean_512x2_actadapter_8tok_dualcontrast \
  --max-rows 512 \
  --epochs 2 \
  --grad-accum 8 \
  --lr 3e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --injection-scale 57.58 \
  --num-injection-tokens 8 \
  --train-activation-adapter \
  --activation-adapter-lr 1e-4 \
  --contrastive-shuffle-weight 1.0 \
  --contrastive-margin 0.02 \
  --response-contrastive-weight 1.0 \
  --response-contrastive-margin 0.02
```

### Qwen3-VL COCO object-token target

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/qwen3vl/extract_qwen3vl_coco_object_tokens.py \
  --out-dir data/qwen3vl_coco_L15_object_bbox_mean_n128_seed2027 \
  --num-samples 128 \
  --seed 2027 \
  --layer-index 15 \
  --target-token object_bbox_mean
```

Then train/evaluate with the same Qwen3-VL AV trainer and evaluator, using the generated COCO AV parquet.

## Main Lessons So Far

1. **VLM-NLA is feasible.** VLM language models can receive activation-derived embeddings the same way text-only NLA injects an activation vector.
2. **Model choice matters.** LLaVA-1.5 is easy to instrument, but Qwen3-VL gives much cleaner activation-conditioned explanations.
3. **Multiple AV special tokens help.** Mapping one activation into 8 injected visual placeholder embeddings works well and is better than a single-token bottleneck.
4. **Local token groups are better than single tokens.** For COCO, bbox-local image-token groups are much more reliable than a single object-center image token.
5. **The next step is not just more training.** Better evaluation and harder negatives are needed: unique-label ranking, semantic match metrics, hard-negative mining, and multi-layer/local-token targets.

## Related Work and Directions

- **Natural Language Autoencoders:** the original AV/AR framing for text-only LLM activations.
- **Multimodal hallucination:** [Hallucinatory Image Tokens / EAZY](https://openaccess.thecvf.com/content/ICCV2025/papers/Che_Hallucinatory_Image_Tokens_A_Training-free_EAZY_Approach_to_Detecting_and_ICCV_2025_paper.pdf) studies image tokens tied to hallucination behavior. VLM-NLA could verbalize those tokens before and during hallucinated decoding.
- **VLM shortcuts and circuits:** [Do Vision-Language Models Really Understand Visual Language?](https://openreview.net/forum?id=ZPQU4uGMBA) and [Circuit Tracing in Vision-Language Models](https://arxiv.org/abs/2602.20330) point toward shortcut/circuit-level analyses of VLM behavior. VLM-NLA can provide natural-language probes for the activations used in those circuits.
- **Multi-layer NLA:** extend AV and AR from one activation vector to tuples such as `(L10 object tokens, L15 object tokens, L20 object tokens)`, with separate token blocks or adapters per layer.
- **Object-grounded explanations:** use segmentation masks, bbox token groups, OCR regions, or detected hallucination tokens as explanation targets.

## Current Report

The most complete human-readable writeup is:

```text
reports/vlm_nla_research_proposal_preliminary_results.html
```

It is bilingual English/Chinese, written as a research proposal plus preliminary-results report. All figures are embedded into the HTML file as base64 data URIs, so the file can be opened standalone.
