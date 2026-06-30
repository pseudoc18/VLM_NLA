# VLM-NLA 后续假设验证实验设计中文说明

这份文档是 `docs/hypothesis_validation_experiments.md` 的中文 companion。英文文档是完整 protocol；这里用更直接的语言说明：我们接下来要验证什么、怎么验证、成功或失败分别意味着什么。

## 总目标

当前探索已经说明：Qwen3-VL 的 image-token activation 可以被 NLA-style AV 读出，且 object bbox 内的一小片 local image tokens 比单个 image token 更稳定。下一步不是简单“多跑一些 epoch”，而是要把几个关键假设拆开，用严谨对照实验验证。

核心要求：

- 每个实验都要有明确 hypothesis。
- 每个实验都要有 matched control。
- 所有 train/val/test split 必须 image-disjoint。
- 记录 seed、commit、命令、模型版本、数据路径、指标和失败案例。
- 大文件不进 Git，但路径和 checksum 必须写进 run record。

## 怎么使用这套 protocol

后续每个实验都分成三类：

| 类型 | 作用 | 能不能支撑最终结论 |
|---|---|---|
| smoke | 检查 extraction/training/eval pipeline 能不能跑通 | 不能 |
| exploratory | 找 layer、token group、loss、prompt 的候选设计 | 只能提供线索 |
| confirmatory | 固定 hypothesis、split、seed、metric、control 后正式验证 | 可以 |

开始任何 run 前先初始化实验目录：

```bash
python3 tools/init_experiment_run.py \
  --run-id H1_A4_Q3VL_COCO_L15_BBOX8_SEED4201 \
  --hypothesis H1 \
  --study A \
  --status planned
```

它会创建：

```text
experiments/runs/{run_id}/
  run_record.md
  config.yaml
  command_log.txt
  environment.json
  git_diff.patch
  train_summary.json
  sensitivity.json
  ranking.json
  semantic_eval.json
  qualitative_panels/
  failure_cases.json
```

之后所有命令都追加到 `command_log.txt`，所有大文件放到 `outputs/{run_id}/`，并在 `run_record.md` 里写路径和 checksum。正式 confirmatory run 尽量从 clean commit 开始；dirty working tree 只适合 smoke/debug。

## 当前代码支持情况

现在可以直接跑的部分：

- Qwen3-VL single-layer activation extraction：可指定 `--layer-index`。
- `object_center` single token。
- `object_bbox_mean`，可通过 `--max-bbox-tokens` 做 max4/max8/max16。
- short local label rewrite：用 `make_qwen3vl_coco_short_label_parquet.py` 去掉 full COCO caption。
- 1/4/8/12 个 AV injection tokens 的 LoRA AV 训练。
- sensitivity 和 candidate ranking eval。
- 显式 image-disjoint split manifest：用 `build_coco_object_split_manifest.py` 生成，再让 extractor 通过 `--image-ids-json` 和 `--image-ids-key train|val|test` 读取。

还需要补代码后才能做正式 main run 的部分：

- center 2x2 local region。
- segmentation mask token mean。
- random same-area region control。
- 多层 activation tuple 的 AV/AR。
- hallucination token intervention pipeline。
- counterfactual shortcut dataset/evaluator。

一个很重要的数据规范：旧 COCO extractor 会把 `The full COCO caption is ...` 拼进 response。正式 local-token interpretation 不应该用这个 response，因为它把全图 caption 信息混进来了。主实验必须先用 short-label rewrite，只保留类似：

```text
These image tokens represent a train in the bottom region of the image.
```

正式 COCO split 应该先这样生成：

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

然后 extraction 时指定：

```bash
--image-ids-json outputs/{run_id}/split_manifest.json --image-ids-key train
```

## H1：object semantics 是 local token group 表示的，而不是 single token 表示的

### 直觉

一个真实 object 通常占据图像里一块区域。Qwen3-VL 的 image-token grid 也是空间网格，所以一个 object 的语义很可能分布在一小片相邻 tokens 上，而不是只压在 bbox center 的一个 token 上。

### 实验

固定：

- 模型：`Qwen/Qwen3-VL-8B-Instruct`
- layer：15
- 标签：short label，例如 `These image tokens represent a train in the bottom region of the image.`
- 数据：COCO val2017，image-disjoint train/val/test

比较 target：

| 条件 | target | 目的 |
|---|---|---|
| A1 | bbox center 单 token | single-token baseline |
| A2 | center 周围 2x2 token mean | 看最小邻域是否有帮助 |
| A3 | bbox 内最多 4 tokens mean | 小 local group |
| A4 | bbox 内最多 8 tokens mean | 当前最有希望 baseline |
| A5 | bbox 内最多 16 tokens mean | 检查 token 太多是否稀释语义 |
| A6 | segmentation mask tokens mean | 如果可用，测试更准 object region |
| A7 | random same-area region | 随机区域负对照 |
| A8 | full image_mean | 全图强 baseline，但粒度粗 |

### 判断标准

如果 A3/A4/A6 在 sensitivity delta、unique-label top5、semantic object accuracy 上显著强于 A1，同时 A7 明显差，那么 H1 被支持。

如果 single token 和 local group 差不多，甚至 single token 更强，就说明当前“object 分布在 local neighborhood”这个假设需要重新审视。

### 第一批正式实验矩阵

优先跑下面这些，因为大部分已经被当前代码支持：

| run family | target | split | seeds | 主要目的 |
|---|---|---|---|---|
| H1-A1 | `object_center` | 1024/256/512 | 3 | single-token baseline |
| H1-A3 | `object_bbox_mean`, max4 | 1024/256/512 | 3 | 小 local group |
| H1-A4 | `object_bbox_mean`, max8 | 1024/256/512 | 3 | 当前最有希望的 local group |
| H1-A5 | `object_bbox_mean`, max16 | 1024/256/512 | 3 | 检查 token 太多是否稀释 |
| H1-A8 | `image_mean` | 1024/256/512 | 3 | global upper bound，但不算 local explanation |

每个 run 都要保存：

- train/test parquet 路径和 checksum；
- LoRA adapter 路径和 checksum；
- `sensitivity.json`；
- `ranking.json`，至少包括 raw NLL 和 activation-gain 两种 scoring；
- semantic eval：object accuracy、region accuracy、unique-label topK；
- 30 个固定 test images 的 qualitative panels；
- 失败案例列表，标注是 wrong object、wrong region、duplicate label、small object 还是 background contamination。

核心比较不是看某一个数字，而是看 paired difference：

```text
same image/object 上，A4 - A1 的 sensitivity_delta、unique-label top5、semantic object accuracy 是否稳定为正。
```

## H2：不同 layer 表示不同层次的信息，多层 activation 可能比单层更好

### 直觉

LVLM 的不同层可能做不同事情：

- 早层：颜色、边缘、局部纹理、位置
- 中层：object category、region、attribute
- 后层：answer planning、language prior、任务格式

如果只看 layer 15，可能漏掉其他层的信息。

### 实验

先做 single-layer scan：

```text
layers: 6, 10, 15, 20, 28, 34
target: object_bbox_mean_max8
```

每一层都训练同样的 AV，并跑轻量 probe：

- object category
- region
- attribute

然后做 multi-layer AV：

| 条件 | layers | token allocation | 目的 |
|---|---|---|---|
| B1 | L15 | 8 tokens | 单层 baseline |
| B2 | L10+L15+L20 | 8 total tokens | concat adapter，固定 token budget |
| B3 | L10+L15+L20 | 每层 4 tokens，共 12 | layer-block injection |
| B4 | L6+L15+L28 | 每层 4 tokens，共 12 | 更分散的层组合 |
| B5 | L15 | 12 tokens | capacity-matched control |

B5 很重要：如果 B3 比 B1 好，可能只是因为 B3 有 12 个 tokens，不一定是多层有用。B5 用来控制容量。

### 判断标准

如果不同层在 object/region/attribute 上表现出稳定差异，并且 multi-layer 在 capacity-matched control 后仍有提升，H2 被支持。

## H3：要接近真正 NLA，需要 AR 闭环

### 直觉

AV 生成的 explanation 可能听起来合理，但不一定真的保留 activation 信息。原始 NLA 的关键是 round trip：

```text
activation -> explanation -> reconstructed activation
```

如果 explanation 能让 AR reconstruct 原 activation，说明它更可能是 faithful 的。

### 实验

用 H1/H2 中最好的 target，比如：

```text
L15 object_bbox_mean_max8
```

训练 AR：

| 条件 | AR 类型 | 目的 |
|---|---|---|
| C1 | ridge regression | 简单下界 |
| C2 | LoRA AR + one reconstruction head | 主要 AR baseline |
| C3 | multi-layer AR + per-layer heads | 多层闭环 |
| C4 | shuffled text control | 检查是否只是预测平均向量 |

### 指标

- normalized MSE，越低越好
- cosine similarity，越高越好
- FVE vs mean baseline，越高越好
- FVE vs shuffle baseline，越高越好
- reconstructed activation retrieval topK

### 判断标准

如果 matched explanation 的 reconstruction 明显强于 shuffled explanation，并且 round-trip score 和 semantic correctness 正相关，H3 被支持。

## H4：hallucination 可能来自 high-impact image tokens

### 直觉

EAZY / Hallucinatory Image Tokens 方向提示：hallucination 可能由少数 image tokens 驱动。VLM-NLA 可以进一步问：这些 high-impact tokens 在模型内部“像什么”？

### 实验

数据：

- COCO + POPE-style object existence questions
- 或 CHAIR-style generated captions

流程：

1. 让 Qwen3-VL 回答 object existence/caption 问题。
2. 标注 hallucinated object。
3. 找到 high-impact image tokens：
   - zero-out/patch 后 hallucinated object likelihood 降最多的 tokens
   - 或生成 hallucinated noun 时 attention 高的 image tokens
4. 用 VLM-NLA 解释这些 tokens。

对照：

| token set | 作用 |
|---|---|
| high-impact hallucination tokens | 主要对象 |
| true-object bbox tokens | grounded positive control |
| random background tokens | negative control |
| low-impact image tokens | causal control |

### 指标

- hallucinated-object explanation score
- topK explanation 是否提到 hallucinated object
- intervention effect 和 NLA hallucination score 的相关性
- zero-out NLA-flagged tokens 是否比 random tokens 更能降低 hallucination

### 判断标准

如果 high-impact tokens 比 random/background tokens 更常 verbalize 成 hallucinated object，并且这个分数和 causal intervention effect 正相关，H4 被支持。

## H5：shortcut 会表现为 explanation 不随视觉 counterfactual 改变

### 直觉

如果模型真的看图，图片里的 object/relation 改了，相关 image-token activation 的 explanation 应该跟着变。如果图片变了但 explanation 和 answer 都不变，说明模型可能依赖 shortcut。

### 实验

构造 paired counterfactual images：

- synthetic shapes：颜色、形状、位置、数量变化
- diagram：箭头方向、连接关系变化
- COCO-derived edits：object crop replace、blur、remove、inpaint

每对数据包含：

```text
image_A, image_B
same prompt
expected_answer_A, expected_answer_B
edited_region_tokens
```

比较：

- object-region image token explanation 是否变化
- last_prompt activation explanation 是否变化
- final answer 是否变化

### 指标

- answer flip accuracy
- explanation flip accuracy
- explanation-change score
- old-prior explanation vs new-visual-evidence explanation 的 NLL 差
- explanation change 和 answer correctness 的相关性

### 判断标准

如果正确回答伴随 region-token explanation 改变，而 shortcut failure 里 explanation 不随视觉证据改变，H5 被支持。

## 实验记录要求

每个 run 必须创建：

```text
experiments/runs/{run_id}/
  run_record.md
  config.yaml
  command_log.txt
  train_summary.json
  sensitivity.json
  ranking.json
  semantic_eval.json
  qualitative_panels/
  failure_cases.json
  git_diff.patch
```

run record 必须记录：

- run_id
- hypothesis
- git commit
- model revision
- data split seed
- train command
- eval command
- activation target
- token selection rule
- train/val/test rows
- metrics
- confidence interval
- qualitative success/failure cases
- artifact paths and checksums

## 推荐执行顺序

1. H1 smoke：128/64/64 COCO split，确认 target extraction 和 evaluation 没问题。
2. H1 main：1024/256/512，3 seeds，验证 local group 是否稳定强于 single token。
3. H2 probe：先跑 cheap layer probes，选层。
4. H2 AV：训练 single-layer 和 multi-layer AV。
5. H3 AR：对 H1/H2 最好的 target 做 AR 闭环。
6. H4 smoke：小规模 POPE/COCO hallucination token pipeline。
7. H5 smoke：synthetic counterfactual shortcut pairs。
8. H4/H5 main：在 smoke 有信号后扩大规模。

## 最重要的判定原则

不要只看一个漂亮例子。最终结论必须同时满足：

- held-out test 有提升；
- 至少 3 个 seeds 稳定；
- 有 random/shuffle controls；
- 有 semantic metric，不只 raw sample-id ranking；
- 有 qualitative failure analysis；
- 有完整 run record 可以复查。
