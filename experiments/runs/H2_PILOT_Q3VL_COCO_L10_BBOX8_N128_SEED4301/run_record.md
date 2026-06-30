# Run Record: H2_PILOT_Q3VL_COCO_L10_BBOX8_N128_SEED4301

## 1. Identity

| Field | Value |
|---|---|
| run_id |  |
| hypothesis |  |
| study |  |
| date_started |  |
| date_finished |  |
| status | planned / running / completed / failed / superseded |
| owner |  |
| hostname |  |
| run_type | smoke / exploratory / confirmatory |
| initialized_with | `tools/init_experiment_run.py` yes / no |

## 2. Research Question

What exact question does this run answer?

```text

```

Expected result if the hypothesis is true:

```text

```

Expected result if the hypothesis is false:

```text

```

Pre-registered decision rule for this run:

```text

```

## 3. Code and Environment

| Field | Value |
|---|---|
| repo_git_commit |  |
| git_branch |  |
| working_tree_clean_at_start | yes / no |
| git_diff_saved_to |  |
| environment_json |  |
| python_version |  |
| torch_version |  |
| transformers_version |  |
| peft_version |  |
| cuda_version |  |
| gpu_type |  |
| model_id |  |
| model_revision_or_local_path |  |
| tokenizer_revision_or_local_path |  |

Special tokens:

| Token | ID |
|---|---:|
| `<|vision_start|>` |  |
| `<|image_pad|>` |  |
| `<|vision_end|>` |  |

## 4. Data

| Field | Value |
|---|---|
| dataset_name |  |
| train_split |  |
| val_split |  |
| test_split |  |
| split_seed |  |
| train_rows |  |
| val_rows |  |
| test_rows |  |
| split_manifest |  |
| image_disjoint | yes / no |
| category_stratified | yes / no |
| parquet_paths |  |
| dataset_checksums |  |

Label policy:

```text

```

Leakage checks:

- [ ] train/val/test image IDs are disjoint
- [ ] candidate pool does not contain duplicate sample IDs
- [ ] duplicate semantic labels are tracked
- [ ] random controls use the same image split
- [ ] full COCO captions are excluded from the primary local-token label unless this is explicitly a caption stress test
- [ ] all controls use the same prompt, label format, optimizer settings, and candidate pool unless the ablation says otherwise

## 5. Activation Target

| Field | Value |
|---|---|
| model_layer_or_layers |  |
| target_token_type |  |
| token_selection_rule |  |
| num_selected_image_tokens |  |
| pooling_rule | center / mean / weighted mean / other |
| current_code_support | supported / requires extractor change / requires trainer change |
| activation_dim |  |
| activation_norm_mean |  |
| activation_norm_std |  |
| activation_parquet_sha256 |  |

## 6. Training Configuration

| Field | Value |
|---|---|
| train_script |  |
| train_command_log |  |
| max_rows |  |
| epochs |  |
| batch_size |  |
| grad_accum |  |
| learning_rate |  |
| lora_r |  |
| lora_alpha |  |
| lora_dropout |  |
| target_modules |  |
| train_activation_adapter | yes / no |
| activation_adapter_lr |  |
| num_injection_tokens |  |
| injection_scale |  |
| contrastive_shuffle_weight |  |
| contrastive_margin |  |
| response_contrastive_weight |  |
| response_contrastive_margin |  |
| seed |  |

Full command:

```bash

```

## 7. Evaluation Configuration

Sensitivity command:

```bash

```

Ranking command:

```bash

```

Semantic evaluation command:

```bash

```

Candidate pool:

| Field | Value |
|---|---|
| num_candidates |  |
| candidate_sampling_seed |  |
| candidate_mode | all / sampled / hard_negative |
| duplicate_label_handling | raw / unique-label / semantic |
| hard_negative_policy |  |

## 8. Results

### 8.1 Sensitivity

| Metric | Value |
|---|---:|
| matched_mean_nll |  |
| shuffled_mean_nll |  |
| sensitivity_delta |  |
| matched_better_fraction |  |

### 8.2 Ranking

| Metric | Value |
|---|---:|
| mean_rank |  |
| median_rank |  |
| raw_top1 |  |
| raw_top3 |  |
| raw_top5 |  |
| activation_gain_top1 |  |
| activation_gain_top5 |  |
| unique_label_top1 |  |
| unique_label_top5 |  |

### 8.3 Semantic Metrics

| Metric | Value |
|---|---:|
| object_accuracy |  |
| region_accuracy |  |
| attribute_accuracy |  |
| semantic_exact_match |  |

### 8.4 AR Metrics

| Metric | Value |
|---|---:|
| normalized_mse |  |
| cosine_similarity |  |
| fve_vs_mean |  |
| fve_vs_shuffle |  |
| retrieval_top1 |  |
| retrieval_top5 |  |

## 9. Confidence Intervals and Significance

Bootstrap settings:

```text

```

Comparison runs:

| Comparison | Test | p-value | 95% CI | Conclusion |
|---|---|---:|---|---|
|  |  |  |  |  |

Seed aggregation:

| Seed group | Runs included | Mean | 95% CI | Notes |
|---|---|---:|---|---|
|  |  |  |  |  |

## 10. Qualitative Analysis

Representative successes:

| sample_id | image_path | target_tokens | correct_response | top_response | note |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

Representative failures:

| sample_id | image_path | target_tokens | correct_response | top_response | failure_type |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

Failure type tags:

- language_prior
- duplicate_label
- wrong_region
- wrong_object_same_supercategory
- small_object
- bbox_background_contamination
- hallucination_related
- shortcut_related
- extraction_bug

## 11. Interpretation

What did this run show?

```text

```

Does it support the hypothesis?

```text

```

What should change in the next run?

```text

```

## 12. Artifact Paths

| Artifact | Path |
|---|---|
| train parquet |  |
| val parquet |  |
| test parquet |  |
| adapter |  |
| activation_adapter.pt |  |
| train_summary.json |  |
| sensitivity.json |  |
| ranking.json |  |
| semantic_eval.json |  |
| qualitative_panels |  |
| failure_cases.json |  |
| command_log.txt |  |
| environment.json |  |
| git_diff.patch |  |

Checksums:

| Artifact | SHA256 |
|---|---|
| train parquet |  |
| val parquet |  |
| test parquet |  |
| adapter |  |
| activation_adapter.pt |  |
