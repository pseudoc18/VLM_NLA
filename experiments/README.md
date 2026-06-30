# Experiment Tracking

This directory is for planning and tracking future VLM-NLA experiments.

The main protocol is:

```text
docs/hypothesis_validation_experiments.md
```

Use the templates here for every run:

```text
experiments/templates/run_record_template.md
experiments/templates/config_template.yaml
experiments/registry.csv
```

Recommended directory layout for completed runs:

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

Large artifacts such as activation parquets, model adapters, downloaded images, and `.pt` files should not be committed. Record their paths and checksums in the run record.

## Run Lifecycle

1. Pick a stable `run_id` from `experiments/registry.csv`, or add a new row before starting.
2. Create the run directory and environment snapshot:

```bash
python3 tools/init_experiment_run.py \
  --run-id H1_A4_Q3VL_COCO_L15_BBOX8_SEED4201 \
  --hypothesis H1 \
  --study A \
  --status planned
```

3. Edit `experiments/runs/{run_id}/config.yaml` only if this run intentionally differs from the template.
4. For COCO main runs, create one split manifest with `scripts/qwen3vl/build_coco_object_split_manifest.py`, then pass it to extraction with `--image-ids-json` and `--image-ids-key`.
5. Append exact extraction, training, evaluation, and visualization commands to `command_log.txt`.
6. Copy or symlink small JSON summaries into the run directory. Keep large parquets, adapters, model files, and image dumps outside Git.
7. Fill `run_record.md` after each stage: data built, training finished, evaluation finished, qualitative analysis finished.
8. Update `experiments/registry.csv` from `planned` to `running`, `completed`, `failed`, or `superseded`.

The `init_experiment_run.py` helper writes:

```text
environment.json
git_diff.patch
config.yaml
run_record.md
command_log.txt
train_summary.json
sensitivity.json
ranking.json
semantic_eval.json
failure_cases.json
qualitative_panels/
```

For confirmatory results, do not rely on a dirty working tree. Commit the code first, then initialize the run so `environment.json` points to a stable commit.
