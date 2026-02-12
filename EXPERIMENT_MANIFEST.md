# Experiment Manifest (IQA Router v2)

Date: 2026-02-12
Owner: b-s-farrahi
Root: /Users/mamadfarrahi/Desktop/thesisproject/nn-dataset/iqa_router_v2
Outputs (local mirror of server): /Users/mamadfarrahi/Desktop/thesisproject/nn-dataset/iqa_router_v2/outputs

## 0) Environment (set once per shell)

```bash
export ROOT="/Users/mamadfarrahi/Desktop/thesisproject/nn-dataset"
export PROJ="$ROOT/iqa_router_v2"
export PIPE="$PROJ/pyiqa_pipeline"
export OUT="$PROJ/outputs"
```

## 1) Build per-dataset modeling tables (labels + expert preds)

### KonIQ-10k (test)

```bash
python -u "$PIPE/03_build_modeling_table.py" \
  --labels "$OUT/labels_koniq10k_test.csv" \
  --preds-dir "$OUT/preds/koniq10k/test" \
  --output "$OUT/koniq10k_test_modeling.csv"
```

### LIVE-C (val)

```bash
python -u "$PIPE/03_build_modeling_table.py" \
  --labels "$OUT/labels_livec_val.csv" \
  --preds-dir "$OUT/preds/livec/val" \
  --output "$OUT/livec_val_modeling.csv"
```

### SPAQ (val)

```bash
python -u "$PIPE/03_build_modeling_table.py" \
  --labels "$OUT/labels_spaq_val.csv" \
  --preds-dir "$OUT/preds/spaq/val" \
  --output "$OUT/spaq_val_modeling.csv"
```

### TID2013 (test)

```bash
python -u "$PIPE/03_build_modeling_table.py" \
  --labels "$OUT/labels_tid2013_test.csv" \
  --preds-dir "$OUT/preds/tid2013/test" \
  --output "$OUT/tid2013_test_modeling.csv"
```

## 2) Evaluate per-model metrics (single expert baselines)

### KonIQ-10k

```bash
python -u "$PIPE/04_eval_metrics_pyiqa.py" \
  --modeling-csv "$OUT/koniq10k_test_modeling.csv" \
  --output "$OUT/koniq10k_test_metrics.json"
```

### LIVE-C

```bash
python -u "$PIPE/04_eval_metrics_pyiqa.py" \
  --modeling-csv "$OUT/livec_val_modeling.csv" \
  --output "$OUT/livec_val_metrics.json"
```

### SPAQ

```bash
python -u "$PIPE/04_eval_metrics_pyiqa.py" \
  --modeling-csv "$OUT/spaq_val_modeling.csv" \
  --output "$OUT/spaq_val_metrics.json"
```

### TID2013

```bash
python -u "$PIPE/04_eval_metrics_pyiqa.py" \
  --modeling-csv "$OUT/tid2013_test_modeling.csv" \
  --output "$OUT/tid2013_test_metrics.json"
```

## 3) Router training/eval (classical features)

### Train on KonIQ, test on LIVE-C

```bash
# TODO: fill in actual train/eval scripts and config paths
# Example placeholders:
# python -u "$PROJ/pyiqa_pipeline/05_train_router.py" --config "$PROJ/configs/router_classical_koniq.yaml"
# python -u "$PROJ/pyiqa_pipeline/06_eval_router.py" --config "$PROJ/configs/router_classical_koniq.yaml" --split livec_val
```

## 4) Router training/eval (CNN features)

```bash
# TODO: fill in actual train/eval scripts and config paths
```

## 5) Router training/eval (CLIP features)

```bash
# TODO: fill in actual train/eval scripts and config paths
```

## 6) Oracle upper bound

```bash
# TODO: add script for oracle routing or notebook path
```

## 7) Lambda sweep (latency vs quality)

```bash
# TODO: add script/flags for lambda sweep and output location
```

## 8) Deliverables

- Tables:
  - Single-model metrics (KonIQ, LIVE-C, SPAQ, TID2013)
  - Router variants (classical, CNN, CLIP)
  - Ablations (no latency, no features)
- Figures:
  - Oracle upper bound vs best router vs best single model
  - Lambda sweep Pareto (quality vs latency)

## 9) Notes

- Predictions already mirrored from server in $OUT.
- Make sure labels CSVs match the exact split names used on the server.
