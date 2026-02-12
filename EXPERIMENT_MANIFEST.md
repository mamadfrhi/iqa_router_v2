# Experiment Manifest (IQA Router v2)

Date: 2026-02-12
Owner: b-s-farrahi
Root: /shared/ssd/home/b-s-farrahi/iqa_router_v2
Outputs (local mirror of server): /shared/ssd/home/b-s-farrahi/iqa_router_v2/outputs

## 0) Environment (set once per shell)

```bash
export ROOT="/shared/ssd/home/b-s-farrahi"
export PROJ="$ROOT/iqa_router_v2"
export PIPE="$PROJ/pyiqa_pipeline"
export OUT="$PROJ/outputs"
export DATA_ROOT="/shared/hdd/data/b-s-farrahi"
export DATASETS_DIR="$DATA_ROOT"
export META_INFO_DIR="$DATA_ROOT/meta_info"
```

## 0.1) Python environment (create once)

```bash
# Option A: venv (recommended for quick setup)
python -m venv "$ROOT/.venv/iqa_router_v2"
source "$ROOT/.venv/iqa_router_v2/bin/activate"
python -m pip install --upgrade pip

# Torch: choose ONE line based on your hardware
# CPU-only:
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.1 torchvision==0.17.1
# CUDA 12.1:
# python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1+cu121 torchvision==0.17.1+cu121

# Core deps (mirrors Dockerfile pins)
python -m pip install \
  "pyiqa==0.1.14.1" --no-deps
python -m pip install \
  "numpy<2" "scipy<1.15" pandas timm pillow opencv-python-headless scikit-image joblib tqdm yacs natsort \
  "transformers==4.37.2" "accelerate<=1.1.0" einops sentencepiece safetensors "huggingface-hub<1.0" datasets \
  openai-clip facexlib lmdb addict icecream protobuf bitsandbytes future pre-commit pytest ruff tensorboard yapf
```

Notes:
- DATA_ROOT points to your HDD dataset root. This folder must contain the paths expected by PyIQA (see pyiqa_pipeline/pyiqa_utils.py).
- Expected layout under $DATASETS_DIR (now the HDD root):
  - meta_info/meta_info_*.csv
  - koniq10k/512x384/
  - SPAQ/TestImage/
  - LIVEC/
  - tid2013/distorted_images/ and reference_images/
```

## 1) Export labels and build per-dataset modeling tables (labels + expert preds)

### 1.1) Export labels via PyIQA

```bash
# KonIQ-10k (test)
python -u "$PIPE/01_export_labels_from_pyiqa.py" \
  --dataset koniq10k \
  --data-root "$DATA_ROOT" \
  --phase test \
  --output "$OUT/labels_koniq10k_test.csv"

# LIVE-C (val)
python -u "$PIPE/01_export_labels_from_pyiqa.py" \
  --dataset livec \
  --data-root "$DATA_ROOT" \
  --phase val \
  --output "$OUT/labels_livec_val.csv"

# SPAQ (val)
python -u "$PIPE/01_export_labels_from_pyiqa.py" \
  --dataset spaq \
  --data-root "$DATA_ROOT" \
  --phase val \
  --output "$OUT/labels_spaq_val.csv"

# TID2013 (test)
python -u "$PIPE/01_export_labels_from_pyiqa.py" \
  --dataset tid2013 \
  --data-root "$DATA_ROOT" \
  --phase test \
  --output "$OUT/labels_tid2013_test.csv"
```

### 1.2) Run experts (if preds are missing)

```bash
# Example: run all experts for KonIQ-10k test
python -u "$PIPE/02_run_experts_pyiqa.py" \
  --labels "$OUT/labels_koniq10k_test.csv" \
  --experts-json "$PROJ/configs/experts.json" \
  --device cuda \
  --output-dir "$OUT/preds/koniq10k/test" \
  --resume
```

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

## 3) Router training/eval (content features)

### 3.1) Extract content features (must match gating content-size)

```bash
# KonIQ-10k (test)
python -u "$PIPE/05_extract_features.py" \
  --labels "$OUT/labels_koniq10k_test.csv" \
  --content-size 512x384 \
  --output "$OUT/features_koniq10k_test_512x384.csv"

# LIVE-C (val)
python -u "$PIPE/05_extract_features.py" \
  --labels "$OUT/labels_livec_val.csv" \
  --content-size 512x384 \
  --output "$OUT/features_livec_val_512x384.csv"

# SPAQ (val)
python -u "$PIPE/05_extract_features.py" \
  --labels "$OUT/labels_spaq_val.csv" \
  --content-size 512x384 \
  --output "$OUT/features_spaq_val_512x384.csv"

# TID2013 (test)
python -u "$PIPE/05_extract_features.py" \
  --labels "$OUT/labels_tid2013_test.csv" \
  --content-size 512x384 \
  --output "$OUT/features_tid2013_test_512x384.csv"
```

### 3.2) Train on KonIQ, test on LIVE-C

```bash
# Train router on KonIQ (content features)
python -u "$PIPE/06_train_router_pyiqa.py" \
  --modeling-csv "$OUT/koniq10k_test_modeling.csv" \
  --features-csv "$OUT/features_koniq10k_test_512x384.csv" \
  --output-dir "$OUT/routers/koniq10k_content_512x384"

# Gate LIVE-C with trained router (run chosen experts to get raw_score)
python -u "$PIPE/07_gate_router_pyiqa.py" \
  --labels "$OUT/labels_livec_val.csv" \
  --features-csv "$OUT/features_livec_val_512x384.csv" \
  --content-size 512x384 \
  --router-dir "$OUT/routers/koniq10k_content_512x384" \
  --device cuda \
  --lambda 0,0.05,0.1,0.2,0.5,0.6,0.7,0.8,0.9,1.0 \
  --run-chosen \
  --output "$OUT/routers/koniq10k_content_512x384/livec_gate_lam{lam:.2f}.csv"

# Evaluate a single lambda output
python -u "$PIPE/08_eval_router_outputs.py" \
  --router-csv "$OUT/routers/koniq10k_content_512x384/livec_gate_lam0.10.csv" \
  --output "$OUT/routers/koniq10k_content_512x384/livec_metrics_lam0.10.json"
```

```bash
# TODO: fill in actual train/eval scripts and config paths
# Example placeholders:
# python -u "$PROJ/pyiqa_pipeline/05_train_router.py" --config "$PROJ/configs/router_classical_koniq.yaml"
# python -u "$PROJ/pyiqa_pipeline/06_eval_router.py" --config "$PROJ/configs/router_classical_koniq.yaml" --split livec_val
```

## 4) Router training/eval (CNN features)

```bash
# NOTE: No CNN feature extractor script is implemented yet in this repo.
# If you have precomputed features, first align them with image_path:
# python - <<'PY'
# import pandas as pd
# labels = pd.read_csv("$OUT/labels_koniq10k_test.csv")
# feats = pd.read_csv("$ROOT/IQA/features/koniq_features_512x384.csv")
# feats["image"] = feats["image"].astype(str).str.strip()
# labels["image"] = labels["image_path"].astype(str).str.split("/").str[-1]
# merged = labels[["image_path","image"]].merge(feats, on="image", how="left")
# merged.drop(columns=["image"], inplace=True)
# merged.to_csv("$OUT/features_koniq10k_cnn_512x384.csv", index=False)
# PY
# Then reuse 06_train_router_pyiqa.py + 07_gate_router_pyiqa.py with --features-csv.
```

## 5) Router training/eval (CLIP features)

```bash
# NOTE: No CLIP feature extractor script is implemented yet in this repo.
# If you have CLIP features with image_path, train the router the same way as section 3.2.
```

## 6) Oracle upper bound

```bash
# Oracle + mean-ensemble baselines from preds_*.csv
python -u "$PIPE/10_oracle_ensemble_baselines.py" \
  --preds-dir "$OUT/preds/koniq10k/test" \
  --labels "$OUT/labels_koniq10k_test.csv" \
  --output-json "$OUT/koniq10k_test_oracle.json" \
  --output-csv "$OUT/koniq10k_test_oracle.csv"
```

## 7) Lambda sweep (latency vs quality)

```bash
# Lambda sweep is done via --lambda in gate_router_pyiqa.py
# Example: sweep multiple lambdas and evaluate all outputs
python -u "$PIPE/07_gate_router_pyiqa.py" \
  --labels "$OUT/labels_livec_val.csv" \
  --features-csv "$OUT/features_livec_val_512x384.csv" \
  --content-size 512x384 \
  --router-dir "$OUT/routers/koniq10k_content_512x384" \
  --device cuda \
  --lambda 0,0.02,0.05,0.1,0.2,0.5,1.0 \
  --run-chosen \
  --output "$OUT/routers/koniq10k_content_512x384/livec_gate_lam{lam:.2f}.csv"

# Evaluate all lambda outputs (example loop)
for f in "$OUT/routers/koniq10k_content_512x384"/livec_gate_lam*.csv; do
  base=$(basename "$f" .csv)
  python -u "$PIPE/08_eval_router_outputs.py" \
    --router-csv "$f" \
    --output "$OUT/routers/koniq10k_content_512x384/${base}_metrics.json"
done
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
