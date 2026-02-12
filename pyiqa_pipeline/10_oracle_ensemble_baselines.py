import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyiqa
from pyiqa.metrics import calculate_plcc, calculate_srcc, calculate_krcc

sys.path.insert(0, os.path.dirname(__file__))

from pyiqa_utils import DATASET_CONFIGS, load_labels_df


def _list_pred_files(preds_dir: str) -> List[str]:
    out = []
    for fn in os.listdir(preds_dir):
        if fn.startswith("preds_") and fn.endswith(".csv"):
            out.append(os.path.join(preds_dir, fn))
    return sorted(out)


def _expert_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    return base.replace("preds_", "").replace(".csv", "")


def _basename(path: str) -> str:
    return os.path.basename(str(path)).strip()


def _with_merge_key(df: pd.DataFrame, image_col: str = "image_path") -> pd.DataFrame:
    out = df.copy()
    if "image" in out.columns:
        out["_merge_key"] = out["image"].astype(str).str.strip()
    else:
        out["_merge_key"] = out[image_col].astype(str).map(_basename)
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    srcc = float(calculate_srcc(y_pred, y_true))
    plcc = float(calculate_plcc(y_pred, y_true))
    krcc = float(calculate_krcc(y_pred, y_true))
    return {
        "SRCC": srcc,
        "PLCC": plcc,
        "KRCC": krcc,
        "MAE": _mae(y_true, y_pred),
        "RMSE": _rmse(y_true, y_pred),
    }


def _load_labels(args: argparse.Namespace) -> pd.DataFrame:
    if args.labels:
        df = pd.read_csv(args.labels)
        if "image_path" not in df.columns:
            raise ValueError("labels must include column 'image_path'")
        if "mos" not in df.columns:
            raise ValueError("labels must include column 'mos'")
        return _with_merge_key(df)

    if not args.dataset or not args.data_root:
        raise ValueError("Provide --labels or --dataset with --data-root")

    if not args.skip_path_check:
        return _with_merge_key(
            load_labels_df(
            dataset=args.dataset,
            data_root=args.data_root,
            phase=args.phase,
            split_index=args.split_index,
            mos_normalize=bool(args.mos_normalize),
            )
        )

    dataset_opts = {
        "phase": args.phase,
        "mos_normalize": bool(args.mos_normalize),
    }
    if args.split_index:
        dataset_opts["split_index"] = args.split_index
    if args.dataset in DATASET_CONFIGS:
        for key, value in DATASET_CONFIGS[args.dataset].items():
            if key not in dataset_opts:
                dataset_opts[key] = value

    ds = pyiqa.load_dataset(args.dataset, data_root=Path(args.data_root), dataset_opts=dataset_opts)
    rows = []
    for item in ds.paths_mos:
        if len(item) == 2:
            image_path = str(item[0])
            ref_path = ""
            mos = float(item[1])
        elif len(item) == 3:
            ref_path = str(item[0])
            image_path = str(item[1])
            mos = float(item[2])
        else:
            raise ValueError("Unsupported paths_mos format from dataset")
        rows.append(
            {
                "image_path": image_path,
                "ref_path": ref_path,
                "mos": mos,
                "image": os.path.basename(image_path),
            }
        )
    return pd.DataFrame(rows)


def _merge_preds(labels: pd.DataFrame, preds_dir: str) -> pd.DataFrame:
    pred_files = _list_pred_files(preds_dir)
    if not pred_files:
        raise ValueError("No preds_*.csv files found in preds-dir")

    base = labels.copy()
    for p in pred_files:
        expert = _expert_name_from_path(p)
        df = pd.read_csv(p)
        if "image_path" not in df.columns or "raw_score" not in df.columns:
            raise ValueError(f"Preds file missing required columns: {p}")
        df = df[["image_path", "raw_score", "latency_ms"]].copy()
        df = _with_merge_key(df)
        df = df.drop(columns=["image_path"], errors="ignore")
        df = df.rename(
            columns={
                "raw_score": f"pred_{expert}",
                "latency_ms": f"latency_ms_{expert}",
            }
        )
        base = base.merge(df, on="_merge_key", how="left")

    return base


def _compute_oracle(preds: np.ndarray, mos: np.ndarray) -> np.ndarray:
    abs_err = np.abs(preds - mos[:, None])
    best_idx = np.argmin(abs_err, axis=1)
    return preds[np.arange(len(mos)), best_idx], best_idx


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute oracle and mean-ensemble baselines from preds_*.csv"
    )
    ap.add_argument("--preds-dir", required=True, type=str)
    ap.add_argument("--labels", type=str, default="")
    ap.add_argument("--dataset", type=str, default="")
    ap.add_argument("--data-root", type=str, default="")
    ap.add_argument("--phase", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-index", type=str, default="")
    ap.add_argument("--mos-normalize", action="store_true")
    ap.add_argument(
        "--skip-path-check",
        action="store_true",
        help="Skip path existence checks when loading labels from PyIQA",
    )
    ap.add_argument("--output-json", required=True, type=str)
    ap.add_argument("--output-csv", type=str, default="")
    args = ap.parse_args()

    labels = _load_labels(args)
    df = _merge_preds(labels, args.preds_dir)

    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    lat_cols = [c for c in df.columns if c.startswith("latency_ms_")]
    if not pred_cols:
        raise ValueError("No pred_* columns found after merge")

    valid_mask = ~df[pred_cols].isna().any(axis=1)
    dropped = int((~valid_mask).sum())
    df = df[valid_mask].copy()
    if len(df) == 0:
        raise ValueError("No rows left after dropping missing predictions")

    y = df["mos"].values.astype(np.float64)
    preds = df[pred_cols].values.astype(np.float64)

    mean_pred = preds.mean(axis=1)
    oracle_pred, oracle_idx = _compute_oracle(preds, y)

    out: Dict[str, object] = {
        "dataset": args.dataset or "custom_labels",
        "phase": args.phase,
        "num_samples": int(len(df)),
        "num_models": int(len(pred_cols)),
        "dropped_rows": dropped,
        "baselines": {
            "mean_ensemble": _metrics(y, mean_pred),
            "oracle": _metrics(y, oracle_pred),
        },
    }

    if lat_cols:
        lat = df[lat_cols].values.astype(np.float64)
        mean_lat = float(np.mean(np.nanmean(lat, axis=1)))
        oracle_lat = float(np.mean(lat[np.arange(len(y)), oracle_idx]))
        out["baselines"]["mean_ensemble"]["avg_latency_ms"] = mean_lat
        out["baselines"]["oracle"]["avg_latency_ms"] = oracle_lat

    if "_merge_key" in df.columns:
        df = df.drop(columns=["_merge_key"])

    if args.output_csv:
        base = pd.DataFrame(
            {
                "image_path": df["image_path"].values,
                "mos": y,
                "pred_mean_ensemble": mean_pred,
                "pred_oracle": oracle_pred,
                "oracle_model": [pred_cols[i].replace("pred_", "") for i in oracle_idx],
            }
        )
        if lat_cols:
            base["latency_ms_mean_ensemble"] = np.nanmean(lat, axis=1)
            base["latency_ms_oracle"] = lat[np.arange(len(y)), oracle_idx]
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        base.to_csv(args.output_csv, index=False)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote baselines to {args.output_json}")


if __name__ == "__main__":
    main()
