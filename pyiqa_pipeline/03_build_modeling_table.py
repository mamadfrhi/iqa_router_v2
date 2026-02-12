import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from pyiqa_utils import load_labels_df


def _list_pred_files(preds_dir: str) -> List[str]:
    out = []
    for fn in os.listdir(preds_dir):
        if fn.startswith("preds_") and fn.endswith(".csv"):
            out.append(os.path.join(preds_dir, fn))
    return sorted(out)


def _expert_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name = base.replace("preds_", "").replace(".csv", "")
    return name


def _basename(path: str) -> str:
    return os.path.basename(str(path)).strip()


def _with_merge_key(df: pd.DataFrame, image_col: str = "image_path") -> pd.DataFrame:
    out = df.copy()
    if "image" in out.columns:
        out["_merge_key"] = out["image"].astype(str).str.strip()
    else:
        out["_merge_key"] = out[image_col].astype(str).map(_basename)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build modeling table: labels + pred_* + latency_ms_*.")
    ap.add_argument("--labels", required=False, type=str, default="")
    ap.add_argument("--dataset", type=str, default="", help="PyIQA dataset name (e.g., koniq10k, spaq, livec)")
    ap.add_argument("--data-root", type=str, default="", help="Root datasets folder used by PyIQA")
    ap.add_argument("--phase", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-index", type=str, default="")
    ap.add_argument("--mos-normalize", action="store_true")
    ap.add_argument("--preds-dir", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    args = ap.parse_args()

    if args.labels:
        labels = pd.read_csv(args.labels)
        if "image_path" not in labels.columns:
            raise ValueError("labels must include column 'image_path'")
        base = _with_merge_key(labels)
    else:
        if not args.dataset or not args.data_root:
            raise ValueError("Provide --labels or --dataset with --data-root")
        base = load_labels_df(
            dataset=args.dataset,
            data_root=args.data_root,
            phase=args.phase,
            split_index=args.split_index,
            mos_normalize=bool(args.mos_normalize),
        )
        base = _with_merge_key(base)

    pred_files = _list_pred_files(args.preds_dir)
    if not pred_files:
        raise ValueError("No preds_*.csv files found in preds-dir")

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

    pred_cols = [c for c in base.columns if c.startswith("pred_")]
    miss = int(np.isnan(base[pred_cols].values).sum())
    if miss > 0:
        print(f"Warning: {miss} missing prediction values after merge")

    if "_merge_key" in base.columns:
        base = base.drop(columns=["_merge_key"])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    base.to_csv(args.output, index=False)
    print(f"Wrote modeling table with {len(base)} rows to {args.output}")


if __name__ == "__main__":
    main()
