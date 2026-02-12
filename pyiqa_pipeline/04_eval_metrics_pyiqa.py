import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from pyiqa.metrics import calculate_plcc, calculate_srcc, calculate_krcc


def _list_pred_cols(cols) -> list:
    return sorted([c for c in cols if c.startswith("pred_")])


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _metrics(y_true, y_pred) -> Dict[str, float]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute metrics for each expert and baselines")
    ap.add_argument("--modeling-csv", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    args = ap.parse_args()

    df = pd.read_csv(args.modeling_csv)
    if "mos" not in df.columns:
        raise ValueError("modeling CSV must include mos")

    y = df["mos"].values.astype(np.float64)
    pred_cols = _list_pred_cols(df.columns)
    if not pred_cols:
        raise ValueError("No pred_* columns found")

    out: Dict = {"experts": {}, "baselines": {}}

    for c in pred_cols:
        name = c.replace("pred_", "")
        y_pred = df[c].values.astype(np.float64)
        mask = ~np.isnan(y_pred) & ~np.isnan(y)
        if not np.any(mask):
            out["experts"][name] = {"error": "no valid samples"}
            continue
        out["experts"][name] = _metrics(y[mask], y_pred[mask])

    yhat_uniform = df[pred_cols].mean(axis=1).values.astype(np.float64)
    mask_uniform = ~np.isnan(yhat_uniform) & ~np.isnan(y)
    if np.any(mask_uniform):
        out["baselines"]["uniform"] = _metrics(y[mask_uniform], yhat_uniform[mask_uniform])
    else:
        out["baselines"]["uniform"] = {"error": "no valid samples"}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
