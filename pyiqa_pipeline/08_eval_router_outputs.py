import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from pyiqa.metrics import calculate_plcc, calculate_srcc, calculate_krcc


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _metrics(y_true, y_pred) -> dict:
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
    ap = argparse.ArgumentParser(description="Evaluate router outputs using PyIQA metrics")
    ap.add_argument("--router-csv", required=True, type=str, help="CSV from gate_router with raw_score and mos")
    ap.add_argument("--output", required=True, type=str)
    args = ap.parse_args()

    df = pd.read_csv(args.router_csv)
    if "mos" not in df.columns or "raw_score" not in df.columns:
        raise ValueError("router CSV must include mos and raw_score")

    y = df["mos"].values.astype(np.float64)
    yhat = df["raw_score"].values.astype(np.float64)

    out: Dict[str, Any] = {"metrics": _metrics(y, yhat)}
    if "latency_ms" in df.columns:
        lat_vals = np.asarray(df["latency_ms"].values, dtype=np.float64)
        out["avg_latency_ms"] = float(np.mean(lat_vals))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote router metrics to {args.output}")


if __name__ == "__main__":
    main()
