import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib


def _list_cols(prefix: str, cols: List[str]) -> List[str]:
    return sorted([c for c in cols if c.startswith(prefix)])


def _expert_from_pred_col(col: str) -> str:
    assert col.startswith("pred_"), col
    return col[len("pred_") :]


def _lat_col_for_expert(expert: str) -> str:
    return f"latency_ms_{expert}"


def _best_single_by_srcc(y: np.ndarray, preds: Dict[str, np.ndarray]) -> str:
    best_name: Optional[str] = None
    best_srcc = -np.inf
    for name, p in preds.items():
        srcc = np.corrcoef(y, p)[0, 1]
        if np.isnan(srcc):
            continue
        if srcc > best_srcc:
            best_srcc = srcc
            best_name = name
    if best_name is None:
        best_name = list(preds.keys())[0]
    return best_name


def main() -> None:
    ap = argparse.ArgumentParser(description="Train content-based router using PyIQA modeling table")
    ap.add_argument("--modeling-csv", required=True, type=str)
    ap.add_argument("--features-csv", required=False, type=str, default=None)
    ap.add_argument("--output-dir", required=True, type=str)
    ap.add_argument("--latency-norm", type=str, default="log_quantile", choices=["global_minmax", "quantile", "log_quantile"])
    ap.add_argument("--latency-quantiles", type=str, default="0.01,0.99")
    ap.add_argument("--latency-transform", type=str, default=None, choices=["none", "log1p", None])
    ap.add_argument("--exclude-experts", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.modeling_csv)
    if args.features_csv:
        feats = pd.read_csv(args.features_csv)
        if "image_path" not in feats.columns:
            raise ValueError("features-csv must include column 'image_path'")
        df = df.merge(feats, on="image_path", how="left")

    cols = df.columns.tolist()
    pred_cols = _list_cols("pred_", cols)
    excl = set([s.strip() for s in str(args.exclude_experts).split(",") if s.strip()])
    if excl:
        pred_cols = [pc for pc in pred_cols if _expert_from_pred_col(pc) not in excl]
    if not pred_cols:
        raise ValueError("No pred_* columns found in modeling CSV")

    ignore_prefixes = ("pred_", "latency_ms_")
    base_cols = {"image_path", "ref_path", "mos", "image"}
    feature_cols = [c for c in cols if c not in base_cols and not any(c.startswith(p) for p in ignore_prefixes)]
    if feature_cols:
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("No numeric feature columns found; provide --features-csv")

    lat_cols_aligned: List[Optional[str]] = []
    for pc in pred_cols:
        expert = _expert_from_pred_col(pc)
        lc = _lat_col_for_expert(expert)
        lat_cols_aligned.append(lc if lc in cols else None)
    if any(lc is None for lc in lat_cols_aligned):
        missing = [_expert_from_pred_col(pc) for pc, lc in zip(pred_cols, lat_cols_aligned) if lc is None]
        raise ValueError(f"Missing latency columns for experts: {missing}")

    need_cols = ["mos"] + pred_cols + feature_cols + [c for c in lat_cols_aligned if c is not None]
    Xnf = df[need_cols].to_numpy(dtype=np.float64)
    mask = np.isfinite(Xnf).all(axis=1)
    df = df.loc[mask.tolist()].reset_index(drop=True)
    y = df["mos"].to_numpy(dtype=np.float64)
    P = df[pred_cols].to_numpy(dtype=np.float64)
    C = df[feature_cols].to_numpy(dtype=np.float64)

    err_models: List[GradientBoostingRegressor] = []
    for i in range(len(pred_cols)):
        target_err = np.abs(y - P[:, i])
        mdl = GradientBoostingRegressor(random_state=42)
        mdl.fit(C, target_err)
        err_models.append(mdl)

    lat_models: List[Optional[GradientBoostingRegressor]] = [None] * len(pred_cols)
    all_lat_values: List[np.ndarray] = []
    for i, lc in enumerate(lat_cols_aligned):
        if lc is None:
            continue
        y_lat = df[lc].to_numpy(dtype=np.float64)
        if not np.isfinite(y_lat).all():
            raise ValueError(f"Non-finite values in latency column: {lc}")
        lmdl = GradientBoostingRegressor(random_state=42)
        lmdl.fit(C, y_lat)
        lat_models[i] = lmdl
        all_lat_values.append(y_lat)

    lat_norm = args.latency_norm
    lat_min = None
    lat_max = None
    lat_q_low = None
    lat_q_high = None
    lat_transform = args.latency_transform
    if lat_transform is None:
        lat_transform = "log1p" if lat_norm == "log_quantile" else "none"

    lat_mean_per_expert: List[Optional[float]] = [None] * len(pred_cols)
    if all_lat_values:
        all_lat_concat = np.concatenate(all_lat_values, axis=0)
        base_vals = all_lat_concat.astype(np.float64)
        if lat_norm == "global_minmax":
            tr_vals = base_vals if lat_transform == "none" else np.log1p(base_vals)
            mn = float(np.min(tr_vals))
            mx = float(np.max(tr_vals))
        elif lat_norm in ("quantile", "log_quantile"):
            try:
                ql_str, qh_str = str(args.latency_quantiles).split(",")
                lat_q_low = float(ql_str)
                lat_q_high = float(qh_str)
            except Exception:
                lat_q_low, lat_q_high = 0.01, 0.99
            lat_q_low = max(0.0, min(1.0, lat_q_low))
            lat_q_high = max(0.0, min(1.0, lat_q_high))
            if not (0.0 <= lat_q_low < lat_q_high <= 1.0):
                raise ValueError("Invalid --latency-quantiles; must satisfy 0 <= low < high <= 1")
            tr_vals = base_vals if lat_transform == "none" else np.log1p(base_vals)
            mn = float(np.quantile(tr_vals, lat_q_low))
            mx = float(np.quantile(tr_vals, lat_q_high))
        else:
            raise ValueError(f"Unsupported latency_norm: {lat_norm}")
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            raise ValueError("Invalid latency normalization min/max computed from data")
        lat_min, lat_max = mn, mx
        for i, lc in enumerate(lat_cols_aligned):
            if lc is None:
                continue
            lat_mean_per_expert[i] = float(np.mean(df[lc].to_numpy(dtype=np.float64)))

    pred_map = {pred_cols[i]: P[:, i] for i in range(len(pred_cols))}
    best_single = _best_single_by_srcc(y, pred_map)

    assets_path = os.path.join(args.output_dir, "router_assets.joblib")
    assets = {"err_models": err_models, "lat_models": lat_models}
    joblib.dump(assets, assets_path)

    meta = {
        "source_modeling_csv": args.modeling_csv,
        "pred_cols": pred_cols,
        "latency_cols": lat_cols_aligned,
        "feature_cols": feature_cols,
        "features_mode": "content",
        "include_features": True,
        "latency_norm": lat_norm,
        "latency_min": lat_min,
        "latency_max": lat_max,
        "latency_transform": lat_transform,
        "latency_q_low": lat_q_low,
        "latency_q_high": lat_q_high,
        "latency_mean_per_expert": lat_mean_per_expert,
        "has_lat_models": any(m is not None for m in lat_models),
        "best_single": best_single,
        "excluded_experts": sorted(list(excl)) if excl else [],
    }
    meta_path = os.path.join(args.output_dir, "router_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(meta_path)


if __name__ == "__main__":
    main()
