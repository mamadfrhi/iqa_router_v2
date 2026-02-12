import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
import torch
import pyiqa
from PIL import Image

try:
    from PIL.Image import Resampling
    RESAMPLE: Any = Resampling.BILINEAR
except Exception:
    RESAMPLE = 2

sys.path.insert(0, os.path.dirname(__file__))

from pyiqa_utils import load_labels_df


def _expert_from_pred_col(col: str) -> str:
    assert col.startswith("pred_"), col
    return col[len("pred_") :]


def _parse_size(s: str) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    try:
        w_str, h_str = str(s).lower().split("x")
        return int(w_str), int(h_str)
    except Exception:
        return None


def _to_numpy_rgb(path: str, target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    if target_size is not None:
        resample = cast(Any, RESAMPLE)
        try:
            img.thumbnail((int(target_size[0]), int(target_size[1])), resample)
        except Exception:
            pass
        try:
            img = img.resize((int(target_size[0]), int(target_size[1])), resample)
        except Exception:
            pass
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
    return arr, gray, (orig_w, orig_h)


def _laplacian_var(gray: np.ndarray) -> float:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad_h, pad_w = 1, 1
    padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    H, W = gray.shape
    out = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            region = padded[i : i + 3, j : j + 3]
            out[i, j] = float(np.sum(region * kernel))
    return float(np.var(out))


def _gradient_energy(gray: np.ndarray) -> float:
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    g2 = gx * gx + gy * gy
    return float(np.mean(g2))


def _edge_density(gray: np.ndarray) -> float:
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    thr = float(np.mean(mag) + np.std(mag))
    if thr <= 0:
        return 0.0
    return float(np.mean(mag > thr))


def _entropy(gray: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram(np.clip(gray * 255.0, 0, 255).astype(np.uint8), bins=bins, range=(0, 255))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _colorfulness(rgb: np.ndarray) -> float:
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def _saturation_stats(rgb: np.ndarray) -> Tuple[float, float]:
    mx = np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
    mn = np.minimum(np.minimum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
    sat = np.zeros_like(mx)
    denom = mx + 1e-8
    sat[mx > 0] = 1.0 - (mn[mx > 0] / denom[mx > 0])
    return float(np.mean(sat)), float(np.std(sat))


def _features_for_path(path: str, target_size: Optional[Tuple[int, int]]) -> Dict[str, float]:
    rgb, gray, (orig_w, orig_h) = _to_numpy_rgb(path, target_size=target_size)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    entropy = _entropy(gray)
    lapvar = _laplacian_var(gray)
    grad_energy = _gradient_energy(gray)
    edge_dens = _edge_density(gray)
    colorf = _colorfulness(rgb)
    sat_mean, sat_std = _saturation_stats(rgb)
    mp = float(orig_h * orig_w) / 1e6
    return {
        "brightness": brightness,
        "contrast": contrast,
        "entropy": entropy,
        "laplacian_var": lapvar,
        "gradient_energy": grad_energy,
        "edge_density": edge_dens,
        "colorfulness": colorf,
        "saturation_mean": sat_mean,
        "saturation_std": sat_std,
        "width": float(orig_w),
        "height": float(orig_h),
        "megapixels": mp,
    }


def _normalize_latency(pred_lat: np.ndarray, lat_min: Optional[float], lat_max: Optional[float], transform: Optional[str] = None) -> np.ndarray:
    if lat_min is None or lat_max is None or not np.isfinite(lat_min) or not np.isfinite(lat_max) or lat_max <= lat_min:
        raise ValueError("Invalid latency normalization parameters (latency_min/max)")
    x = pred_lat.astype(np.float64)
    if (transform or "none").lower() == "log1p":
        x = np.log1p(x)
    x = np.clip(x, float(lat_min), float(lat_max))
    out = (x - float(lat_min)) / (float(lat_max) - float(lat_min))
    return out


def _parse_lambdas(s: str) -> List[float]:
    vals: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals


def _sync_device(device: str) -> None:
    dev = (device or "").lower()
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate images using a trained router (PyIQA inference)")
    ap.add_argument("--labels", required=False, type=str, default="", help="CSV with image_path, ref_path, mos")
    ap.add_argument("--dataset", type=str, default="", help="PyIQA dataset name (e.g., koniq10k, spaq, livec)")
    ap.add_argument("--data-root", type=str, default="", help="Root datasets folder used by PyIQA")
    ap.add_argument("--phase", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-index", type=str, default="")
    ap.add_argument("--mos-normalize", action="store_true")
    ap.add_argument("--features-csv", type=str, default="")
    ap.add_argument("--content-size", required=True, type=str)
    ap.add_argument("--router-dir", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--lambda", dest="lam", type=str, default="0,0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--run-chosen", action="store_true")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--exclude-experts", type=str, default="")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--error-margin", type=float, default=0.0)
    args = ap.parse_args()

    meta_path = os.path.join(args.router_dir, "router_meta.json")
    assets_path = os.path.join(args.router_dir, "router_assets.joblib")
    if not os.path.isfile(meta_path) or not os.path.isfile(assets_path):
        raise FileNotFoundError("router_meta.json or router_assets.joblib not found in --router-dir")

    with open(meta_path, "r") as f:
        rmeta = json.load(f)
    assets = joblib.load(assets_path)

    err_models: List = assets.get("err_models", [])
    lat_models: List = assets.get("lat_models", [])
    if not err_models:
        raise ValueError("router assets missing err_models")
    if (not isinstance(lat_models, list)) or (len(lat_models) != len(err_models)) or any(lm is None for lm in lat_models):
        raise ValueError("router assets must include complete lat_models aligned with err_models")

    pred_cols: List[str] = rmeta["pred_cols"]
    feature_cols: List[str] = rmeta.get("feature_cols", [])
    lat_min = rmeta.get("latency_min")
    lat_max = rmeta.get("latency_max")
    lat_transform = rmeta.get("latency_transform", "none")

    experts: List[str] = [_expert_from_pred_col(c) for c in pred_cols]

    excl_set = set()
    if args.exclude_experts:
        tokens = [t.strip() for t in str(args.exclude_experts).split(",") if t.strip()]
        excl_set = set(t.upper() for t in tokens)
    if excl_set:
        keep_idx = [i for i, e in enumerate(experts) if e.upper() not in excl_set]
        if not keep_idx:
            raise ValueError("All experts excluded; nothing to gate")
        experts = [experts[i] for i in keep_idx]
        err_models = [err_models[i] for i in keep_idx]
        lat_models = [lat_models[i] for i in keep_idx]

    target_size = _parse_size(args.content_size)
    if not target_size:
        raise ValueError("Content size must be provided via --content-size as WxH")

    if args.labels:
        labels = pd.read_csv(args.labels)
        if "image_path" not in labels.columns:
            raise ValueError("labels must include column 'image_path'")
    else:
        if not args.dataset or not args.data_root:
            raise ValueError("Provide --labels or --dataset with --data-root")
        labels = load_labels_df(
            dataset=args.dataset,
            data_root=args.data_root,
            phase=args.phase,
            split_index=args.split_index,
            mos_normalize=bool(args.mos_normalize),
        )

    feat_rows_exact: Dict[str, Dict] = {}
    feat_rows_lower: Dict[str, Dict] = {}
    if args.features_csv:
        fdf = pd.read_csv(args.features_csv)
        if "image_path" not in fdf.columns:
            raise ValueError("features-csv must include column 'image_path'")
        for _, rr in fdf.iterrows():
            name = str(rr["image_path"]).strip()
            d = rr.to_dict()
            feat_rows_exact[name] = d
            feat_rows_lower[name.lower()] = d

    lambdas = _parse_lambdas(args.lam)
    runners: Dict[str, Any] = {}

    for lam_value in lambdas:
        rows: List[Dict] = []
        n = 0
        total = len(labels)
        print(f"=== lambda={lam_value} ===")
        for _, r in labels.iterrows():
            img_path = str(r["image_path"]).strip()
            if not img_path:
                continue

            t_gate_start = time.perf_counter()
            t0 = time.perf_counter()
            if args.features_csv:
                row = feat_rows_exact.get(img_path) or feat_rows_lower.get(img_path.lower())
                if row is None:
                    continue
                feats = dict(row)
                t1 = time.perf_counter()
            else:
                feats = _features_for_path(img_path, target_size)
                t1 = time.perf_counter()

            vals: List[float] = []
            for k in feature_cols:
                v = feats.get(k)
                if v is None:
                    vf = np.nan
                else:
                    try:
                        vf = float(v)
                    except Exception:
                        vf = np.nan
                if not np.isfinite(vf):
                    vf = np.nan
                vals.append(vf)
            X = np.array(vals, dtype=np.float64).reshape(1, -1)
            if not np.isfinite(X).all():
                continue

            t2 = time.perf_counter()
            pred_errs = np.array([float(m.predict(X)[0]) for m in err_models], dtype=np.float64)
            t3 = time.perf_counter()
            pred_lats = np.array([float(lm.predict(X)[0]) for lm in lat_models], dtype=np.float64)
            t4 = time.perf_counter()
            if not (np.isfinite(pred_errs).all() and np.isfinite(pred_lats).all()):
                continue

            t5 = time.perf_counter()
            lat_normed = _normalize_latency(pred_lats, lat_min, lat_max, lat_transform)
            cost = pred_errs + float(lam_value) * lat_normed
            choice: Optional[int] = None

            if args.error_margin and float(args.error_margin) > 0.0:
                e_min = float(np.min(pred_errs))
                tau = float(args.error_margin)
                mask = pred_errs <= (e_min + tau)
                cand_idx = np.where(mask)[0]
                if cand_idx.size == 0:
                    cand_idx = np.arange(len(experts))
                if args.top_k and int(args.top_k) > 0:
                    k = int(args.top_k)
                    k = max(1, min(k, cand_idx.size))
                    order_local = np.argsort(pred_errs[cand_idx])
                    cand_idx = cand_idx[order_local[:k]]
                choice = int(cand_idx[int(np.argmin(cost[cand_idx]))])
            elif args.top_k and int(args.top_k) > 0:
                k = int(args.top_k)
                k = max(1, min(k, len(experts)))
                order = np.argsort(pred_errs)
                cand = order[:k]
                choice = int(cand[int(np.argmin(cost[cand]))])
            else:
                choice = int(np.argmin(cost))

            chosen = experts[choice]
            t6 = time.perf_counter()

            gate_ms = (t6 - t_gate_start) * 1000.0
            feat_ms = (t1 - t0) * 1000.0
            err_pred_ms = (t3 - t2) * 1000.0
            lat_pred_ms = (t4 - t3) * 1000.0
            select_ms = (t6 - t5) * 1000.0

            rec: Dict = {
                "image_path": img_path,
                "chosen_expert": chosen,
                "chosen_err_pred": float(pred_errs[choice]),
                "chosen_lat_pred_ms": float(pred_lats[choice]),
                "chosen_cost": float(cost[choice]),
                "gate_ms": float(gate_ms),
                "feat_ms": float(feat_ms),
                "err_pred_ms": float(err_pred_ms),
                "lat_pred_ms": float(lat_pred_ms),
                "select_ms": float(select_ms),
            }

            if "mos" in labels.columns:
                try:
                    rec["mos"] = float(r["mos"]) if np.isfinite(float(r["mos"])) else None
                except Exception:
                    rec["mos"] = None

            if args.run_chosen:
                runner = runners.get(chosen)
                if runner is None:
                    runner = pyiqa.create_metric(chosen, device=args.device)
                    runners[chosen] = runner
                t_run0 = time.perf_counter()
                y = runner(img_path)
                _sync_device(args.device)
                t_run1 = time.perf_counter()
                score = float(y.detach().cpu().item())
                expert_latency_ms = (t_run1 - t_run0) * 1000.0
                rec.update({
                    "raw_score": score,
                    "expert_latency_ms": float(expert_latency_ms),
                    "latency_ms": float(gate_ms + expert_latency_ms),
                })

            rows.append(rec)
            n += 1
            if args.progress_every and (n % args.progress_every == 0 or n == total):
                print(f"[gate] {n}/{total} processed")

        base_out = args.output or "gating_results_lam{lam:.2f}.csv"
        if "{lam" in base_out:
            try:
                out_path = base_out.format(lam=lam_value)
            except Exception:
                root, ext = os.path.splitext(base_out)
                out_path = f"{root}_lam{lam_value:.2f}{ext or '.csv'}"
        elif base_out.lower().endswith(".csv"):
            out_path = base_out
        else:
            out_dir = base_out
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"gating_results_lam{lam_value:.2f}.csv")

        out_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved gating results: {out_path} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
