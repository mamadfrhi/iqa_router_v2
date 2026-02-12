import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from PIL import Image

try:
    from PIL.Image import Resampling
    RESAMPLE: Any = Resampling.BILINEAR
except Exception:
    RESAMPLE = 2

sys.path.insert(0, os.path.dirname(__file__))

from pyiqa_utils import load_labels_df


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


def _conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    H, W = img.shape
    out = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            region = padded[i : i + kh, j : j + kw]
            out[i, j] = float(np.sum(region * kernel))
    return out


def _laplacian_var(gray: np.ndarray) -> float:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    resp = _conv2d(gray, kernel)
    return float(np.var(resp))


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract simple content features for routing")
    ap.add_argument("--labels", required=False, type=str, default="", help="CSV with image_path")
    ap.add_argument("--dataset", type=str, default="", help="PyIQA dataset name (e.g., koniq10k, spaq, livec)")
    ap.add_argument("--data-root", type=str, default="", help="Root datasets folder used by PyIQA")
    ap.add_argument("--phase", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-index", type=str, default="")
    ap.add_argument("--mos-normalize", action="store_true")
    ap.add_argument("--content-size", required=False, type=str, default=None)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    if args.labels:
        df = pd.read_csv(args.labels)
        if "image_path" not in df.columns:
            raise ValueError("labels must include column 'image_path'")
    else:
        if not args.dataset or not args.data_root:
            raise ValueError("Provide --labels or --dataset with --data-root")
        df = load_labels_df(
            dataset=args.dataset,
            data_root=args.data_root,
            phase=args.phase,
            split_index=args.split_index,
            mos_normalize=bool(args.mos_normalize),
        )

    target_size = _parse_size(args.content_size)

    rows: List[Dict] = []
    n = 0
    total = len(df)
    for _, r in df.iterrows():
        path = str(r["image_path"]).strip()
        if not path:
            continue
        feats = _features_for_path(path, target_size)
        row = {"image_path": path, **feats}
        rows.append(row)
        n += 1
        if args.progress_every and (n % args.progress_every == 0 or n == total):
            print(f"features: {n}/{total} processed")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
