import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple, cast, Iterable

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

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
    if _HAS_CV2:
        return cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REFLECT)
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
    if _HAS_CV2:
        resp = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        return float(np.var(resp))
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    resp = _conv2d(gray, kernel)
    return float(np.var(resp))


def _gradient_energy(gray: np.ndarray) -> float:
    if _HAS_CV2:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        g2 = gx * gx + gy * gy
        return float(np.mean(g2))
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    g2 = gx * gx + gy * gy
    return float(np.mean(g2))


def _edge_density(gray: np.ndarray) -> float:
    if _HAS_CV2:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        thr = float(np.mean(mag) + np.std(mag))
        if thr <= 0:
            return 0.0
        return float(np.mean(mag > thr))
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


def _compute_row(arg: Tuple[str, Optional[Tuple[int, int]]]) -> Dict[str, float]:
    path, target_size = arg
    feats = _features_for_path(path, target_size)
    return {"image_path": path, **feats}


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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0, help="0=auto, 1=disable parallel")
    ap.add_argument("--backend", type=str, default="processes", choices=["threads", "processes"])
    ap.add_argument("--chunksize", type=int, default=50)
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

    if args.limit is not None:
        df = df.head(int(args.limit))

    target_size = _parse_size(args.content_size)

    task_args: List[Tuple[str, Optional[Tuple[int, int]]]] = []
    for _, r in df.iterrows():
        path = str(r["image_path"]).strip()
        if not path:
            continue
        task_args.append((path, target_size))

    rows: List[Dict] = []
    total = len(task_args)
    progress_every = int(max(1, args.progress_every))

    if args.workers <= 1 or total == 0:
        for i, arg in enumerate(task_args, start=1):
            rows.append(_compute_row(arg))
            if args.progress_every and (i % progress_every == 0 or i == total):
                print(f"features: {i}/{total} processed")
    else:
        max_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count())
        Executor = ProcessPoolExecutor if args.backend == "processes" else ThreadPoolExecutor
        with Executor(max_workers=max_workers) as ex:
            if args.backend == "processes":
                it: Iterable[Dict] = ex.map(_compute_row, task_args, chunksize=max(1, int(args.chunksize)))
            else:
                it = ex.map(_compute_row, task_args)
            for i, row in enumerate(it, start=1):
                rows.append(row)
                if args.progress_every and (i % progress_every == 0 or i == total):
                    print(f"features: {i}/{total} processed")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
