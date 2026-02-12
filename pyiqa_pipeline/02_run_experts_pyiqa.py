import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
import pyiqa

sys.path.insert(0, os.path.dirname(__file__))

from pyiqa_utils import load_labels_df


def _parse_expert_list(s: str) -> list:
    out = []
    for tok in str(s).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _load_experts_from_json(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return list(data.get("experts", []))


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
    ap = argparse.ArgumentParser(description="Run PyIQA experts from label CSV")
    ap.add_argument("--labels", required=False, type=str, default="", help="CSV with image_path, ref_path, mos")
    ap.add_argument("--dataset", type=str, default="", help="PyIQA dataset name (e.g., koniq10k, spaq, livec)")
    ap.add_argument("--data-root", type=str, default="", help="Root datasets folder used by PyIQA")
    ap.add_argument("--phase", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-index", type=str, default="")
    ap.add_argument("--mos-normalize", action="store_true")
    ap.add_argument("--experts", type=str, default=None)
    ap.add_argument("--experts-json", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--output-dir", required=True, type=str)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    experts = []
    if args.experts_json:
        experts = _load_experts_from_json(args.experts_json)
    if args.experts:
        experts = _parse_expert_list(args.experts)
    if not experts:
        raise ValueError("Provide --experts or --experts-json")

    if args.labels:
        df = pd.read_csv(args.labels)
        if "image_path" not in df.columns:
            raise ValueError("labels must include column 'image_path'")
        if "ref_path" not in df.columns:
            df["ref_path"] = ""
    else:
        if not args.dataset or not args.data_root:
            raise ValueError("Provide --labels or --dataset with --data-root")
        print(f"Loading dataset {args.dataset} with phase='{args.phase}', split_index='{args.split_index}' from {args.data_root}...")
        df = load_labels_df(
            dataset=args.dataset,
            data_root=args.data_root,
            phase=args.phase,
            split_index=args.split_index,
            mos_normalize=bool(args.mos_normalize),
        )
        print(f"✓ Dataset {args.dataset} loaded successfully with {len(df)} samples from phase='{args.phase}'")
    if args.limit is not None:
        df = df.head(int(args.limit))

    os.makedirs(args.output_dir, exist_ok=True)

    for expert_name in experts:
        out_path = os.path.join(args.output_dir, f"preds_{expert_name}.csv")
        done = set()
        prev_df = pd.DataFrame()  # For appending to existing results
        if args.resume and os.path.exists(out_path):
            try:
                prev_df = pd.read_csv(out_path)
                done = set(prev_df["image_path"].astype(str).str.strip().tolist())
                print(f"Resuming {expert_name}: {len(done)} images already processed")
            except Exception as e:
                print(f"Warning: Could not read existing {out_path}: {e}")
                done = set()
                prev_df = pd.DataFrame()

        model = pyiqa.create_metric(expert_name, device=args.device)
        
        # Warmup immediately after model creation
        if args.warmup > 0 and len(df) > 0:
            first_img = str(df.iloc[0]["image_path"]).strip()
            first_ref = str(df.iloc[0].get("ref_path", "")).strip()
            try:
                for _ in range(int(args.warmup)):
                    if first_ref and getattr(model, "metric_mode", "NR") == "FR":
                        _ = model(first_img, first_ref)
                    else:
                        _ = model(first_img)
                _sync_device(args.device)
                print(f"Warmup done with {args.warmup} iterations on first image.")
            except Exception as e:
                print(f"Warmup failed: {e}")
        
        rows = []
        n = 0
        total = len(df)
        print(f"Running expert '{expert_name}' on {total} images ({args.dataset} phase='{args.phase}' split='{args.split_index}')")
        print(f"{'Resuming' if args.resume else 'Starting fresh'} (done={len(done)})..." if args.resume else "")

        for _, r in df.iterrows():
            img_path = str(r["image_path"]).strip()
            ref_path = str(r.get("ref_path", "")).strip()
            if args.resume and img_path in done:
                n += 1
                continue

            t0 = time.perf_counter()
            if ref_path and getattr(model, "metric_mode", "NR") == "FR":
                y = model(img_path, ref_path)
            else:
                y = model(img_path)
            _sync_device(args.device)
            t1 = time.perf_counter()
            score = float(y.detach().cpu().item())
            latency_ms = (t1 - t0) * 1000.0

            rows.append({
                "image_path": img_path,
                "raw_score": score,
                "latency_ms": latency_ms,
            })
            n += 1
            if args.progress_every and (n % args.progress_every == 0 or n == total):
                print(f"{expert_name}: {n}/{total} processed")

        # IMPORTANT: Merge new results with any existing results to preserve previous runs
        if args.resume and len(prev_df) > 0:
            out_df = pd.DataFrame(rows)
            out_df = pd.concat([prev_df, out_df], ignore_index=True)
            out_df = out_df.drop_duplicates(subset=["image_path"], keep="last")  # Keep latest scores
            print(f"Merged {len(rows)} new results with {len(prev_df)} existing → {len(out_df)} total")
        else:
            out_df = pd.DataFrame(rows)
        
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
