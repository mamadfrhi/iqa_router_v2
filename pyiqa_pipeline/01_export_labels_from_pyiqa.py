import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pyiqa_utils import load_labels_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Export standardized labels from PyIQA dataset API")
    ap.add_argument("--dataset", required=True, type=str, help="e.g., koniq10k, spaq, livec")
    ap.add_argument("--data-root", required=True, type=str, help="Root datasets folder used by PyIQA")
    ap.add_argument("--phase", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-index", type=str, default="", help="Split index (e.g., official_split)")
    ap.add_argument("--mos-normalize", action="store_true", help="Normalize MOS to [0,1] if supported by dataset config")
    ap.add_argument("--output", required=True, type=str)
    args = ap.parse_args()

    df = load_labels_df(
        dataset=args.dataset,
        data_root=args.data_root,
        phase=args.phase,
        split_index=args.split_index,
        mos_normalize=bool(args.mos_normalize),
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
