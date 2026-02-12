import argparse
import json

import pyiqa


def main() -> None:
    ap = argparse.ArgumentParser(description="List available PyIQA models")
    ap.add_argument("--metric-mode", type=str, default="NR", choices=["NR", "FR"])
    ap.add_argument("--filter", type=str, default="")
    ap.add_argument("--exclude", type=str, default="")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    models = pyiqa.list_models(metric_mode=args.metric_mode, filter=args.filter, exclude_filters=args.exclude)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"metric_mode": args.metric_mode, "models": models}, f, indent=2)
        print(f"Wrote {len(models)} models to {args.output}")
    else:
        for m in models:
            print(m)


if __name__ == "__main__":
    main()
