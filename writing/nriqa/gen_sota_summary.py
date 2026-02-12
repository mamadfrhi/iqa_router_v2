import csv
import re

csv_path = "/Users/mamadfarrahi/Desktop/thesisproject/nn-dataset/iqa_scripts/nriqa/nriqa_sota_tracker.csv"

TARGET_DATASETS = ["KonIQ-10k", "SPAQ", "KADID-10k", "CLIVE"]

score_re = re.compile(r"\b(SRCC|PLCC|KRCC):\s*([0-9\.,\s]+)")

best = {d: {"SRCC": (None, None), "PLCC": (None, None)} for d in TARGET_DATASETS}

with open(csv_path, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        datasets = row.get("datasets", "")
        if not datasets:
            continue
        dataset_list = [d.strip() for d in datasets.split(";") if d.strip()]
        scores_text = row.get("reported_scores", "")
        if not scores_text:
            continue
        scores = {}
        for m in score_re.finditer(scores_text):
            metric = m.group(1)
            values = [v.strip() for v in m.group(2).split(",") if v.strip()]
            if not values:
                continue
            try:
                val = float(values[0])
            except ValueError:
                continue
            scores[metric] = val
        for d in dataset_list:
            if d not in TARGET_DATASETS:
                continue
            for metric in ("SRCC", "PLCC"):
                if metric in scores:
                    current_val, _ = best[d][metric]
                    if current_val is None or scores[metric] > current_val:
                        best[d][metric] = (scores[metric], row)

lines = []
lines.append("| Dataset | Best SRCC (paper) | Best PLCC (paper) | Notes |")
lines.append("|---|---|---|---|")

for d in TARGET_DATASETS:
    srcc_val, srcc_row = best[d]["SRCC"]
    plcc_val, plcc_row = best[d]["PLCC"]
    if srcc_row:
        srcc = f"{srcc_val:.4f} - {srcc_row['title']} ({srcc_row['venue']} {srcc_row['year']})"
    else:
        srcc = "N/A"
    if plcc_row:
        plcc = f"{plcc_val:.4f} - {plcc_row['title']} ({plcc_row['venue']} {plcc_row['year']})"
    else:
        plcc = "N/A"
    note = "auto-parsed; verify in paper"
    lines.append(f"| {d} | {srcc} | {plcc} | {note} |")

out_path = "/Users/mamadfarrahi/Desktop/thesisproject/nn-dataset/iqa_scripts/nriqa/sota_summary.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote {out_path}")
