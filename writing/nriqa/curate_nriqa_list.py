import csv
import os
import re
import time
import urllib.error
import urllib.request
from typing import List, Tuple

from PyPDF2 import PdfReader

TRACKER_PATH = "/Users/mamadfarrahi/Desktop/thesisproject/nn-dataset/iqa_scripts/nriqa/nriqa_sota_tracker.csv"
OUT_PATH = "/Users/mamadfarrahi/Desktop/thesisproject/nn-dataset/iqa_scripts/nriqa/nriqa_valuable_list.md"
CACHE_DIR = "/Users/mamadfarrahi/Desktop/thesisproject/nn-dataset/iqa_scripts/nriqa/cache/pdfs"

MAX_PAPERS = 30
MAX_PAGES = 3

EXCLUDE_TERMS = [
    "face",
    "facial",
    "fiqa",
    "medical",
    "mri",
    "radiotherapy",
    "pathology",
    "ultrasound",
    "ct",
    "x-ray",
    "xray",
    "retina",
    "spect",
    "photoacoustic",
    "histopath",
    "fundus",
    "microscopy",
]


def is_excluded_title(title: str) -> bool:
    title_lower = title.lower()
    return any(term in title_lower for term in EXCLUDE_TERMS)


def arxiv_pdf_from_abs(abs_link: str) -> str:
    match = re.search(r"arxiv\.org/abs/([^\s]+)", abs_link)
    if not match:
        return ""
    arxiv_id = match.group(1).split("v", 1)[0]
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return cleaned[:120].strip("_") or "paper"


def download_pdf(url: str, dest_path: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def extract_text_from_pdf(pdf_path: str, max_pages: int = MAX_PAGES) -> str:
    reader = PdfReader(pdf_path)
    text_parts = []
    total_pages = min(len(reader.pages), max_pages)
    for i in range(total_pages):
        try:
            text_parts.append(reader.pages[i].extract_text() or "")
        except Exception:
            continue
    return "\n".join(text_parts)


def extract_abstract(text: str) -> str:
    text_clean = re.sub(r"\s+", " ", text).strip()
    match = re.search(r"\babstract\b", text_clean, re.IGNORECASE)
    if not match:
        return ""
    start = match.end()
    tail = text_clean[start:]
    intro_match = re.search(r"\bintroduction\b", tail, re.IGNORECASE)
    if intro_match:
        tail = tail[: intro_match.start()]
    tail = tail.strip()
    if not tail:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", tail)
    summary = " ".join(sentences[:3]).strip()
    return summary[:800]


def parse_scores(text: str) -> Tuple[str, str]:
    srcc_match = re.search(r"SRCC:\s*([0-1]\.\d{2,4})", text)
    plcc_match = re.search(r"PLCC:\s*([0-1]\.\d{2,4})", text)
    srcc = srcc_match.group(1) if srcc_match else ""
    plcc = plcc_match.group(1) if plcc_match else ""
    return srcc, plcc


def sort_key(row):
    year_value = row.get("year", "")
    year = int(year_value) if year_value.isdigit() else 0
    has_scores = bool(row.get("reported_scores"))
    has_datasets = bool(row.get("datasets"))
    source = row.get("source", "")
    source_rank = 0 if source == "CVF" else 1
    return (-has_scores, -has_datasets, -year, source_rank, row.get("title", ""))


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open(TRACKER_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    filtered = [r for r in rows if not is_excluded_title(r.get("title", ""))]
    filtered.sort(key=sort_key)

    selected = []
    for row in filtered:
        if len(selected) >= MAX_PAPERS:
            break
        selected.append(row)

    lines: List[str] = []
    lines.append("# NR-IQA Valuable Papers (Top 30)")
    lines.append("")

    for idx, row in enumerate(selected, start=1):
        title = row.get("title", "").strip()
        venue = row.get("venue", "")
        year = row.get("year", "")
        cvf = row.get("cvf_page", "")
        arxiv = row.get("arxiv_link", "")
        pdf = row.get("pdf_link", "") or arxiv_pdf_from_abs(arxiv)
        datasets = row.get("datasets", "")
        scores = row.get("reported_scores", "")

        abstract_summary = ""
        if pdf:
            cache_name = safe_filename(title) + ".pdf"
            cache_path = os.path.join(CACHE_DIR, cache_name)
            if not os.path.exists(cache_path):
                ok = download_pdf(pdf, cache_path)
                if not ok:
                    cache_path = ""
                time.sleep(0.2)
            if cache_path and os.path.exists(cache_path):
                try:
                    text = extract_text_from_pdf(cache_path)
                    abstract_summary = extract_abstract(text)
                except Exception:
                    abstract_summary = ""

        srcc, plcc = parse_scores(scores)

        lines.append(f"## {idx}. {title}")
        lines.append("")
        lines.append(f"- Venue/Year: {venue} {year}")
        if datasets:
            lines.append(f"- Datasets: {datasets}")
        if srcc or plcc:
            parts = []
            if srcc:
                parts.append(f"SRCC {srcc}")
            if plcc:
                parts.append(f"PLCC {plcc}")
            lines.append(f"- Best metrics (auto-parsed): {', '.join(parts)}")
        if cvf:
            lines.append(f"- CVF: {cvf}")
        if arxiv:
            lines.append(f"- arXiv: {arxiv}")
        if pdf:
            lines.append(f"- PDF: {pdf}")
        if abstract_summary:
            lines.append(f"- Summary: {abstract_summary}")
        lines.append("")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
