import csv
import datetime as dt
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

TRACKER_PATH = "/Users/mamadfarrahi/Desktop/thesisproject/nn-dataset/iqa_scripts/nriqa/nriqa_sota_tracker.csv"
CVF_RESOLVED_PATH = "/Users/mamadfarrahi/Desktop/thesisproject/nn-dataset/cvf_papers_resolved.csv"

IQA_TERMS = [
    "image quality assessment",
    "blind image quality",
    "no-reference image quality",
    "no reference image quality",
    "iqa",
    "biqa",
    "nriqa",
    "fiqa",
]

IMAGE_CONTEXT_TERMS = [
    "image",
    "photo",
    "photograph",
    "face",
]

EXCLUDE_TERMS = [
    "video quality",
    "video",
    "point cloud",
    "action quality",
    "speech",
    "audio",
]

ARXIV_QUERIES = [
    'ti:"image quality assessment" OR abs:"image quality assessment"',
    'ti:"no-reference" OR abs:"no-reference" OR ti:"blind image quality" OR abs:"blind image quality"',
    'ti:"IQA" OR abs:"IQA"',
]

HF_SEARCH_TERMS = [
    "image quality assessment",
    "IQA",
    "NR-IQA",
]


def normalize_title(title):
    return re.sub(r"\s+", " ", title.strip().lower())


def normalize_arxiv_link(link):
    if not link:
        return ""
    match = re.search(r"arxiv\.org/abs/([^\s]+)", link)
    if not match:
        return link
    arxiv_id = match.group(1)
    base_id = arxiv_id.split("v", 1)[0]
    return f"https://arxiv.org/abs/{base_id}"


def make_key(row):
    title_key = normalize_title(row.get("title", ""))
    arxiv_link = normalize_arxiv_link(row.get("arxiv_link", ""))
    if arxiv_link:
        return (title_key, arxiv_link)
    link_key = row.get("cvf_page") or row.get("pdf_link") or ""
    return (title_key, link_key)


def is_nriqa_text(text):
    text_lower = text.lower()
    if any(term in text_lower for term in EXCLUDE_TERMS):
        return False
    if any(term in text_lower for term in IQA_TERMS):
        return True
    if "quality assessment" in text_lower and any(term in text_lower for term in IMAGE_CONTEXT_TERMS):
        return True
    return False


def is_nriqa_title_summary(title, summary):
    combined = f"{title} {summary}".strip()
    return is_nriqa_text(combined)


def parse_year_from_cvf(url):
    match = re.search(r"/(CVPR|ICCV)(\d{4})/", url)
    if match:
        return int(match.group(2)), match.group(1)
    return None, None


def load_existing_rows():
    rows = []
    existing = set()
    with open(TRACKER_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            row["arxiv_link"] = normalize_arxiv_link(row.get("arxiv_link", ""))
            existing.add(make_key(row))
    return rows, existing, reader.fieldnames


def add_row(rows, existing, fieldnames, row):
    row["arxiv_link"] = normalize_arxiv_link(row.get("arxiv_link", ""))
    title_key = normalize_title(row.get("title", ""))
    if not title_key:
        return
    key = make_key(row)
    if key in existing:
        return
    for field in fieldnames:
        row.setdefault(field, "")
    rows.append(row)
    existing.add(key)


def parse_cvf(rows, existing, fieldnames, min_year):
    with open(CVF_RESOLVED_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for item in reader:
            title = item.get("title", "")
            cvf_page = item.get("cvf_page", "")
            pdf_link = item.get("pdf_link", "")
            year, venue = parse_year_from_cvf(cvf_page)
            if not year or year < min_year:
                continue
            if not is_nriqa_text(title):
                continue
            add_row(
                rows,
                existing,
                fieldnames,
                {
                    "source": "CVF",
                    "title": title,
                    "venue": venue or "CVF",
                    "year": str(year),
                    "cvf_page": cvf_page,
                    "pdf_link": pdf_link,
                    "arxiv_link": "",
                    "datasets": "",
                    "metrics": "",
                    "reported_scores": "",
                    "notes": "",
                },
            )


def arxiv_fetch(query, start=0, max_results=100):
    base = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def parse_arxiv(rows, existing, fieldnames, min_year):
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for query in ARXIV_QUERIES:
        data = arxiv_fetch(query, start=0, max_results=200)
        root = ET.fromstring(data)
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            link = entry.findtext("atom:id", default="", namespaces=ns).strip()
            published = entry.findtext("atom:published", default="", namespaces=ns).strip()
            if not title or not link or not published:
                continue
            year = int(published[:4])
            if year < min_year:
                continue
            if not is_nriqa_title_summary(title, summary):
                continue
            add_row(
                rows,
                existing,
                fieldnames,
                {
                    "source": "arXiv",
                    "title": re.sub(r"\s+", " ", title),
                    "venue": "arXiv",
                    "year": str(year),
                    "cvf_page": "",
                    "pdf_link": "",
                    "arxiv_link": normalize_arxiv_link(link),
                    "datasets": "",
                    "metrics": "",
                    "reported_scores": "",
                    "notes": "",
                },
            )


def hf_fetch(term):
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode({"search": term})
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_hf(rows, existing, fieldnames, min_year):
    seen = set()
    for term in HF_SEARCH_TERMS:
        try:
            items = hf_fetch(term)
        except Exception:
            continue
        for item in items:
            model_id = item.get("modelId") or item.get("id")
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            model_id_lower = model_id.lower()
            tags = [t.lower() for t in (item.get("tags") or [])]
            if "image-quality-assessment" not in tags and "iqa" not in model_id_lower and "quality" not in model_id_lower:
                continue
            last_modified = item.get("lastModified") or ""
            year = None
            if len(last_modified) >= 4 and last_modified[:4].isdigit():
                year = int(last_modified[:4])
            if year is None or year < min_year:
                continue
            add_row(
                rows,
                existing,
                fieldnames,
                {
                    "source": "HuggingFace",
                    "title": model_id,
                    "venue": "HF Model",
                    "year": str(year),
                    "cvf_page": f"https://huggingface.co/{model_id}",
                    "pdf_link": "",
                    "arxiv_link": "",
                    "datasets": "",
                    "metrics": "",
                    "reported_scores": "",
                    "notes": "",
                },
            )


def main():
    rows, existing, fieldnames = load_existing_rows()
    if not fieldnames:
        fieldnames = [
            "source",
            "title",
            "venue",
            "year",
            "cvf_page",
            "pdf_link",
            "arxiv_link",
            "datasets",
            "metrics",
            "reported_scores",
            "notes",
        ]
    today = dt.date.today()
    min_year = today.year - 2

    filtered_rows = []
    filtered_existing = set()
    for row in rows:
        title = row.get("title", "")
        source = row.get("source", "")
        year_value = row.get("year", "")
        year = int(year_value) if year_value.isdigit() else None
        if year and year < min_year:
            continue
        if source != "HuggingFace" and not is_nriqa_text(title):
            continue
        add_row(filtered_rows, filtered_existing, fieldnames, row)

    rows = filtered_rows
    existing = filtered_existing
    parse_cvf(rows, existing, fieldnames, min_year)
    parse_arxiv(rows, existing, fieldnames, min_year)
    parse_hf(rows, existing, fieldnames, min_year)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r.get("source", ""),
            r.get("year", ""),
            r.get("title", ""),
        ),
    )

    with open(TRACKER_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(f"Updated {TRACKER_PATH} with {len(rows_sorted)} rows")


if __name__ == "__main__":
    main()
