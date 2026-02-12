This document consolidates the current SOTA tracking status and the benchmarking policy for NR-IQA (no-reference image quality assessment).

Scope and sources
- Task: NR-IQA (natural images), last 2 years.
- Sources: CVF (ICCV2023, CVPR2024, ICCV2025 day=all) + arXiv.
- Current arXiv coverage: top 50 recent keyword matches (not exhaustive).
- Exclusions: medical-domain IQA and face-specific IQA (e.g., MRI, pathology, FIQA).

Current tracking outputs
- Curated Top 30 list with PDF-derived summaries and metrics: iqa_router_v2/writing/nriqa/nriqa_valuable_list.md
- Auto-parsed SOTA snapshot table (per dataset): iqa_router_v2/writing/nriqa/sota_summary.md
- Source tracker CSV: iqa_router_v2/writing/nriqa/nriqa_sota_tracker.csv

Benchmarking policy (what to compare)
- Primary datasets: KonIQ-10k, SPAQ, KADID-10k, CLIVE.
- Secondary datasets (if available): LIVE, CSIQ, TID2013, AVA.
- Metrics: SRCC and PLCC are required; KRCC optional.
- Report per-dataset results; avoid only averaged scores unless the paper also provides per-dataset numbers.
- Prefer models that report cross-dataset testing or generalization claims.
- Always verify auto-parsed values against the PDF before final claims.

Auto-parsed SOTA snapshot (verify in papers)
| Dataset | Best SRCC (paper) | Best PLCC (paper) | Notes |
|---|---|---|---|
| KonIQ-10k | 0.9590 - Describe-to-Score: Text-Guided Efficient Image Complexity Assessment (arXiv 2025) | 0.9576 - Describe-to-Score: Text-Guided Efficient Image Complexity Assessment (arXiv 2025) | auto-parsed; verify in paper |
| SPAQ | 0.9430 - Don't Judge Before You CLIP: A Unified Approach for Perceptual Tasks (arXiv 2025) | 0.9530 - BPCLIP: A Bottom-up Image Quality Assessment from Distortion to Semantics Based on CLIP (arXiv 2025) | auto-parsed; verify in paper |
| KADID-10k | 0.9590 - Describe-to-Score: Text-Guided Efficient Image Complexity Assessment (arXiv 2025) | 0.9576 - Describe-to-Score: Text-Guided Efficient Image Complexity Assessment (arXiv 2025) | auto-parsed; verify in paper |
| CLIVE | 0.9086 - DocIQ: A Benchmark Dataset and Feature Fusion Network for Document Image Quality Assessment (arXiv 2025) | 0.9530 - BPCLIP: A Bottom-up Image Quality Assessment from Distortion to Semantics Based on CLIP (arXiv 2025) | auto-parsed; verify in paper |

Top-10 (2024-2026, NR-IQA shortlist for citations)
Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment (2026, arXiv)
https://arxiv.org/abs/2511.09948
Note: CLIP-based, training-free; strong SOTA, AAAI 2026.

Enhancing Image Quality Assessment Ability of LMMs via Retrieval-Augmented Generation (2026, arXiv)
https://arxiv.org/abs/2601.08311
Note: RAG-augmented LMMs for NR-IQA; strong multi-dataset results.

Revisiting Vision-Language Foundations for No-Reference Image Quality Assessment (2025, arXiv)
https://arxiv.org/abs/2509.17374
Note: VLM backbone study + activation selection; SOTA on key benchmarks.

HiRQA: Hierarchical Ranking and Quality Alignment for Opinion-Unaware IQA (2025, arXiv)
https://arxiv.org/abs/2508.15130
Note: Self-supervised ranking/contrastive alignment; strong generalization.

TRIQA: IQA by Contrastive Pretraining on Ordered Distortion Triplets (2025, arXiv)
https://arxiv.org/abs/2507.12687
Note: Contrastive pretraining for NR-IQA; data-efficient, strong SRCC.

BPCLIP: Bottom-up IQA from Distortion to Semantics Based on CLIP (2025, arXiv)
https://arxiv.org/abs/2506.17969
Note: CLIP-guided multiscale cross-attention; ICME 2025.

DGIQA: Depth-guided Feature Attention and Refinement for Generalizable IQA (2025, arXiv)
https://arxiv.org/abs/2505.24002
Note: Depth-guided attention; strong cross-dataset results.

VisualQuality-R1: Reasoning-Induced IQA via Reinforcement Learning to Rank (2025, arXiv)
https://arxiv.org/abs/2505.14460
Note: RL-to-rank for NR-IQA.

CoDI-IQA: Content-Distortion High-Order Interaction for BIQA (2025, arXiv)
https://arxiv.org/abs/2504.05076
Note: Explicit content-distortion interaction modeling.

Compare2Score: Teaching LMMs to Compare for Adaptive IQA (2024, arXiv)
https://arxiv.org/abs/2405.19298
Note: Comparison-to-score conversion; strong multi-dataset performance.

Core baselines (pre-2024 but still mandatory in comparisons)
MANIQA (2022, arXiv) - https://arxiv.org/abs/2204.08958
MUSIQ (2021, arXiv) - https://arxiv.org/abs/2108.05997
TOPIQ (2023, arXiv) - https://arxiv.org/abs/2308.03060
LIQE (2023, arXiv) - https://arxiv.org/abs/2303.14968
CLIP-IQA (2022, arXiv) - https://arxiv.org/abs/2207.12396
ARNIQA (2023, arXiv) - https://arxiv.org/abs/2310.14918

Next actions to improve coverage
- Expand arXiv coverage beyond top 50 (e.g., top 200) and rerun the parser.
- Tighten filtering to remove non-NR or benchmark-only papers.
- Replace auto-parsed metrics with verified values from the PDFs.


MUSIQ — https://arxiv.org/abs/2108.05997
MANIQA — https://arxiv.org/abs/2205.01389
TOPIQ — https://arxiv.org/abs/2204.12485
TReS — https://arxiv.org/abs/2104.00409
HyperIQA — https://arxiv.org/abs/2008.06894
LIQE — https://arxiv.org/abs/2111.11952
ARNIQA — https://arxiv.org/abs/2206.12890
CLIP-IQA / CLIPIQA+ — https://arxiv.org/abs/2207.12396
PaQ-2-PiQ — https://arxiv.org/abs/2002.09516
DBCNN — https://arxiv.org/abs/1807.00209