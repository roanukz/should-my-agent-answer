"""Write data/manifest.json: what ran, over what, and how much of it there was.

The corpus commit is the important field. FastAPI's documentation moves every
week, so a finding that does not name the commit it was computed against is not
reproducible and not really checkable either.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GRAPH = DATA / "graph"


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def pipeline_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def main() -> int:
    fetch = read(DATA / "raw" / "fetch_meta.json", {})
    nodes = read(GRAPH / "nodes.json", {"nodes": []})["nodes"]
    edges = read(GRAPH / "edges.json", {"edges": []})["edges"]
    findings = read(DATA / "findings.json", {"findings": [], "counts": {}})
    extraction = read(GRAPH / "extraction_report.json", {})
    questions = read(GRAPH / "question_report.json", {})
    validation = read(DATA / "validation.json", {})
    answers = read(DATA / "answers.json", {})

    manifest = {
        "schema_version": 1,
        "corpus": fetch.get("corpus", "fastapi/fastapi"),
        "corpus_commit": fetch.get("corpus_commit", ""),
        "corpus_commit_date": fetch.get("corpus_commit_date", ""),
        "docs_pages": len(fetch.get("docs", [])),
        "discussion_threads": len(fetch.get("threads", [])),
        "run_date": date.today().isoformat(),
        "nodes": len(nodes),
        "edges": len(edges),
        "nodes_by_type": dict(Counter(n["type"] for n in nodes)),
        "edges_by_type": dict(Counter(e["type"] for e in edges)),
        "findings": findings.get("counts", {}),
        "extractor": "claude-code-batch",
        "embedding_model": answers.get("embedding_model", ""),
        "generator": answers.get("generator", ""),
        "extraction": {
            "batches": extraction.get("batches_processed", 0),
            "edges_proposed": extraction.get("raw_edges", 0),
            "edges_kept": extraction.get("edges_kept", 0),
            "dropped_span_not_found": extraction.get("dropped_no_span", 0),
            "dropped_short_span": extraction.get("dropped_short_span", 0),
            "dropped_self_reference": extraction.get("dropped_self_reference", 0),
            "dropped_duplicate": extraction.get("dropped_duplicate", 0),
        },
        "question_mapping": {
            "batches": questions.get("batches_processed", 0),
            "edges": questions.get("asks_about_edges", 0),
            "questions_mapped": questions.get("questions_mapped", 0),
            "questions_total": questions.get("questions_total", 0),
        },
        "validation": validation.get("by_type", {}),
        "retrieval_comparison": answers.get("summary", {}),
        "pipeline_commit": pipeline_commit(),
    }

    (DATA / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    missing = [e["id"] for e in edges if not (e.get("evidence") or {}).get("span")]
    print(f"Wrote data/manifest.json: {len(nodes)} nodes, {len(edges)} edges, "
          f"{sum(manifest['findings'].values())} findings")
    print(f"  corpus {manifest['corpus']} at {manifest['corpus_commit'][:12]}")
    print(f"  edges with no evidence span: {len(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
