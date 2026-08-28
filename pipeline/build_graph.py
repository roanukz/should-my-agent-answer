"""Assemble the networkx graph and add the edges that need no model.

extract.py produces the edges that need judgment: DEFINES, REQUIRES, MENTIONS
from the documentation, and ASKS_ABOUT from the questions. Three more edge types
are pure structure and are built here, because a model would only add noise:

  CONTAINS     Doc -> Section, one per section
  LINKS_TO     Section -> Doc, one per internal documentation link
  ANSWERED_BY  Question -> Answer, one per accepted answer

They carry evidence spans like every other edge. For CONTAINS the span is the
heading line itself, for LINKS_TO it is the link markdown, for ANSWERED_BY it is
the line in the thread that records who GitHub marked as answering. No
exceptions to the invariant, including for the easy cases.

Also does the graph sanity pass: degree distributions, isolated nodes, and the
DEFINES/REQUIRES balance that decides whether the extraction is worth trusting.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data" / "graph"
RAW = ROOT / "data" / "raw"

MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def log(msg: str) -> None:
    print(msg, flush=True)


def load() -> tuple[list[dict], list[dict]]:
    nodes = json.loads((GRAPH / "nodes.json").read_text(encoding="utf-8"))["nodes"]
    edges = json.loads((GRAPH / "edges.json").read_text(encoding="utf-8"))["edges"]
    return nodes, edges


# --------------------------------------------------------------------------
# Structural edges
# --------------------------------------------------------------------------

def contains_edges(sections: list[dict]) -> list[dict]:
    out = []
    for sec in sections:
        first_line = sec["text"].splitlines()[0] if sec["text"] else sec["heading"]
        out.append({
            "id": "",
            "type": "CONTAINS",
            "from": sec["doc_id"],
            "to": sec["id"],
            "evidence": {
                "span": first_line.strip(),
                "path": sec["path"],
                "lines": [sec["lines"][0], sec["lines"][0]],
            },
            "extractor": "structural",
            "confidence": "high",
        })
    return out


def resolve_link(target: str, from_rel: str, known: set[str]) -> str | None:
    """Turn a markdown link target into a doc id, or None if it leaves the corpus."""
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    target = target.split("#", 1)[0].strip()
    if not target or not target.endswith(".md"):
        return None
    base = Path(from_rel).parent
    resolved = (base / target).as_posix()
    parts: list[str] = []
    for piece in resolved.split("/"):
        if piece in ("", "."):
            continue
        if piece == "..":
            if parts:
                parts.pop()
            continue
        parts.append(piece)
    doc_id = "doc:" + "/".join(parts)[: -len(".md")]
    return doc_id if doc_id in known else None


def links_to_edges(sections: list[dict], doc_ids: set[str]) -> list[dict]:
    out = []
    seen: set[tuple[str, str]] = set()
    for sec in sections:
        from_rel = sec["path"].split("docs/en/docs/", 1)[-1]
        lines = sec["text"].splitlines()
        for offset, line in enumerate(lines):
            for match in MD_LINK_RE.finditer(line):
                doc_id = resolve_link(match.group(2), from_rel, doc_ids)
                if doc_id is None or doc_id == sec["doc_id"]:
                    continue
                key = (sec["id"], doc_id)
                if key in seen:
                    continue
                seen.add(key)
                line_no = sec["lines"][0] + offset
                out.append({
                    "id": "",
                    "type": "LINKS_TO",
                    "from": sec["id"],
                    "to": doc_id,
                    "evidence": {
                        "span": match.group(0),
                        "path": sec["path"],
                        "lines": [line_no, line_no],
                    },
                    "extractor": "structural",
                    "confidence": "high",
                })
    return out


def question_answer_nodes() -> tuple[list[dict], list[dict]]:
    """Question and Answer nodes, plus the ANSWERED_BY edge between them."""
    nodes: list[dict] = []
    edges: list[dict] = []
    for path in sorted(RAW.glob("discussions/*.json"), key=lambda p: int(p.stem)):
        rec = json.loads(path.read_text(encoding="utf-8"))
        number = rec["number"]
        thread_md = RAW / "discussions" / f"{number}.md"
        rel_md = str(thread_md.relative_to(ROOT))
        qid, aid = f"q:{number}", f"ans:{number}"
        nodes.append({
            "id": qid,
            "type": "Question",
            "number": number,
            "title": rec["title"],
            "body": rec["body_text"],
            "created_at": rec["created_at"],
            "url": rec["url"],
            "answered": rec["answered"],
            "raw_path": rel_md,
        })
        answer = rec["answer"]
        nodes.append({
            "id": aid,
            "type": "Answer",
            "question_id": qid,
            "author": answer["author"],
            "is_maintainer": answer["is_maintainer"],
            "author_association": answer["author_association"],
            "body": answer["body_text"],
            "url": answer["url"],
            "raw_path": rel_md,
        })

        # The evidence for "this reply answers this question" is the line where
        # the thread records GitHub's accepted-answer marker.
        md_lines = thread_md.read_text(encoding="utf-8").splitlines()
        line_no = next(
            (i + 1 for i, ln in enumerate(md_lines) if ln.startswith("## Accepted answer")),
            1,
        )
        edges.append({
            "id": "",
            "type": "ANSWERED_BY",
            "from": qid,
            "to": aid,
            "evidence": {
                "span": md_lines[line_no - 1],
                "path": rel_md,
                "lines": [line_no, line_no],
            },
            "extractor": "structural",
            "confidence": "high",
        })
    return nodes, edges


# --------------------------------------------------------------------------

def build(nodes: list[dict], edges: list[dict]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node in nodes:
        graph.add_node(node["id"], **node)
    for edge in edges:
        graph.add_edge(edge["from"], edge["to"], key=edge["id"], **edge)
    return graph


def sanity(graph: nx.MultiDiGraph, nodes: list[dict], edges: list[dict]) -> dict:
    by_type = Counter(n["type"] for n in nodes)
    edge_types = Counter(e["type"] for e in edges)

    concepts = [n for n in nodes if n["type"] == "Concept"]
    in_by_type: dict[str, Counter] = defaultdict(Counter)
    for edge in edges:
        if edge["to"].startswith("concept:"):
            in_by_type[edge["to"]][edge["type"]] += 1

    defined = sum(1 for c in concepts if in_by_type[c["id"]]["DEFINES"] > 0)
    required = sum(1 for c in concepts if in_by_type[c["id"]]["REQUIRES"] > 0)
    orphan = sum(
        1 for c in concepts
        if in_by_type[c["id"]]["DEFINES"] == 0
        and (in_by_type[c["id"]]["REQUIRES"] + in_by_type[c["id"]]["MENTIONS"]) > 0
    )
    sections_touching = defaultdict(set)
    for edge in edges:
        if edge["to"].startswith("concept:") and edge["from"].startswith("sec:"):
            sections_touching[edge["to"]].add(edge["from"])
    multi_section = sum(1 for c in concepts if len(sections_touching[c["id"]]) >= 2)

    isolated = [n["id"] for n in nodes if graph.degree(n["id"]) == 0]

    report = {
        "nodes": len(nodes),
        "edges": len(edges),
        "nodes_by_type": dict(by_type),
        "edges_by_type": dict(edge_types),
        "concepts_total": len(concepts),
        "concepts_defined_somewhere": defined,
        "concepts_required_somewhere": required,
        "concepts_orphan": orphan,
        "concepts_in_two_or_more_sections": multi_section,
        "isolated_nodes": len(isolated),
        "isolated_examples": isolated[:10],
    }

    log("")
    log(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
    log("  nodes  " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))
    log("  edges  " + ", ".join(f"{k} {v}" for k, v in sorted(edge_types.items())))
    log(f"  concepts: {len(concepts)} total, {defined} defined somewhere, "
        f"{orphan} orphan, {multi_section} appear in two or more sections")
    if isolated:
        log(f"  {len(isolated)} isolated nodes, e.g. {isolated[:3]}")
    return report


def main() -> int:
    nodes, edges = load()
    sections = [n for n in nodes if n["type"] == "Section"]
    doc_ids = {n["id"] for n in nodes if n["type"] == "Doc"}

    # Structural edges are rebuilt every run; model edges are kept as they are.
    edges = [e for e in edges if e["extractor"] != "structural"]
    nodes = [n for n in nodes if n["type"] not in ("Question", "Answer")]

    log("Adding structural edges")
    structural = contains_edges(sections)
    links = links_to_edges(sections, doc_ids)
    log(f"  CONTAINS {len(structural)}, LINKS_TO {len(links)}")

    qa_nodes, qa_edges = question_answer_nodes()
    log(f"  Question and Answer nodes {len(qa_nodes)}, ANSWERED_BY {len(qa_edges)}")

    nodes = nodes + qa_nodes
    edges = edges + structural + links + qa_edges

    # A concept whose every edge was dropped by the validator is not a concept,
    # it is the residue of one. Prune it rather than shipping a node nothing
    # points at.
    referenced = {e["from"] for e in edges} | {e["to"] for e in edges}
    pruned = [n for n in nodes if n["type"] == "Concept" and n["id"] not in referenced]
    if pruned:
        log(f"  pruning {len(pruned)} concepts left with no surviving edge")
        nodes = [n for n in nodes if n["id"] not in {p["id"] for p in pruned}]

    # Renumber every edge so ids are stable and dense across the whole graph.
    edges.sort(key=lambda e: (e["type"], e["from"], e["to"]))
    for i, edge in enumerate(edges):
        edge["id"] = f"e:{i:05d}"

    missing = [e["id"] for e in edges if not (e.get("evidence") or {}).get("span")]
    if missing:
        raise SystemExit(f"INVARIANT BROKEN: {len(missing)} edges with no evidence span")

    (GRAPH / "nodes.json").write_text(
        json.dumps({"schema_version": 1, "nodes": nodes}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (GRAPH / "edges.json").write_text(
        json.dumps({"schema_version": 1, "edges": edges}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    graph = build(nodes, edges)
    report = sanity(graph, nodes, edges)
    (GRAPH / "graph_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    log("")
    log("Wrote data/graph/nodes.json, data/graph/edges.json, data/graph/graph_report.json")
    log("Invariant holds: zero edges with a null or empty evidence span")
    return 0


if __name__ == "__main__":
    sys.exit(main())
