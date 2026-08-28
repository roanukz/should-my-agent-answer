"""Find the articles that will make an assistant answer confidently and wrong.

Three finding types, in priority order, each carrying a proof path made of real
edges with real evidence spans.

  F1 near_miss           A concept that questions ask about, that a section's
                         instructions depend on, and that NO section in the
                         corpus explains. The section looks like the answer. It
                         is not, because it rests on something never taught.

  F2 orphan_concept      A concept the documentation leans on or refers to and
                         never explains. The same shape as F1 without the
                         demand, so it is a backlog rather than a fire.

  F3 retrieval_collision Two sections a single question maps to with comparable
                         concept overlap, where only one carries the answer. The
                         vaguer one can win retrieval.

Everything is ranked by demand: the number of distinct discussion threads that
touch the concept. A gap nobody asks about is not worth writing.

`missing` and `confirming_answer` are left empty here and filled by a separate
model pass (describe.py), because naming the absent fact needs judgement and
this file must stay a deterministic function of the graph.
"""

from __future__ import annotations

import builtins
import json
import keyword
import re
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data" / "graph"
DATA = ROOT / "data"
WORK = DATA / "work"

# A concept referenced by exactly one section is nearly always a name local to
# that page's code sample, not something the documentation owes the reader an
# explanation of. F1 is protected from those by requiring real demand; F2 has no
# such gate, so it gets this one.
MIN_SECTIONS_FOR_ORPHAN = 2

# --------------------------------------------------------------------------
# What is not a documentation gap
#
# Added after the first validation round came back at 9 valid out of 33. Seven
# of the twenty-four rejections were the same complaint: "the FastAPI docs never
# explain `str`" is true, and useless, because explaining `str` is Python's job.
#
# The rule this encodes: a gap is only a gap if closing it is THIS documentation
# set's responsibility. Python's own language surface is not, and neither is a
# third-party tool that has its own documentation.
#
# The language part is generated rather than hand-written, from Python's keyword
# list and its builtins, so it cannot be quietly tuned to make a number look
# better. The short list underneath it is the part that is a judgement call, so
# it is kept small, written out in full, and every entry is here because the
# thing has its own documentation elsewhere.
# --------------------------------------------------------------------------

PYTHON_SURFACE = (
    {k.lower() for k in keyword.kwlist}
    | {k.lower() for k in keyword.softkwlist}
    | {name.lower() for name in dir(builtins) if not name.startswith("_")}
)

OWNED_ELSEWHERE = {
    "type annotation", "type annotations", "type hint", "type hints",
    "annotation", "annotations",          # Python's typing surface
    "pip", "uv", "uv run", "venv",        # packaging tools, own docs
    "deployment", "deploy",               # a lifecycle stage, not a thing
    "python", "fastapi",                  # the subject, not a dependency
}


def out_of_scope(label: str) -> bool:
    """True when defining this belongs to somebody other than these 60 pages.

    A label made only of Python keywords and builtins counts, so `async def`,
    `async for` and `await` all go, while `response_model` and `jsonable_encoder`
    stay. A multi-word label counts only if EVERY word is language surface,
    which is what keeps "async generator" and "path operation" in scope.
    """
    normalised = label.strip().lower()
    if normalised in OWNED_ELSEWHERE:
        return True
    words = [w for w in re.split(r"[^a-z0-9_]+", normalised) if w]
    if not words:
        return True
    return all(w in PYTHON_SURFACE for w in words)

# F3: how close the runner-up's concept overlap has to be to the winner's before
# the two are genuinely competing for the same retrieval slot.
COLLISION_RATIO = 0.8
MIN_COLLISION_OVERLAP = 2


def log(msg: str) -> None:
    print(msg, flush=True)


def load():
    nodes = json.loads((GRAPH / "nodes.json").read_text(encoding="utf-8"))["nodes"]
    edges = json.loads((GRAPH / "edges.json").read_text(encoding="utf-8"))["edges"]
    return nodes, edges


class Index:
    """Everything the three finders need, computed once."""

    def __init__(self, nodes: list[dict], edges: list[dict]) -> None:
        self.node = {n["id"]: n for n in nodes}
        self.sections = [n for n in nodes if n["type"] == "Section"]
        self.concepts = [n for n in nodes if n["type"] == "Concept"]
        self.questions = [n for n in nodes if n["type"] == "Question"]
        self.edges = edges
        self.edge = {e["id"]: e for e in edges}

        # concept -> {edge type -> [edges]}
        self.into_concept: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list))
        # section -> {edge type -> [concept ids]}
        self.section_concepts: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list))
        # question -> [ASKS_ABOUT edges]
        self.question_asks: dict[str, list[dict]] = defaultdict(list)
        self.answer_of: dict[str, dict] = {}

        for edge in edges:
            etype = edge["type"]
            if etype in ("DEFINES", "REQUIRES", "MENTIONS"):
                self.into_concept[edge["to"]][etype].append(edge)
                self.section_concepts[edge["from"]][etype].append(edge["to"])
            elif etype == "ASKS_ABOUT":
                self.into_concept[edge["to"]]["ASKS_ABOUT"].append(edge)
                self.question_asks[edge["from"]].append(edge)
            elif etype == "ANSWERED_BY":
                self.answer_of[edge["from"]] = self.node[edge["to"]]

        # A section's full concept set, used for overlap scoring.
        self.section_concept_set: dict[str, set[str]] = {
            sid: set(sum(types.values(), []))
            for sid, types in self.section_concepts.items()
        }
        self.question_concept_set: dict[str, set[str]] = {
            qid: {e["to"] for e in asks} for qid, asks in self.question_asks.items()
        }
        self.total_sections = len(self.sections)

    def defines(self, cid: str) -> list[dict]:
        return self.into_concept[cid]["DEFINES"]

    def requires(self, cid: str) -> list[dict]:
        return self.into_concept[cid]["REQUIRES"]

    def mentions(self, cid: str) -> list[dict]:
        return self.into_concept[cid]["MENTIONS"]

    def asked_by(self, cid: str) -> list[dict]:
        return self.into_concept[cid]["ASKS_ABOUT"]

    def has_prose_evidence(self, cid: str) -> bool:
        """Does the corpus ever discuss this in prose, rather than only show it?

        Reported on every finding, never used to filter one out. It was a filter
        for one revision, and it removed four findings an independent check had
        already confirmed, because a concept that only ever appears inside code
        samples is exactly what a real gap in third-party API surface looks like.
        As a label on the card it tells a reader how the corpus treats the thing.
        Related: `out_of_scope`, which IS a filter, and says so.
        """
        for etype in ("REQUIRES", "MENTIONS"):
            for edge in self.into_concept[cid][etype]:
                if edge["evidence"].get("kind") == "prose":
                    return True
        return False


def evidence_of(edge: dict) -> dict:
    return {
        "span": edge["evidence"]["span"],
        "path": edge["evidence"]["path"],
        "lines": edge["evidence"]["lines"],
    }


def section_summary(idx: Index, sid: str) -> dict:
    sec = idx.node[sid]
    return {
        "id": sid,
        "heading": sec["heading"],
        "doc_title": sec["doc_title"],
        "url": sec["url"],
        "path": sec["path"],
        "lines": sec["lines"],
    }


# --------------------------------------------------------------------------
# F1: near miss
# --------------------------------------------------------------------------

def pick_requiring_section(idx: Index, cid: str, askers: list[str]) -> dict:
    """Of the sections that depend on this concept, which one looks like the answer?

    The one a reader would land on: the section that shares the most OTHER
    concepts with the people asking, so it genuinely appears to be on topic.
    Prose evidence beats an identifier lifted out of a code sample, because
    "this section's instructions assume X" is a claim about what the page says.
    """
    asker_concepts: set[str] = set()
    for qid in askers:
        asker_concepts |= idx.question_concept_set.get(qid, set())
    asker_concepts.discard(cid)

    best = None
    for edge in idx.requires(cid):
        sid = edge["from"]
        shared = idx.section_concept_set.get(sid, set()) & asker_concepts
        prose = 1 if edge["evidence"].get("kind") == "prose" else 0
        conf = {"high": 2, "medium": 1, "low": 0}.get(edge["confidence"], 1)
        key = (len(shared), prose, conf, len(idx.section_concept_set.get(sid, set())))
        if best is None or key > best[0]:
            best = (key, edge, shared)
    edge, shared = best[1], best[2]
    return {"edge": edge, "shared_concepts": sorted(shared)}


def find_near_miss(idx: Index) -> list[dict]:
    out = []
    for concept in idx.concepts:
        cid = concept["id"]
        if out_of_scope(concept["label"]):
            continue
        asks = idx.asked_by(cid)
        requires = idx.requires(cid)
        if not asks or not requires:
            continue
        if idx.defines(cid):
            continue

        askers = sorted({e["from"] for e in asks})
        chosen = pick_requiring_section(idx, cid, askers)
        req_edge = chosen["edge"]
        sid = req_edge["from"]
        section = idx.node[sid]

        # The strongest asking question is the one whose thread carries the
        # fullest answer, because that answer is what confirms the gap.
        def answer_weight(qid: str) -> tuple:
            answer = idx.answer_of.get(qid)
            if answer is None:
                return (0, 0, 0)
            return (1 if answer["is_maintainer"] else 0, len(answer["body"]), 0)

        askers_ranked = sorted(askers, key=answer_weight, reverse=True)
        lead_q = askers_ranked[0]
        lead_ask = next(e for e in asks if e["from"] == lead_q)

        out.append({
            "id": "",
            "type": "near_miss",
            "demand": len(askers),
            "concept": cid,
            "concept_label": concept["label"],
            "concept_kind": concept["kind"],
            "section": sid,
            "section_heading": section["heading"],
            "doc_title": section["doc_title"],
            "doc_url": section["url"],
            "section_path": section["path"],
            "section_lines": section["lines"],
            "missing": "",
            "proof_path": [
                {
                    "hop": "Question ASKS_ABOUT Concept",
                    "edge": lead_ask["id"],
                    "from": lead_q,
                    "to": cid,
                    "evidence": evidence_of(lead_ask),
                },
                {
                    "hop": "Section REQUIRES Concept",
                    "edge": req_edge["id"],
                    "from": sid,
                    "to": cid,
                    "evidence": evidence_of(req_edge),
                },
                {
                    "hop": "no Section DEFINES Concept",
                    "sections_checked": idx.total_sections,
                    "defines_edges_found": 0,
                },
            ],
            "questions": askers_ranked,
            "question_titles": [idx.node[q]["title"] for q in askers_ranked[:8]],
            "shared_concepts": chosen["shared_concepts"][:12],
            "other_requiring_sections": sorted(
                {e["from"] for e in requires if e["from"] != sid}),
            "mentions_count": len(idx.mentions(cid)),
            "requires_count": len(requires),
            "requires_evidence_kind": req_edge["evidence"].get("kind", "prose"),
            "code_samples_only": not idx.has_prose_evidence(cid),
            "confirming_answer": None,
            "validated": None,
            "validation_note": "",
            "gap_note": "",
        })

    out.sort(key=lambda f: (-f["demand"], -f["requires_count"], f["concept"]))
    for i, finding in enumerate(out, 1):
        finding["id"] = f"F1-{i:04d}"
    return out


# --------------------------------------------------------------------------
# F2: orphan concept
# --------------------------------------------------------------------------

def find_orphans(idx: Index, near_miss_concepts: set[str]) -> list[dict]:
    out = []
    for concept in idx.concepts:
        cid = concept["id"]
        if cid in near_miss_concepts:
            continue  # already reported, with more evidence, as F1
        if out_of_scope(concept["label"]):
            continue
        if idx.defines(cid):
            continue
        requires = idx.requires(cid)
        mentions = idx.mentions(cid)
        if not requires and not mentions:
            continue
        sections = {e["from"] for e in requires + mentions}
        if len(sections) < MIN_SECTIONS_FOR_ORPHAN:
            continue

        asks = idx.asked_by(cid)
        in_edges = sorted(requires + mentions, key=lambda e: (e["type"], e["from"]))
        out.append({
            "id": "",
            "type": "orphan_concept",
            "demand": len({e["from"] for e in asks}),
            "concept": cid,
            "concept_label": concept["label"],
            "concept_kind": concept["kind"],
            "section": in_edges[0]["from"],
            "section_heading": idx.node[in_edges[0]["from"]]["heading"],
            "doc_title": idx.node[in_edges[0]["from"]]["doc_title"],
            "doc_url": idx.node[in_edges[0]["from"]]["url"],
            "missing": "",
            "proof_path": [
                {
                    "hop": f"Section {e['type']} Concept",
                    "edge": e["id"],
                    "from": e["from"],
                    "to": cid,
                    "evidence": evidence_of(e),
                }
                for e in in_edges[:6]
            ] + [
                {
                    "hop": "no Section DEFINES Concept",
                    "sections_checked": idx.total_sections,
                    "defines_edges_found": 0,
                }
            ],
            "sections_referencing": sorted(sections),
            "requires_count": len(requires),
            "mentions_count": len(mentions),
            "questions": sorted({e["from"] for e in asks}),
            "in_edges_total": len(in_edges),
            "code_samples_only": not idx.has_prose_evidence(cid),
        })

    out.sort(key=lambda f: (-f["demand"], -f["in_edges_total"], f["concept"]))
    for i, finding in enumerate(out, 1):
        finding["id"] = f"F2-{i:04d}"
    return out


# --------------------------------------------------------------------------
# F3: retrieval collision
# --------------------------------------------------------------------------

def find_collisions(idx: Index, answer_similarity) -> list[dict]:
    """Two sections that look equally on topic, where only one holds the answer.

    Concept overlap is the retrieval proxy: a question and a section that share
    concepts are what a topical retriever brings back together. The answer is
    located separately, by similarity to what the maintainer actually replied,
    so "looks relevant" and "contains the answer" are measured by two different
    things rather than by the same one twice.
    """
    grouped: dict[tuple[str, str], dict] = {}

    for question in idx.questions:
        qid = question["id"]
        qconcepts = idx.question_concept_set.get(qid, set())
        if len(qconcepts) < 2:
            continue

        scored = []
        for sid, sconcepts in idx.section_concept_set.items():
            overlap = qconcepts & sconcepts
            if len(overlap) >= MIN_COLLISION_OVERLAP:
                scored.append((len(overlap), sid, sorted(overlap)))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda x: (-x[0], x[1]))

        top_score = scored[0][0]
        competing = [s for s in scored if s[0] >= top_score * COLLISION_RATIO][:5]
        if len(competing) < 2:
            continue

        sims = {sid: answer_similarity(qid, sid) for _, sid, _ in competing}
        if any(v is None for v in sims.values()):
            continue
        holder = max(sims, key=lambda s: sims[s])
        rival = max((s for _, s, _ in competing if s != holder),
                    key=lambda s: (dict((c[1], c[0]) for c in competing)[s], -sims[s]))

        overlap_of = {sid: score for score, sid, _ in competing}
        # The collision only bites when the section that does NOT hold the
        # answer looks at least as on-topic as the one that does.
        if overlap_of[rival] < overlap_of[holder]:
            continue
        if sims[holder] - sims[rival] < 0.02:
            continue  # too close to call which one holds the answer

        key = tuple(sorted((holder, rival)))
        entry = grouped.setdefault(key, {
            "sections": [holder, rival],
            "answer_holder": holder,
            "rival": rival,
            "questions": [],
            "overlaps": [],
            "similarity": {holder: sims[holder], rival: sims[rival]},
            "shared": {},
        })
        entry["questions"].append(qid)
        entry["overlaps"].append({
            "question": qid,
            "answer_holder_overlap": overlap_of[holder],
            "rival_overlap": overlap_of[rival],
            "shared_with_answer_holder": sorted(qconcepts & idx.section_concept_set[holder]),
            "shared_with_rival": sorted(qconcepts & idx.section_concept_set[rival]),
        })

    out = []
    for entry in grouped.values():
        holder, rival = entry["answer_holder"], entry["rival"]
        sample = entry["overlaps"][0]
        out.append({
            "id": "",
            "type": "retrieval_collision",
            "demand": len(set(entry["questions"])),
            "concept": "",
            "concept_label": "",
            "section": rival,
            "section_heading": idx.node[rival]["heading"],
            "doc_title": idx.node[rival]["doc_title"],
            "doc_url": idx.node[rival]["url"],
            "missing": "",
            "answer_holder": section_summary(idx, holder),
            "rival": section_summary(idx, rival),
            "proof_path": [
                {
                    "hop": "Question maps to both Sections",
                    "question": sample["question"],
                    "question_title": idx.node[sample["question"]]["title"],
                    "shared_with_answer_holder": sample["shared_with_answer_holder"],
                    "shared_with_rival": sample["shared_with_rival"],
                },
                {
                    "hop": "concept overlap is comparable",
                    "answer_holder_overlap": sample["answer_holder_overlap"],
                    "rival_overlap": sample["rival_overlap"],
                },
                {
                    "hop": "only one matches the maintainer answer",
                    "answer_holder_similarity": round(entry["similarity"][holder], 4),
                    "rival_similarity": round(entry["similarity"][rival], 4),
                },
            ],
            "questions": sorted(set(entry["questions"])),
            "question_titles": [idx.node[q]["title"] for q in sorted(set(entry["questions"]))[:8]],
            "overlaps": entry["overlaps"][:8],
        })

    out.sort(key=lambda f: (-f["demand"], f["section"]))
    for i, finding in enumerate(out, 1):
        finding["id"] = f"F3-{i:04d}"
    return out


# --------------------------------------------------------------------------

def build_similarity(idx: Index):
    """Cosine similarity between a section and the answer on a thread.

    Uses the same pinned local model as the retrieval comparison, and caches to
    data/work/embeddings.npz so compare.py does not pay for it twice.
    """
    import numpy as np

    cache = WORK / "embeddings.npz"
    section_ids = [s["id"] for s in idx.sections]
    answer_ids = [f"ans:{q['number']}" for q in idx.questions]

    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        if (list(blob["section_ids"]) == section_ids
                and list(blob["answer_ids"]) == answer_ids):
            sec_vecs, ans_vecs = blob["sections"], blob["answers"]
        else:
            sec_vecs = ans_vecs = None
    else:
        sec_vecs = ans_vecs = None

    if sec_vecs is None:
        from sentence_transformers import SentenceTransformer
        log("  embedding sections and answers with all-MiniLM-L6-v2")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        secs = json.loads((GRAPH / "sections.json").read_text(encoding="utf-8"))["sections"]
        embed_text = {s["id"]: s["embed_text"] for s in secs}
        sec_vecs = model.encode([embed_text[s] for s in section_ids],
                                normalize_embeddings=True, batch_size=32,
                                show_progress_bar=False)
        ans_vecs = model.encode([idx.node[a]["body"] for a in answer_ids],
                                normalize_embeddings=True, batch_size=32,
                                show_progress_bar=False)
        WORK.mkdir(parents=True, exist_ok=True)
        np.savez(cache, sections=sec_vecs, answers=ans_vecs,
                 section_ids=np.array(section_ids), answer_ids=np.array(answer_ids))

    sec_row = {sid: i for i, sid in enumerate(section_ids)}
    ans_row = {aid: i for i, aid in enumerate(answer_ids)}

    def similarity(qid: str, sid: str):
        aid = "ans:" + qid.split(":", 1)[1]
        if aid not in ans_row or sid not in sec_row:
            return None
        return float(np.dot(ans_vecs[ans_row[aid]], sec_vecs[sec_row[sid]]))

    return similarity


def main() -> int:
    nodes, edges = load()
    idx = Index(nodes, edges)
    log(f"Graph: {len(idx.sections)} sections, {len(idx.concepts)} concepts, "
        f"{len(idx.questions)} questions")

    # networkx is what makes the traversal in compare.py cheap; building it here
    # too is the sanity check that every edge endpoint actually exists.
    graph = nx.MultiDiGraph()
    for node in nodes:
        graph.add_node(node["id"], type=node["type"])
    for edge in edges:
        if edge["from"] not in graph or edge["to"] not in graph:
            raise SystemExit(f"dangling edge {edge['id']}: {edge['from']} -> {edge['to']}")
        graph.add_edge(edge["from"], edge["to"], key=edge["id"], type=edge["type"])

    log("F1 near miss")
    f1 = find_near_miss(idx)
    log(f"  {len(f1)} findings, demand {sum(f['demand'] for f in f1)} question hits")

    log("F2 orphan concept")
    f2 = find_orphans(idx, {f["concept"] for f in f1})
    log(f"  {len(f2)} findings")

    log("F3 retrieval collision")
    similarity = build_similarity(idx)
    f3 = find_collisions(idx, similarity)
    log(f"  {len(f3)} findings")

    findings = f1 + f2 + f3
    (DATA / "findings.json").write_text(json.dumps({
        "schema_version": 1,
        "manifest_ref": "data/manifest.json",
        "counts": {"near_miss": len(f1), "orphan_concept": len(f2),
                   "retrieval_collision": len(f3)},
        "findings": findings,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    log("")
    log(f"Wrote data/findings.json: {len(findings)} findings")
    if f1:
        log("Top ten near misses by demand:")
        for finding in f1[:10]:
            log(f"  {finding['demand']:>3}  {finding['concept_label']:<28} "
                f"{finding['doc_title']} / {finding['section_heading']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
