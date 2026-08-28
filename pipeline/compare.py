"""Answer every question twice, once from vector retrieval and once from the graph.

Sections 5 to 8 of the design build and analyse a graph. That is knowledge graph
work. GraphRAG means the graph drives RETRIEVAL that feeds GENERATION, and this
is the file that closes that loop, so it is also what turns a diagnosis into a
comparison.

  Path A, vector baseline. Embed every Section and every Question with
  sentence-transformers all-MiniLM-L6-v2, pinned, local, CPU. Take the top five
  sections by cosine similarity.

  Path B, graph retrieval. Take the question's ASKS_ABOUT concepts. Retrieve
  every Section that DEFINES them. Then follow one hop: for each retrieved
  section, also retrieve the sections that DEFINE anything that section
  REQUIRES. Cap at five to match Path A.

The traversal is the whole point. Path B can pull in the prerequisite a section
depends on. Path A cannot, because a prerequisite is not lexically similar to
the question that needs it.

Both answers are generated blind, by separate agents that never see each other's
sections, and then scored against the maintainer's reply on the thread. The
headline metric is the wrong_confident rate per path.

Subcommands:
  retrieve  build both retrieval sets and write the generation payloads
  prep-score  write the scoring payloads once answers exist
  assemble  read answers and verdicts, write data/answers.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data" / "graph"
DATA = ROOT / "data"
WORK = DATA / "work" / "compare"

TOP_K = 5
MAX_SECTION_CHARS = 3000
QUESTIONS_PER_GEN_BATCH = 5
QUESTIONS_PER_SCORE_BATCH = 10

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GENERATION_PROMPT = """\
Answer the user's question using ONLY the documentation sections provided below.
Do not use anything you know about this software from elsewhere. If the sections
do not contain enough to answer, say exactly: INSUFFICIENT CONTEXT, and name
what is missing.

Be direct. Do not hedge unless the sections genuinely leave it ambiguous.

QUESTION:
{question_title}
{question_body}

SECTIONS:
{sections}
"""

SCORING_PROMPT = """\
You are scoring a generated answer against a known correct answer written by a
project maintainer. Return JSON only.

Verdicts:
  correct         - matches the maintainer answer on the substance
  wrong_confident - asserts something that contradicts or omits the key fact,
                    with no acknowledgment of uncertainty
  declined        - says it cannot answer from the material provided

An answer that is vague but not wrong is still wrong_confident if it presents
itself as an answer. Declined requires an explicit statement of insufficiency.

Return: {"verdict": "...", "reason": "one sentence"}

QUESTION: {question_title}
MAINTAINER ANSWER: {maintainer_answer}
GENERATED ANSWER: {generated_answer}
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def load():
    nodes = json.loads((GRAPH / "nodes.json").read_text(encoding="utf-8"))["nodes"]
    edges = json.loads((GRAPH / "edges.json").read_text(encoding="utf-8"))["edges"]
    sections = json.loads((GRAPH / "sections.json").read_text(encoding="utf-8"))["sections"]
    return nodes, edges, sections


# --------------------------------------------------------------------------
# Path A
# --------------------------------------------------------------------------

def vector_retrieval(sections: list[dict], questions: list[dict]) -> dict[str, list[str]]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    log(f"  embedding {len(sections)} sections and {len(questions)} questions")
    model = SentenceTransformer(EMBED_MODEL)
    sec_ids = [s["id"] for s in sections]
    sec_vecs = model.encode([s["embed_text"] for s in sections],
                            normalize_embeddings=True, batch_size=32,
                            show_progress_bar=False)
    q_text = [f"{q['title']}\n\n{q['body']}" for q in questions]
    q_vecs = model.encode(q_text, normalize_embeddings=True, batch_size=32,
                          show_progress_bar=False)

    scores = q_vecs @ sec_vecs.T
    out = {}
    for i, question in enumerate(questions):
        order = np.argsort(-scores[i])[:TOP_K]
        out[question["id"]] = [sec_ids[j] for j in order]
    return out


# --------------------------------------------------------------------------
# Path B
# --------------------------------------------------------------------------

class GraphIndex:
    def __init__(self, edges: list[dict]) -> None:
        self.defines_by_concept: dict[str, list[str]] = defaultdict(list)
        self.requires_by_concept: dict[str, list[str]] = defaultdict(list)
        self.mentions_by_concept: dict[str, list[str]] = defaultdict(list)
        self.requires_of_section: dict[str, list[str]] = defaultdict(list)
        self.asks: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge["type"] == "DEFINES":
                self.defines_by_concept[edge["to"]].append(edge["from"])
            elif edge["type"] == "REQUIRES":
                self.requires_by_concept[edge["to"]].append(edge["from"])
                self.requires_of_section[edge["from"]].append(edge["to"])
            elif edge["type"] == "MENTIONS":
                self.mentions_by_concept[edge["to"]].append(edge["from"])
            elif edge["type"] == "ASKS_ABOUT":
                self.asks[edge["from"]].append(edge["to"])


def graph_retrieval(gi: GraphIndex, qid: str) -> tuple[list[str], list[str]]:
    """Return (section ids, human-readable traversal steps).

    Seed on the sections that DEFINE what the question asks about. Where a
    concept has no definition anywhere - which is exactly the near-miss case -
    fall back to the sections that depend on it, because those are the pages
    that look like the answer, and record that the fallback fired.
    """
    concepts = gi.asks.get(qid, [])
    if not concepts:
        return [], [f"{qid} has no ASKS_ABOUT concepts; graph retrieval has no seed"]

    steps: list[str] = []
    seed_score: dict[str, int] = defaultdict(int)
    for cid in concepts:
        definers = gi.defines_by_concept.get(cid, [])
        if definers:
            for sid in definers:
                seed_score[sid] += 3
            steps.append(f"{qid} ASKS_ABOUT {cid} -> DEFINES -> "
                         f"{len(definers)} section(s)")
        else:
            fallback = gi.requires_by_concept.get(cid, []) + gi.mentions_by_concept.get(cid, [])
            for sid in fallback:
                seed_score[sid] += 1
            steps.append(f"{qid} ASKS_ABOUT {cid} -> no DEFINES anywhere, "
                         f"fell back to {len(set(fallback))} section(s) that depend on it")

    seeds = sorted(seed_score, key=lambda s: (-seed_score[s], s))[:TOP_K]

    # One hop: whatever the seed sections themselves assume, pull in its
    # definition. This is the move a lexical search cannot make.
    expansion_score: dict[str, int] = defaultdict(int)
    for sid in seeds:
        for cid in gi.requires_of_section.get(sid, []):
            for definer in gi.defines_by_concept.get(cid, []):
                if definer in seeds:
                    continue
                expansion_score[definer] += 1
                steps.append(f"{sid} REQUIRES {cid} -> DEFINES -> {definer}")

    expansion = sorted(expansion_score, key=lambda s: (-expansion_score[s], s))
    retrieved = seeds + [s for s in expansion if s not in seeds]
    return retrieved[:TOP_K], steps


# --------------------------------------------------------------------------

def render_sections(section_by_id: dict[str, dict], ids: list[str]) -> str:
    if not ids:
        return "(no sections were retrieved)"
    parts = []
    for sid in ids:
        sec = section_by_id[sid]
        text = sec["embed_text"]
        if len(text) > MAX_SECTION_CHARS:
            text = text[:MAX_SECTION_CHARS] + "\n[section truncated]"
        parts.append(
            f"--- SECTION: {sec['doc_title']} / {sec['heading']}\n"
            f"--- SOURCE: {sec['path']} lines {sec['lines'][0]}-{sec['lines'][1]}\n"
            f"{text}"
        )
    return "\n\n".join(parts)


def cmd_retrieve() -> int:
    nodes, edges, sections = load()
    questions = [n for n in nodes if n["type"] == "Question"]
    answers = {n["question_id"]: n for n in nodes if n["type"] == "Answer"}
    section_by_id = {s["id"]: s for s in sections}

    log("Path A: vector baseline")
    vector = vector_retrieval(sections, questions)

    log("Path B: graph retrieval")
    gi = GraphIndex(edges)
    graph: dict[str, dict] = {}
    for question in questions:
        retrieved, steps = graph_retrieval(gi, question["id"])
        graph[question["id"]] = {"retrieved": retrieved, "traversal": steps}

    empty_graph = sum(1 for q in questions if not graph[q["id"]]["retrieved"])
    identical = sum(1 for q in questions
                    if set(vector[q["id"]]) == set(graph[q["id"]]["retrieved"]))
    reached = sum(1 for q in questions
                  if set(graph[q["id"]]["retrieved"]) - set(vector[q["id"]]))
    log(f"  {empty_graph} questions with no graph seed, "
        f"{identical} where both paths retrieved the same set, "
        f"{reached} where the graph reached a section the vector path missed")

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "retrieval.json").write_text(json.dumps({
        "top_k": TOP_K,
        "embedding_model": EMBED_MODEL,
        "questions": [
            {
                "id": q["id"],
                "title": q["title"],
                "url": q["url"],
                "vector": vector[q["id"]],
                "graph": graph[q["id"]]["retrieved"],
                "traversal": graph[q["id"]]["traversal"],
            }
            for q in questions
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generation payloads, one directory per path so an agent working on one
    # path cannot see the other path's sections.
    for path_name, retrieved_by_q in (("vector", vector),
                                      ("graph", {k: v["retrieved"] for k, v in graph.items()})):
        indir = WORK / path_name / "in"
        (WORK / path_name / "out").mkdir(parents=True, exist_ok=True)
        indir.mkdir(parents=True, exist_ok=True)
        batches = []
        for i in range(0, len(questions), QUESTIONS_PER_GEN_BATCH):
            chunk = questions[i: i + QUESTIONS_PER_GEN_BATCH]
            batch_id = f"gen-{path_name}-{i // QUESTIONS_PER_GEN_BATCH:03d}"
            batches.append(batch_id)
            blocks = []
            for question in chunk:
                blocks.append("=" * 70)
                blocks.append(f"ANSWER ID: {question['id']}")
                blocks.append(GENERATION_PROMPT.format(
                    question_title=question["title"],
                    question_body=question["body"][:4000],
                    sections=render_sections(section_by_id, retrieved_by_q[question["id"]]),
                ))
                blocks.append("")
            (indir / f"{batch_id}.txt").write_text("\n".join(blocks), encoding="utf-8")
        (WORK / path_name / "batches.json").write_text(
            json.dumps({"batches": batches}, indent=2), encoding="utf-8")
        log(f"  {len(batches)} generation batches for the {path_name} path")

    (WORK / "answers_reference.json").write_text(json.dumps({
        q["id"]: {
            "title": q["title"],
            "maintainer_answer": answers[q["id"]]["body"],
            "is_maintainer": answers[q["id"]]["is_maintainer"],
            "url": answers[q["id"]]["url"],
        }
        for q in questions
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Wrote retrieval sets and generation payloads to {WORK.relative_to(ROOT)}")
    return 0


def load_generated(path_name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    outdir = WORK / path_name / "out"
    for path in sorted(outdir.glob("gen-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log(f"  unreadable: {path.name}")
            continue
        for qid, answer in payload.items():
            if isinstance(answer, dict):
                answer = answer.get("answer", "")
            if isinstance(answer, str) and answer.strip():
                out[qid] = answer.strip()
    return out


def cmd_prep_score() -> int:
    reference = json.loads((WORK / "answers_reference.json").read_text(encoding="utf-8"))
    generated = {p: load_generated(p) for p in ("vector", "graph")}
    log(f"  vector answers {len(generated['vector'])}, graph answers {len(generated['graph'])}")

    qids = [q for q in reference if q in generated["vector"] or q in generated["graph"]]
    indir = WORK / "score" / "in"
    indir.mkdir(parents=True, exist_ok=True)
    (WORK / "score" / "out").mkdir(parents=True, exist_ok=True)

    batches = []
    for i in range(0, len(qids), QUESTIONS_PER_SCORE_BATCH):
        chunk = qids[i: i + QUESTIONS_PER_SCORE_BATCH]
        batch_id = f"score-{i // QUESTIONS_PER_SCORE_BATCH:03d}"
        batches.append(batch_id)
        blocks = []
        for qid in chunk:
            for path_name in ("vector", "graph"):
                if qid not in generated[path_name]:
                    continue
                blocks.append("=" * 70)
                blocks.append(f"SCORE ID: {qid}::{path_name}")
                blocks.append(SCORING_PROMPT.format(
                    question_title=reference[qid]["title"],
                    maintainer_answer=reference[qid]["maintainer_answer"][:6000],
                    generated_answer=generated[path_name][qid][:6000],
                ))
                blocks.append("")
        (indir / f"{batch_id}.txt").write_text("\n".join(blocks), encoding="utf-8")
    (WORK / "score" / "batches.json").write_text(
        json.dumps({"batches": batches}, indent=2), encoding="utf-8")
    log(f"  {len(batches)} scoring batches over {len(qids)} questions")
    return 0


def cmd_assemble() -> int:
    retrieval = json.loads((WORK / "retrieval.json").read_text(encoding="utf-8"))
    reference = json.loads((WORK / "answers_reference.json").read_text(encoding="utf-8"))
    generated = {p: load_generated(p) for p in ("vector", "graph")}

    verdicts: dict[str, dict] = {}
    for path in sorted((WORK / "score" / "out").glob("score-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log(f"  unreadable: {path.name}")
            continue
        verdicts.update(payload)

    valid = {"correct", "wrong_confident", "declined"}
    results = []
    summary = {p: {"correct": 0, "wrong_confident": 0, "declined": 0, "unscored": 0}
               for p in ("vector", "graph")}

    for question in retrieval["questions"]:
        qid = question["id"]
        entry = {
            "question_id": qid,
            "question_title": question["title"],
            "question_url": question["url"],
            "maintainer_answer_url": reference.get(qid, {}).get("url"),
        }
        for path_name in ("vector", "graph"):
            answer = generated[path_name].get(qid, "")
            record = verdicts.get(f"{qid}::{path_name}") or {}
            verdict = str(record.get("verdict", "")).lower()
            if verdict not in valid:
                verdict = "unscored"
            summary[path_name][verdict] = summary[path_name].get(verdict, 0) + 1
            block = {
                "retrieved": question[path_name],
                "answer": answer,
                "verdict": verdict,
                "reason": record.get("reason", ""),
            }
            if path_name == "graph":
                block["traversal"] = question["traversal"]
            entry[path_name] = block
        results.append(entry)

    def rate(path_name: str, bucket: str) -> float:
        scored = sum(v for k, v in summary[path_name].items() if k != "unscored")
        return round(summary[path_name][bucket] / scored, 4) if scored else 0.0

    payload = {
        "schema_version": 1,
        "embedding_model": EMBED_MODEL,
        "generator": "claude-code-batch",
        "top_k": TOP_K,
        "results": results,
        "summary": {
            "n": len(results),
            "vector": summary["vector"],
            "graph": summary["graph"],
            "wrong_confident_rate": {
                "vector": rate("vector", "wrong_confident"),
                "graph": rate("graph", "wrong_confident"),
            },
            "correct_rate": {
                "vector": rate("vector", "correct"),
                "graph": rate("graph", "correct"),
            },
            "declined_rate": {
                "vector": rate("vector", "declined"),
                "graph": rate("graph", "declined"),
            },
        },
    }
    (DATA / "answers.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    log("")
    log(f"Wrote data/answers.json over {len(results)} questions")
    for path_name in ("vector", "graph"):
        counts = summary[path_name]
        log(f"  {path_name:<7} correct {counts['correct']}, "
            f"wrong_confident {counts['wrong_confident']}, "
            f"declined {counts['declined']}, unscored {counts['unscored']}")
    log(f"  wrong_confident rate: vector {payload['summary']['wrong_confident_rate']['vector']}, "
        f"graph {payload['summary']['wrong_confident_rate']['graph']}")
    return 0



def _llm():
    """Imported lazily so the deterministic steps never need the module."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import llm
    return llm

def cmd_run(which: str) -> int:
    """API path for generation and scoring.

    The two retrieval paths are generated in separate calls that never see each
    other's sections, which is the whole point of the comparison.
    """
    llm = _llm()
    if not llm.available():
        print("No API key. Have a Claude Code session fill the .txt payloads under "
              f"{WORK.relative_to(ROOT)} into matching .json files.")
        return 1
    print(f"Running {which} with {llm.model_id()}")
    if which == "score":
        return llm.run_batches(WORK / "score" / "in", WORK / "score" / "out", "score-*")
    return llm.run_batches(WORK / which / "in", WORK / which / "out", f"gen-{which}-*")


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "retrieve"
    if command == "retrieve":
        return cmd_retrieve()
    if command == "prep-score":
        return cmd_prep_score()
    if command == "assemble":
        return cmd_assemble()
    if command in ("run-vector", "run-graph", "run-score"):
        return cmd_run(command[len("run-"):])
    print(f"unknown subcommand: {command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
