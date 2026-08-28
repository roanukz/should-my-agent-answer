"""Name the missing fact on every near-miss finding.

find.py is a deterministic function of the graph and stays that way. But a
finding that says "concept:engine is undefined" is a graph fact, not something a
content team can act on. What they need is the sentence: "the documentation
never says how the engine is configured for a database other than SQLite."

Naming an absent fact needs judgment, so it gets its own model pass, and its
own file, kept away from find.py so the deterministic part stays deterministic.

This runs BEFORE validate.py and is deliberately separate from it. This pass
describes the gap; validate.py independently decides whether the gap is real,
without seeing what this pass wrote. Letting one agent do both would be marking
its own homework.

Subcommands:
  prep      write payloads to data/work/describe/in/
  assemble  fold the descriptions back into data/findings.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GRAPH = DATA / "graph"
WORK = DATA / "work" / "describe"

FINDINGS_PER_BATCH = 6
MAX_ANSWER_CHARS = 4000


def log(msg: str) -> None:
    print(msg, flush=True)


def load():
    findings = json.loads((DATA / "findings.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in
             json.loads((GRAPH / "nodes.json").read_text(encoding="utf-8"))["nodes"]}
    sections = {s["id"]: s for s in
                json.loads((GRAPH / "sections.json").read_text(encoding="utf-8"))["sections"]}
    return findings, nodes, sections


HEADER = """\
You are naming the fact a documentation set never supplies.

Each item below is a candidate gap found in the FastAPI documentation. The graph
says: people asked about a concept, one documentation section's instructions
depend on that concept, and no section in the 60-page corpus explains it. On the
thread, a maintainer or another user answered the question, so the correct
information exists somewhere outside the documentation.

For each item, return:

  "missing"    ONE clause naming the specific fact the documentation never
               supplies. Start with a noun or a "how"/"what"/"which" phrase, no
               leading capital, no full stop. Concrete, not abstract.
               Good:  how the engine is configured for a database other than SQLite
               Bad:   more information about engines
               Bad:   The documentation is missing details.

  "confirming_excerpt"
               A VERBATIM excerpt from the ANSWER shown, between 20 and 300
               characters, that supplies that fact. Copy it exactly. If the
               answer does not actually supply the missing fact, return an empty
               string, and say so in "note".

  "note"       One sentence, at most 25 words, on what the answer supplies that
               the documentation does not. If you think this candidate is not a
               real gap, say that here instead.

Write plainly. Two house rules, both checked afterwards: use US spelling (color,
behavior, normalize, recognize, labeled), and do not use an em dash or an en dash
anywhere in your output.

Return JSON only, keyed by finding id:

{
  "F1-0012": {
    "missing": "how the engine is configured for a database other than SQLite",
    "confirming_excerpt": "You change the connection string. For Postgres it is postgresql://",
    "note": "The answer gives the connection string format, which no page in the corpus shows."
  }
}
"""


def cmd_prep() -> int:
    payload, nodes, sections = load()
    targets = [f for f in payload["findings"] if f["type"] == "near_miss"]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "in").mkdir(exist_ok=True)
    (WORK / "out").mkdir(exist_ok=True)

    batches = []
    for i in range(0, len(targets), FINDINGS_PER_BATCH):
        chunk = targets[i: i + FINDINGS_PER_BATCH]
        batch_id = f"desc-{i // FINDINGS_PER_BATCH:03d}"
        batches.append(batch_id)
        blocks = [HEADER, ""]
        for finding in chunk:
            section = sections[finding["section"]]
            lead_q = finding["questions"][0]
            question = nodes[lead_q]
            answer = nodes.get(f"ans:{question['number']}", {})
            requires_hop = finding["proof_path"][1]
            asks_hop = finding["proof_path"][0]
            blocks += [
                "=" * 70,
                f"FINDING ID: {finding['id']}",
                f"CONCEPT: {finding['concept_label']}  ({finding['concept_kind']})",
                f"ASKED BY: {finding['demand']} distinct threads",
                "",
                f"SECTION THAT DEPENDS ON IT: {section['doc_title']} / {section['heading']}",
                f"  source: {section['path']} lines {section['lines'][0]}-{section['lines'][1]}",
                f"  the span that shows the dependency: {requires_hop['evidence']['span']!r}",
                "",
                "SECTION TEXT:",
                section["embed_text"][:5000],
                "",
                f"LEAD QUESTION ({lead_q}): {question['title']}",
                f"  the span that shows what they asked: {asks_hop['evidence']['span']!r}",
                question["body"][:2000],
                "",
                f"ANSWER on that thread (by {answer.get('author')}, "
                f"maintainer: {answer.get('is_maintainer')}):",
                (answer.get("body") or "")[:MAX_ANSWER_CHARS],
                "",
                "OTHER QUESTIONS ASKING ABOUT THE SAME CONCEPT:",
                *[f"  - {t}" for t in finding["question_titles"][1:6]],
                "",
            ]
        (WORK / "in" / f"{batch_id}.txt").write_text("\n".join(blocks), encoding="utf-8")

    (WORK / "batches.json").write_text(json.dumps({"batches": batches}, indent=2),
                                       encoding="utf-8")
    log(f"{len(batches)} describe batches over {len(targets)} near-miss findings")
    return 0


def cmd_assemble() -> int:
    payload, nodes, _sections = load()
    described: dict[str, dict] = {}
    for path in sorted((WORK / "out").glob("desc-*.json")):
        try:
            described.update(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log(f"  unreadable: {path.name}")

    filled = 0
    with_excerpt = 0
    for finding in payload["findings"]:
        record = described.get(finding["id"])
        if not record:
            continue
        missing = str(record.get("missing", "")).strip()
        if missing:
            finding["missing"] = missing
            filled += 1
        # Its own field. `validation_note` belongs to validate.py, and having
        # both write the same key meant whichever ran last silently won.
        note = str(record.get("note", "")).strip()
        if note:
            finding["gap_note"] = note
        excerpt = str(record.get("confirming_excerpt", "")).strip()
        if excerpt and finding["questions"]:
            lead = finding["questions"][0]
            number = nodes[lead]["number"]
            answer = nodes.get(f"ans:{number}")
            if answer:
                finding["confirming_answer"] = {
                    "id": answer["id"],
                    "excerpt": excerpt,
                    "author": answer["author"],
                    "is_maintainer": answer["is_maintainer"],
                    "url": answer["url"],
                    "question_id": lead,
                    "question_title": nodes[lead]["title"],
                }
                with_excerpt += 1

    (DATA / "findings.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(1 for f in payload["findings"] if f["type"] == "near_miss")
    log(f"described {filled}/{total} near-miss findings, "
        f"{with_excerpt} with a confirming answer excerpt")
    return 0



def _llm():
    """Imported lazily so the deterministic steps never need the module."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import llm
    return llm

def cmd_run() -> int:
    llm = _llm()
    if not llm.available():
        print("No API key. Have a Claude Code session fill "
              f"{(WORK / 'in').relative_to(ROOT)}/*.txt -> {(WORK / 'out').relative_to(ROOT)}/*.json")
        return 1
    print(f"Describing gaps with {llm.model_id()}")
    return llm.run_batches(WORK / "in", WORK / "out", "desc-*")


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "prep"
    if command == "prep":
        return cmd_prep()
    if command == "assemble":
        return cmd_assemble()
    if command == "run":
        return cmd_run()
    print(f"unknown subcommand: {command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
