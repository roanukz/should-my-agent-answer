"""Concept and edge extraction, and the machinery that keeps it honest.

This is the only step that needs a model. Two paths, as the PRD requires:

  1. ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment -> call it directly.
  2. Neither -> `python extract.py prep` writes one payload file per batch, a
     Claude Code session fills in the matching output file, and
     `python extract.py assemble` turns those into nodes and edges.

Both paths produce the same committed JSON, and both go through the same
validator, which is the part that actually matters.

THE VALIDATOR IS THE INVARIANT. A model asked for a verbatim span will
sometimes paraphrase, normalise whitespace, or quietly invent one. So no span is
believed. Every span is searched for in the file the edge claims it came from,
and an edge whose span is not found there is dropped, not downgraded. The count
of dropped edges is reported and lands in the manifest, because a silent drop
would be exactly the kind of invisible failure this whole project is about.

Subcommands:
  prep                write section batch payloads to data/work/extract/in/
  assemble            validate data/work/extract/out/ and write nodes.json + edges.json
  status              how many section batches are done
  prep-questions      write question batch payloads to data/work/questions/in/
  assemble-questions  validate answers and write ASKS_ABOUT edges
  status-questions    how many question batches are done
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data" / "graph"
WORK = ROOT / "data" / "work" / "extract"
WORK_IN = WORK / "in"
WORK_OUT = WORK / "out"

MAX_SECTIONS_PER_BATCH = 12
MAX_CHARS_PER_BATCH = 18_000

EDGE_TYPES = {"DEFINES", "REQUIRES", "MENTIONS"}
CONCEPT_KINDS = {"parameter", "object", "class", "function", "config_key", "decorator", "term"}
CONFIDENCES = {"high", "medium", "low"}

MIN_SPAN_CHARS = 12


# --------------------------------------------------------------------------
# The prompt. Kept here so the API path and the Claude Code path cannot drift.
# --------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are extracting a knowledge graph from technical documentation. You will be
given one section of a documentation page. Return JSON only, no prose.

For every distinct named thing in this section (a parameter, class, function,
config key, decorator, or domain term), decide which ONE of these applies:

  DEFINES  - this section explains what the thing is or how to use it. A reader
             who knows nothing could learn it here.
  REQUIRES - this section's instructions assume the reader already knows this
             thing. Following the instructions without knowing it would fail.
  MENTIONS - referenced in passing, neither explained nor depended upon.

The distinction between DEFINES and REQUIRES is the entire point of this task.
Be strict. A thing named in a code sample without explanation is REQUIRES or
MENTIONS, never DEFINES.

For EVERY relationship you emit you MUST include the verbatim span of text from
this section that justifies it. Copy it exactly. Never paraphrase. Never invent
a span. If you cannot point at a span, do not emit the relationship.

Return:
{
  "concepts": [
    {"label": "engine", "kind": "object", "aliases": ["create_engine"]}
  ],
  "edges": [
    {
      "type": "REQUIRES",
      "concept": "engine",
      "span": "we need to create the tables using the engine we defined earlier",
      "confidence": "high"
    }
  ]
}

kind is one of: parameter, object, class, function, config_key, decorator, term.
confidence is one of: high, medium, low.

SECTION PATH: {path}
SECTION LINES: {start}-{end}
SECTION HEADING: {heading}

SECTION TEXT:
{text}
"""

# Additions the PRD's prompt does not carry but the corpus needs. Kept separate
# and appended, so the PRD prompt above stays readable as written.
EXTRACTION_PROMPT_NOTES = """\

ADDITIONAL RULES FOR THIS CORPUS

A span may be copied from the SECTION TEXT or from any CODE SAMPLE block shown
below the section text. Set "span_source" to "section" or to the code sample's
path. Default is "section".

Do not emit a concept for FastAPI itself or for Python. Those are the subject,
not a dependency.

DO emit a concept that shares its name with the page or the section heading. The
page called "Middleware" is exactly the page that explains what middleware is,
and that DEFINES edge is the most valuable one on the page. Skipping it makes
the corpus look as though it never explains its own subjects.

Prefer the name a reader would search for. `Query` not "the Query function".
Strip backticks and markdown from the label. Lowercase a plain-English term;
keep the exact casing of an identifier.

A section can DEFINE a thing and REQUIRE other things at the same time. Emit
every relationship you can evidence, not just one per section.

Return one JSON object per section, keyed by section id, like:

{
  "sec:tutorial/sql-databases#create-an-engine": {
    "concepts": [...],
    "edges": [...]
  }
}
"""


# --------------------------------------------------------------------------
# Normalisation used by the validator
# --------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Collapse the differences a model introduces without changing meaning.

    Unicode NFKC (so a curly quote matches a straight one), whitespace runs to a
    single space, and case folded. Nothing else: this must not be loose enough
    to let a paraphrase through.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


MARKER_RE = re.compile(r"[`*_~]")
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def strip_markdown_markers(text: str) -> str:
    """Drop the characters that are markup rather than content.

    Used for ONE case, and only that case: a span copied out of a question.
    GitHub's API serves a discussion body twice, once as markdown and once as
    the plain text it renders to. The question mapper reads the plain text; the
    thread file on disk holds the markdown. The difference between them is
    exactly these characters, so removing them from both sides is matching like
    with like, not loosening the match. It is still a substring search.
    """
    text = LINK_RE.sub(r"\1", text)
    return MARKER_RE.sub("", text)


def locate_span(span: str, haystack: str, base_line: int,
                transform=None) -> tuple[int, int] | None:
    """Find `span` in `haystack` and return its 1-based line range.

    base_line is the file line number of haystack's first line.
    """
    transform = transform or normalise
    norm_span = transform(span)
    if len(norm_span) < MIN_SPAN_CHARS:
        return None

    lines = haystack.splitlines()
    # Build a normalised copy of the haystack plus a map back to line numbers.
    pieces: list[str] = []
    line_of_char: list[int] = []
    for offset, line in enumerate(lines):
        norm_line = transform(line)
        if pieces and norm_line:
            pieces.append(" ")
            line_of_char.append(base_line + offset)
        for _ in norm_line:
            line_of_char.append(base_line + offset)
        pieces.append(norm_line)
    norm_hay = "".join(pieces)

    index = norm_hay.find(norm_span)
    if index < 0:
        return None
    end = index + len(norm_span) - 1
    if end >= len(line_of_char):
        end = len(line_of_char) - 1
    return line_of_char[index], line_of_char[end]


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def load_sections() -> tuple[list[dict], list[dict]]:
    data = json.loads((GRAPH / "sections.json").read_text(encoding="utf-8"))
    return data["docs"], data["sections"]


def build_batches(sections: list[dict]) -> list[dict]:
    """One batch per page, split further when a page is long.

    Batching by page rather than by a flat count of ten gives the model the
    surrounding page, which is what the DEFINES/REQUIRES call turns on: whether
    the thing is explained HERE or assumed from somewhere else.
    """
    by_doc: dict[str, list[dict]] = {}
    for sec in sections:
        by_doc.setdefault(sec["doc_id"], []).append(sec)

    batches: list[dict] = []
    for doc_id in sorted(by_doc):
        current: list[dict] = []
        chars = 0
        part = 1
        pages = by_doc[doc_id]
        for sec in pages:
            size = len(sec["embed_text"])
            if current and (len(current) >= MAX_SECTIONS_PER_BATCH
                            or chars + size > MAX_CHARS_PER_BATCH):
                batches.append(make_batch(doc_id, part, current))
                current, chars, part = [], 0, part + 1
            current.append(sec)
            chars += size
        if current:
            batches.append(make_batch(doc_id, part, current))
    for i, batch in enumerate(batches):
        batch["batch_id"] = f"batch-{i:03d}"
    return batches


def make_batch(doc_id: str, part: int, sections: list[dict]) -> dict:
    slug = doc_id[len("doc:"):].replace("/", "__")
    return {
        "batch_id": "",
        "name": f"{slug}--{part}",
        "doc_id": doc_id,
        "doc_title": sections[0]["doc_title"],
        "path": sections[0]["path"],
        "sections": [
            {
                "id": s["id"],
                "heading": s["heading"],
                "level": s["level"],
                "path": s["path"],
                "lines": s["lines"],
                "text": s["text"],
                "code": [
                    {"path": c["path"], "lines": c["lines"], "text": c["text"]}
                    for c in s["code"]
                ],
            }
            for s in sections
        ],
    }


def render_batch_prompt(batch: dict) -> str:
    """The exact text a model sees for one batch."""
    header = (
        EXTRACTION_PROMPT.split("SECTION PATH:")[0].rstrip()
        + EXTRACTION_PROMPT_NOTES
    )
    parts = [header, "", f"PAGE: {batch['doc_title']}  ({batch['path']})", ""]
    for sec in batch["sections"]:
        parts.append("=" * 70)
        parts.append(f"SECTION ID: {sec['id']}")
        parts.append(f"SECTION PATH: {sec['path']}")
        parts.append(f"SECTION LINES: {sec['lines'][0]}-{sec['lines'][1]}")
        parts.append(f"SECTION HEADING: {sec['heading']}")
        parts.append("")
        parts.append("SECTION TEXT:")
        parts.append(sec["text"])
        for code in sec["code"]:
            parts.append("")
            parts.append(f"CODE SAMPLE {code['path']} lines {code['lines'][0]}-{code['lines'][1]}:")
            parts.append(code["text"])
        parts.append("")
    return "\n".join(parts)


def cmd_prep() -> int:
    _docs, sections = load_sections()
    batches = build_batches(sections)
    WORK_IN.mkdir(parents=True, exist_ok=True)
    WORK_OUT.mkdir(parents=True, exist_ok=True)
    for batch in batches:
        (WORK_IN / f"{batch['batch_id']}.json").write_text(
            json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        (WORK_IN / f"{batch['batch_id']}.prompt.txt").write_text(
            render_batch_prompt(batch), encoding="utf-8")
    index = [
        {"batch_id": b["batch_id"], "doc_id": b["doc_id"], "name": b["name"],
         "sections": len(b["sections"]),
         "chars": sum(len(s["text"]) for s in b["sections"])}
        for b in batches
    ]
    (WORK / "batches.json").write_text(
        json.dumps({"batches": index}, indent=2), encoding="utf-8")
    print(f"{len(batches)} batches over {len(sections)} sections -> {WORK_IN.relative_to(ROOT)}")
    print(f"Write each answer to {WORK_OUT.relative_to(ROOT)}/<batch_id>.json")
    return 0


def cmd_status() -> int:
    index = json.loads((WORK / "batches.json").read_text(encoding="utf-8"))["batches"]
    done = {p.stem for p in WORK_OUT.glob("batch-*.json")}
    missing = [b["batch_id"] for b in index if b["batch_id"] not in done]
    print(f"{len(done)}/{len(index)} batches complete")
    if missing:
        print("missing: " + " ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    return 0


# --------------------------------------------------------------------------
# Assembly and validation
# --------------------------------------------------------------------------

def concept_id(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").lower()
    return f"concept:{slug}"


def clean_label(label: str) -> str:
    label = label.strip().strip("`").strip()
    label = re.sub(r"^\*+|\*+$", "", label).strip()
    return label


def cmd_assemble() -> int:
    docs, sections = load_sections()
    by_id = {s["id"]: s for s in sections}

    outputs = sorted(WORK_OUT.glob("batch-*.json"))
    if not outputs:
        print("No batch outputs under data/work/extract/out/. Run prep first, then "
              "have a model fill them in.")
        return 1

    concepts: dict[str, dict] = {}
    edges: list[dict] = []
    stats = {
        "raw_edges": 0, "dropped_no_span": 0, "dropped_bad_type": 0,
        "dropped_unknown_section": 0, "dropped_short_span": 0,
        "dropped_heading_only_span": 0, "dropped_duplicate": 0,
    }
    dropped_examples: list[dict] = []
    seen_edge_keys: set[tuple] = set()

    for path in outputs:
        batch_id = path.stem
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  {batch_id}: unreadable JSON ({exc}); skipped")
            continue

        for section_id, result in payload.items():
            section = by_id.get(section_id)
            if section is None:
                stats["dropped_unknown_section"] += len(result.get("edges", []))
                continue

            for concept in result.get("concepts", []):
                label = clean_label(str(concept.get("label", "")))
                if not label:
                    continue
                cid = concept_id(label)
                kind = concept.get("kind")
                if kind not in CONCEPT_KINDS:
                    kind = "term"
                entry = concepts.setdefault(cid, {
                    "id": cid, "type": "Concept", "label": label,
                    "kind": kind, "aliases": [], "kinds_seen": {}, "labels_seen": {},
                })
                entry["kinds_seen"][kind] = entry["kinds_seen"].get(kind, 0) + 1
                entry["labels_seen"][label] = entry["labels_seen"].get(label, 0) + 1
                for alias in concept.get("aliases", []) or []:
                    alias = clean_label(str(alias))
                    if alias and alias != label and alias not in entry["aliases"]:
                        entry["aliases"].append(alias)

            for edge in result.get("edges", []):
                stats["raw_edges"] += 1
                etype = str(edge.get("type", "")).upper()
                if etype not in EDGE_TYPES:
                    stats["dropped_bad_type"] += 1
                    continue
                label = clean_label(str(edge.get("concept", "")))
                if not label:
                    stats["dropped_bad_type"] += 1
                    continue
                cid = concept_id(label)
                span = str(edge.get("span", "") or "")
                if len(normalise(span)) < MIN_SPAN_CHARS:
                    stats["dropped_short_span"] += 1
                    continue

                # A section titled "Middleware" that explains what middleware is
                # is the canonical DEFINES edge, so a concept matching the
                # heading is fine. What is not fine is a span that is nothing
                # but the heading line, which evidences the title and not the
                # claim.
                heading_line = section["text"].splitlines()[0] if section["text"] else ""
                if normalise(span) == normalise(heading_line):
                    stats["dropped_heading_only_span"] += 1
                    continue

                found = locate_span(span, section["text"], section["lines"][0])
                span_path = section["path"]
                if found is None:
                    for code in section["code"]:
                        found = locate_span(span, code["text"], code["lines"][0])
                        if found is not None:
                            span_path = code["path"]
                            break
                if found is None:
                    stats["dropped_no_span"] += 1
                    if len(dropped_examples) < 40:
                        dropped_examples.append({
                            "batch": batch_id, "section": section_id,
                            "concept": label, "type": etype, "span": span[:160],
                        })
                    continue

                key = (section_id, cid, etype)
                if key in seen_edge_keys:
                    stats["dropped_duplicate"] += 1
                    continue
                seen_edge_keys.add(key)

                confidence = str(edge.get("confidence", "medium")).lower()
                if confidence not in CONFIDENCES:
                    confidence = "medium"

                entry = concepts.setdefault(cid, {
                    "id": cid, "type": "Concept", "label": label,
                    "kind": "term", "aliases": [], "kinds_seen": {"term": 1},
                    "labels_seen": {},
                })
                entry["labels_seen"][label] = entry["labels_seen"].get(label, 0) + 1
                edges.append({
                    "id": "",
                    "type": etype,
                    "from": section_id,
                    "to": cid,
                    "evidence": {
                        "span": span.strip(),
                        "path": span_path,
                        "lines": [found[0], found[1]],
                        "kind": "prose" if span_path == section["path"] else "code",
                    },
                    "extractor": batch_id,
                    "confidence": confidence,
                })

    # Settle each concept on the kind and the surface form its extractors used
    # most often. `response_class` and "response class" already share an id
    # because the slug collapses punctuation; this decides which one shows.
    for entry in concepts.values():
        kinds = entry.pop("kinds_seen")
        labels = entry.pop("labels_seen") or {entry["label"]: 1}
        entry["kind"] = max(kinds.items(), key=lambda kv: (kv[1], kv[0]))[0]
        entry["label"] = max(labels.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
        entry["surface_forms"] = sorted(labels)
        entry["aliases"] = sorted(set(entry["aliases"]) - {entry["label"]})

    for i, edge in enumerate(sorted(edges, key=lambda e: (e["from"], e["to"], e["type"]))):
        edge["id"] = f"e:{i:05d}"
    edges = sorted(edges, key=lambda e: e["id"])

    write_graph(docs, sections, concepts, edges, stats, dropped_examples, len(outputs))
    return 0


def write_graph(docs, sections, concepts, edges, stats, dropped_examples, batches_done) -> None:
    nodes = []
    nodes.extend(docs)
    for sec in sections:
        nodes.append({k: v for k, v in sec.items() if k != "embed_text"})
    nodes.extend(sorted(concepts.values(), key=lambda c: c["id"]))

    (GRAPH / "nodes.json").write_text(
        json.dumps({"schema_version": 1, "nodes": nodes}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (GRAPH / "edges.json").write_text(
        json.dumps({"schema_version": 1, "edges": edges}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (GRAPH / "extraction_report.json").write_text(
        json.dumps({
            "batches_processed": batches_done,
            "concepts": len(concepts),
            "edges_kept": len(edges),
            **stats,
            "dropped_examples": dropped_examples,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    kept = len(edges)
    raw = stats["raw_edges"]
    print(f"{batches_done} batches -> {len(concepts)} concepts, {kept} edges kept of {raw}")
    for key in ("dropped_no_span", "dropped_short_span", "dropped_bad_type",
                "dropped_heading_only_span", "dropped_duplicate", "dropped_unknown_section"):
        if stats[key]:
            print(f"  {key}: {stats[key]}")
    by_type: dict[str, int] = {}
    for edge in edges:
        by_type[edge["type"]] = by_type.get(edge["type"], 0) + 1
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))
    assert all(e["evidence"] and e["evidence"].get("span") for e in edges), \
        "an edge reached edges.json without a span"
    print("  invariant holds: zero edges with a null or empty evidence span")




# --------------------------------------------------------------------------
# Question mapping: which concepts is each thread actually about
# --------------------------------------------------------------------------

QWORK = ROOT / "data" / "work" / "questions"
QWORK_IN = QWORK / "in"
QWORK_OUT = QWORK / "out"
QUESTIONS_PER_BATCH = 10

QUESTION_PROMPT = """\
You are mapping a user question to the concepts it is about. You will be given
one discussion thread title and body, and a list of known concept labels from a
documentation set. Return JSON only.

Return the concepts this question is actually asking about. Do not return a
concept merely because the word appears. The test is whether answering the
question requires understanding that concept.

For every concept you return you MUST include the verbatim span from the
question that justifies it.

Return:
{
  "asks_about": [
    {
      "concept": "engine",
      "span": "point the engine at Postgres instead of SQLite",
      "confidence": "high"
    }
  ]
}
"""

QUESTION_PROMPT_NOTES = """\
ADDITIONAL RULES FOR THIS CORPUS

Use a concept label EXACTLY as it appears in KNOWN CONCEPTS. A label that is not
on the list is dropped by the validator, so do not invent one and do not correct
the spelling or casing of one that is on the list.

The span must be copied verbatim from the question's TITLE or BODY, at least 12
characters, and contiguous. Whitespace runs and curly-vs-straight quotes are
normalised for you. Everything else must match, and a span the validator cannot
find in the thread file loses its edge.

Return between one and six concepts per question. Prefer the few that the
question genuinely turns on over a long list of things it touches. A question
with nothing on the list that it truly turns on should return an empty array,
and that is a valid answer.

Return one JSON object per question, keyed by question id:

{
  "q:11836": {"asks_about": [{"concept": "engine", "span": "...", "confidence": "high"}]}
}
"""


def candidate_concepts() -> list[dict]:
    """Concepts worth offering to the question mapper.

    A concept that is only ever MENTIONED is page furniture, not something a
    reader depends on, so it is left off the list. Concepts that are defined or
    required somewhere stay, ordered by how much of the corpus touches them, so
    the model reads the load-bearing ones first.
    """
    nodes = json.loads((GRAPH / "nodes.json").read_text(encoding="utf-8"))["nodes"]
    edges = json.loads((GRAPH / "edges.json").read_text(encoding="utf-8"))["edges"]
    concepts = {n["id"]: n for n in nodes if n["type"] == "Concept"}
    counts: dict[str, dict[str, int]] = {cid: {"DEFINES": 0, "REQUIRES": 0, "MENTIONS": 0}
                                         for cid in concepts}
    sections: dict[str, set] = {cid: set() for cid in concepts}
    for edge in edges:
        if edge["to"] in counts and edge["type"] in counts[edge["to"]]:
            counts[edge["to"]][edge["type"]] += 1
            sections[edge["to"]].add(edge["from"])

    out = []
    for cid, node in concepts.items():
        c = counts[cid]
        if c["DEFINES"] == 0 and c["REQUIRES"] == 0:
            continue
        out.append({
            "id": cid,
            "label": node["label"],
            "kind": node["kind"],
            "aliases": node["aliases"],
            "sections": len(sections[cid]),
            "defines": c["DEFINES"],
            "requires": c["REQUIRES"],
        })
    out.sort(key=lambda c: (-c["sections"], c["label"].lower()))
    return out


def load_questions() -> list[dict]:
    questions = []
    for path in sorted((ROOT / "data" / "raw" / "discussions").glob("*.json"),
                       key=lambda p: -int(p.stem)):
        rec = json.loads(path.read_text(encoding="utf-8"))
        questions.append({
            "id": f"q:{rec['number']}",
            "number": rec["number"],
            "title": rec["title"],
            "body": rec["body_text"],
            "url": rec["url"],
            "raw_path": f"data/raw/discussions/{rec['number']}.md",
        })
    return questions


def render_question_prompt(batch: dict, labels: list[dict]) -> str:
    lines = [QUESTION_PROMPT, "", QUESTION_PROMPT_NOTES, "", "KNOWN CONCEPTS:"]
    for c in labels:
        alias = f"  (also: {', '.join(c['aliases'][:3])})" if c["aliases"] else ""
        lines.append(f"  {c['label']}  [{c['kind']}, in {c['sections']} sections]{alias}")
    lines.append("")
    for q in batch["questions"]:
        lines.append("=" * 70)
        lines.append(f"QUESTION ID: {q['id']}")
        lines.append(f"QUESTION TITLE: {q['title']}")
        lines.append("QUESTION BODY:")
        lines.append(q["body"][:6000])
        lines.append("")
    return "\n".join(lines)


def cmd_prep_questions() -> int:
    labels = candidate_concepts()
    questions = load_questions()
    QWORK_IN.mkdir(parents=True, exist_ok=True)
    QWORK_OUT.mkdir(parents=True, exist_ok=True)

    batches = []
    for i in range(0, len(questions), QUESTIONS_PER_BATCH):
        chunk = questions[i: i + QUESTIONS_PER_BATCH]
        batch = {"batch_id": f"qbatch-{i // QUESTIONS_PER_BATCH:03d}", "questions": chunk}
        batches.append(batch)
        (QWORK_IN / f"{batch['batch_id']}.json").write_text(
            json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        (QWORK_IN / f"{batch['batch_id']}.prompt.txt").write_text(
            render_question_prompt(batch, labels), encoding="utf-8")

    (QWORK / "batches.json").write_text(json.dumps(
        {"concept_labels": len(labels),
         "batches": [{"batch_id": b["batch_id"], "questions": len(b["questions"])}
                     for b in batches]}, indent=2), encoding="utf-8")
    print(f"{len(batches)} question batches over {len(questions)} threads, "
          f"{len(labels)} concept labels offered")
    print(f"Write each answer to {QWORK_OUT.relative_to(ROOT)}/<batch_id>.json")
    return 0


def cmd_status_questions() -> int:
    index = json.loads((QWORK / "batches.json").read_text(encoding="utf-8"))["batches"]
    done = {p.stem for p in QWORK_OUT.glob("qbatch-*.json")}
    missing = [b["batch_id"] for b in index if b["batch_id"] not in done]
    print(f"{len(done)}/{len(index)} question batches complete")
    if missing:
        print("missing: " + " ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    return 0


def cmd_assemble_questions() -> int:
    """Turn the mapped concepts into ASKS_ABOUT edges, validating every span.

    Spans are checked against the thread's markdown copy, not its JSON, because
    a JSON string puts the whole body on one line and a line range into it would
    tell a reader nothing.
    """
    nodes = json.loads((GRAPH / "nodes.json").read_text(encoding="utf-8"))["nodes"]
    edges = json.loads((GRAPH / "edges.json").read_text(encoding="utf-8"))["edges"]
    by_label = {}
    for node in nodes:
        if node["type"] != "Concept":
            continue
        by_label[normalise(node["label"])] = node["id"]
        for surface in node.get("surface_forms", []):
            by_label.setdefault(normalise(surface), node["id"])
        for alias in node.get("aliases", []):
            by_label.setdefault(normalise(alias), node["id"])

    questions = {q["id"]: q for q in load_questions()}
    outputs = sorted(QWORK_OUT.glob("qbatch-*.json"))
    if not outputs:
        print("No question batch outputs under data/work/questions/out/.")
        return 1

    new_edges: list[dict] = []
    stats = {"raw": 0, "dropped_unknown_concept": 0, "dropped_no_span": 0,
             "dropped_short_span": 0, "dropped_unknown_question": 0,
             "dropped_duplicate": 0, "recovered_after_markdown_strip": 0}
    seen: set[tuple] = set()
    unknown_labels: dict[str, int] = {}

    for path in outputs:
        batch_id = path.stem
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  {batch_id}: unreadable JSON ({exc}); skipped")
            continue
        for qid, result in payload.items():
            question = questions.get(qid)
            if question is None:
                stats["dropped_unknown_question"] += len(result.get("asks_about", []))
                continue
            thread_text = (ROOT / question["raw_path"]).read_text(encoding="utf-8")
            for item in result.get("asks_about", []) or []:
                stats["raw"] += 1
                label = clean_label(str(item.get("concept", "")))
                cid = by_label.get(normalise(label))
                if cid is None:
                    stats["dropped_unknown_concept"] += 1
                    unknown_labels[label] = unknown_labels.get(label, 0) + 1
                    continue
                span = str(item.get("span", "") or "")
                if len(normalise(span)) < MIN_SPAN_CHARS:
                    stats["dropped_short_span"] += 1
                    continue
                found = locate_span(span, thread_text, 1)
                recovered = False
                if found is None:
                    # The mapper read GitHub's plain-text rendering of the body
                    # while this file holds the markdown source. Try again with
                    # the markup characters removed from both sides.
                    found = locate_span(
                        span, thread_text, 1,
                        transform=lambda t: strip_markdown_markers(normalise(t)))
                    recovered = found is not None
                if found is None:
                    stats["dropped_no_span"] += 1
                    continue
                if recovered:
                    stats["recovered_after_markdown_strip"] += 1
                key = (qid, cid)
                if key in seen:
                    stats["dropped_duplicate"] += 1
                    continue
                seen.add(key)
                confidence = str(item.get("confidence", "medium")).lower()
                if confidence not in CONFIDENCES:
                    confidence = "medium"
                new_edges.append({
                    "id": "",
                    "type": "ASKS_ABOUT",
                    "from": qid,
                    "to": cid,
                    "evidence": {
                        "span": span.strip(),
                        "path": question["raw_path"],
                        "lines": [found[0], found[1]],
                        "kind": "question",
                    },
                    "extractor": batch_id,
                    "confidence": confidence,
                })

    edges = [e for e in edges if e["type"] != "ASKS_ABOUT"] + new_edges
    edges.sort(key=lambda e: (e["type"], e["from"], e["to"]))
    for i, edge in enumerate(edges):
        edge["id"] = f"e:{i:05d}"
    (GRAPH / "edges.json").write_text(
        json.dumps({"schema_version": 1, "edges": edges}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    mapped = len({e["from"] for e in new_edges})
    report = {"batches_processed": len(outputs), "asks_about_edges": len(new_edges),
              "questions_mapped": mapped, "questions_total": len(questions),
              **stats,
              "unknown_label_examples": sorted(unknown_labels.items(),
                                               key=lambda kv: -kv[1])[:25]}
    (GRAPH / "question_report.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")
    print(f"{len(outputs)} batches -> {len(new_edges)} ASKS_ABOUT edges "
          f"over {mapped}/{len(questions)} threads, from {stats['raw']} proposed")
    for key in ("dropped_unknown_concept", "dropped_no_span", "dropped_short_span",
                "dropped_duplicate", "dropped_unknown_question",
                "recovered_after_markdown_strip"):
        if stats[key]:
            print(f"  {key}: {stats[key]}")
    assert all(e["evidence"].get("span") for e in edges), "edge with no span"
    print("  invariant holds: zero edges with a null or empty evidence span")
    return 0



def _llm():
    """Imported lazily so the deterministic steps never need the module."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import llm
    return llm

def cmd_run() -> int:
    """API path for the section batches. Falls back with a clear message."""
    llm = _llm()
    if not llm.available():
        print("No ANTHROPIC_API_KEY or OPENAI_API_KEY. Have a Claude Code session fill")
        print(f"  {WORK_IN.relative_to(ROOT)}/*.prompt.txt -> {WORK_OUT.relative_to(ROOT)}/<batch_id>.json")
        return 1
    print(f"Extracting with {llm.model_id()}")
    return llm.run_batches(WORK_IN, WORK_OUT, "batch-*", ".prompt.txt")


def cmd_run_questions() -> int:
    llm = _llm()
    if not llm.available():
        print("No ANTHROPIC_API_KEY or OPENAI_API_KEY. Have a Claude Code session fill")
        print(f"  {QWORK_IN.relative_to(ROOT)}/*.prompt.txt -> {QWORK_OUT.relative_to(ROOT)}/<batch_id>.json")
        return 1
    print(f"Mapping questions with {llm.model_id()}")
    return llm.run_batches(QWORK_IN, QWORK_OUT, "qbatch-*", ".prompt.txt")



# --------------------------------------------------------------------------
# Definition sweep: the completeness pass over concepts nothing defines
# --------------------------------------------------------------------------

DWORK = ROOT / "data" / "work" / "definitions"
DWORK_IN = DWORK / "in"
DWORK_OUT = DWORK / "out"
CONCEPTS_PER_SWEEP_BATCH = 8

DEFINITION_PROMPT_HEADER = """\
You are looking for definitions the first pass missed.

Every concept below currently has NO section in the corpus that DEFINES it,
while several sections depend on it or refer to it. That is what this project
reports as a gap. Before it is reported, someone has to go and check, because
the likeliest way this is wrong is simple: the section that does explain the
thing is sitting in the corpus and the first pass did not emit the edge.

There is a known reason for that. The first pass was told not to emit a concept
for the name of the page it was reading. So the page called "Middleware", which
is exactly the page that explains what middleware is, was the one page that
never said so. Concepts whose name matches a page title or a section heading are
therefore the most likely to be wrong here, not the least.

THE CORPUS IS EXACTLY THESE FILES AND NOTHING ELSE:
  data/raw/docs/          the 60 markdown pages
  data/raw/docs_src/      the code samples those pages pull in

A definition on fastapi.tiangolo.com in a page that is not under data/raw/docs/
does not count. Neither do the Starlette, Pydantic or SQLModel docs.

FOR EACH CONCEPT, in this order. An earlier version of this pass missed real
definitions and an independent check caught them, so these steps are the exact
places it was looking in the wrong order.

  1. READ THE SECTIONS LISTED UNDER "REFERENCED BY", STARTING WITH THE FIRST.
     The commonest miss by a distance: the sentence that looks like a dependency
     is ALSO the defining sentence. A section headed "What is Form Data" that
     says HTML forms send data in a special encoding both depends on the term
     and teaches it. Open those files and read them before you search anything.

  2. READ THE WHOLE PAGE around each hit, not the matching line. A term is often
     taught by its parts in neighbouring subsections: "path operation" is
     defined by a subsection called "Path" followed by one called "Operation".
     A term is also often introduced a few sections ABOVE the one that leans on
     it, on the same page.

  3. NOW grep data/raw/docs/ for the label, for every surface form listed, and
     for the individual words. Check page titles and headings first: the first
     pass was told to skip the concept named after the page it was reading, so
     the page that explains a thing is the page most likely to be missing its
     edge.

  4. Also check data/raw/docs_src/ prose comments, and check whether an install
     or setup instruction teaches the thing in passing. "Install HTTPX, the
     Requests-style client TestClient is built on" is a definition.

  5. Decide. A definition TEACHES: a reader who knew nothing could learn what the
     thing is or how to use it. A name in a code sample with no surrounding
     explanation is not one. A sentence that merely uses the term correctly is
     not one. Be strict about that, and equally strict about not missing a real
     one.

  6. If you find one, return the file path, the line number, and the VERBATIM
     span that does the teaching, copied character for character out of the file
     with a tool. A validator searches for your span in that file and drops it if
     it is not there, so copy and paste; do not retype.

  7. If you genuinely do not find one, say so. That is the right answer for many
     of these. Do not manufacture a definition out of incidental usage.

Return JSON only, keyed by concept id:

{
  "concept:middleware": {
    "defined": true,
    "path": "data/raw/docs/tutorial/middleware.md",
    "line": 5,
    "span": "A \\"middleware\\" is a function that works with every **request** before it is processed by any specific *path operation*",
    "confidence": "high"
  },
  "concept:some_other_thing": {
    "defined": false,
    "reason": "only ever appears inside code samples, never explained"
  }
}
"""


def cmd_prep_definitions() -> int:
    """Write one payload per batch of orphan concepts worth re-checking."""
    nodes = json.loads((GRAPH / "nodes.json").read_text(encoding="utf-8"))["nodes"]
    edges = json.loads((GRAPH / "edges.json").read_text(encoding="utf-8"))["edges"]
    node = {n["id"]: n for n in nodes}
    concepts = {n["id"]: n for n in nodes if n["type"] == "Concept"}

    defines: dict[str, int] = {cid: 0 for cid in concepts}
    refs: dict[str, list] = {cid: [] for cid in concepts}
    asked: dict[str, set] = {cid: set() for cid in concepts}
    for edge in edges:
        if edge["to"] not in concepts:
            continue
        if edge["type"] == "DEFINES":
            defines[edge["to"]] += 1
        elif edge["type"] in ("REQUIRES", "MENTIONS"):
            refs[edge["to"]].append(edge)
        elif edge["type"] == "ASKS_ABOUT":
            asked[edge["to"]].add(edge["from"])

    # Worth re-checking: nothing defines it, and either two sections lean on it
    # or somebody asked about it. Those are exactly the ones that become
    # findings, so they are the ones a false positive would cost most.
    targets = [
        cid for cid, c in concepts.items()
        if defines[cid] == 0
        and ({e["from"] for e in refs[cid]}.__len__() >= 2 or asked[cid])
    ]
    targets.sort(key=lambda cid: (-len({e["from"] for e in refs[cid]}), cid))

    DWORK_IN.mkdir(parents=True, exist_ok=True)
    DWORK_OUT.mkdir(parents=True, exist_ok=True)
    batches = []
    for i in range(0, len(targets), CONCEPTS_PER_SWEEP_BATCH):
        chunk = targets[i: i + CONCEPTS_PER_SWEEP_BATCH]
        batch_id = f"def-{i // CONCEPTS_PER_SWEEP_BATCH:03d}"
        batches.append(batch_id)
        blocks = [DEFINITION_PROMPT_HEADER, ""]
        for cid in chunk:
            concept = concepts[cid]
            sections = sorted({e["from"] for e in refs[cid]})
            blocks += [
                "=" * 70,
                f"CONCEPT ID: {cid}",
                f"LABEL: {concept['label']}   kind: {concept['kind']}",
                f"SURFACE FORMS: {', '.join(concept.get('surface_forms', []))}",
                f"ALIASES: {', '.join(concept.get('aliases', [])) or 'none'}",
                f"ASKED ABOUT BY: {len(asked[cid])} threads",
                f"REFERENCED BY {len(sections)} sections:",
            ]
            for sid in sections[:10]:
                sec = node[sid]
                blocks.append(f"  {sec['path']} lines {sec['lines'][0]}-{sec['lines'][1]}"
                              f"  ({sec['heading']})")
            blocks += ["  how they refer to it:"]
            for edge in refs[cid][:4]:
                blocks.append(f"    {edge['type']}: {edge['evidence']['span'][:140]!r}")
            blocks.append("")
        (DWORK_IN / f"{batch_id}.txt").write_text("\n".join(blocks), encoding="utf-8")

    (DWORK / "batches.json").write_text(
        json.dumps({"targets": len(targets), "batches": batches}, indent=2),
        encoding="utf-8")
    print(f"{len(batches)} definition-sweep batches over {len(targets)} "
          f"concepts nothing currently defines")
    return 0


def cmd_run_definitions() -> int:
    llm = _llm()
    if not llm.available():
        print("No API key. Have a Claude Code session fill "
              f"{DWORK_IN.relative_to(ROOT)}/*.txt -> {DWORK_OUT.relative_to(ROOT)}/*.json")
        return 1
    return llm.run_batches(DWORK_IN, DWORK_OUT, "def-*")


def cmd_assemble_definitions() -> int:
    """Fold any definitions the sweep found back in as real DEFINES edges.

    They go through the same validator as everything else. A span the sweep
    reports that is not in the file it names is dropped, exactly as it would be
    from the first pass.
    """
    nodes = json.loads((GRAPH / "nodes.json").read_text(encoding="utf-8"))["nodes"]
    edges = json.loads((GRAPH / "edges.json").read_text(encoding="utf-8"))["edges"]
    sections = [n for n in nodes if n["type"] == "Section"]
    concepts = {n["id"] for n in nodes if n["type"] == "Concept"}

    # A reported file and line maps to whichever section owns that line.
    by_raw: dict[str, list] = {}
    for sec in sections:
        by_raw.setdefault(sec["raw_path"], []).append(sec)

    found: dict[str, dict] = {}
    for path in sorted(DWORK_OUT.glob("def-*.json")):
        try:
            found.update(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"  unreadable: {path.name}")

    stats = {"reported": 0, "not_defined": 0, "added": 0, "dropped_no_span": 0,
             "dropped_unknown_section": 0, "dropped_unknown_concept": 0,
             "already_present": 0}
    new_edges = []
    existing = {(e["from"], e["to"], e["type"]) for e in edges}

    for cid, record in found.items():
        if not record.get("defined"):
            stats["not_defined"] += 1
            continue
        stats["reported"] += 1
        if cid not in concepts:
            stats["dropped_unknown_concept"] += 1
            continue
        raw_path = str(record.get("path", "")).strip()
        span = str(record.get("span", "") or "")
        candidates = by_raw.get(raw_path, [])
        if not candidates:
            stats["dropped_unknown_section"] += 1
            continue

        hit = None
        for sec in candidates:
            located = locate_span(span, sec["text"], sec["lines"][0])
            if located:
                hit = (sec, located, sec["path"])
                break
            for code in sec["code"]:
                located = locate_span(span, code["text"], code["lines"][0])
                if located:
                    hit = (sec, located, code["path"])
                    break
            if hit:
                break
        if hit is None:
            stats["dropped_no_span"] += 1
            continue

        sec, located, span_path = hit
        key = (sec["id"], cid, "DEFINES")
        if key in existing:
            stats["already_present"] += 1
            continue
        existing.add(key)
        new_edges.append({
            "id": "",
            "type": "DEFINES",
            "from": sec["id"],
            "to": cid,
            "evidence": {
                "span": span.strip(),
                "path": span_path,
                "lines": [located[0], located[1]],
                "kind": "prose" if span_path == sec["path"] else "code",
            },
            "extractor": "definition-sweep",
            "confidence": str(record.get("confidence", "medium")).lower()
            if str(record.get("confidence", "")).lower() in CONFIDENCES else "medium",
        })
        stats["added"] += 1

    edges = edges + new_edges
    edges.sort(key=lambda e: (e["type"], e["from"], e["to"]))
    for i, edge in enumerate(edges):
        edge["id"] = f"e:{i:05d}"
    (GRAPH / "edges.json").write_text(
        json.dumps({"schema_version": 1, "edges": edges}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (GRAPH / "definition_sweep_report.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print(f"definition sweep: {stats['added']} DEFINES edges added, "
          f"{stats['not_defined']} concepts confirmed as defined nowhere")
    for key in ("dropped_no_span", "dropped_unknown_section", "dropped_unknown_concept"):
        if stats[key]:
            print(f"  {key}: {stats[key]}")
    return 0


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "assemble"
    if command == "prep":
        return cmd_prep()
    if command == "status":
        return cmd_status()
    if command == "assemble":
        return cmd_assemble()
    if command == "prep-questions":
        return cmd_prep_questions()
    if command == "status-questions":
        return cmd_status_questions()
    if command == "assemble-questions":
        return cmd_assemble_questions()
    if command == "run":
        return cmd_run()
    if command == "run-questions":
        return cmd_run_questions()
    if command == "prep-definitions":
        return cmd_prep_definitions()
    if command == "run-definitions":
        return cmd_run_definitions()
    if command == "assemble-definitions":
        return cmd_assemble_definitions()
    print(f"unknown subcommand: {command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
