"""Sample findings, check them independently, and report intervals rather than a gate.

Fifty near-miss findings, drawn uniformly at random with a fixed seed. For each,
an independent check: does the discussion answer supply a fact that is absent
from all 60 pages? The checker is given the finding and told to go looking for a
definition in the corpus itself, because the most likely way a near miss is
wrong is that the extractor missed a DEFINES edge that is sitting in the docs.

There is no pass or fail gate, on purpose. At n=50 the Wilson 95 percent
interval around 43/50 runs from about 0.74 to 0.93, so a binary threshold at 90
percent would be decided by two or three items. RefusalBench human validated 180
items and scored 93.1 percent, a rate whose Wilson 95 percent interval runs from
0.88 to 0.96, so a 90 percent gate could not separate work done by a funded team
with expert annotators from failure. Report the observed rate and the interval,
and let the reader judge.

If the floor fires, the run is not a result. Fix the extraction, rerun, and
report the rerun. A rate produced by a build with a known defect describes the
defect, not the corpus, and publishing it alongside the corrected number invites
a reader to average two things only one of which measures anything.

The one number that does trigger action is the floor in the build plan: under
roughly 70 percent, stop and fix extraction rather than build anything on top.

Subcommands:
  prep      draw the sample and write payloads to data/work/validate/in/
  assemble  read verdicts, compute Wilson intervals, write data/validation.json
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GRAPH = DATA / "graph"
WORK = DATA / "work" / "validate"

SAMPLE_SIZE = 50
SEED = 20260906
FINDINGS_PER_BATCH = 5
STOP_BELOW = 0.70


def log(msg: str) -> None:
    print(msg, flush=True)


def wilson(successes: int, total: int, z: float = 1.959963985) -> list[float]:
    """Wilson score interval. Behaves at the extremes where the normal one does not."""
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (z / denominator) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def load():
    payload = json.loads((DATA / "findings.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in
             json.loads((GRAPH / "nodes.json").read_text(encoding="utf-8"))["nodes"]}
    sections = {s["id"]: s for s in
                json.loads((GRAPH / "sections.json").read_text(encoding="utf-8"))["sections"]}
    return payload, nodes, sections


def draw_sample(findings: list[dict]) -> tuple[list[dict], bool]:
    """Returns the drawn findings and whether it turned out to be a census.

    The design asked for 50 drawn uniformly at random. When the population is
    smaller than that, drawing 50 means checking all of them, and saying
    "sampled 50" would be wrong in a way that matters: a census has no sampling
    error at all, and the interval then says something different.
    """
    pool = [f for f in findings if f["type"] == "near_miss"]
    if len(pool) <= SAMPLE_SIZE:
        return sorted(pool, key=lambda f: f["id"]), True
    rng = random.Random(SEED)
    return sorted(rng.sample(pool, SAMPLE_SIZE), key=lambda f: f["id"]), False


HEADER = """\
You are checking whether a reported documentation gap is real. Be sceptical. The
job is to catch false positives, not to confirm the finding.

Each item claims: a concept that users ask about, whose meaning or usage NO
section of the 60-page FastAPI corpus explains, and which one section's
instructions nevertheless depend on. It also shows the answer someone gave on
the discussion thread.

The corpus you are judging against is exactly the 60 markdown files under
  {docs_root}
and the code samples under
  {src_root}
Nothing else counts. A definition on fastapi.tiangolo.com in a page that is not
in that directory, or in the SQLModel or Starlette or Pydantic docs, does NOT
make the finding invalid. It has to be in these files.

DO THIS FOR EACH ITEM, and actually run the searches, do not judge from memory:
  1. Search the corpus for the concept and for its aliases. grep is the tool.
     Look for a section that explains what it is or how to use it, not one that
     merely names it.
  2. Read the surrounding text of the strongest hits. A name inside a code
     sample with no explanation is NOT a definition. A sentence that teaches the
     reader what the thing is or how to set it IS one.
  3. Decide whether the thread answer supplies a fact that is genuinely absent
     from those files.

Verdicts:
  valid    No section of the corpus explains the concept, AND the thread answer
           supplies a fact the corpus does not contain.
  invalid  A section of the corpus does explain it and the extractor missed the
           DEFINES edge, OR the corpus already contains the fact the answer
           gives, OR the concept is too generic to be a real gap (a bare Python
           keyword, a name local to one code sample, the page's own subject).

Return JSON only, keyed by finding id. "reason" is one sentence. When you mark
something invalid, "evidence" must name the file and line where the definition
actually sits.

{{
  "F1-0087": {{
    "verdict": "invalid",
    "reason": "the concept is defined in advanced/settings.md, the extractor missed the DEFINES edge",
    "evidence": "data/raw/docs/advanced/settings.md:120"
  }}
}}
"""


def cmd_prep() -> int:
    payload, nodes, sections = load()
    sample, census = draw_sample(payload["findings"])
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "in").mkdir(exist_ok=True)
    (WORK / "out").mkdir(exist_ok=True)

    header = HEADER.format(docs_root="data/raw/docs/", src_root="data/raw/docs_src/")
    batches = []
    for i in range(0, len(sample), FINDINGS_PER_BATCH):
        chunk = sample[i: i + FINDINGS_PER_BATCH]
        batch_id = f"val-{i // FINDINGS_PER_BATCH:03d}"
        batches.append(batch_id)
        blocks = [header, ""]
        for finding in chunk:
            concept = nodes[finding["concept"]]
            section = sections[finding["section"]]
            lead_q = finding["questions"][0]
            question = nodes[lead_q]
            answer = nodes.get(f"ans:{question['number']}", {})
            blocks += [
                "=" * 70,
                f"FINDING ID: {finding['id']}",
                f"CONCEPT: {concept['label']}   kind: {concept['kind']}",
                f"ALIASES: {', '.join(concept.get('aliases', [])) or 'none'}",
                f"SURFACE FORMS SEEN: {', '.join(concept.get('surface_forms', []))}",
                f"ASKED BY: {finding['demand']} distinct threads",
                f"REFERENCED BY: {finding['requires_count']} REQUIRES edges, "
                f"{finding['mentions_count']} MENTIONS edges",
                "",
                f"SECTION SAID TO DEPEND ON IT: {section['doc_title']} / {section['heading']}",
                f"  {section['path']} lines {section['lines'][0]}-{section['lines'][1]}",
                f"  dependency span: {finding['proof_path'][1]['evidence']['span']!r}",
                "",
                f"LEAD QUESTION: {question['title']}",
                f"  {question['url']}",
                question["body"][:1500],
                "",
                f"THREAD ANSWER (by {answer.get('author')}, "
                f"maintainer: {answer.get('is_maintainer')}):",
                (answer.get("body") or "")[:3000],
                "",
            ]
        (WORK / "in" / f"{batch_id}.txt").write_text("\n".join(blocks), encoding="utf-8")

    (WORK / "sample.json").write_text(json.dumps({
        "sample_size": len(sample),
        "seed": SEED,
        "census": census,
        "sampling": (
            "every finding of type near_miss; the population is smaller than the "
            f"target of {SAMPLE_SIZE}, so this is a census rather than a sample"
            if census else
            f"uniform random over findings of type near_miss, seed {SEED}"),
        "finding_ids": [f["id"] for f in sample],
        "batches": batches,
    }, indent=2), encoding="utf-8")
    kind = "every" if census else "a random sample of"
    log(f"{len(batches)} validation batches over {kind} {len(sample)} near-miss findings")
    log(f"  population: {sum(1 for f in payload['findings'] if f['type'] == 'near_miss')}")
    return 0


def cmd_assemble() -> int:
    payload, _nodes, _sections = load()
    meta = json.loads((WORK / "sample.json").read_text(encoding="utf-8"))
    by_id = {f["id"]: f for f in payload["findings"]}

    # Each finding may be read by several independent checkers. The verdict is
    # the majority, and the split is recorded, because a 2-1 verdict and a 3-0
    # verdict are not the same evidence and reporting them as one number would
    # hide that. A single reader per finding still works: the majority of one
    # is itself.
    ballots: dict[str, list[dict]] = {}
    for path in sorted((WORK / "out").glob("val-*.json")):
        try:
            ballot_file = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log(f"  unreadable: {path.name}")
            continue
        for fid, record in ballot_file.items():
            ballots.setdefault(fid, []).append(record)

    results = []
    counts: Counter = Counter()
    unanimous = 0
    for fid in meta["finding_ids"]:
        cast = ballots.get(fid, [])
        votes = [str(r.get("verdict", "")).lower() for r in cast]
        votes = [v for v in votes if v in ("valid", "invalid")]
        if not votes:
            verdict = "unchecked"
        else:
            verdict = "valid" if votes.count("valid") * 2 > len(votes) else "invalid"
            if votes.count(verdict) == len(votes):
                unanimous += 1
        counts[verdict] += 1
        agreeing = [r for r, v in zip(cast, [str(x.get("verdict", "")).lower() for x in cast])
                    if v == verdict]
        lead = agreeing[0] if agreeing else (cast[0] if cast else {})
        results.append({
            "finding_id": fid,
            "concept_label": by_id[fid]["concept_label"] if fid in by_id else "",
            "verdict": verdict,
            "readers": len(votes),
            "votes_valid": votes.count("valid"),
            "reason": lead.get("reason", ""),
            "evidence": lead.get("evidence", ""),
        })
        if fid in by_id and verdict != "unchecked":
            by_id[fid]["validated"] = verdict == "valid"
            if lead.get("reason"):
                by_id[fid]["validation_note"] = lead["reason"]

    checked = counts["valid"] + counts["invalid"]
    observed = round(counts["valid"] / checked, 4) if checked else 0.0
    interval = wilson(counts["valid"], checked)

    out = {
        "schema_version": 1,
        "sampled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample_size": meta["sample_size"],
        "census": meta.get("census", False),
        "sampling": meta["sampling"],
        "population": {
            "near_miss": sum(1 for f in payload["findings"] if f["type"] == "near_miss"),
        },
        "readers_per_finding": max((r["readers"] for r in results), default=0),
        "unanimous": unanimous,
        "results": results,
        "by_type": {
            "near_miss": {
                "n": checked,
                "valid": counts["valid"],
                "invalid": counts["invalid"],
                "unchecked": counts["unchecked"],
                "observed_rate": observed,
                "wilson_95": interval,
            }
        },
        "note": (
            ("Every near-miss finding was checked, so the observed rate has no "
             "sampling error: it is exactly the rate for this corpus and this run. "
             "The interval is reported anyway, as the uncertainty about the method "
             "rather than about the sample, and it is wide because the population "
             "is small.")
            if meta.get("census") else
            "Interval is wide at n=50 by design. Reported, not gated. The observed "
            "rate is the point estimate and the interval is what the sample supports."),
    }
    (DATA / "validation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    (DATA / "findings.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                        encoding="utf-8")

    log("")
    log(f"near_miss: {counts['valid']}/{checked} valid, observed {observed:.3f}, "
        f"Wilson 95 percent [{interval[0]:.3f}, {interval[1]:.3f}]")
    if results and results[0]["readers"] > 1:
        log(f"  {results[0]['readers']} readers per finding, "
            f"{unanimous}/{checked} unanimous")
    if counts["unchecked"]:
        log(f"  {counts['unchecked']} sampled findings came back without a verdict")
    if checked and observed < STOP_BELOW:
        log("")
        log(f"OBSERVED VALIDITY IS BELOW {STOP_BELOW:.0%}. Per the build plan, stop and fix "
            "extraction before building anything on top of these findings.")
        return 3
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
    print(f"Checking the sample with {llm.model_id()}")
    return llm.run_batches(WORK / "in", WORK / "out", "val-*")


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
