"""Split the fetched markdown into Doc and Section nodes.

This is the deterministic half of extraction. No model is involved, and it must
stay that way: every downstream evidence span is checked against the line range
a section claims, so the line arithmetic here is what makes the whole "no edge
without evidence" invariant auditable.

Sections are LEAF units: a heading owns the text up to the next heading of ANY
level, not just the next heading of the same or higher level. Nested headings
therefore do not duplicate their children's text. This matches how retrieval
software actually chunks a page - non-overlapping pieces - and it is the same
choice made in the sibling repo, Will My Agent Answer This.

FastAPI's pages pull their code samples in with a macro rather than writing them
inline:

    {* ../../docs_src/sql_databases/tutorial001_an_py310.py ln[14:18] hl[14:15] *}

A reader of the published page sees that code. A reader of the raw markdown does
not. We resolve the macro and attach the referenced lines to the section as a
separate `code` entry that carries its OWN path and line range, so a span quoted
from a code sample still points at a real file and real lines.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW_DOCS = RAW / "docs"
RAW_SRC = RAW
DOCS_ROOT = "docs/en/docs"
SITE_BASE = "https://fastapi.tiangolo.com/"
USER_AGENT = "should-my-agent-answer/1.0"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
ANCHOR_RE = re.compile(r"\s*\{\s*#([A-Za-z0-9_\-]+)\s*\}\s*$")
INCLUDE_RE = re.compile(r"\{\*\s*(\S+?)(?:\s+([^*]*?))?\s*\*\}")
LN_RE = re.compile(r"ln\[(\d+):(\d+)\]")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def log(msg: str) -> None:
    print(msg, flush=True)


def slugify(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*?([^*]*)\*\*?", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "section"


def clean_heading(text: str) -> tuple[str, str | None]:
    """Return the display heading and the explicit anchor, if the page set one."""
    anchor = None
    match = ANCHOR_RE.search(text)
    if match:
        anchor = match.group(1)
        text = text[: match.start()]
    display = text.strip()
    display = re.sub(r"`([^`]*)`", r"\1", display)
    display = re.sub(r"\*\*([^*]*)\*\*", r"\1", display)
    display = re.sub(r"<abbr[^>]*>(.*?)</abbr>", r"\1", display)
    return display.strip(), anchor


def doc_url(rel_path: str) -> str:
    slug = rel_path[:-3]
    if slug.endswith("/index"):
        slug = slug[: -len("/index")]
    elif slug == "index":
        return SITE_BASE
    return f"{SITE_BASE}{slug}/"


# --------------------------------------------------------------------------
# Code includes
# --------------------------------------------------------------------------

def collect_includes(commit: str) -> None:
    """Download every docs_src file the 60 pages point at, once."""
    wanted: set[str] = set()
    for md in sorted(RAW_DOCS.rglob("*.md")):
        for match in INCLUDE_RE.finditer(md.read_text(encoding="utf-8")):
            wanted.add(resolve_src_path(match.group(1)))
    log(f"  {len(wanted)} distinct code files referenced by the 60 pages")
    missing = [p for p in sorted(wanted) if not (RAW_SRC / p).exists()]
    if not missing:
        log("  all already present under data/raw/docs_src/")
        return
    for i, rel in enumerate(missing, 1):
        url = f"https://raw.githubusercontent.com/fastapi/fastapi/{commit}/{rel}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
        out = RAW_SRC / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        if i % 25 == 0 or i == len(missing):
            log(f"  fetched {i}/{len(missing)} code files")


def resolve_src_path(raw: str) -> str:
    """`../../docs_src/x.py` resolves against docs/en/, giving `docs_src/x.py`.

    The macro is configured relative to the mkdocs config directory, not the
    docs directory. Verified: `docs/docs_src/` does not exist in the repo and
    `docs_src/` does.
    """
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    while parts and parts[0] == "..":
        parts.pop(0)
    return "/".join(parts)


def expand_includes(body: str) -> list[dict]:
    """Return one entry per code include found in this section's markdown."""
    out: list[dict] = []
    for match in INCLUDE_RE.finditer(body):
        rel = resolve_src_path(match.group(1))
        options = match.group(2) or ""
        path = RAW_SRC / rel
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        ln = LN_RE.search(options)
        if ln:
            start, end = int(ln.group(1)), int(ln.group(2))
        else:
            start, end = 1, len(lines)
        start = max(1, start)
        end = min(len(lines), end)
        if end < start:
            continue
        out.append({
            "path": rel,
            "raw_path": str((RAW_SRC / rel).relative_to(ROOT)),
            "lines": [start, end],
            "text": "\n".join(lines[start - 1: end]),
        })
    return out


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def find_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line_index, level, raw_heading_text) for every heading outside a fence."""
    found: list[tuple[int, int, str]] = []
    fence: str | None = None
    for i, line in enumerate(lines):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0] * 3
            elif line.strip().startswith(fence):
                fence = None
            continue
        if fence is not None:
            continue
        heading = HEADING_RE.match(line)
        if heading:
            found.append((i, len(heading.group(1)), heading.group(2)))
    return found


def split_page(rel_path: str, text: str) -> tuple[dict, list[dict]]:
    lines = text.splitlines()
    headings = find_headings(lines)

    doc_id = f"doc:{rel_path[:-3]}"
    title = None
    for _, level, raw in headings:
        if level == 1:
            title, _anchor = clean_heading(raw)
            break
    if title is None:
        title = rel_path[:-3].split("/")[-1].replace("-", " ").title()

    doc = {
        "id": doc_id,
        "type": "Doc",
        "path": f"{DOCS_ROOT}/{rel_path}",
        "raw_path": str((RAW_DOCS / rel_path).relative_to(ROOT)),
        "title": title,
        "url": doc_url(rel_path),
        "lines": len(lines),
    }

    sections: list[dict] = []
    seen_anchors: dict[str, int] = {}

    def emit(heading_text: str, level: int, anchor: str | None, start: int, end: int) -> None:
        """start/end are 1-based inclusive line numbers into the source file."""
        body = "\n".join(lines[start - 1: end])
        if not body.strip():
            return
        slug = anchor or slugify(heading_text)
        count = seen_anchors.get(slug, 0)
        seen_anchors[slug] = count + 1
        if count:
            slug = f"{slug}-{count + 1}"
        code = expand_includes(body)
        embed_parts = [heading_text, body]
        embed_parts += [c["text"] for c in code]
        sections.append({
            "id": f"sec:{rel_path[:-3]}#{slug}",
            "type": "Section",
            "doc_id": doc_id,
            "doc_title": title,
            "heading": heading_text,
            "level": level,
            "path": f"{DOCS_ROOT}/{rel_path}",
            "raw_path": str((RAW_DOCS / rel_path).relative_to(ROOT)),
            "lines": [start, end],
            "anchor": slug,
            "url": doc["url"] + (f"#{slug}" if level > 1 else ""),
            "text": body,
            "code": code,
            "embed_text": "\n\n".join(p for p in embed_parts if p.strip()),
            "chars": len(body),
        })

    if not headings:
        emit(title, 1, None, 1, len(lines))
        return doc, sections

    # Anything before the first heading is front matter, not a section.
    for idx, (line_index, level, raw) in enumerate(headings):
        heading_text, anchor = clean_heading(raw)
        start = line_index + 1               # the heading line itself, 1-based
        if idx + 1 < len(headings):
            end = headings[idx + 1][0]       # line before the next heading, 1-based
        else:
            end = len(lines)
        emit(heading_text, level, anchor, start, end)

    return doc, sections


def main() -> int:
    meta = json.loads((RAW / "fetch_meta.json").read_text(encoding="utf-8"))
    log("Resolving code includes")
    collect_includes(meta["corpus_commit"])

    log("Splitting pages into sections")
    docs: list[dict] = []
    sections: list[dict] = []
    for record in meta["docs"]:
        rel = record["rel_path"]
        text = (RAW_DOCS / rel).read_text(encoding="utf-8")
        doc, secs = split_page(rel, text)
        docs.append(doc)
        sections.extend(secs)

    # A section that claims lines [a, b] must contain exactly those lines.
    for sec in sections:
        source = (ROOT / sec["raw_path"]).read_text(encoding="utf-8").splitlines()
        a, b = sec["lines"]
        assert "\n".join(source[a - 1: b]) == sec["text"], f"line range wrong: {sec['id']}"

    out = ROOT / "data" / "graph" / "sections.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"schema_version": 1, "docs": docs, "sections": sections},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sizes = sorted(s["chars"] for s in sections)
    with_code = sum(1 for s in sections if s["code"])
    log("")
    log(f"{len(docs)} docs, {len(sections)} sections")
    log(f"  median {sizes[len(sizes) // 2]} chars, "
        f"largest {sizes[-1]}, {sum(1 for s in sizes if s < 200)} under 200 chars")
    log(f"  {with_code} sections carry a resolved code sample")
    log(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
