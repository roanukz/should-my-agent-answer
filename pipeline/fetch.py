"""Fetch the corpus and the demand signal.

Two sources, both from github.com/fastapi/fastapi:

  1. Documentation pages. Plain markdown, pulled unauthenticated from
     raw.githubusercontent.com at a pinned commit so the run is repeatable.
  2. GitHub Discussions in the "Questions" category, answered only. These are
     the demand signal AND the ground truth: if someone asked and a maintainer
     answered, the question is real and the correct answer already exists.

Discussions are only reachable through GitHub's GraphQL API, which always
requires a token. Docs need no token at all. See DECISIONS.md, row "Discussion
auth", for why that is not the "no auth required" the PRD asked for.

Everything lands under data/raw/ and is committed, so a stranger with no token
can still re-run every step after this one.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_OWNER = "fastapi"
REPO_NAME = "fastapi"
DOCS_ROOT = "docs/en/docs"
MKDOCS_PATH = "docs/en/mkdocs.yml"
SITE_BASE = "https://fastapi.tiangolo.com/"

# The "Questions" category on fastapi/fastapi. Resolved once and pinned rather
# than looked up every run, so a category rename cannot silently change what we
# fetch. fetch.py verifies the name still matches before it trusts the id.
QUESTIONS_CATEGORY_ID = "MDE4OkRpc2N1c3Npb25DYXRlZ29yeTMyMDAxNDM0"
QUESTIONS_CATEGORY_NAME = "Questions"

DOC_CAP = 60
THREAD_CAP = 300

# Substance floor. An accepted answer of "+1" or "yes" is answered by GitHub's
# definition and useless as ground truth: there is no fact in it to be missing
# from the docs. Recorded in DECISIONS.md.
MIN_ANSWER_CHARS = 120
MIN_QUESTION_CHARS = 80

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW_DOCS = RAW / "docs"
RAW_DISCUSSIONS = RAW / "discussions"

USER_AGENT = "should-my-agent-answer/1.0 (+https://github.com/Roanukz/should-my-agent-answer)"


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def http_get(url: str, token: str | None = None, retries: int = 4) -> bytes:
    """GET with a short exponential backoff. Raises on final failure."""
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            # 403 here is nearly always the unauthenticated rate limit.
            if exc.code in (403, 429, 500, 502, 503, 504):
                wait = 2 ** attempt
                log(f"    {exc.code} on {url}, retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}: {last}")


def resolve_token() -> str | None:
    """GITHUB_TOKEN, then GH_TOKEN, then whatever the gh CLI is holding."""
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name)
        if value:
            log(f"  auth: using ${name}")
            return value
    if shutil.which("gh"):
        try:
            out = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout.strip()
            if out:
                log("  auth: using the token from the gh CLI")
                return out
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    log("  auth: none found")
    return None


def graphql(query: str, variables: dict, token: str) -> dict:
    """One GraphQL POST, with backoff on secondary rate limits."""
    payload = json.dumps({"query": query, "variables": variables}).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    for attempt in range(5):
        req = urllib.request.Request(
            "https://api.github.com/graphql", data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429, 502, 503):
                wait = 5 * (attempt + 1)
                log(f"    graphql {exc.code}, waiting {wait}s")
                time.sleep(wait)
                continue
            raise
        if "errors" in body:
            raise RuntimeError(f"GraphQL error: {body['errors']}")
        return body["data"]
    raise RuntimeError("GraphQL failed after 5 attempts")


# --------------------------------------------------------------------------
# Step 1: pin the corpus commit
# --------------------------------------------------------------------------

def resolve_commit(token: str | None) -> dict:
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/master"
    data = json.loads(http_get(url, token))
    return {
        "sha": data["sha"],
        "date": data["commit"]["committer"]["date"],
        "message": data["commit"]["message"].splitlines()[0],
    }


# --------------------------------------------------------------------------
# Step 2: choose 60 pages, in the documentation's own reading order
# --------------------------------------------------------------------------

def parse_nav_order(mkdocs_text: str) -> list[str]:
    """Pull the ordered page list out of the mkdocs nav block.

    A real YAML parse would need a dependency for a file whose nav is a flat
    list of quoted-or-bare paths one per line. We take the lines between `nav:`
    and the next top-level key and keep anything ending in `.md`. Order is what
    matters here and order survives.
    """
    lines = mkdocs_text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.rstrip() == "nav:")
    except StopIteration:
        return []
    order: list[str] = []
    for ln in lines[start + 1:]:
        # A non-indented, non-list line ends the nav block.
        if ln and not ln[0].isspace() and not ln.startswith("-"):
            break
        match = re.search(r"([A-Za-z0-9_\-./]+\.md)\s*$", ln)
        if match:
            path = match.group(1)
            if path not in order:
                order.append(path)
    return order


def select_pages(nav_order: list[str], all_md: list[str]) -> list[str]:
    """tutorial/ first, then advanced/, then top-level pages, capped at 60.

    Within each group we keep the nav order, which is the order a reader is
    meant to move through the documentation. Anything the nav does not list
    falls to the back of its group in path order, so the selection is total and
    deterministic even if the nav and the tree disagree.
    """
    nav_rank = {p: i for i, p in enumerate(nav_order)}

    def group_of(path: str) -> int:
        if path.startswith("tutorial/"):
            return 0
        if path.startswith("advanced/"):
            return 1
        if "/" not in path:
            return 2
        return 3  # everything else, only reached if the first three run dry

    def sort_key(path: str):
        return (group_of(path), nav_rank.get(path, 10_000), path)

    ordered = sorted(all_md, key=sort_key)
    return ordered[:DOC_CAP]


def doc_url(rel_path: str) -> str:
    """Map a markdown path to the published page URL."""
    slug = rel_path[:-3]  # strip .md
    if slug.endswith("/index"):
        slug = slug[: -len("/index")]
    elif slug == "index":
        return SITE_BASE
    return f"{SITE_BASE}{slug}/"


def fetch_docs(commit_sha: str, token: str | None) -> list[dict]:
    log("Fetching documentation pages")
    tree_url = (
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
        f"/git/trees/{commit_sha}?recursive=1"
    )
    tree = json.loads(http_get(tree_url, token))
    if tree.get("truncated"):
        raise RuntimeError("git tree came back truncated; cannot select pages reliably")

    prefix = DOCS_ROOT + "/"
    all_md = sorted(
        t["path"][len(prefix):]
        for t in tree["tree"]
        if t["type"] == "blob" and t["path"].startswith(prefix) and t["path"].endswith(".md")
    )
    log(f"  {len(all_md)} markdown pages under {DOCS_ROOT}")

    mkdocs_raw = http_get(
        f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{commit_sha}/{MKDOCS_PATH}"
    ).decode("utf-8")
    nav_order = parse_nav_order(mkdocs_raw)
    log(f"  nav lists {len(nav_order)} pages in reading order")

    selected = select_pages(nav_order, all_md)
    log(f"  selected {len(selected)} pages "
        f"({sum(1 for p in selected if p.startswith('tutorial/'))} tutorial, "
        f"{sum(1 for p in selected if p.startswith('advanced/'))} advanced, "
        f"{sum(1 for p in selected if '/' not in p)} top level)")

    records = []
    for i, rel in enumerate(selected, 1):
        raw_url = (
            f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{commit_sha}"
            f"/{DOCS_ROOT}/{rel}"
        )
        text = http_get(raw_url, token).decode("utf-8")
        out = RAW_DOCS / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        records.append({
            "rel_path": rel,
            "source_path": f"{DOCS_ROOT}/{rel}",
            "raw_path": str(out.relative_to(ROOT)),
            "url": doc_url(rel),
            "bytes": len(text.encode("utf-8")),
            "lines": text.count("\n") + 1,
        })
        if i % 10 == 0 or i == len(selected):
            log(f"  {i}/{len(selected)} pages")
    return records


# --------------------------------------------------------------------------
# Step 3: discussions
# --------------------------------------------------------------------------

DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $categoryId: ID!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(
      first: 50
      categoryId: $categoryId
      answered: true
      orderBy: {field: CREATED_AT, direction: DESC}
      after: $after
    ) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        body
        bodyText
        createdAt
        url
        isAnswered
        category { name }
        answer {
          body
          bodyText
          url
          createdAt
          upvoteCount
          authorAssociation
          author { login }
        }
      }
    }
  }
}
"""

MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


def thread_markdown(node: dict) -> str:
    """A readable, line-addressable copy of the thread.

    Evidence spans for question edges point at THIS file, not at the JSON,
    because a JSON string escapes every newline onto one line and a line range
    into it would tell a reader nothing.
    """
    answer = node.get("answer") or {}
    author = (answer.get("author") or {}).get("login") or "unknown"
    parts = [
        f"# {node['title']}",
        "",
        f"Discussion #{node['number']} · {node['createdAt']} · {node['url']}",
        "",
        "## Question",
        "",
        (node.get("body") or "").strip(),
        "",
        f"## Accepted answer by {author} ({answer.get('authorAssociation', 'NONE')})",
        "",
        (answer.get("body") or "").strip(),
        "",
    ]
    return "\n".join(parts)


def fetch_discussions(token: str) -> list[dict]:
    log("Fetching discussion threads")
    kept: list[dict] = []
    skipped_thin = 0
    scanned = 0
    after: str | None = None
    total_available = None

    while len(kept) < THREAD_CAP:
        data = graphql(
            DISCUSSIONS_QUERY,
            {
                "owner": REPO_OWNER,
                "name": REPO_NAME,
                "categoryId": QUESTIONS_CATEGORY_ID,
                "after": after,
            },
            token,
        )
        conn = data["repository"]["discussions"]
        if total_available is None:
            total_available = conn["totalCount"]
            log(f"  {total_available} answered threads in the Questions category")

        for node in conn["nodes"]:
            scanned += 1
            if node["category"]["name"] != QUESTIONS_CATEGORY_NAME:
                continue
            answer = node.get("answer") or {}
            body_text = (node.get("bodyText") or "").strip()
            answer_text = (answer.get("bodyText") or "").strip()
            if len(answer_text) < MIN_ANSWER_CHARS or len(body_text) < MIN_QUESTION_CHARS:
                skipped_thin += 1
                continue

            number = node["number"]
            record = {
                "number": number,
                "title": node["title"],
                "body": node.get("body") or "",
                "body_text": body_text,
                "created_at": node["createdAt"],
                "url": node["url"],
                "answered": bool(node["isAnswered"]),
                "answer": {
                    "body": answer.get("body") or "",
                    "body_text": answer_text,
                    "url": answer.get("url"),
                    "created_at": answer.get("createdAt"),
                    "upvotes": answer.get("upvoteCount", 0),
                    "author": (answer.get("author") or {}).get("login"),
                    "author_association": answer.get("authorAssociation"),
                    "is_maintainer": answer.get("authorAssociation") in MAINTAINER_ASSOCIATIONS,
                },
            }
            (RAW_DISCUSSIONS / f"{number}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (RAW_DISCUSSIONS / f"{number}.md").write_text(
                thread_markdown(node), encoding="utf-8"
            )
            kept.append({
                "number": number,
                "title": node["title"],
                "url": node["url"],
                "created_at": node["createdAt"],
                "answer_chars": len(answer_text),
                "is_maintainer": record["answer"]["is_maintainer"],
            })
            if len(kept) >= THREAD_CAP:
                break

        if not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]
        log(f"  {len(kept)}/{THREAD_CAP} kept, {scanned} scanned")

    log(f"  kept {len(kept)} threads, skipped {skipped_thin} with a thin question or answer")
    maintainer = sum(1 for k in kept if k["is_maintainer"])
    log(f"  {maintainer} answered by a maintainer, {len(kept) - maintainer} by another user")
    return kept


# --------------------------------------------------------------------------

def main() -> int:
    RAW_DOCS.mkdir(parents=True, exist_ok=True)
    RAW_DISCUSSIONS.mkdir(parents=True, exist_ok=True)

    token = resolve_token()
    commit = resolve_commit(token)
    log(f"Corpus pinned at {commit['sha'][:12]} ({commit['date']})")

    docs = fetch_docs(commit["sha"], token)

    if token:
        threads = fetch_discussions(token)
    else:
        existing = sorted(RAW_DISCUSSIONS.glob("*.json"))
        if not existing:
            log("")
            log("No token, and data/raw/discussions/ is empty.")
            log("GitHub serves Discussions only through the GraphQL API, which always")
            log("requires a token. Set GITHUB_TOKEN, or run `gh auth login`, or use the")
            log("threads already committed in this repo.")
            return 1
        log(f"No token: reusing the {len(existing)} threads committed under data/raw/")
        threads = []
        for path in existing:
            rec = json.loads(path.read_text(encoding="utf-8"))
            threads.append({
                "number": rec["number"],
                "title": rec["title"],
                "url": rec["url"],
                "created_at": rec["created_at"],
                "answer_chars": len(rec["answer"]["body_text"]),
                "is_maintainer": rec["answer"]["is_maintainer"],
            })

    meta = {
        "schema_version": 1,
        "corpus": f"{REPO_OWNER}/{REPO_NAME}",
        "corpus_commit": commit["sha"],
        "corpus_commit_date": commit["date"],
        "docs_root": DOCS_ROOT,
        "doc_cap": DOC_CAP,
        "thread_cap": THREAD_CAP,
        "min_answer_chars": MIN_ANSWER_CHARS,
        "min_question_chars": MIN_QUESTION_CHARS,
        "docs": docs,
        "threads": sorted(threads, key=lambda t: -t["number"]),
    }
    (RAW / "fetch_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log("")
    log(f"Wrote {len(docs)} pages and {len(threads)} threads to data/raw/")
    log(f"Manifest of the fetch: data/raw/fetch_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
