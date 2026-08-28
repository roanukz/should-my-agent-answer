# Should My Agent Answer This?

## What this is
A pipeline that maps a documentation set as a knowledge graph, crosses it against
questions real users asked, and produces a ranked list of the articles most likely to
make an AI assistant answer confidently and wrong, naming the specific missing fact in
each case. The corpus is the FastAPI documentation. The demand signal and the ground
truth are both GitHub Discussions on the same repository.

Three artifacts: `index.html` is the teardown essay, `tool.html` is the findings
explorer, `pipeline/` is the Python that produced the committed results.

## The hard invariant: no edge without evidence
**Every edge in the graph carries the verbatim source span that justifies it, plus the
file path and the line range where that span lives.** An edge without a span is a bug,
not a low-confidence edge. `data/graph/edges.json` must contain zero edges with a null
or empty `evidence` object, and `pipeline/extract.py` asserts this before it writes.

This is not a style preference. The whole claim of the project is "here is the sentence
the graph read", not "the graph says so". A finding a reader cannot audit is a finding
nobody acts on.

The invariant is enforced mechanically, never by trusting the model. `extract.py`
searches for every returned span in the file the edge claims it came from and DROPS any
edge whose span is not found there. Dropped counts are reported and land in
`data/manifest.json`. Do not relax the search into fuzzy matching. The permitted
normalizations are Unicode NFKC, whitespace collapsing, quote and dash flattening, and
case folding.

There is exactly one addition, and only for a span copied out of a question. GitHub's
API serves a discussion body twice, once as markdown and once as the plain text it
renders to. The question mapper reads the plain text; the thread file on disk holds the
markdown. So a question span that fails the first search is searched once more with the
markup characters removed from BOTH sides, which is matching like with like rather than
loosening the match. It is still a substring search, it is counted separately in
`data/graph/question_report.json`, and it is not available to any other edge type.

## Other constraints, all deliberate
- **The published pages never call a model, and never call the network at all.** Both
  read committed JSON from `data/`. Extraction and generation happen offline, in the
  pipeline, and their results are committed.
- **No build step for the site.** No bundler, no package manager, no framework. Open
  `index.html` from disk and it renders. This is the save-the-dates pattern.
- **Everything is reproducible.** `pipeline/run.sh` regenerates `data/findings.json`
  from `data/raw/` on a clean checkout.
- **`data/raw/` is committed.** The corpus commit is pinned in `data/manifest.json`, so
  a re-run reads the same bytes even after FastAPI's docs move on.
- **No analytics, no telemetry, no accounts, no database.**
- **No claim of methodological novelty anywhere in the repo.** Google Research's
  sufficient-context work is the method behind the headline finding. Ragas already
  builds a corpus knowledge graph. Say so plainly.

## Not in scope, each considered and cut
No Claude Code plugin, skill packaging, or MCP server. No CI check that runs on
documentation changes. No generalization to a second repository. No live model calls
from either page. No hosted service. No downstream impact claim: this measures
documentation, not a deployed assistant.

## House style for anything a reader sees
- **US spelling throughout.** Color, behavior, normalize, recognize, prioritize,
  labeled, center, gray, license, practice, while, among. This covers prose, code
  identifiers and comments, and any model-generated string that renders into a
  page. The exceptions are things that are not English words: `aria-labelledby`,
  `CancelledError`, and anything quoted verbatim from a source file or a
  discussion thread, which is data and is never edited.
- No em dashes and no en dashes anywhere in visible copy, including `<title>` and
  `data/` strings that render into a page. Rebuild the sentence with a conjunction
  rather than swapping in a full stop, which reads choppy.
- "Cut", never "kill", for scope decisions. That includes section ids and anchors.
- Numeric ranges spell out: "2024 to 2026", not "2024-2026".
- Numbering follows ISO 2145: Arabic numerals for body parts, letters for appendices,
  and the contents rail carries the same label text as the body kicker.
- Report the real number whatever it is. If validity comes in low, fix the build and
  rerun: the corrected run is the result. Do not narrate the superseded one in the
  teardown. A number produced by a build with a known defect measures the defect, and
  walking a reader through it costs them time and tells them nothing about the tool
  they can actually use. Record what changed in `DECISIONS.md`, which is where that
  belongs.

## Files
- `pipeline/fetch.py` docs and discussions into `data/raw/`, corpus commit pinned
- `pipeline/sections.py` deterministic split into Doc and Section nodes
- `pipeline/extract.py` concepts and edges, and the span validator
- `pipeline/build_graph.py` networkx graph over the nodes and edges
- `pipeline/find.py` F1 near miss, F2 orphan concept, F3 retrieval collision
- `pipeline/compare.py` vector retrieval against graph retrieval, scored
- `pipeline/validate.py` 50-item sample, Wilson intervals, no pass or fail gate
- `index.html` teardown, `tool.html` explorer, `app.js` explorer logic
- `tokens.css` design tokens shared with the sibling repos, `style.css` components

## Rules for you
- Ask before adding a dependency. The list is networkx, requests, markdown-it-py,
  sentence-transformers, and nothing else.
- Do not refactor code you were not asked to change.
- Keep functions short and comment them in plain English.
- Record every deviation from the PRD as a row in `DECISIONS.md`, with the reason.
- After each change, say exactly what to run to test it.
