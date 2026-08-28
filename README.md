# Should My Agent Answer This?

**[Read the teardown](https://roanukz.github.io/should-my-agent-answer/)** ·
**[Try the explorer](https://roanukz.github.io/should-my-agent-answer/tool.html)**

Maps a documentation set as a knowledge graph and finds the articles that will
make an AI assistant answer confidently and wrong.

## The problem

Point an assistant at your documentation and three things can happen.

It works. The answer is in the docs, retrieval finds it, the assistant answers
correctly.

It finds nothing. No article is close, so the assistant declines. Annoying, but
safe, and visible: someone complains, a ticket gets filed, the content team
learns something is missing.

It finds something close but incomplete. An article looks relevant, gets
retrieved, and is missing one fact the answer depends on. The assistant answers
confidently and wrong. Nobody sees it, because the assistant resolved the
request and the deflection metric counted it as a success. The article looks
healthy forever.

Google Research measured the gap. Models handed no supporting context answered
incorrectly about 10 percent of the time. Models handed **insufficient** context
answered incorrectly about 66 percent of the time. Partial information is far
more dangerous than none.
([arXiv 2411.06037](https://arxiv.org/abs/2411.06037))

Content teams prioritize from complaints, escalations, and searches that return
nothing. All three surface only the second case. The third case generates no
signal at all, so it never gets prioritized, and it is the one doing the damage.

## What it does

Builds a knowledge graph over a documentation set where every edge carries the
verbatim sentence that justifies it, crosses that graph against questions real
users actually asked, and produces a ranked list of the articles most likely to
produce a confident wrong answer, naming the specific missing fact in each case.

Three kinds of finding, each with a proof path you can check line by line:

| | What it is | Why it matters |
|---|---|---|
| **Near miss** | A concept people ask about, that a section's instructions depend on, and that no section in the corpus explains | The article looks like the answer. It is not, because it rests on something never taught |
| **Orphan concept** | A concept the documentation leans on across several pages and never explains | The same shape, without anyone asking yet. A backlog rather than a fire |
| **Retrieval collision** | Two sections a single question maps to with comparable concept overlap, where only one carries the answer | The vaguer one can win retrieval |

Everything is ranked by demand, meaning the number of distinct discussion
threads that touch the concept. A gap nobody asks about is not worth writing.

The repository also answers every question twice, once from a vector baseline
and once by traversing the graph, and scores both against the maintainer's own
reply, so the headline number is a comparison rather than an assertion.

## Scope

**Corpus.** 60 pages of the FastAPI documentation, taken in the documentation's
own reading order: all 51 tutorial pages, then the first 9 advanced pages.
Pinned to one commit and committed under `data/raw/`.

**Demand and ground truth.** 300 answered threads from the Questions category of
the same repository's GitHub Discussions. If a question was answered in a
discussion and that answer is not in the documentation, the gap is real and the
correct content already exists in someone's reply. This is free, verifiable
ground truth, and it is the decision the whole project rests on, because it
removes the need to hand label anything.

**Not a full audit.** 60 of 155 English pages, and 300 of roughly 3,900 answered
threads. Extraction quality bounds every number downstream of it, which is why a
random sample is independently rechecked and the result reported with an
interval rather than as a pass or a fail.

## Constraints

Each of these is a decision, not an accident.

- **No build step for the site.** No bundler, no package manager, no framework.
  Open `index.html` from disk and it renders. The explorer ships its data as a
  script rather than fetching it, because every browser blocks `fetch()` against
  a `file://` URL and "open it from disk" should be true rather than nearly true.
- **No network calls from either page.** Ever. Not for fonts, not for analytics,
  not for a model. Everything either page shows is committed in this repository.
- **No analytics, no telemetry, no accounts, no database.**
- **Every graph edge carries its evidence span.** Verbatim source text, file
  path, line range. No exceptions, including for the structural edges where the
  span is obvious. `data/graph/edges.json` contains zero edges with a null or
  empty `evidence` object.
- **The invariant is enforced, not trusted.** Every span a model returns is
  searched for in the file its edge claims it came from, and an edge whose span
  is not found there is dropped rather than downgraded. The dropped count is
  reported in `data/manifest.json`.
- **All results are committed and reproducible.** `pipeline/run.sh` regenerates
  `data/findings.json` from `data/raw/` on a clean checkout.
- **No claim of methodological novelty.** Google Research's sufficient-context
  work is the method behind the headline finding. Ragas already builds a
  knowledge graph from a corpus. The contribution here is routing the finding to
  the person who owns the content, not the technique.

## Layout

```
index.html          the teardown
tool.html           the findings explorer
tokens.css          design tokens, shared unchanged with the sibling projects
style.css           components for both pages
app.js              explorer logic

pipeline/
  fetch.py          docs and threads into data/raw/, corpus commit pinned
  sections.py       deterministic split into Doc and Section nodes
  extract.py        concepts, edges, question mapping, and the span validator
  build_graph.py    networkx graph, structural edges, sanity report
  find.py           F1, F2 and F3 with proof paths, ranked by demand
  describe.py       names the missing fact on each near miss
  compare.py        vector retrieval against graph retrieval, generated and scored
  validate.py       50-item random sample, Wilson intervals
  manifest.py       what ran, over what, and how much of it there was
  site_data.py      the payload the two pages read
  llm.py            the API path for every step that needs a model
  run.sh            one command, end to end

tests/
  test_find.py      fixture graphs with known F1, F2 and F3 findings
  test_invariant.py the committed data, re-checked from outside the pipeline

data/
  raw/              the fetched pages, code samples and threads
  graph/            nodes.json and edges.json, every edge with its span
  findings.json     what the explorer reads
  answers.json      both retrieval paths, verdicts, summary
  validation.json   sample verdicts and intervals
  manifest.json     corpus commit, run date, counts
```

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r pipeline/requirements.txt
./pipeline/run.sh
```

Refetching the discussion threads needs a GitHub token, because GitHub serves
Discussions only through its GraphQL API and that API always requires one. Set
`GITHUB_TOKEN`, or be logged in with the `gh` CLI. The threads are committed, so
every step after the fetch runs with no token at all.

Four steps need a model. If `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is in the
environment, those steps run themselves. If neither is, each writes its prompt
payloads and tells you which directory a Claude Code session has to fill. Both
paths write the same JSON to the same place and everything downstream validates
it the same way. The committed results here came from the second path.

## Testing

```bash
python3 -m unittest discover -s tests -v
```

44 tests in two files.

`test_find.py` runs the finders over fixture graphs small enough to reason about
completely, where the right answer is known before the code runs: a near miss
that should fire and three neighbors that should not, an orphan that should fire
and two that should not, a collision that should fire and two arrangements that
should not, the demand ranking, the scope rule, the span validator's rejection
cases, the exact McNemar test, and the Wilson interval against its published
value.

`test_invariant.py` checks the committed data rather than the code that wrote it.
The one that matters re-reads all 5,399 edges and searches for each evidence span
in the source file that edge names, and separately checks that the line range it
cites is where the span actually sits. The invariant is only worth anything if it
can be checked from outside the extractor that produced it.

## What it found

| | Count | Held up on an independent check |
|---|---|---|
| Near miss | 14 | 11 of 14, unanimous across three readers each. Wilson 95% 0.52 to 0.92 |
| Orphan concept | 40 | not checked; these are the tail nobody has asked about |
| Retrieval collision | 99 | not checked |

Three near misses did not hold up, and they stay in the table and in the
explorer marked as such, because a gap-finding tool is judged on its false
positives and a list that quietly drops its own failures is not evidence of
anything.

Checking the same fourteen against the 95 English pages that are **not** in the
corpus knocked three more out: `APIRoute`, `host` and `virtual environment` are
documented on FastAPI pages outside the index. So precision is 11 of 14 as a
claim about the index and 8 of 14 as a claim about FastAPI. Both are published,
and the cheapest improvement available is not a better extractor, it is indexing
the other 95 pages.

Answering all 300 questions both ways: ordinary similarity search produced 24
correct answers and 5 confidently wrong, graph retrieval 23 correct and 3
confidently wrong. Only 17 questions had exactly one path right, split 9 to 8, so
graph retrieval did not make the assistant more right on this corpus. It made it
more cautious, and the difference is too small to claim either way. The
comparison is underpowered, and for a reason worth knowing: 282 of the 300
threads carry FastAPI's Questions template, in which the asker confirms they
already read the tutorial and did not find the answer. The sample is selected
for being unanswerable from the documentation, so both paths declining about 90%
of it is close to correct behavior rather than a retrieval failure.

## Related

[Will My Agent Answer This?](https://github.com/Roanukz/agent-answer) scores a
single knowledge-base article for whether an agent can answer from it. This one
maps a whole documentation set and finds which articles will make an agent
answer wrongly. That one is a linter for the page in front of you; this one is a
map of everything you own.

## License

MIT. See [LICENSE](LICENSE).
