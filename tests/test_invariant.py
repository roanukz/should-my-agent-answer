"""Check the committed data, not the code that wrote it.

test_find.py checks the finders against fixtures. This file checks the artifacts
that actually ship: the graph in data/graph/, the findings the explorer reads,
and the numbers the teardown quotes. If someone edits a JSON file by hand, or a
pipeline change quietly drops a field, this is what notices.

The important one is test_every_span_is_still_in_its_file. It re-reads all 5,399
edges and searches for each evidence span in the source file that edge names,
independently of the extractor that put it there. The invariant is only worth
anything if it can be checked from outside.

Run with:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import extract  # noqa: E402


def load(name: str):
    path = ROOT / "data" / name
    if not path.exists():
        raise unittest.SkipTest(f"{name} has not been generated yet")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str) -> Path | None:
    """Map the path an edge records to the committed copy of that file."""
    if path.startswith("docs/en/docs/"):
        return ROOT / "data" / "raw" / path[len("docs/en/"):]
    if path.startswith("docs_src/"):
        return ROOT / "data" / "raw" / path
    if path.startswith("data/raw/"):
        return ROOT / path
    return None


class EvidenceInvariantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.edges = load("graph/edges.json")["edges"]
        cls.nodes = load("graph/nodes.json")["nodes"]
        cls.ids = {n["id"] for n in cls.nodes}

    def test_no_edge_without_an_evidence_span(self) -> None:
        """S1. The one rule the whole project hangs off."""
        missing = [e["id"] for e in self.edges
                   if not (e.get("evidence") or {}).get("span")]
        self.assertEqual(missing, [], f"{len(missing)} edges carry no span")

    def test_every_span_carries_a_path_and_a_line_range(self) -> None:
        for edge in self.edges:
            evidence = edge["evidence"]
            self.assertTrue(evidence.get("path"), edge["id"])
            self.assertEqual(len(evidence.get("lines", [])), 2, edge["id"])
            self.assertLessEqual(evidence["lines"][0], evidence["lines"][1], edge["id"])

    def test_every_span_is_still_in_its_file(self) -> None:
        """The invariant, re-derived from the committed files themselves."""
        cache: dict[Path, str] = {}
        unfindable = []
        checked = 0
        for edge in self.edges:
            evidence = edge["evidence"]
            source = resolve(evidence["path"])
            if source is None or not source.exists():
                unfindable.append((edge["id"], "no such file: " + evidence["path"]))
                continue
            if source not in cache:
                cache[source] = extract.normalize(source.read_text(encoding="utf-8"))
            checked += 1
            span = extract.normalize(evidence["span"])
            if span in cache[source]:
                continue
            # The one extra normalization, and only for a question span.
            loose_span = extract.strip_markdown_markers(span)
            loose_hay = extract.strip_markdown_markers(cache[source])
            if loose_span not in loose_hay:
                unfindable.append((edge["id"], evidence["span"][:60]))
        self.assertGreater(checked, 0)
        self.assertEqual(unfindable, [], f"{len(unfindable)} spans could not be found")

    def test_no_edge_points_at_a_node_that_does_not_exist(self) -> None:
        dangling = [e["id"] for e in self.edges
                    if e["from"] not in self.ids or e["to"] not in self.ids]
        self.assertEqual(dangling, [])

    def test_the_line_range_matches_where_the_span_actually_sits(self) -> None:
        """A span in the right file at the wrong lines is still a broken citation."""
        cache: dict[Path, list[str]] = {}
        wrong = []
        for edge in self.edges:
            evidence = edge["evidence"]
            source = resolve(evidence["path"])
            if source is None or not source.exists():
                continue
            if source not in cache:
                cache[source] = source.read_text(encoding="utf-8").splitlines()
            lines = cache[source]
            start, end = evidence["lines"]
            if start < 1 or end > len(lines):
                wrong.append((edge["id"], "line range outside the file"))
                continue
            window = extract.normalize("\n".join(lines[start - 1: end]))
            span = extract.normalize(evidence["span"])
            if span in window:
                continue
            if extract.strip_markdown_markers(span) in extract.strip_markdown_markers(window):
                continue
            wrong.append((edge["id"], f"{evidence['path']}:{start}-{end}"))
        self.assertEqual(wrong[:5], [], f"{len(wrong)} spans cited at the wrong lines")


class FindingsShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.findings = load("findings.json")["findings"]

    def test_every_finding_has_a_non_empty_proof_path(self) -> None:
        """S2."""
        bare = [f["id"] for f in self.findings if not f.get("proof_path")]
        self.assertEqual(bare, [])

    def test_every_finding_has_an_integer_demand(self) -> None:
        """S3."""
        for finding in self.findings:
            self.assertIsInstance(finding.get("demand"), int, finding["id"])

    def test_ids_are_unique_and_typed(self) -> None:
        ids = [f["id"] for f in self.findings]
        self.assertEqual(len(ids), len(set(ids)))
        for finding in self.findings:
            self.assertIn(finding["type"],
                          {"near_miss", "orphan_concept", "retrieval_collision"})

    def test_no_near_miss_concept_is_defined_anywhere(self) -> None:
        """The whole claim of an F1 finding, checked against the graph."""
        edges = load("graph/edges.json")["edges"]
        defined = {e["to"] for e in edges if e["type"] == "DEFINES"}
        offenders = [f["id"] for f in self.findings
                     if f["type"] == "near_miss" and f["concept"] in defined]
        self.assertEqual(offenders, [])

    def test_no_em_or_en_dash_in_anything_a_reader_sees(self) -> None:
        """House style, and the explorer renders these strings straight onto the page."""
        offenders = []
        for finding in self.findings:
            for field in ("missing", "validation_note", "gap_note", "concept_label"):
                value = finding.get(field) or ""
                if "—" in value or "–" in value:
                    offenders.append((finding["id"], field))
        self.assertEqual(offenders, [])


class ReportedNumbersTest(unittest.TestCase):
    """The teardown quotes these. They have to come from the files."""

    def test_validation_reports_a_rate_and_an_interval(self) -> None:
        """S4."""
        by_type = load("validation.json")["by_type"]
        self.assertIn("near_miss", by_type)
        entry = by_type["near_miss"]
        self.assertEqual(entry["n"], entry["valid"] + entry["invalid"])
        low, high = entry["wilson_95"]
        self.assertLessEqual(low, entry["observed_rate"])
        self.assertLessEqual(entry["observed_rate"], high)

    def test_both_retrieval_paths_are_scored_over_every_question(self) -> None:
        """S8."""
        answers = load("answers.json")
        summary = answers["summary"]
        self.assertEqual(summary["n"], len(answers["results"]))
        for path in ("vector", "graph"):
            self.assertIn("wrong_confident", summary[path])
            self.assertEqual(sum(summary[path].values()), summary["n"])

    def test_the_manifest_pins_the_corpus_commit(self) -> None:
        manifest = load("manifest.json")
        self.assertRegex(manifest["corpus_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(manifest["corpus"], "fastapi/fastapi")

    def test_the_site_payload_matches_the_findings(self) -> None:
        site = (ROOT / "data" / "site.js")
        if not site.exists():
            self.skipTest("site.js has not been generated yet")
        text = site.read_text(encoding="utf-8")
        payload = json.loads(text[text.index("=") + 1: text.rstrip().rfind(";")])
        self.assertEqual(len(payload["findings"]), len(load("findings.json")["findings"]))


if __name__ == "__main__":
    unittest.main()


class DiagramAccuracyTest(unittest.TestCase):
    """The schema figure quotes counts. Pin them to the graph so they cannot drift.

    A diagram is the part of a page a reader trusts most and checks least, so
    the numbers drawn inside it are held to the same standard as the tables:
    derived from data/graph/edges.json, never typed in and left.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.page = (ROOT / "index.html").read_text(encoding="utf-8")
        cls.flat = re.sub(r"\s+", " ", cls.page)
        cls.nodes = load("graph/nodes.json")["nodes"]
        cls.edges = load("graph/edges.json")["edges"]
        cls.by_id = {n["id"]: n for n in cls.nodes}

    def counts(self, key: str) -> int:
        from collections import Counter
        return Counter(e["type"] for e in self.edges)[key]

    def test_the_figure_quotes_the_real_edge_counts(self) -> None:
        for edge_type in ("CONTAINS", "DEFINES", "REQUIRES", "MENTIONS",
                          "ASKS_ABOUT", "ANSWERED_BY", "LINKS_TO"):
            n = self.counts(edge_type)
            drawn = f"{edge_type} {n:,}"
            self.assertIn(drawn, self.page,
                          f"the figure should say {drawn!r}")

    def test_the_figure_quotes_the_real_node_counts(self) -> None:
        from collections import Counter
        counted = Counter(n["type"] for n in self.nodes)
        # Each node box carries its type name and its count on the next line.
        for node_type, n in counted.items():
            self.assertRegex(
                self.page,
                rf'>{node_type}</text>\s*<text[^>]*>{n:,}</text>',
                f"the figure should draw {node_type} with {n:,}")

    def test_every_edge_type_joins_exactly_one_ordered_pair_of_node_types(self) -> None:
        """The page calls the schema strict. Check that it is."""
        from collections import defaultdict
        pairs = defaultdict(set)
        for edge in self.edges:
            pairs[edge["type"]].add(
                (self.by_id[edge["from"]]["type"], self.by_id[edge["to"]]["type"]))
        loose = {k: v for k, v in pairs.items() if len(v) != 1}
        self.assertEqual(loose, {}, "an edge type joins more than one pair of types")

    def test_the_graph_is_directed_with_no_reciprocated_pair(self) -> None:
        seen = {(e["from"], e["to"]) for e in self.edges}
        both_ways = [(a, b) for a, b in seen if (b, a) in seen]
        self.assertEqual(both_ways, [])

    def test_the_multigraph_claim_matches_the_data(self) -> None:
        """The page says 48 ordered pairs carry two edges. Recount them."""
        from collections import Counter
        pair = Counter((e["from"], e["to"]) for e in self.edges)
        parallel = sum(1 for v in pair.values() if v > 1)
        self.assertIn(f"{parallel} times the extractor produced two relations", self.flat)

    def test_defines_wins_where_a_pair_carries_two_relations(self) -> None:
        """The page claims the cautious direction is taken. Check the finders."""
        from collections import defaultdict
        relations = defaultdict(set)
        for edge in self.edges:
            if edge["type"] in ("DEFINES", "REQUIRES", "MENTIONS"):
                relations[(edge["from"], edge["to"])].add(edge["type"])
        conflicted = {c for (_s, c), rel in relations.items()
                      if "DEFINES" in rel and len(rel) > 1}
        reported = {f["concept"] for f in load("findings.json")["findings"]
                    if f["type"] in ("near_miss", "orphan_concept")}
        self.assertEqual(conflicted & reported, set(),
                         "a concept defined somewhere was still reported as a gap")

    def test_the_retrieval_comparison_row_matches_answers_json(self) -> None:
        """The costs table quotes the measured result; it must be the real one."""
        summary = load("answers.json")["summary"]
        self.assertEqual(summary["vector"]["wrong_confident"], 5)
        self.assertEqual(summary["graph"]["wrong_confident"], 3)
        self.assertIn("3 over the same 300", self.page)

    def test_both_figures_are_labelled_for_a_screen_reader(self) -> None:
        for token in ('role="img"', "schema-title", "schema-desc",
                      "nearmiss-title", "nearmiss-desc"):
            self.assertIn(token, self.page)

    def test_the_figures_reference_no_external_resource(self) -> None:
        import re
        for match in re.finditer(r'<(?:image|use)\b[^>]*>', self.page):
            self.fail(f"figure pulls an external resource: {match.group(0)[:80]}")
        self.assertNotIn("url(http", self.page)


class PublishedNumbersTest(unittest.TestCase):
    """Every figure the essay prints, recomputed from the data it cites.

    Added after an independent review found nine stale or wrong numbers on the
    page, including an edge count that contradicted the diagram fifty lines
    above it. Each of these is a number a reader can check, so each one is a
    number the test suite should check first.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.page = (ROOT / "index.html").read_text(encoding="utf-8")
        cls.findings = load("findings.json")["findings"]
        cls.answers = load("answers.json")

    def of_type(self, kind: str):
        return [f for f in self.findings if f["type"] == kind]

    def threads_reached(self, kind: str) -> int:
        return len({q for f in self.of_type(kind) for q in f["questions"]})

    def test_the_threads_reached_column_counts_distinct_threads(self) -> None:
        """Not the sum of demand: one thread raised two near misses."""
        self.assertEqual(self.threads_reached("near_miss"), 20)
        self.assertEqual(self.threads_reached("retrieval_collision"), 116)
        for value in (20, 116):
            self.assertIn(f'<td class="num">{value}</td>', self.page)

    def test_findings_counts_on_the_page_match_the_file(self) -> None:
        for kind, n in (("near_miss", 14), ("orphan_concept", 40),
                        ("retrieval_collision", 99)):
            self.assertEqual(len(self.of_type(kind)), n)

    def test_the_answers_table_and_its_caption_agree(self) -> None:
        summary = self.answers["summary"]
        scored = sum(
            v for path in ("vector", "graph")
            for k, v in summary[path].items() if k != "unscored")
        self.assertEqual(scored, 598)
        self.assertIn("598 of 600 answers scored", self.page)

    def test_the_paired_comparison_runs_on_correctness_discordant_pairs(self) -> None:
        paired = self.answers["summary"]["paired"]
        discordant = paired["vector_only_correct"] + paired["graph_only_correct"]
        self.assertEqual(discordant, 17)
        self.assertIn("the 17 where exactly one of them got it", self.page)
        self.assertIn("that split is 9 to 8", self.page)

    def test_the_page_does_not_claim_the_scoring_was_blind(self) -> None:
        """It was not: the scoring prompt names the path in every block."""
        self.assertNotIn("without being told which retrieval path", self.page)
        self.assertIn("The scorer was not blind", self.page)

    def test_every_traversal_string_on_the_page_is_verbatim_in_the_data(self) -> None:
        """The guard against the worst kind of error this page could make.

        A callout headed "as committed" once carried a traversal line that was
        reconstructed rather than copied. On a page whose argument is that every
        claim is checkable in the repository, a quote that is not in the
        repository is the one defect that cannot be excused, so it is tested.
        """
        import html as htmllib
        import re
        committed = set()
        for result in self.answers["results"]:
            for step in result["graph"].get("traversal", []):
                committed.add(re.sub(r"\s+", " ", step).strip())

        quoted = re.findall(
            r"<code[^>]*>\s*(q:\d+ ASKS_ABOUT[^<]*|sec:\S[^<]*REQUIRES[^<]*?)\s*</code\s*>",
            self.page)
        self.assertTrue(quoted, "no traversal quotes found to check")
        for raw in quoted:
            step = re.sub(r"\s+", " ", htmllib.unescape(raw)).strip()
            self.assertIn(step, committed,
                          f"traversal quoted on the page is not in answers.json: {step}")

    def test_the_worked_example_is_a_question_the_hop_actually_changed(self) -> None:
        import re
        result = next(r for r in self.answers["results"] if r["question_id"] == "q:13828")
        retrieved = set(result["graph"]["retrieved"])
        hopped = {m.group(1)
                  for step in result["graph"]["traversal"]
                  for m in [re.search(r"REQUIRES \S+ -> DEFINES -> (sec:\S+)$", step)]
                  if m}
        self.assertTrue(retrieved & hopped,
                        "the example must be one where a hop reached the final set")
        self.assertEqual(result["graph"]["verdict"], "correct")
        self.assertNotEqual(result["vector"]["verdict"], "correct")

    def test_the_definition_sweep_reports_what_landed(self) -> None:
        # Derived from the graph, because the report file describes only the
        # invocation that wrote it and the adjudication pass ran last.
        edges = load("graph/edges.json")["edges"]
        landed = sum(1 for e in edges if e["extractor"] == "definition-sweep")
        self.assertEqual(landed, 101)

    def test_the_page_reports_the_validation_result_the_data_holds(self) -> None:
        by_type = load("validation.json")["by_type"]["near_miss"]
        self.assertEqual((by_type["n"], by_type["valid"]), (14, 11))
        self.assertIn('<td class="num">11</td>', self.page)
        self.assertIn("0.52 to 0.92", self.page)

    def test_the_three_that_failed_are_still_shown(self) -> None:
        """A list that drops its own failures is not evidence of anything."""
        failed = [f for f in self.of_type("near_miss") if f.get("validated") is False]
        self.assertEqual(len(failed), 3)
        for finding in failed:
            self.assertIn(finding["id"], self.page,
                          f"{finding['id']} did not hold up and is not on the page")

    def test_no_result_from_a_superseded_build_is_discussed(self) -> None:
        """The shipped tool is the baseline. Earlier broken runs are not results."""
        import re
        for pattern in (r"\bround one\b", r"\bround two\b", r"\bfirst round\b",
                        r"\bsecond round\b", r"\b9 of 33\b", r"\b27\.3\b",
                        r"first validation round"):
            self.assertIsNone(
                re.search(pattern, self.page, re.I),
                f"page still discusses a superseded run: {pattern}")
        self.assertFalse((ROOT / "data" / "validation-round-1.json").exists(),
                         "an artifact of a superseded run is still committed")


class CorpusBoundaryTest(unittest.TestCase):
    """The measurement that separates a gap in FastAPI from a gap in the index.

    Added after a review pointed out that everything else on the page measures
    the findings against the 60 pages that were indexed, which is not the
    question a reader thinks is being answered.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.page = (ROOT / "index.html").read_text(encoding="utf-8")
        cls.flat = re.sub(r"\s+", " ", cls.page)
        cls.boundary = load("corpus_boundary.json")
        cls.near = {f["id"]: f for f in load("findings.json")["findings"]
                    if f["type"] == "near_miss"}

    def test_every_near_miss_was_checked_outside_the_corpus(self) -> None:
        self.assertEqual(set(self.boundary), set(self.near))

    def test_each_verdict_carries_a_majority_of_its_readers(self) -> None:
        for fid, rec in self.boundary.items():
            self.assertGreaterEqual(rec["readers"], 3, fid)
            outside = rec["verdict"] == "documented_outside"
            self.assertEqual(outside, rec["votes_outside"] * 2 > rec["readers"], fid)

    def test_a_definition_found_outside_names_a_real_file(self) -> None:
        for fid, rec in self.boundary.items():
            if rec["verdict"] != "documented_outside":
                continue
            self.assertTrue(rec["path"], fid)
            self.assertTrue((ROOT / rec["path"]).exists(),
                            f"{fid} cites {rec['path']}, which is not there")
            self.assertNotIn("data/raw/docs/", rec["path"],
                             f"{fid} cites a page that IS in the corpus")

    def test_the_two_by_two_on_the_page_matches_the_data(self) -> None:
        cells = {}
        for fid, rec in self.boundary.items():
            key = (self.near[fid].get("validated") is True,
                   rec["verdict"] == "documented_outside")
            cells[key] = cells.get(key, 0) + 1
        self.assertEqual(cells[(True, False)], 8)
        self.assertEqual(cells[(True, True)], 3)
        self.assertEqual(cells[(False, False)], 2)
        self.assertEqual(cells[(False, True)], 1)
        self.assertIn("drops from 11 of 14 to 8 of 14", self.flat,
                      "the page should still state both denominators")

    def test_the_page_names_the_three_that_are_index_artifacts(self) -> None:
        artifacts = {self.near[fid]["concept_label"]
                     for fid, rec in self.boundary.items()
                     if rec["verdict"] == "documented_outside"
                     and self.near[fid].get("validated") is True}
        self.assertEqual(artifacts, {"APIRoute", "host", "virtual environment"})
        for label in artifacts:
            self.assertIn(label, self.page)


class ManifestLedgerTest(unittest.TestCase):
    """The extraction ledger has to account for every edge that did not survive."""

    def test_the_drops_sum_to_the_gap(self) -> None:
        e = load("manifest.json")["extraction"]
        gap = e["edges_proposed"] - e["edges_kept"]
        drops = sum(v for k, v in e.items() if k.startswith("dropped"))
        self.assertEqual(
            gap, drops,
            "the manifest lost an edge between proposed and kept without saying why")


class SpellingTest(unittest.TestCase):
    """US spelling, in everything the project writes.

    Source data is exempt: data/raw/ holds fetched FastAPI pages and discussion
    threads, and those are quoted verbatim elsewhere, so editing them would
    break the evidence invariant. Two identifiers are exempt because they are
    not English words.
    """

    # Two rules, because one list rots. The -ise/-isation family is caught
    # generically and filtered against the words that legitimately end that way
    # in American English; everything else is an explicit list. A leading prefix
    # must not hide the stem: "recolour" slipped past a pattern that anchored on
    # a word boundary immediately before it, so the stem lists carry no leading
    # boundary. An earlier version of this test named too few words and passed
    # while the repository still held 18 British spellings.
    US_ISE = (
        "advertise advise apprise arise chastise circumcise comprise compromise "
        "concise cruise bruise demise despise devise disguise enterprise excise "
        "exercise expertise franchise guise improvise incise merchandise noise "
        "paradise poise praise appraise braise precise premise prise promise "
        "raise revise rise sunrise supervise surmise surprise televise treatise "
        "malaise valise reprise anise turquoise tortoise porpoise fundraise"
    ).split()

    # Candidates first, then filtered in Python. A regex lookahead cannot do
    # this filtering: it has to reject "raised" via "raise" and accept
    # "characterised" despite the "ris" sitting inside it, and an unanchored
    # alternation gets that backwards.
    ISE = re.compile(
        r"\b\w*?is(?:e|es|ed|ing|er|ers|ation|ations|ational|able|ability)\b",
        re.I)

    # Words the generic rule catches by accident: the "is" it keys on is not a
    # suffix here. "disable" is "dis" + "able"; "wiser" is "wise" + "r".
    NOT_A_SUFFIX = ("disable disabled disables disabling disablement advisable "
                    "database databases miser misers wiser wisest").split()

    @classmethod
    def us_ise_forms(cls):
        forms = set(cls.NOT_A_SUFFIX)
        for word in cls.US_ISE:
            stem = word[:-1]
            forms.update({word, word + "s", stem + "ed", stem + "ing",
                          stem + "er", stem + "ers", stem + "able",
                          stem + "ation", stem + "ations"})
        return forms

    BRITISH = re.compile(
        r"\b\w*?("
        # -our
        r"colour\w*|behaviour\w*|favour\w*|honour\w*|labour\w*|neighbour\w*"
        r"|humour\w*|rumour\w*|flavour\w*|harbour\w*|armour\w*|endeavour\w*"
        r"|vapour\w*|savour\w*|odour\w*|vigour\w*|rigour\w*|candour|clamour"
        r"|demeanour|saviour|splendour|parlour|valour|tumour"
        # doubled l
        r"|labelled|labelling|unlabelled|modelling|modelled|cancelled|cancelling"
        r"|totalling|levelled|levelling|travelled|travelling|traveller\w*"
        r"|marvellous|counsellor\w*|signalled|signalling|fuelled|fuelling"
        r"|channelled|channelling|jewellery|woollen"
        # -re
        r"|centre[sd]?|centring|metre[s]?|litre[s]?|fibre[s]?|theatre[s]?"
        r"|calibre|sombre|spectre|lustre|manoeuvre[sd]?"
        # -ce / -se
        r"|licence[sd]?|defence[s]?|offence[s]?|pretence|practise\w*"
        # -yse
        r"|analyse|analysed|analysing|analyser|paralyse\w*|catalyse\w*"
        # -logue and the rest
        r"|analogue[s]?|catalogue[sd]?|programme[s]?|judgement[s]?"
        r"|acknowledgement[s]?|grey|greys|greyed|greyish|greyscale\w*"
        r"|whilst|amongst|maths|aluminium|ageing|sceptic\w*|artefact[s]?"
        r"|enquir\w*|speciality|specialities|fulfil|fulfils|instalment[s]?"
        r"|skilful\w*|wilful\w*|storey[s]?|kerb\w*|cheque[s]?|moustache"
        r"|mediaeval|encyclopaedia|sulphur\w*|tranquillity|manoeuvre"
        r"|learnt|burnt|spelt|dreamt"
        r")\b",
        re.I)

    EXEMPT = ("aria-labelledby", "labelledby", "CancelledError")

    # Walked, not listed. A hand-kept list silently stops covering new files,
    # which is half of why the earlier version of this test read clean.
    SUFFIXES = (".html", ".js", ".css", ".md", ".py", ".sh", ".json", ".txt")

    def sources(self):
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or path.suffix not in self.SUFFIXES:
                continue
            rel = path.relative_to(ROOT)
            parts = set(rel.parts)
            # data/ is fetched source and generated artifacts built from it.
            if "data" in parts or ".git" in parts or "node_modules" in parts:
                continue
            yield rel.as_posix(), path

    def scrub(self, text: str) -> str:
        for token in self.EXEMPT:
            text = text.replace(token, " ")
        return text

    def offences(self, rel: str, body: str):
        allowed = self.us_ise_forms()
        for rx in (self.BRITISH, self.ISE):
            for match in rx.finditer(body):
                word = match.group(0).lower()
                if rx is self.ISE:
                    # "otherwise", "turquoise" and the -ise verbs American
                    # English keeps are not misspellings.
                    if word.endswith(("wise", "oise")):
                        continue
                    # Exact match, or a real prefix on one. Not endswith:
                    # "linearise" ends with "arise" and is still wrong.
                    if word in allowed or any(
                            word[len(pre):] in allowed
                            for pre in ("un", "re", "dis", "mis", "over",
                                        "under", "pre", "non", "de", "co",
                                        "inter")
                            if word.startswith(pre)):
                        continue
                line = body[:match.start()].count("\n") + 1
                yield f"{rel}:{line} {match.group(0)}"

    def test_no_british_spelling_in_anything_this_project_writes(self) -> None:
        offenders = []
        for rel, path in self.sources():
            try:
                body = self.scrub(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                continue
            # This test file names the forms it bans, so skip its own pattern.
            if rel == "tests/test_invariant.py":
                body = body.split("class SpellingTest")[0]
            offenders.extend(self.offences(rel, body))
        self.assertEqual(offenders, [], f"{len(offenders)} British spellings")

    def test_no_british_spelling_in_strings_that_render_to_a_reader(self) -> None:
        """Model-written prose ships to the page, so it is held to the same rule."""
        offenders = []
        for finding in load("findings.json")["findings"]:
            for field in ("missing", "validation_note", "gap_note", "concept_label"):
                value = finding.get(field) or ""
                for hit in self.offences(f"{finding['id']}.{field}",
                                         self.scrub(value)):
                    offenders.append(hit)
        self.assertEqual(offenders, [])


class ShareCardTest(unittest.TestCase):
    """Every page that gets linked needs a share card that actually resolves.

    A posted link with no preview, next to sibling links that have one, reads as
    the unfinished project. The card is part of the build, so it is tested like
    the rest of it.
    """

    PAGES = {"index.html": "og-image", "tool.html": "og-tool"}
    BASE = "https://roanukz.github.io/should-my-agent-answer/"

    def png_size(self, path: Path) -> tuple:
        """Width and height straight out of the PNG header, no dependency."""
        import struct
        blob = path.read_bytes()
        self.assertEqual(blob[:8], b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG")
        width, height = struct.unpack(">II", blob[16:24])
        return width, height

    def test_both_cards_exist_as_svg_source_and_rendered_png(self) -> None:
        for stem in self.PAGES.values():
            self.assertTrue((ROOT / f"{stem}.svg").exists(), f"{stem}.svg missing")
            self.assertTrue((ROOT / f"{stem}.png").exists(), f"{stem}.png missing")

    def test_the_pngs_are_rendered_at_two_times(self) -> None:
        """1200x627 coordinate space, rendered at 2x so it stays sharp."""
        for stem in self.PAGES.values():
            self.assertEqual(self.png_size(ROOT / f"{stem}.png"), (2400, 1254), stem)

    def test_the_svg_declares_the_open_graph_coordinate_space(self) -> None:
        for stem in self.PAGES.values():
            svg = (ROOT / f"{stem}.svg").read_text(encoding="utf-8")
            self.assertIn('viewBox="0 0 1200 627"', svg, stem)

    def test_every_element_survives_the_square_crop(self) -> None:
        """LinkedIn's Featured section center crops, so nothing may sit outside
        the 540px center column at x 330 to 870."""
        import re
        for stem in self.PAGES.values():
            svg = (ROOT / f"{stem}.svg").read_text(encoding="utf-8")
            body = re.sub(r"<!--.*?-->", "", svg, flags=re.S)
            for match in re.finditer(r'<rect[^>]*?x="([\d.]+)"[^>]*?width="([\d.]+)"', body):
                x, w = float(match.group(1)), float(match.group(2))
                if w >= 1200:
                    continue  # the full-bleed background
                self.assertGreaterEqual(x, 330, f"{stem}: a rect starts left of the column")
                self.assertLessEqual(x + w, 870, f"{stem}: a rect ends right of the column")
            for match in re.finditer(r'<line[^>]*?x1="([\d.]+)"[^>]*?x2="([\d.]+)"', body):
                for value in (float(match.group(1)), float(match.group(2))):
                    self.assertGreaterEqual(value, 330, f"{stem}: a line leaves the column")
                    self.assertLessEqual(value, 870, f"{stem}: a line leaves the column")

    def test_the_cards_fetch_nothing(self) -> None:
        for stem in self.PAGES.values():
            svg = (ROOT / f"{stem}.svg").read_text(encoding="utf-8")
            for banned in ("<image", "xlink:href", "@import", "url(http", ".woff", "<use"):
                self.assertNotIn(banned, svg, f"{stem} reaches outside itself")

    def test_both_pages_name_their_author(self) -> None:
        for page in self.PAGES:
            html = (ROOT / page).read_text(encoding="utf-8")
            self.assertIn('<meta name="author" content="Roanuk Zaman" />', html, page)

    def test_each_page_points_at_its_own_card_with_dimensions_and_alt(self) -> None:
        for page, stem in self.PAGES.items():
            html = (ROOT / page).read_text(encoding="utf-8")
            self.assertIn(f'content="{self.BASE}{stem}.png"', html, page)
            self.assertIn('property="og:image:width" content="2400"', html, page)
            self.assertIn('property="og:image:height" content="1254"', html, page)
            self.assertIn('property="og:image:alt"', html, page)
            self.assertIn('name="twitter:card" content="summary_large_image"', html, page)
            self.assertIn('property="og:url"', html, page)
