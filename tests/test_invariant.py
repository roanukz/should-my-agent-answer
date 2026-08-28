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
                cache[source] = extract.normalise(source.read_text(encoding="utf-8"))
            checked += 1
            span = extract.normalise(evidence["span"])
            if span in cache[source]:
                continue
            # The one extra normalisation, and only for a question span.
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
            window = extract.normalise("\n".join(lines[start - 1: end]))
            span = extract.normalise(evidence["span"])
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
        self.assertIn(f"{parallel} ordered pairs carry two", self.page)

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
        for path in ("vector", "graph"):
            self.assertEqual(summary[path]["wrong_confident"], 5)
        self.assertIn("p = 0.70", self.page)
        self.assertAlmostEqual(summary["paired"]["mcnemar_p_correct"], 0.7011, places=3)

    def test_both_figures_are_labelled_for_a_screen_reader(self) -> None:
        for token in ('role="img"', "schema-title", "schema-desc",
                      "nearmiss-title", "nearmiss-desc"):
            self.assertIn(token, self.page)

    def test_the_figures_reference_no_external_resource(self) -> None:
        import re
        for match in re.finditer(r'<(?:image|use)\b[^>]*>', self.page):
            self.fail(f"figure pulls an external resource: {match.group(0)[:80]}")
        self.assertNotIn("url(http", self.page)
