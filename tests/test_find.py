"""Fixture graphs with known findings, asserted.

The finders are the part of the pipeline where a subtle mistake is invisible: a
wrong result still looks like a list of plausible findings. So each type gets a
hand-built graph small enough to reason about completely, where the right answer
is known before the code runs.

Run with:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import find  # noqa: E402
import extract  # noqa: E402


def doc(doc_id: str, title: str) -> dict:
    return {"id": doc_id, "type": "Doc", "title": title,
            "path": f"docs/{doc_id}.md", "url": f"https://example.test/{doc_id}/"}


def section(sid: str, doc_id: str, heading: str, concepts=()) -> dict:
    return {"id": sid, "type": "Section", "doc_id": doc_id, "doc_title": doc_id,
            "heading": heading, "level": 2, "path": f"docs/{doc_id}.md",
            "lines": [1, 10], "url": f"https://example.test/{doc_id}/#{heading}",
            "text": "body", "code": [], "chars": 4}


def concept(cid: str, label: str) -> dict:
    return {"id": cid, "type": "Concept", "label": label, "kind": "term",
            "aliases": [], "surface_forms": [label]}


def question(qid: str, number: int, title: str) -> dict:
    return {"id": qid, "type": "Question", "number": number, "title": title,
            "body": "body", "created_at": "2026-01-01T00:00:00Z",
            "url": f"https://example.test/q/{number}", "answered": True,
            "raw_path": f"data/raw/discussions/{number}.md"}


def answer(qid: str, number: int, body: str, maintainer=True) -> dict:
    return {"id": f"ans:{number}", "type": "Answer", "question_id": qid,
            "author": "someone", "is_maintainer": maintainer,
            "author_association": "MEMBER", "body": body,
            "url": f"https://example.test/q/{number}#a"}


def edge(eid: str, etype: str, src: str, dst: str, span: str = "a justifying span",
         kind: str = "prose") -> dict:
    return {"id": eid, "type": etype, "from": src, "to": dst,
            "evidence": {"span": span, "path": "docs/x.md", "lines": [1, 1], "kind": kind},
            "extractor": "fixture", "confidence": "high"}


class NearMissTest(unittest.TestCase):
    """F1 fires only when demand, a dependency, and no definition all hold."""

    def setUp(self) -> None:
        self.nodes = [
            doc("d1", "Databases"),
            section("sec:d1#tables", "d1", "Create the tables"),
            section("sec:d1#models", "d1", "Create models"),
            concept("concept:engine", "engine"),
            concept("concept:session", "session"),
            concept("concept:model", "model"),
            question("q:1", 1, "How do I point the engine at Postgres?"),
            answer("q:1", 1, "You change the connection string."),
            question("q:2", 2, "Engine against MySQL?"),
            answer("q:2", 2, "Same, a different connection string."),
        ]
        self.edges = [
            # engine: asked about twice, required once, defined nowhere -> F1
            edge("e:1", "REQUIRES", "sec:d1#tables", "concept:engine"),
            edge("e:2", "ASKS_ABOUT", "q:1", "concept:engine"),
            edge("e:3", "ASKS_ABOUT", "q:2", "concept:engine"),
            # session: required and asked about, but also DEFINED -> not a finding
            edge("e:4", "REQUIRES", "sec:d1#tables", "concept:session"),
            edge("e:5", "DEFINES", "sec:d1#models", "concept:session"),
            edge("e:6", "ASKS_ABOUT", "q:1", "concept:session"),
            # model: required, never defined, but nobody asks -> not F1
            edge("e:7", "REQUIRES", "sec:d1#models", "concept:model"),
            edge("e:8", "ANSWERED_BY", "q:1", "ans:1"),
            edge("e:9", "ANSWERED_BY", "q:2", "ans:2"),
        ]
        self.idx = find.Index(self.nodes, self.edges)

    def test_finds_only_the_undefined_and_asked_about_concept(self) -> None:
        findings = find.find_near_miss(self.idx)
        self.assertEqual([f["concept"] for f in findings], ["concept:engine"])

    def test_demand_counts_distinct_questions(self) -> None:
        finding = find.find_near_miss(self.idx)[0]
        self.assertEqual(finding["demand"], 2)
        self.assertEqual(sorted(finding["questions"]), ["q:1", "q:2"])

    def test_proof_path_has_three_hops_and_names_the_absence(self) -> None:
        finding = find.find_near_miss(self.idx)[0]
        self.assertEqual(len(finding["proof_path"]), 3)
        self.assertEqual(finding["proof_path"][0]["hop"], "Question ASKS_ABOUT Concept")
        self.assertEqual(finding["proof_path"][1]["hop"], "Section REQUIRES Concept")
        absent = finding["proof_path"][2]
        self.assertEqual(absent["defines_edges_found"], 0)
        self.assertEqual(absent["sections_checked"], 2)

    def test_every_evidenced_hop_carries_a_span(self) -> None:
        for finding in find.find_near_miss(self.idx):
            for hop in finding["proof_path"]:
                if "evidence" in hop:
                    self.assertTrue(hop["evidence"]["span"])
                    self.assertTrue(hop["evidence"]["path"])
                    self.assertEqual(len(hop["evidence"]["lines"]), 2)

    def test_a_definition_anywhere_cancels_the_finding(self) -> None:
        edges = self.edges + [edge("e:10", "DEFINES", "sec:d1#models", "concept:engine")]
        findings = find.find_near_miss(find.Index(self.nodes, edges))
        self.assertEqual(findings, [])

    def test_prose_evidence_wins_the_representative_section(self) -> None:
        """Two sections require it; the one evidenced by prose is the one shown."""
        nodes = self.nodes + [section("sec:d1#extra", "d1", "Another")]
        edges = self.edges + [
            edge("e:11", "REQUIRES", "sec:d1#extra", "concept:engine",
                 span="engine = create_engine(url)", kind="code"),
        ]
        # Give the code-evidenced section the same shared-concept score as the
        # prose one so the tie is decided by evidence kind alone.
        edges.append(edge("e:12", "MENTIONS", "sec:d1#extra", "concept:session"))
        finding = find.find_near_miss(find.Index(nodes, edges))[0]
        self.assertEqual(finding["requires_evidence_kind"], "prose")


class OrphanTest(unittest.TestCase):
    """F2 fires on undefined concepts that more than one section leans on."""

    def setUp(self) -> None:
        self.nodes = [
            doc("d1", "Guide"),
            section("sec:d1#a", "d1", "A"),
            section("sec:d1#b", "d1", "B"),
            concept("concept:lifespan", "lifespan"),
            concept("concept:localname", "localname"),
            concept("concept:taught", "taught"),
        ]
        self.edges = [
            # two sections lean on it, nothing defines it -> F2
            edge("e:1", "REQUIRES", "sec:d1#a", "concept:lifespan"),
            edge("e:2", "MENTIONS", "sec:d1#b", "concept:lifespan"),
            # only one section -> almost always a name local to a code sample
            edge("e:3", "REQUIRES", "sec:d1#a", "concept:localname"),
            # defined, so not an orphan however often it is referenced
            edge("e:4", "REQUIRES", "sec:d1#a", "concept:taught"),
            edge("e:5", "MENTIONS", "sec:d1#b", "concept:taught"),
            edge("e:6", "DEFINES", "sec:d1#b", "concept:taught"),
        ]
        self.idx = find.Index(self.nodes, self.edges)

    def test_single_section_concepts_are_not_reported(self) -> None:
        findings = find.find_orphans(self.idx, set())
        self.assertEqual([f["concept"] for f in findings], ["concept:lifespan"])

    def test_near_miss_concepts_are_not_repeated_as_orphans(self) -> None:
        findings = find.find_orphans(self.idx, {"concept:lifespan"})
        self.assertEqual(findings, [])

    def test_proof_lists_the_in_edges_and_the_empty_define_set(self) -> None:
        finding = find.find_orphans(self.idx, set())[0]
        hops = finding["proof_path"]
        self.assertEqual(len(hops), 3)
        self.assertEqual(hops[-1]["defines_edges_found"], 0)
        self.assertEqual({h["hop"] for h in hops[:2]},
                         {"Section REQUIRES Concept", "Section MENTIONS Concept"})


class CollisionTest(unittest.TestCase):
    """F3 fires when the section without the answer looks at least as relevant."""

    def setUp(self) -> None:
        self.nodes = [
            doc("d1", "Guide"),
            section("sec:d1#vague", "d1", "Overview"),
            section("sec:d1#real", "d1", "The actual answer"),
            concept("concept:a", "a"),
            concept("concept:b", "b"),
            question("q:1", 1, "How do I do the thing?"),
            answer("q:1", 1, "Here is how."),
        ]
        self.edges = [
            edge("e:1", "ASKS_ABOUT", "q:1", "concept:a"),
            edge("e:2", "ASKS_ABOUT", "q:1", "concept:b"),
            edge("e:3", "MENTIONS", "sec:d1#vague", "concept:a"),
            edge("e:4", "MENTIONS", "sec:d1#vague", "concept:b"),
            edge("e:5", "DEFINES", "sec:d1#real", "concept:a"),
            edge("e:6", "DEFINES", "sec:d1#real", "concept:b"),
            edge("e:7", "ANSWERED_BY", "q:1", "ans:1"),
        ]
        self.idx = find.Index(self.nodes, self.edges)

    def test_reports_the_pair_when_overlap_ties(self) -> None:
        similarity = {"sec:d1#real": 0.62, "sec:d1#vague": 0.20}
        findings = find.find_collisions(self.idx, lambda q, s: similarity[s])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["answer_holder"]["id"], "sec:d1#real")
        self.assertEqual(findings[0]["rival"]["id"], "sec:d1#vague")
        self.assertEqual(findings[0]["demand"], 1)

    def test_silent_when_the_answer_section_is_clearly_more_relevant(self) -> None:
        """Drop the rival's overlap below the answer holder's and it is not a collision."""
        edges = [e for e in self.edges if e["id"] != "e:4"]
        idx = find.Index(self.nodes, edges)
        similarity = {"sec:d1#real": 0.62, "sec:d1#vague": 0.20}
        self.assertEqual(find.find_collisions(idx, lambda q, s: similarity[s]), [])

    def test_silent_when_neither_section_matches_the_answer_better(self) -> None:
        similarity = {"sec:d1#real": 0.40, "sec:d1#vague": 0.39}
        self.assertEqual(find.find_collisions(self.idx, lambda q, s: similarity[s]), [])


class RankingTest(unittest.TestCase):
    def test_findings_are_ranked_by_demand(self) -> None:
        nodes = [doc("d1", "Guide"), section("sec:d1#a", "d1", "A")]
        edges = []
        for i, (label, askers) in enumerate([("low", 1), ("high", 3), ("mid", 2)]):
            cid = f"concept:{label}"
            nodes.append(concept(cid, label))
            edges.append(edge(f"e:r{i}", "REQUIRES", "sec:d1#a", cid))
            for n in range(askers):
                qid = f"q:{label}{n}"
                nodes.append(question(qid, 100 + i * 10 + n, f"about {label}"))
                nodes.append(answer(qid, 100 + i * 10 + n, "reply"))
                edges.append(edge(f"e:q{i}{n}", "ASKS_ABOUT", qid, cid))
                edges.append(edge(f"e:x{i}{n}", "ANSWERED_BY", qid, f"ans:{100 + i * 10 + n}"))
        findings = find.find_near_miss(find.Index(nodes, edges))
        self.assertEqual([f["demand"] for f in findings], [3, 2, 1])
        self.assertEqual([f["id"] for f in findings], ["F1-0001", "F1-0002", "F1-0003"])


class SpanValidatorTest(unittest.TestCase):
    """The invariant is only worth anything if the validator actually rejects."""

    HAYSTACK = "We need to create the tables\nusing the engine we defined earlier.\nDone."

    def test_finds_a_span_that_wraps_across_lines(self) -> None:
        found = extract.locate_span(
            "create the tables using the engine", self.HAYSTACK, base_line=10)
        self.assertEqual(found, (10, 11))

    def test_tolerates_curly_quotes_and_case(self) -> None:
        found = extract.locate_span(
            "USING THE ENGINE WE DEFINED", self.HAYSTACK, base_line=1)
        self.assertEqual(found, (2, 2))

    def test_rejects_a_paraphrase(self) -> None:
        self.assertIsNone(extract.locate_span(
            "you must build the tables with the engine", self.HAYSTACK, base_line=1))

    def test_rejects_a_span_stitched_from_two_places(self) -> None:
        self.assertIsNone(extract.locate_span(
            "We need to create the tables Done.", self.HAYSTACK, base_line=1))

    def test_rejects_a_span_that_is_too_short(self) -> None:
        self.assertIsNone(extract.locate_span("Done.", self.HAYSTACK, base_line=1))

    def test_reports_the_real_line_range(self) -> None:
        found = extract.locate_span("Done.  and more", "a\nb\nDone. and more",
                                    base_line=100)
        self.assertEqual(found, (102, 102))


class WilsonTest(unittest.TestCase):
    def test_matches_the_published_interval(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
        import validate
        low, high = validate.wilson(43, 50)
        self.assertAlmostEqual(low, 0.7381, places=3)
        self.assertAlmostEqual(high, 0.9305, places=3)

    def test_never_leaves_the_unit_interval(self) -> None:
        import validate
        self.assertEqual(validate.wilson(50, 50)[1], 1.0)
        self.assertEqual(validate.wilson(0, 50)[0], 0.0)
        self.assertEqual(validate.wilson(0, 0), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
