"""End-to-end tests for extraction, traversal and question answering.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor import bootstrap_knowledge_graph  # noqa: E402
from graph_engine import EntityNotFoundError, EntityType, KnowledgeGraph, RelationType  # noqa: E402
from multi_hop import GraphRAGPipeline, QueryMode, graph_search, link_entities  # noqa: E402


@pytest.fixture()
def kg() -> KnowledgeGraph:
    graph, _ = bootstrap_knowledge_graph()
    return graph


def test_extraction_builds_expected_facts(kg: KnowledgeGraph) -> None:
    keys = {e.key for e in kg.edges()}
    assert ("ApexCorp", "TechNova", "SUPPLIES") in keys
    assert ("TechNova", "NovaPhone", "MANUFACTURES") in keys
    assert ("GlobalLogistics", "TechNova", "MANAGES_SHIPPING_FOR") in keys
    assert ("European Dockworker Strike", "GlobalLogistics", "DELAYS") in keys
    assert ("OmegaCorp", "TechNova", "COMPETES_WITH") in keys
    assert ("BetaTech", "OmegaCorp", "SUPPLIES") in keys
    # Coreference: "these batteries" resolved to Lithium Batteries.
    assert ("Lithium Batteries", "NovaPhone", "COMPONENT_OF") in keys
    assert kg.entity_type("European Dockworker Strike") is EntityType.RISK


def test_ingestion_is_idempotent(kg: KnowledgeGraph) -> None:
    from extractor import SEED_DOCUMENTS, KnowledgeExtractor

    before = kg.stats()
    KnowledgeExtractor(kg).ingest_many(SEED_DOCUMENTS)
    after = kg.stats()
    assert (before["nodes"], before["edges"]) == (after["nodes"], after["edges"])


def test_entity_resolution(kg: KnowledgeGraph) -> None:
    assert kg.resolve_entity("European Strike") == "European Dockworker Strike"
    assert kg.resolve_entity("novaphone") == "NovaPhone"
    assert kg.resolve_entity("TechNova's") == "TechNova"
    with pytest.raises(EntityNotFoundError):
        kg.require_entity("Acme Rockets")


def test_graph_search_finds_three_hop_path(kg: KnowledgeGraph) -> None:
    result = graph_search(kg, "European Strike", max_hops=3, target="NovaPhone")
    path = result.target_path
    assert path is not None
    assert path.nodes == ("European Dockworker Strike", "GlobalLogistics", "TechNova", "NovaPhone")
    assert [h.edge.relation for h in path.hops] == [
        RelationType.DELAYS,
        RelationType.MANAGES_SHIPPING_FOR,
        RelationType.MANUFACTURES,
    ]
    assert 0 < path.score < 1
    # Competitors are not impacted: COMPETES_WITH does not propagate impact.
    assert "OmegaCorp" not in result.impacted()


def test_graph_search_respects_horizon(kg: KnowledgeGraph) -> None:
    assert graph_search(kg, "European Strike", max_hops=2, target="NovaPhone").target_path is None


def test_one_hop_question(kg: KnowledgeGraph) -> None:
    resp = GraphRAGPipeline(kg).answer("Who supplies batteries to TechNova?")
    assert resp.mode is QueryMode.ONE_HOP
    assert resp.context.answer_nodes == ["ApexCorp"]
    assert "ApexCorp" in resp.answer and "[Doc 1]" in resp.answer


def test_multi_hop_question(kg: KnowledgeGraph) -> None:
    resp = GraphRAGPipeline(kg).answer("How does the European strike impact the NovaPhone?")
    assert resp.mode is QueryMode.MULTI_HOP
    assert resp.paths[0].render() == (
        "European Dockworker Strike -[DELAYS]-> GlobalLogistics "
        "-[MANAGES_SHIPPING_FOR]-> TechNova -[MANUFACTURES]-> NovaPhone"
    )
    assert "OmegaCorp" in resp.answer  # lateral competitive insight
    assert resp.highlight_edges == resp.paths[0].edge_keys


def test_unlinked_question_does_not_hallucinate(kg: KnowledgeGraph) -> None:
    resp = GraphRAGPipeline(kg).answer("What is the weather in Paris?")
    assert resp.mode is QueryMode.NO_MATCH


def test_entity_linking_prefers_longest_match(kg: KnowledgeGraph) -> None:
    mentions = link_entities("Did the dockworker strike in Europe hit TechNova?", kg)
    assert [m.node_id for m in mentions][:1] == ["European Dockworker Strike"]


def test_export_html(kg: KnowledgeGraph, tmp_path: Path) -> None:
    out = kg.export_html(tmp_path / "graph.html", highlight_edges={("ApexCorp", "TechNova", "SUPPLIES")})
    content = out.read_text(encoding="utf-8")
    assert "vis-network" in content and "ApexCorp" in content
