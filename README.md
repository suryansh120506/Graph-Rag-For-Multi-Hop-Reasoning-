# Multi-Hop Graph RAG Analyzer

A local, API-key-free Graph RAG system for supply-chain risk analysis. Unstructured documents are turned into a typed knowledge graph (NetworkX). Questions are answered either by a direct 1-hop lookup or by a multi-hop traversal that follows how a disruption *propagates*. Every answer cites its sources, and the traversal path is drawn on an interactive pyvis graph inside a Streamlit dashboard.

| File | Responsibility |
|---|---|
| `graph_engine.py` | Ontology, `KnowledgeGraph` (MultiDiGraph wrapper), entity resolution, impact view, pyvis export |
| `extractor.py` | Prompt → mock LLM → validated JSON → graph. Handles coreference and subject ellipsis |
| `multi_hop.py` | Entity linking, intent routing, `graph_search`, path scoring, grounded mock-LLM synthesis |
| `app.py` | Streamlit dashboard: graph (top), chat (bottom) |

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py        # dashboard at http://localhost:8501
python multi_hop.py         # terminal demo (also writes graph.html)
python extractor.py         # show extracted triples per document
pytest -q                   # test suite
```

The graph view loads `vis-network` from a CDN, so the browser needs internet access the first time it renders the graph. Everything else runs offline.

## How it works

1. **Extraction**: each document is wrapped in a schema-constrained prompt. `MockLLMExtractor.complete()` returns JSON just like a real model would, and `KnowledgeExtractor` validates it before writing. To use a real model, implement one class with `complete(prompt) -> str`.
2. **Impact view**: each relation type declares a propagation direction (`SUPPLIES` forward, `USES` reverse, `COMPETES_WITH` none). Traversal runs on a view where every edge points the way risk flows.
3. **`graph_search`**: breadth-first enumeration of simple paths up to 3 hops. Path score = Π(confidence × impact weight).
4. **Synthesis**: the retrieved sub-graph becomes a grounded prompt (visible in the UI). The mock generator writes the insight, a competitor/BOM side-analysis and mitigation actions, using only retrieved facts.

## Example questions

- `Who supplies batteries to TechNova?` → 1-hop: ApexCorp [Doc 1]
- `How does the European strike impact the NovaPhone?` → 3-hop: `European Dockworker Strike -[DELAYS]-> GlobalLogistics -[MANAGES_SHIPPING_FOR]-> TechNova -[MANUFACTURES]-> NovaPhone`
- `What are the downstream impacts of the dockworker strike?` → impact report grouped by hop
- `How is OmegaCorp related to the NovaPhone?` → structural (non-causal) connection
