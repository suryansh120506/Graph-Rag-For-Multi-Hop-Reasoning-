"""
graph_engine.py
===============

Knowledge-graph management layer for the Multi-Hop Graph RAG Analyzer.

Responsibilities
----------------
* Own the ontology (entity types, relation types, and how risk *propagates*
  along each relation).
* Wrap a directed ``networkx.MultiDiGraph`` with safe, idempotent
  ``add_entity`` / ``add_relationship`` methods that merge provenance when the
  same fact is extracted from several documents.
* Resolve free-text mentions ("European strike", "novaphone") to canonical
  node IDs (alias index -> token overlap -> fuzzy string match).
* Produce an *impact view* of the graph in which every edge points in the
  direction that a disruption flows. Multi-hop traversal runs on this view.
* Export the current graph state to an interactive ``pyvis`` HTML file, with
  optional highlighting of a traversal path.

A MultiDiGraph (rather than a DiGraph) is used so that two entities may share
more than one typed relationship (e.g. A SUPPLIES B *and* A COMPETES_WITH B)
without one silently overwriting the other. Edge keys are relation names.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

import networkx as nx
from pyvis.network import Network

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Ontology
# --------------------------------------------------------------------------- #
class EntityType(str, Enum):
    COMPANY = "Company"
    COMPONENT = "Component"
    PRODUCT = "Product"
    RISK = "Risk"
    REGION = "Region"
    UNKNOWN = "Entity"


class RelationType(str, Enum):
    SUPPLIES = "SUPPLIES"
    PRODUCES = "PRODUCES"
    USES = "USES"
    COMPONENT_OF = "COMPONENT_OF"
    MANUFACTURES = "MANUFACTURES"
    MANAGES_SHIPPING_FOR = "MANAGES_SHIPPING_FOR"
    DELAYS = "DELAYS"
    OCCURS_IN = "OCCURS_IN"
    COMPETES_WITH = "COMPETES_WITH"
    EXPOSED_TO = "EXPOSED_TO"


class Propagation(str, Enum):
    """Direction in which a disruption travels along a stored edge."""

    FORWARD = "forward"   # trouble at source hurts target
    REVERSE = "reverse"   # trouble at target hurts source
    NONE = "none"         # contextual edge; does not transmit impact


@dataclass(frozen=True)
class RelationSpec:
    phrase: str           # natural-language verb for source -> target
    inverse_phrase: str   # natural-language verb for target -> source
    propagation: Propagation
    impact_weight: float  # prior on how strongly impact transmits (0..1)
    symmetric: bool = False


RELATION_SCHEMA: dict[RelationType, RelationSpec] = {
    RelationType.SUPPLIES: RelationSpec("supplies", "is supplied by", Propagation.FORWARD, 0.80),
    RelationType.PRODUCES: RelationSpec("produces", "is produced by", Propagation.FORWARD, 0.80),
    RelationType.USES: RelationSpec("uses", "is used by", Propagation.REVERSE, 0.85),
    RelationType.COMPONENT_OF: RelationSpec("is a component of", "contains", Propagation.FORWARD, 0.90),
    RelationType.MANUFACTURES: RelationSpec("manufactures", "is manufactured by", Propagation.FORWARD, 0.90),
    RelationType.MANAGES_SHIPPING_FOR: RelationSpec(
        "manages shipping for", "relies on shipping from", Propagation.FORWARD, 0.85
    ),
    RelationType.DELAYS: RelationSpec("delays", "is delayed by", Propagation.FORWARD, 0.75),
    RelationType.OCCURS_IN: RelationSpec("occurs in", "is the location of", Propagation.NONE, 0.0),
    RelationType.COMPETES_WITH: RelationSpec(
        "competes with", "competes with", Propagation.NONE, 0.0, symmetric=True
    ),
    RelationType.EXPOSED_TO: RelationSpec("is exposed to", "is a risk to", Propagation.REVERSE, 0.80),
}

# Severity adverbs captured by the extractor override the default impact weight.
SEVERITY_WEIGHTS: dict[str, float] = {
    "slightly": 0.35,
    "moderately": 0.55,
    "significantly": 0.80,
    "severely": 0.90,
}

NODE_STYLE: dict[EntityType, dict[str, Any]] = {
    EntityType.COMPANY: {"color": "#4C78A8", "shape": "dot", "size": 30},
    EntityType.COMPONENT: {"color": "#F58518", "shape": "diamond", "size": 26},
    EntityType.PRODUCT: {"color": "#54A24B", "shape": "star", "size": 34},
    EntityType.RISK: {"color": "#E45756", "shape": "triangle", "size": 32},
    EntityType.REGION: {"color": "#72B7B2", "shape": "square", "size": 22},
    EntityType.UNKNOWN: {"color": "#9D9D9D", "shape": "dot", "size": 22},
}

HIGHLIGHT_COLOR = "#D62728"
DIMMED_EDGE_COLOR = "#CBD5E1"
EDGE_COLOR = "#94A3B8"

# Canvas units for the deterministic layout (the viewer auto-fits afterwards).
LAYOUT_SCALE_X = 700.0
LAYOUT_SCALE_Y = 360.0
MIN_NODE_GAP = 120.0          # canvas units; keeps shapes and wrapped labels from touching
MAX_RELAX_NODES = 300         # skip the O(n^2) separation pass on very large graphs
LAYOUT_SEED = 7
LABEL_WRAP = 18


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EdgeRecord:
    """An immutable snapshot of one stored edge, safe to hand to callers."""

    source: str
    relation: RelationType
    target: str
    confidence: float
    impact_weight: float
    evidence: tuple[str, ...]
    doc_ids: tuple[str, ...]
    attributes: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.relation.value)

    @property
    def spec(self) -> RelationSpec:
        return RELATION_SCHEMA[self.relation]


class EntityNotFoundError(KeyError):
    """Raised when a mention cannot be resolved to any node in the graph."""


# --------------------------------------------------------------------------- #
# Text normalisation helpers (shared by resolver and question linker)
# --------------------------------------------------------------------------- #
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_key(text: str) -> str:
    """Lower-case, strip punctuation/possessives, and light-stem every token."""
    text = text.lower().replace("'s", "")
    text = _NON_ALNUM.sub(" ", text)
    return " ".join(_stem(t) for t in text.split())


# --------------------------------------------------------------------------- #
# Knowledge graph
# --------------------------------------------------------------------------- #
class KnowledgeGraph:
    """Typed, provenance-aware wrapper around a ``networkx.MultiDiGraph``."""

    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._alias_index: dict[str, str] = {}
        self._impact_view: nx.MultiDiGraph | None = None

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def add_entity(
        self,
        name: str,
        entity_type: EntityType | str = EntityType.UNKNOWN,
        aliases: Iterable[str] = (),
        doc_id: str | None = None,
        **attributes: Any,
    ) -> str:
        """Add (or merge into) an entity node and return its canonical ID.

        If ``name`` or any alias already resolves to a node, that node is
        enriched instead of creating a duplicate. A known type is never
        downgraded to ``UNKNOWN``.
        """
        name = name.strip()
        if not name:
            raise ValueError("Entity name must be non-empty.")
        entity_type = EntityType(entity_type)

        node_id = self._lookup_alias(name) or next(
            (nid for a in aliases if (nid := self._lookup_alias(a))), None
        ) or name

        if node_id in self._g:
            data = self._g.nodes[node_id]
            if data["entity_type"] == EntityType.UNKNOWN and entity_type != EntityType.UNKNOWN:
                data["entity_type"] = entity_type
            data["aliases"].update(a.strip() for a in aliases if a.strip())
            if doc_id:
                data["doc_ids"].add(doc_id)
            data.update(attributes)
        else:
            self._g.add_node(
                node_id,
                entity_type=entity_type,
                aliases={a.strip() for a in aliases if a.strip()},
                doc_ids={doc_id} if doc_id else set(),
                **attributes,
            )
            logger.debug("Added entity %s (%s)", node_id, entity_type.value)

        for alias in (node_id, name, *aliases):
            self._register_alias(alias, node_id)
        self._invalidate()
        return node_id

    def add_relationship(
        self,
        source: str,
        relation: RelationType | str,
        target: str,
        *,
        confidence: float = 1.0,
        evidence: str | None = None,
        doc_id: str | None = None,
        impact_weight: float | None = None,
        **attributes: Any,
    ) -> EdgeRecord:
        """Add (or merge into) a typed directed edge.

        Unknown endpoints are created as ``UNKNOWN`` entities. Re-asserting an
        existing fact merges its provenance and keeps the max confidence, so
        ingestion is idempotent.
        """
        relation = RelationType(relation)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        src = self._lookup_alias(source) or self.add_entity(source)
        dst = self._lookup_alias(target) or self.add_entity(target)
        if src == dst:
            raise ValueError(f"Self-loop rejected: {src} -{relation.value}-> {dst}")

        spec = RELATION_SCHEMA[relation]
        weight = spec.impact_weight if impact_weight is None else impact_weight
        key = relation.value

        if self._g.has_edge(src, dst, key=key):
            data = self._g.edges[src, dst, key]
            data["confidence"] = max(data["confidence"], confidence)
            data["impact_weight"] = max(data["impact_weight"], weight)
            if evidence:
                data["evidence"].add(evidence)
            if doc_id:
                data["doc_ids"].add(doc_id)
            data["attributes"].update(attributes)
        else:
            self._g.add_edge(
                src,
                dst,
                key=key,
                relation=relation,
                confidence=confidence,
                impact_weight=weight,
                evidence={evidence} if evidence else set(),
                doc_ids={doc_id} if doc_id else set(),
                attributes=dict(attributes),
            )
            logger.debug("Added edge %s -%s-> %s", src, key, dst)

        self._invalidate()
        return self._record(src, dst, key)

    def clear(self) -> None:
        self._g.clear()
        self._alias_index.clear()
        self._invalidate()

    # ------------------------------------------------------------------ #
    # Read API
    # ------------------------------------------------------------------ #
    @property
    def graph(self) -> nx.MultiDiGraph:
        """Read-only view of the underlying graph (for analytics)."""
        return self._g.copy(as_view=True)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._g

    def __len__(self) -> int:
        return self._g.number_of_nodes()

    def nodes(self) -> list[str]:
        return list(self._g.nodes)

    def entity_type(self, node_id: str) -> EntityType:
        return self._g.nodes[node_id]["entity_type"]

    def node_data(self, node_id: str) -> dict[str, Any]:
        data = self._g.nodes[node_id]
        return {
            "id": node_id,
            "entity_type": data["entity_type"].value,
            "aliases": sorted(data["aliases"]),
            "doc_ids": sorted(data["doc_ids"]),
        }

    def edges(self) -> Iterator[EdgeRecord]:
        for u, v, k in self._g.edges(keys=True):
            yield self._record(u, v, k)

    def neighbors(
        self,
        node_id: str,
        relation: RelationType | None = None,
        direction: str = "both",
    ) -> list[EdgeRecord]:
        """Return the 1-hop edges incident to ``node_id``.

        ``direction`` is ``"out"``, ``"in"`` or ``"both"``. Symmetric relations
        (e.g. COMPETES_WITH) are returned for either direction.
        """
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in' or 'both'")
        if node_id not in self._g:
            raise EntityNotFoundError(node_id)

        records: list[EdgeRecord] = []
        if direction in {"out", "both"} or relation and RELATION_SCHEMA[relation].symmetric:
            records += [self._record(u, v, k) for u, v, k in self._g.out_edges(node_id, keys=True)]
        if direction in {"in", "both"} or relation and RELATION_SCHEMA[relation].symmetric:
            records += [self._record(u, v, k) for u, v, k in self._g.in_edges(node_id, keys=True)]
        if relation is not None:
            records = [r for r in records if r.relation == relation]
        # de-duplicate while preserving order
        return list({r.key: r for r in records}.values())

    def stats(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        for _, data in self._g.nodes(data=True):
            t = data["entity_type"].value
            type_counts[t] = type_counts.get(t, 0) + 1
        docs = {d for _, _, data in self._g.edges(data=True) for d in data["doc_ids"]}
        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
            "documents": len(docs),
            "density": round(nx.density(self._g), 3) if len(self._g) > 1 else 0.0,
            "weakly_connected_components": nx.number_weakly_connected_components(self._g)
            if len(self._g)
            else 0,
            "entity_types": type_counts,
        }

    def to_triples(self) -> list[dict[str, Any]]:
        """Flat, tabular representation of every edge (for UIs / export)."""
        return [
            {
                "source": r.source,
                "relation": r.relation.value,
                "target": r.target,
                "confidence": round(r.confidence, 2),
                "docs": ", ".join(r.doc_ids),
                "evidence": " | ".join(r.evidence),
            }
            for r in self.edges()
        ]

    def to_json(self) -> str:
        return json.dumps(
            {"nodes": [self.node_data(n) for n in self._g.nodes], "edges": self.to_triples()},
            indent=2,
        )

    # ------------------------------------------------------------------ #
    # Entity resolution
    # ------------------------------------------------------------------ #
    def resolve_entity(self, mention: str, fuzzy_cutoff: float = 0.75) -> str | None:
        """Map a free-text mention to a canonical node ID, or ``None``.

        Strategy (cheapest / most precise first):
          1. exact alias match on the normalised key;
          2. token containment - every query token appears in the node's keys;
          3. ``difflib`` fuzzy match above ``fuzzy_cutoff``.
        """
        if not mention or not mention.strip():
            return None
        if hit := self._lookup_alias(mention):
            return hit

        query_tokens = set(normalize_key(mention).split())
        best: tuple[float, str] | None = None
        for key, node_id in self._alias_index.items():
            key_tokens = set(key.split())
            if query_tokens and query_tokens <= key_tokens:
                score = len(query_tokens) / len(key_tokens)
                if best is None or score > best[0]:
                    best = (score, node_id)
        if best:
            return best[1]

        close = difflib.get_close_matches(
            normalize_key(mention), list(self._alias_index), n=1, cutoff=fuzzy_cutoff
        )
        return self._alias_index[close[0]] if close else None

    def require_entity(self, mention: str) -> str:
        node_id = self.resolve_entity(mention)
        if node_id is None:
            raise EntityNotFoundError(f"No entity in the graph matches '{mention}'.")
        return node_id

    def alias_index(self) -> dict[str, str]:
        """Normalised alias -> node ID (used by the question entity-linker)."""
        return dict(self._alias_index)

    # ------------------------------------------------------------------ #
    # Impact view
    # ------------------------------------------------------------------ #
    def impact_view(self) -> nx.MultiDiGraph:
        """A graph whose edges point in the direction impact propagates.

        FORWARD edges are copied as-is, REVERSE edges are flipped and NONE
        edges are dropped. Every edge keeps a pointer (``origin``) to the
        stored edge it came from plus a ``reversed`` flag, so traversal output
        can always be explained in terms of the original facts.
        """
        if self._impact_view is not None:
            return self._impact_view

        view = nx.MultiDiGraph()
        view.add_nodes_from(self._g.nodes(data=True))
        for u, v, k, data in self._g.edges(keys=True, data=True):
            prop = RELATION_SCHEMA[data["relation"]].propagation
            if prop is Propagation.NONE:
                continue
            a, b, rev = (u, v, False) if prop is Propagation.FORWARD else (v, u, True)
            view.add_edge(
                a,
                b,
                key=f"{k}{'~rev' if rev else ''}",
                origin=(u, v, k),
                reversed=rev,
                weight=data["confidence"] * data["impact_weight"],
            )
        self._impact_view = view
        return view

    def edge_record(self, source: str, target: str, relation: str) -> EdgeRecord:
        return self._record(source, target, relation)

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #
    def export_html(
        self,
        path: str | Path = "graph.html",
        highlight_edges: Iterable[tuple[str, str, str]] = (),
        height: str = "640px",
        cdn_resources: str = "remote",
    ) -> Path:
        """Render the graph to an interactive pyvis HTML file.

        ``highlight_edges`` is an iterable of stored-edge keys
        ``(source, target, relation)`` - typically the path returned by a
        multi-hop search. Those edges are drawn thick and red and every other
        edge is dimmed so the path stands out.

        Usability choices (all aimed at embedding in Streamlit):

        * **Deterministic layout.** Node coordinates are computed here with a
          seeded spring layout and physics is off, so the graph looks the same
          on every rerun instead of being re-scrambled by each question. The
          viewer remembers positions the user dragged (per browser).
        * **No page chrome.** pyvis' empty headings / card padding are hidden
          and the canvas fills the iframe, so there is no inner scrollbar.
        * **Scroll-wheel zoom is off by default** (it hijacks page scrolling);
          the in-page toolbar offers zoom, fit, reset and label toggles.

        ``cdn_resources="remote"`` loads vis-network from a CDN, which is the
        mode that renders reliably inside Streamlit's sandboxed iframe
        (``"local"`` references ``lib/`` files the iframe cannot reach). The
        file is written as UTF-8 explicitly, which avoids pyvis'
        locale-encoding bug on Windows.
        """
        highlight = set(highlight_edges)
        highlight_nodes = {n for u, v, _ in highlight for n in (u, v)}
        positions = self._layout_positions()

        net = Network(
            height=height,
            width="100%",
            directed=True,
            bgcolor="#FFFFFF",
            font_color="#111827",
            cdn_resources=cdn_resources,
        )

        for node_id, data in self._g.nodes(data=True):
            etype: EntityType = data["entity_type"]
            style = NODE_STYLE[etype]
            on_path = node_id in highlight_nodes
            x, y = positions[node_id]
            tooltip = "\n".join(
                [
                    f"{node_id}  [{etype.value}]",
                    f"Aliases: {', '.join(sorted(data['aliases'])) or '-'}",
                    f"Sources: {', '.join(sorted(data['doc_ids'])) or '-'}",
                    f"Degree: in={self._g.in_degree(node_id)} out={self._g.out_degree(node_id)}",
                ]
            )
            net.add_node(
                node_id,
                label=textwrap.fill(node_id, LABEL_WRAP, break_long_words=False),
                title=tooltip,
                x=x,
                y=y,
                color={
                    "background": style["color"],
                    "border": HIGHLIGHT_COLOR if on_path else "#FFFFFF",
                    "highlight": {"background": style["color"], "border": "#111827"},
                    "hover": {"background": style["color"], "border": "#111827"},
                },
                borderWidth=5 if on_path else 2,
                shape=style["shape"],
                size=style["size"] + (8 if on_path else 0),
                group=etype.value,
            )

        seen_direction: dict[tuple[str, str], int] = {}
        for u, v, k, data in self._g.edges(keys=True, data=True):
            rel: RelationType = data["relation"]
            spec = RELATION_SCHEMA[rel]
            on_path = (u, v, k) in highlight
            attrs = data["attributes"]
            tooltip = "\n".join(
                filter(
                    None,
                    [
                        f"{u} -[{rel.value}]-> {v}",
                        f"confidence={data['confidence']:.2f}  impact_weight={data['impact_weight']:.2f}",
                        f"propagation={spec.propagation.value}",
                        f"attributes={attrs}" if attrs else "",
                        f"sources={', '.join(sorted(data['doc_ids']))}",
                        *(f'"{e}"' for e in sorted(data["evidence"])),
                    ],
                )
            )
            # Parallel edges between the same pair get distinct curvature so
            # their labels do not sit on top of one another.
            rank = seen_direction.get((u, v), 0)
            seen_direction[(u, v)] = rank + 1
            if self._g.number_of_edges(u, v) + self._g.number_of_edges(v, u) > 1:
                smooth: dict[str, Any] = {
                    "enabled": True,
                    "type": "curvedCW",
                    "roundness": round(0.18 + 0.2 * rank, 2),
                }
            else:
                smooth = {"enabled": False}

            net.add_edge(
                u,
                v,
                label=rel.value.lower().replace("_", " "),
                title=tooltip,
                color={
                    "color": HIGHLIGHT_COLOR if on_path else (DIMMED_EDGE_COLOR if highlight else EDGE_COLOR),
                    "highlight": "#111827",
                    "hover": "#111827",
                },
                width=6 if on_path else 2,
                dashes=spec.propagation is Propagation.NONE,
                arrows="to" if not spec.symmetric else "to, from",
                smooth=smooth,
            )

        net.set_options(json.dumps(_PYVIS_OPTIONS))
        page = _inject_viewer(net.generate_html(), sig="|".join(sorted(self._g.nodes)))
        out = Path(path)
        out.write_text(page, encoding="utf-8")
        logger.info("Exported graph (%d nodes, %d edges) to %s", len(self._g), self._g.size(), out)
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _layout_positions(self) -> dict[str, tuple[float, float]]:
        """Deterministic 2-D coordinates for every node (seeded spring layout)."""
        n = len(self._g)
        if n == 0:
            return {}
        if n == 1:
            return {next(iter(self._g.nodes)): (0.0, 0.0)}
        simple = nx.Graph()
        simple.add_nodes_from(self._g.nodes)
        simple.add_edges_from((u, v) for u, v in self._g.edges())
        try:
            raw = nx.spring_layout(simple, k=1.6 / math.sqrt(n), iterations=300, seed=LAYOUT_SEED)
        except ImportError:  # numpy missing: fall back to a ring, still deterministic
            raw = nx.circular_layout(simple)
        coords = {node: [float(x) * LAYOUT_SCALE_X, float(y) * LAYOUT_SCALE_Y] for node, (x, y) in raw.items()}
        if n <= MAX_RELAX_NODES:
            _separate(coords, MIN_NODE_GAP)
        return {node: (round(x, 1), round(y, 1)) for node, (x, y) in coords.items()}

    def _register_alias(self, alias: str, node_id: str) -> None:
        key = normalize_key(alias)
        if not key:
            return
        existing = self._alias_index.get(key)
        if existing and existing != node_id:
            logger.warning("Alias '%s' already maps to '%s'; not remapping to '%s'", alias, existing, node_id)
            return
        self._alias_index[key] = node_id
        compact = key.replace(" ", "")
        self._alias_index.setdefault(compact, node_id)

    def _lookup_alias(self, mention: str) -> str | None:
        key = normalize_key(mention)
        return self._alias_index.get(key) or self._alias_index.get(key.replace(" ", ""))

    def _record(self, u: str, v: str, k: str) -> EdgeRecord:
        data = self._g.edges[u, v, k]
        return EdgeRecord(
            source=u,
            relation=data["relation"],
            target=v,
            confidence=data["confidence"],
            impact_weight=data["impact_weight"],
            evidence=tuple(sorted(data["evidence"])),
            doc_ids=tuple(sorted(data["doc_ids"])),
            attributes=dict(data["attributes"]),
        )

    def _invalidate(self) -> None:
        self._impact_view = None


def _separate(coords: dict[str, list[float]], min_gap: float, rounds: int = 60) -> None:
    """Push apart any two points closer than ``min_gap`` (in place, deterministic)."""
    ids = list(coords)
    for _ in range(rounds):
        moved = False
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                dx = coords[b][0] - coords[a][0]
                dy = coords[b][1] - coords[a][1]
                dist = math.hypot(dx, dy)
                if dist >= min_gap:
                    continue
                if dist < 1e-6:  # coincident: separate along a fixed axis
                    dx, dy, dist = 1.0, 0.0, 1.0
                push = (min_gap - dist) / 2 + 0.5
                ux, uy = dx / dist, dy / dist
                coords[a][0] -= ux * push
                coords[a][1] -= uy * push
                coords[b][0] += ux * push
                coords[b][1] += uy * push
                moved = True
        if not moved:
            break


_FONT_FACE = "Arial,Helvetica,sans-serif"  # pyvis strips spaces from the options string

_PYVIS_OPTIONS: dict[str, Any] = {
    "nodes": {
        "font": {"size": 18, "face": _FONT_FACE, "color": "#111827", "strokeWidth": 5, "strokeColor": "#FFFFFF"},
        "shadow": {"enabled": True, "size": 8, "x": 0, "y": 2},
    },
    "edges": {
        "font": {
            "size": 13,
            "face": _FONT_FACE,
            "align": "middle",
            "color": "#374151",
            "strokeWidth": 0,
            "background": "rgba(255,255,255,0.92)",
        },
        "smooth": {"enabled": False},
        "arrows": {"to": {"scaleFactor": 0.8}},
        "hoverWidth": 1.5,
        "selectionWidth": 2,
    },
    # Physics is off by default (stable, draggable layout). The toolbar's
    # "Springy" toggle enables it using these parameters.
    "physics": {
        "enabled": False,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -90,
            "centralGravity": 0.01,
            "springLength": 190,
            "springConstant": 0.05,
            "avoidOverlap": 0.8,
        },
        "stabilization": {"enabled": False},
    },
    "interaction": {
        "hover": True,
        "tooltipDelay": 100,
        "multiselect": True,
        "zoomView": False,
        "navigationButtons": False,
        "keyboard": False,
    },
}


# --------------------------------------------------------------------------- #
# Viewer chrome injected into the pyvis page
# --------------------------------------------------------------------------- #
_VIEWER_CSS = """
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #FFFFFF; }
  center, h1 { display: none !important; }
  .card { border: 0 !important; border-radius: 0 !important; height: 100vh; }
  .card-body { padding: 0 !important; }
  #mynetwork { float: none !important; width: 100% !important; height: 100vh !important;
               box-sizing: border-box; border: 1px solid #E5E7EB !important; border-radius: 8px; }
  #gr-controls { position: fixed; top: 10px; left: 10px; z-index: 20; display: flex; flex-wrap: wrap; gap: 6px;
                 max-width: calc(100% - 20px); font-family: Arial, Helvetica, sans-serif; }
  #gr-controls button { font: 600 13px Arial, Helvetica, sans-serif; padding: 6px 11px; cursor: pointer;
                        color: #1F2937; background: rgba(255,255,255,0.96); border: 1px solid #CBD5E1;
                        border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,.08); }
  #gr-controls button:hover { background: #F1F5F9; }
  #gr-controls button.on { color: #FFFFFF; background: #1F2937; border-color: #1F2937; }
  #gr-hint { position: fixed; left: 12px; bottom: 10px; z-index: 20; pointer-events: none;
             font: 12px Arial, Helvetica, sans-serif; color: #475569; background: rgba(255,255,255,0.9);
             padding: 3px 8px; border-radius: 6px; }
</style>
"""

_VIEWER_BODY = """
<div id="gr-controls">
  <button data-act="zoomin" title="Zoom in">+</button>
  <button data-act="zoomout" title="Zoom out">&minus;</button>
  <button data-act="fit" title="Fit the whole graph in view">Fit</button>
  <button data-act="reset" title="Restore the original layout">Reset layout</button>
  <button data-act="labels" class="on" title="Show or hide relationship labels">Edge labels</button>
  <button data-act="scroll" title="Let the mouse wheel zoom the graph (otherwise it scrolls the page)">Scroll to zoom</button>
  <button data-act="springy" title="Let nodes pull their neighbours when dragged">Springy</button>
</div>
<div id="gr-hint">Drag nodes to move them &middot; drag the background to pan &middot; double-click a node to focus, the background to fit</div>
<script type="text/javascript">
(function () {
  if (typeof network === "undefined" || typeof nodes === "undefined") { return; }
  var STORE_KEY = "graphrag.positions.v1";
  var SIG = "__SIG__";
  var base = network.getPositions();

  function finite(p) { return p && isFinite(p.x) && isFinite(p.y); }
  function applyPositions(pos) {
    Object.keys(pos).forEach(function (id) {
      if (finite(pos[id]) && nodes.get(id) !== null) { network.moveNode(id, pos[id].x, pos[id].y); }
    });
  }
  function loadSaved() {
    try {
      var raw = window.localStorage.getItem(STORE_KEY);
      if (!raw) { return null; }
      var saved = JSON.parse(raw);
      return saved && saved.sig === SIG ? saved.pos : null;   // ignore layouts of a different node set
    } catch (e) { return null; }
  }
  function save() {
    try { window.localStorage.setItem(STORE_KEY, JSON.stringify({ sig: SIG, pos: network.getPositions() })); } catch (e) {}
  }
  function clearSaved() { try { window.localStorage.removeItem(STORE_KEY); } catch (e) {} }
  function fit(ms) { network.fit({ animation: ms ? { duration: ms, easingFunction: "easeInOutQuad" } : false }); }

  var saved = loadSaved();
  if (saved) { applyPositions(saved); }
  fit(0);
  setTimeout(function () { fit(0); }, 120);

  network.on("dragEnd", function (p) { if (p.nodes && p.nodes.length) { save(); } });
  network.on("doubleClick", function (p) {
    if (p.nodes && p.nodes.length) {
      network.focus(p.nodes[0], { scale: 1.3, animation: { duration: 400, easingFunction: "easeInOutQuad" } });
    } else { fit(400); }
  });

  var buttons = {};
  Array.prototype.forEach.call(document.querySelectorAll("#gr-controls button"), function (b) { buttons[b.dataset.act] = b; });
  function zoom(factor) { network.moveTo({ scale: network.getScale() * factor, animation: { duration: 200 } }); }
  var state = { labels: true, scroll: false, springy: false };

  function setLabels(on) {
    state.labels = on;
    network.setOptions({ edges: { font: {
      color: on ? "#374151" : "rgba(0,0,0,0)",
      background: on ? "rgba(255,255,255,0.92)" : "rgba(0,0,0,0)"
    } } });
    buttons.labels.classList.toggle("on", on);
  }
  function setScroll(on) {
    state.scroll = on;
    network.setOptions({ interaction: { zoomView: on } });
    buttons.scroll.classList.toggle("on", on);
  }
  function setSpringy(on) {
    state.springy = on;
    network.setOptions({ physics: { enabled: on } });
    buttons.springy.classList.toggle("on", on);
    if (!on) { save(); }
  }

  var actions = {
    zoomin: function () { zoom(1.3); },
    zoomout: function () { zoom(1 / 1.3); },
    fit: function () { fit(400); },
    reset: function () {
      if (state.springy) { setSpringy(false); }
      clearSaved(); applyPositions(base); fit(400);
    },
    labels: function () { setLabels(!state.labels); },
    scroll: function () { setScroll(!state.scroll); },
    springy: function () { setSpringy(!state.springy); }
  };
  Object.keys(buttons).forEach(function (act) { buttons[act].addEventListener("click", actions[act]); });
})();
</script>
"""


def _inject_viewer(page: str, sig: str) -> str:
    """Add the toolbar, CSS and persistence script to pyvis' generated page."""
    safe_sig = json.dumps(sig)[1:-1].replace("</", "<\\/")  # embed inside a JS string literal
    body = _VIEWER_BODY.replace("__SIG__", safe_sig)
    if "</head>" not in page or "</body>" not in page:
        logger.warning("Unexpected pyvis template; skipping viewer enhancements")
        return page
    page = page.replace("</head>", _VIEWER_CSS + "</head>", 1)
    head, _, tail = page.rpartition("</body>")
    return head + body + "</body>" + tail
