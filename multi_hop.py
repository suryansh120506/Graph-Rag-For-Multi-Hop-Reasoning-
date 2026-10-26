"""
multi_hop.py
============

Graph traversal and retrieval-augmented generation (RAG) logic.

Pipeline for a user question
----------------------------
1. **Entity linking** - longest-match n-gram lookup of the question against the
   graph's alias index ("European strike" -> ``European Dockworker Strike``).
2. **Intent routing** - impact language ("impact", "affect", "exposed", ...)
   routes to multi-hop search; a relation verb ("supplies", "competes")
   routes to a 1-hop lookup; otherwise we fall back to an entity profile or a
   structural connection query.
3. **Retrieval**
   * 1-hop: typed neighbour lookup, with the direction inferred from where
     the anchor entity sits relative to the verb.
   * multi-hop: ``graph_search`` - breadth-first enumeration of simple paths
     (<= ``max_hops``) over the *impact view* of the graph, where each edge
     points in the direction a disruption propagates. Each path is scored as
     the product of ``confidence x impact_weight`` along its hops.
4. **Generation** - the retrieved sub-graph is serialised into a grounded
   prompt and handed to a generator. ``MockLLMGenerator`` is a deterministic
   template-based synthesiser that only uses retrieved facts and cites the
   source document for every claim.

Run ``python multi_hop.py`` for a terminal demo.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Protocol

import networkx as nx

from graph_engine import (
    RELATION_SCHEMA,
    EdgeRecord,
    EntityNotFoundError,
    EntityType,
    KnowledgeGraph,
    RelationType,
    normalize_key,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_HOPS = 3
EXTENDED_HORIZON = 6      # used only to tell the user a path exists beyond their horizon
MAX_ENUMERATED_PATHS = 2_000


# --------------------------------------------------------------------------- #
# Path model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Hop:
    """One traversal step. ``reversed`` means we walked against the stored edge."""

    edge: EdgeRecord
    from_node: str
    to_node: str
    reversed: bool = False

    @property
    def weight(self) -> float:
        return self.edge.confidence * self.edge.impact_weight

    def render(self) -> str:
        rel = self.edge.relation.value
        return f" <-[{rel}]- {self.to_node}" if self.reversed else f" -[{rel}]-> {self.to_node}"

    def narrative(self) -> str:
        """Plain-English statement of the underlying fact, in traversal order."""
        e, spec = self.edge, self.edge.spec
        if self.reversed:
            return f"{self.from_node} {spec.inverse_phrase} {self.to_node}"
        if e.relation is RelationType.SUPPLIES and e.attributes.get("component"):
            return f"{e.source} supplies {e.attributes['component']} to {e.target}"
        severity = e.attributes.get("severity")
        verb = f"{severity} {spec.phrase}" if severity else spec.phrase
        return f"{e.source} {verb} {e.target}"


@dataclass(frozen=True)
class GraphPath:
    start: str
    hops: tuple[Hop, ...] = ()

    @property
    def end(self) -> str:
        return self.hops[-1].to_node if self.hops else self.start

    @property
    def nodes(self) -> tuple[str, ...]:
        return (self.start, *(h.to_node for h in self.hops))

    @property
    def length(self) -> int:
        return len(self.hops)

    @property
    def score(self) -> float:
        s = 1.0
        for h in self.hops:
            s *= h.weight
        return s

    @property
    def edge_keys(self) -> set[tuple[str, str, str]]:
        return {h.edge.key for h in self.hops}

    @property
    def doc_ids(self) -> list[str]:
        return sorted({d for h in self.hops for d in h.edge.doc_ids})

    @property
    def weakest_hop(self) -> Hop | None:
        return min(self.hops, key=lambda h: h.weight, default=None)

    def extend(self, hop: Hop) -> "GraphPath":
        return GraphPath(self.start, (*self.hops, hop))

    def render(self) -> str:
        """ASCII path string, e.g. ``A -[DELAYS]-> B -[MANUFACTURES]-> C``."""
        return self.start + "".join(h.render() for h in self.hops)

    def hop_table(self) -> list[dict[str, object]]:
        rows, cumulative = [], 1.0
        for i, h in enumerate(self.hops, 1):
            cumulative *= h.weight
            rows.append(
                {
                    "hop": i,
                    "from": h.from_node,
                    "relation": h.edge.relation.value,
                    "to": h.to_node,
                    "direction": "reverse" if h.reversed else "forward",
                    "confidence": round(h.edge.confidence, 3),
                    "impact_weight": round(h.edge.impact_weight, 3),
                    "cumulative_score": round(cumulative, 3),
                    "sources": ", ".join(h.edge.doc_ids),
                    "evidence": " | ".join(h.edge.evidence),
                }
            )
        return rows


@dataclass
class TraversalResult:
    start: str
    max_hops: int
    paths: list[GraphPath]
    target: str | None = None
    truncated: bool = False

    def impacted(self) -> dict[str, GraphPath]:
        """Best-scoring path to every reachable node."""
        best: dict[str, GraphPath] = {}
        for p in self.paths:
            if p.end not in best or p.score > best[p.end].score:
                best[p.end] = p
        return best

    def best_path_to(self, node: str) -> GraphPath | None:
        candidates = [p for p in self.paths if p.end == node]
        return max(candidates, key=lambda p: (p.score, -p.length), default=None)

    @property
    def target_path(self) -> GraphPath | None:
        return self.best_path_to(self.target) if self.target else None

    def by_hop(self) -> dict[int, list[GraphPath]]:
        layers: dict[int, list[GraphPath]] = {}
        for node, path in sorted(self.impacted().items(), key=lambda kv: (kv[1].length, -kv[1].score)):
            layers.setdefault(path.length, []).append(path)
        return layers


# --------------------------------------------------------------------------- #
# Multi-hop traversal
# --------------------------------------------------------------------------- #
def graph_search(
    kg: KnowledgeGraph,
    start_entity: str,
    max_hops: int = DEFAULT_MAX_HOPS,
    target: str | None = None,
    max_paths: int = MAX_ENUMERATED_PATHS,
) -> TraversalResult:
    """Find downstream impacts of ``start_entity`` up to ``max_hops`` away.

    Breadth-first enumeration of *simple* paths (no node revisited) over
    ``kg.impact_view()``. Returning every path - not just the shortest - lets
    callers reason about alternative transmission channels and pick the
    highest-scoring one. ``max_paths`` bounds work on dense graphs.

    Raises ``EntityNotFoundError`` if ``start_entity`` (or ``target``) cannot
    be resolved.
    """
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")
    start = kg.require_entity(start_entity)
    target_id = kg.require_entity(target) if target else None
    view = kg.impact_view()

    paths: list[GraphPath] = []
    frontier = [GraphPath(start)]
    truncated = False

    for _depth in range(max_hops):
        next_frontier: list[GraphPath] = []
        for path in frontier:
            visited = set(path.nodes)
            for _, nbr, data in view.out_edges(path.end, data=True):
                if nbr in visited:
                    continue
                u, v, k = data["origin"]
                hop = Hop(kg.edge_record(u, v, k), path.end, nbr, data["reversed"])
                new_path = path.extend(hop)
                paths.append(new_path)
                next_frontier.append(new_path)
                if len(paths) >= max_paths:
                    truncated = True
                    break
            if truncated:
                break
        frontier = next_frontier
        if truncated or not frontier:
            break

    paths.sort(key=lambda p: (p.length, -p.score))
    logger.info("graph_search(%s, hops=%d): %d paths, %d reachable", start, max_hops, len(paths), len({p.end for p in paths}))
    return TraversalResult(start, max_hops, paths, target_id, truncated)


def structural_path(kg: KnowledgeGraph, a: str, b: str) -> GraphPath | None:
    """Shortest path ignoring edge direction - 'how are these related at all?'."""
    undirected = kg.graph.to_undirected()
    try:
        nodes = nx.shortest_path(undirected, a, b)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    path = GraphPath(nodes[0])
    g = kg.graph
    for x, y in zip(nodes, nodes[1:]):
        if g.has_edge(x, y):
            k = next(iter(g[x][y]))
            path = path.extend(Hop(kg.edge_record(x, y, k), x, y, False))
        else:
            k = next(iter(g[y][x]))
            path = path.extend(Hop(kg.edge_record(y, x, k), x, y, True))
    return path


# --------------------------------------------------------------------------- #
# Question understanding
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mention:
    node_id: str
    surface: str
    start: int
    end: int


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

_MULTI_HOP_CUES = re.compile(
    r"\b(impact\w*|affect\w*|effect\w*|expos\w*|disrupt\w*|downstream|ripple|knock-on|"
    r"consequence\w*|propagat\w*|cascad\w*|vulnerab\w*|risk\s+to|implication\w*|hit)\b",
    re.I,
)

_RELATION_CUES: list[tuple[RelationType, re.Pattern[str]]] = [
    (RelationType.SUPPLIES, re.compile(r"\b(suppl\w*|provid\w*|vendor\w*|sourc\w*)\b", re.I)),
    (RelationType.COMPETES_WITH, re.compile(r"\b(compet\w*|rival\w*)\b", re.I)),
    (RelationType.MANAGES_SHIPPING_FOR, re.compile(r"\b(ship\w*|logistic\w*|freight|transport\w*)\b", re.I)),
    (RelationType.MANUFACTURES, re.compile(r"\b(manufactur\w*|makes?|build\w*|assembl\w*)\b", re.I)),
    (RelationType.DELAYS, re.compile(r"\b(delay\w*)\b", re.I)),
    (RelationType.PRODUCES, re.compile(r"\b(produc\w*)\b", re.I)),
    (RelationType.USES, re.compile(r"\b(uses?|using|components?|parts?)\b", re.I)),
    (RelationType.OCCURS_IN, re.compile(r"\b(where|located|occur\w*|happen\w*)\b", re.I)),
]


def link_entities(question: str, kg: KnowledgeGraph, max_ngram: int = 5) -> list[Mention]:
    """Greedy longest-match entity linking against the graph alias index."""
    index = kg.alias_index()
    tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(question)]
    keys = [normalize_key(t) for t, _, _ in tokens]
    taken = [False] * len(tokens)
    mentions: list[Mention] = []

    for n in range(min(max_ngram, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            if any(taken[i : i + n]):
                continue
            key = " ".join(k for k in keys[i : i + n] if k)
            node = index.get(key) or index.get(key.replace(" ", ""))
            if node and node not in {m.node_id for m in mentions}:
                start, end = tokens[i][1], tokens[i + n - 1][2]
                mentions.append(Mention(node, question[start:end], start, end))
                taken[i : i + n] = [True] * n
    return sorted(mentions, key=lambda m: m.start)


def _mask(question: str, mentions: Iterable[Mention]) -> str:
    chars = list(question)
    for m in mentions:
        chars[m.start : m.end] = "_" * (m.end - m.start)
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
class QueryMode(str, Enum):
    ONE_HOP = "1-hop"
    MULTI_HOP = "multi-hop"
    CONNECTION = "connection"
    ENTITY_PROFILE = "entity-profile"
    NO_MATCH = "no-match"


@dataclass
class RetrievedContext:
    """Everything the generator is allowed to know."""

    question: str
    mode: QueryMode
    facts: list[EdgeRecord] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    traversal: TraversalResult | None = None
    anchor: str | None = None
    answer_nodes: list[str] = field(default_factory=list)
    relation: RelationType | None = None
    notes: list[str] = field(default_factory=list)


class Generator(Protocol):
    def synthesize(self, context: RetrievedContext, kg: KnowledgeGraph) -> str:  # pragma: no cover
        ...


def render_prompt(context: RetrievedContext) -> str:
    """The grounded prompt a real LLM would receive (shown in the UI for transparency)."""
    lines = [
        "SYSTEM: You are a supply-chain risk analyst. Answer ONLY from the graph context below.",
        "Cite source documents in [brackets]. If the context is insufficient, say so.",
        "",
        f"RETRIEVAL MODE: {context.mode.value}",
        "GRAPH CONTEXT:",
    ]
    facts = {f.key: f for f in context.facts}
    for p in context.paths:
        facts.update({h.edge.key: h.edge for h in p.hops})
    for i, f in enumerate(facts.values(), 1):
        lines.append(
            f"  {i}. {f.source} -[{f.relation.value}]-> {f.target} "
            f"(conf={f.confidence:.2f}; {', '.join(f.doc_ids)})"
        )
    for i, p in enumerate(context.paths, 1):
        lines.append(f"TRAVERSAL PATH {i} (score={p.score:.3f}): {p.render()}")
    lines += ["", f"QUESTION: {context.question}", "ANSWER:"]
    return "\n".join(lines)


def _cite(edge: EdgeRecord) -> str:
    return f"[{', '.join(edge.doc_ids)}]" if edge.doc_ids else ""


def _risk_label(score: float) -> str:
    return "HIGH" if score >= 0.5 else "MEDIUM" if score >= 0.3 else "LOW"


class MockLLMGenerator:
    """Deterministic, grounded answer synthesiser.

    Behaves like a well-prompted LLM: it never introduces entities that are
    not in the retrieved context, and every factual sentence carries a
    document citation.
    """

    model_name = "mock-analyst-v1"

    def synthesize(self, context: RetrievedContext, kg: KnowledgeGraph) -> str:
        handler = {
            QueryMode.ONE_HOP: self._one_hop,
            QueryMode.MULTI_HOP: self._multi_hop,
            QueryMode.CONNECTION: self._connection,
            QueryMode.ENTITY_PROFILE: self._profile,
            QueryMode.NO_MATCH: self._no_match,
        }[context.mode]
        return handler(context, kg)

    # -- 1-hop ---------------------------------------------------------------- #
    def _one_hop(self, ctx: RetrievedContext, kg: KnowledgeGraph) -> str:
        if not ctx.facts:
            rel = ctx.relation.value if ctx.relation else "matching"
            return (
                f"The knowledge graph has no {rel} relationship for **{ctx.anchor}**. "
                "Try ingesting a document that covers it."
            )
        answer = ", ".join(f"**{n}**" for n in ctx.answer_nodes)
        lines = [f"**Answer:** {answer}", ""]
        for f in ctx.facts:
            hop = Hop(f, f.source, f.target)
            lines.append(f"- {hop.narrative()} {_cite(f)}")
        if ctx.relation is RelationType.SUPPLIES and ctx.anchor and len(ctx.facts) == 1:
            f = ctx.facts[0]
            if f.target == ctx.anchor:
                comp = f.attributes.get("component", "this input")
                lines += [
                    "",
                    f"_Analyst note: {f.source} is the only {comp} supplier to {ctx.anchor} in the "
                    f"graph - a single-source dependency._",
                ]
        lines += ["", "_Retrieval depth: 1 hop (direct relationship)._"]
        return "\n".join(lines)

    # -- multi-hop ------------------------------------------------------------ #
    def _multi_hop(self, ctx: RetrievedContext, kg: KnowledgeGraph) -> str:
        tr = ctx.traversal
        assert tr is not None
        if ctx.paths:
            return self._targeted_insight(ctx, kg, ctx.paths[0])
        return self._impact_report(ctx, kg)

    def _targeted_insight(self, ctx: RetrievedContext, kg: KnowledgeGraph, path: GraphPath) -> str:
        tr = ctx.traversal
        assert tr is not None
        score, label = path.score, _risk_label(path.score)
        impacted = tr.impacted()

        chain = path.hops[0].narrative() + "".join(f", which {self._clause(h)}" for h in path.hops[1:])
        channels = self._channels(path)
        cites = ", ".join(path.doc_ids)

        out = [
            f"**Insight - {path.length}-hop exposure, impact score {score:.2f} ({label})**",
            "",
            f"{chain}. [{cites}]",
            "",
            f"**{path.start}** reaches **{path.end}** through a {channels} channel "
            f"({path.length} degrees of separation). No single document states this link - it only "
            "emerges by chaining facts across sources.",
        ]

        findings = self._component_finding(kg, path, impacted) + self._competitor_findings(kg, path, impacted)
        if findings:
            out += ["", "**What the graph also tells us:**", *[f"- {f}" for f in findings]]

        actions = self._actions(kg, path, impacted)
        if actions:
            out += ["", "**Recommended actions:**", *[f"{i}. {a}" for i, a in enumerate(actions, 1)]]

        weak = path.weakest_hop
        if weak:
            out += [
                "",
                f"_Weakest link: {weak.edge.source} -[{weak.edge.relation.value}]-> {weak.edge.target} "
                f"(confidence x weight = {weak.weight:.2f}). Score = product of hop weights._",
            ]
        return "\n".join(out)

    def _impact_report(self, ctx: RetrievedContext, kg: KnowledgeGraph) -> str:
        tr = ctx.traversal
        assert tr is not None
        layers = tr.by_hop()
        if not layers:
            return (
                f"**{tr.start}** has no downstream impact channels within {tr.max_hops} hops in the "
                "current graph (it may only have contextual links such as location or competition)."
            )
        out = [f"**Downstream impact analysis for {tr.start}** (horizon: {tr.max_hops} hops)", ""]
        for depth, paths in layers.items():
            order = {1: "1st", 2: "2nd", 3: "3rd"}.get(depth, f"{depth}th")
            out.append(f"**{order}-order impacts**")
            for p in paths:
                kind = kg.entity_type(p.end).value
                out.append(
                    f"- **{p.end}** ({kind}) - score {p.score:.2f} ({_risk_label(p.score)}): "
                    f"{p.hops[-1].narrative()} [{', '.join(p.hops[-1].edge.doc_ids)}]"
                )
            out.append("")
        most = max(tr.impacted().values(), key=lambda p: (p.length, p.score))
        out.append(f"Furthest-reaching chain: `{most.render()}`")
        return "\n".join(out)

    # -- other modes ---------------------------------------------------------- #
    def _connection(self, ctx: RetrievedContext, kg: KnowledgeGraph) -> str:
        out = [*ctx.notes]
        if ctx.paths:
            p = ctx.paths[0]
            facts = "; ".join(f"{h.narrative()} {_cite(h.edge)}" for h in p.hops)
            out += [
                "",
                f"The entities are structurally connected in {p.length} hops: {facts}.",
                "This is a relationship path, not a causal impact channel, so no impact score is assigned.",
            ]
        return "\n".join(out).strip()

    def _profile(self, ctx: RetrievedContext, kg: KnowledgeGraph) -> str:
        node = ctx.anchor
        assert node is not None
        etype = kg.entity_type(node).value
        out = [f"**{node}** ({etype}) - {len(ctx.facts)} direct relationships:", ""]
        out += [f"- {Hop(f, f.source, f.target).narrative()} {_cite(f)}" for f in ctx.facts]
        out += ["", "_Ask how it impacts something, or about a specific relationship, for a deeper answer._"]
        return "\n".join(out)

    @staticmethod
    def _no_match(ctx: RetrievedContext, kg: KnowledgeGraph) -> str:
        sample = ", ".join(sorted(kg.nodes())[:8])
        return (
            "I could not link any entity in your question to the knowledge graph, so I can't answer "
            f"without speculating. Known entities include: {sample}."
        )

    # -- reasoning helpers ---------------------------------------------------- #
    @staticmethod
    def _clause(h: Hop) -> str:
        """'X manages shipping for Y' -> 'manages shipping for Y' (subject elided)."""
        text = h.narrative()
        return text[len(h.from_node) + 1 :] if text.startswith(h.from_node + " ") else text

    @staticmethod
    def _channels(path: GraphPath) -> str:
        names: list[str] = []
        for h in path.hops:
            r = h.edge.relation
            name = (
                "logistics"
                if r in {RelationType.DELAYS, RelationType.MANAGES_SHIPPING_FOR}
                else "supply"
                if r in {RelationType.SUPPLIES, RelationType.PRODUCES, RelationType.USES, RelationType.COMPONENT_OF}
                else "production"
                if r is RelationType.MANUFACTURES
                else "risk-exposure"
            )
            if name not in names:
                names.append(name)
        return " -> ".join(names)

    @staticmethod
    def _component_finding(kg: KnowledgeGraph, path: GraphPath, impacted: dict[str, GraphPath]) -> list[str]:
        if kg.entity_type(path.end) is not EntityType.PRODUCT:
            return []
        findings = []
        for e in kg.neighbors(path.end, RelationType.COMPONENT_OF, "in"):
            comp = e.source
            if comp in impacted:
                continue
            suppliers = [s.source for s in kg.neighbors(comp, RelationType.PRODUCES, "in")]
            via = f" from {', '.join(suppliers)}" if suppliers else ""
            findings.append(
                f"The {path.end}'s {comp} supply{via} is **not** on any impact path from {path.start} "
                f"{_cite(e)} - the exposure is in moving goods, not in the bill of materials."
            )
        return findings

    @staticmethod
    def _competitor_findings(kg: KnowledgeGraph, path: GraphPath, impacted: dict[str, GraphPath]) -> list[str]:
        findings = []
        for node in path.nodes:
            if kg.entity_type(node) is not EntityType.COMPANY:
                continue
            for e in kg.neighbors(node, RelationType.COMPETES_WITH, "both"):
                rival = e.target if e.source == node else e.source
                if rival in impacted or rival in path.nodes:
                    continue
                suppliers = [s.source for s in kg.neighbors(rival, RelationType.SUPPLIES, "in")]
                sourcing = f" (it sources from {', '.join(suppliers)})" if suppliers else ""
                findings.append(
                    f"**{rival}**, a competitor of {node} {_cite(e)}, has no path to {path.start} in the "
                    f"graph{sourcing}. It is positioned to capture demand displaced from {path.end}."
                )
        return findings

    @staticmethod
    def _actions(kg: KnowledgeGraph, path: GraphPath, impacted: dict[str, GraphPath]) -> list[str]:
        actions: list[str] = []
        regions = [e.target for e in kg.neighbors(path.start, RelationType.OCCURS_IN, "out")]
        for h in path.hops:
            e = h.edge
            if e.relation is RelationType.MANAGES_SHIPPING_FOR:
                where = f" away from ports in {', '.join(regions)}" if regions else ""
                n_providers = len(kg.neighbors(e.target, RelationType.MANAGES_SHIPPING_FOR, "in"))
                sole = f"; {e.source} is its only shipping partner in the graph" if n_providers == 1 else ""
                actions.append(
                    f"Qualify a secondary logistics provider for {e.target} and reroute priority freight{where}{sole}."
                )
            elif e.relation is RelationType.SUPPLIES:
                actions.append(f"Dual-source {e.attributes.get('component', 'inputs')} for {e.target} to reduce reliance on {e.source}.")
            elif e.relation is RelationType.MANUFACTURES:
                actions.append(
                    f"Quantify days-of-inventory cover and revenue-at-risk for the {e.target}; "
                    "pre-position finished goods ahead of the disruption window."
                )
        for node in path.nodes:
            for e in kg.neighbors(node, RelationType.COMPETES_WITH, "both"):
                rival = e.target if e.source == node else e.source
                if rival not in impacted and rival not in path.nodes:
                    actions.append(f"Monitor {rival}'s pricing and channel activity for share-capture moves.")
        return list(dict.fromkeys(actions))


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
@dataclass
class RAGResponse:
    question: str
    mode: QueryMode
    answer: str
    mentions: list[Mention]
    context: RetrievedContext
    prompt: str
    trace: list[str]
    latency_ms: float

    @property
    def paths(self) -> list[GraphPath]:
        return self.context.paths

    @property
    def highlight_edges(self) -> set[tuple[str, str, str]]:
        keys = {f.key for f in self.context.facts}
        for p in self.context.paths:
            keys |= p.edge_keys
        return keys


class GraphRAGPipeline:
    def __init__(self, kg: KnowledgeGraph, generator: Generator | None = None) -> None:
        self.kg = kg
        self.generator: Generator = generator or MockLLMGenerator()

    def answer(self, question: str, max_hops: int = DEFAULT_MAX_HOPS) -> RAGResponse:
        t0 = time.perf_counter()
        question = question.strip()
        trace: list[str] = []

        mentions = link_entities(question, self.kg)
        trace.append(
            "Entity linking: "
            + (", ".join(f"'{m.surface}' -> {m.node_id}" for m in mentions) if mentions else "no entities found")
        )
        masked = _mask(question, mentions)
        ctx = self._retrieve(question, masked, mentions, max_hops, trace)

        prompt = render_prompt(ctx)
        answer = self.generator.synthesize(ctx, self.kg)
        latency = (time.perf_counter() - t0) * 1000
        trace.append(f"Generation: {getattr(self.generator, 'model_name', type(self.generator).__name__)}")
        return RAGResponse(question, ctx.mode, answer, mentions, ctx, prompt, trace, latency)

    # -- routing -------------------------------------------------------------- #
    def _retrieve(
        self, question: str, masked: str, mentions: list[Mention], max_hops: int, trace: list[str]
    ) -> RetrievedContext:
        if not mentions:
            trace.append("Intent: none (no grounded entities)")
            return RetrievedContext(question, QueryMode.NO_MATCH)

        if _MULTI_HOP_CUES.search(masked):
            trace.append("Intent: multi-hop impact analysis (impact language detected)")
            return self._multi_hop(question, mentions, max_hops, trace)

        relation, cue_pos = self._relation_cue(masked)
        if relation:
            trace.append(f"Intent: 1-hop lookup on relation {relation.value}")
            return self._one_hop(question, mentions, relation, cue_pos, trace)

        if len(mentions) >= 2:
            trace.append("Intent: connection query between two entities")
            return self._connection(question, mentions[0].node_id, mentions[-1].node_id, max_hops, trace)

        trace.append("Intent: entity profile")
        node = mentions[0].node_id
        return RetrievedContext(
            question, QueryMode.ENTITY_PROFILE, facts=self.kg.neighbors(node), anchor=node
        )

    @staticmethod
    def _relation_cue(masked: str) -> tuple[RelationType | None, int]:
        hits = [(m.start(), rel) for rel, rx in _RELATION_CUES if (m := rx.search(masked))]
        if not hits:
            return None, -1
        pos, rel = min(hits, key=lambda h: h[0])
        return rel, pos

    # -- 1-hop ---------------------------------------------------------------- #
    def _one_hop(
        self,
        question: str,
        mentions: list[Mention],
        relation: RelationType,
        cue_pos: int,
        trace: list[str],
    ) -> RetrievedContext:
        # Components named in the question act as filters ("batteries"), not anchors,
        # unless they are the only thing mentioned.
        anchors = [m for m in mentions if self.kg.entity_type(m.node_id) is not EntityType.COMPONENT] or mentions
        filters = [m for m in mentions if m not in anchors]
        filter_tokens = {t for m in filters for t in normalize_key(m.surface).split()}

        for anchor in anchors:
            # Anchor after the verb -> it is the object ("Who supplies ... TechNova?") -> incoming edges.
            preferred = "in" if anchor.start > cue_pos else "out"
            for direction in (preferred, "out" if preferred == "in" else "in"):
                facts = self.kg.neighbors(anchor.node_id, relation, direction)
                if filter_tokens:
                    facts = [f for f in facts if self._matches_filter(f, filter_tokens)]
                if facts:
                    answer_nodes = list(
                        dict.fromkeys(f.target if f.source == anchor.node_id else f.source for f in facts)
                    )
                    trace.append(
                        f"Retrieval: {len(facts)} {relation.value} edge(s) "
                        f"{'into' if direction == 'in' else 'out of'} {anchor.node_id}"
                        + (f", filtered by '{' '.join(sorted(filter_tokens))}'" if filter_tokens else "")
                    )
                    return RetrievedContext(
                        question, QueryMode.ONE_HOP, facts=facts, anchor=anchor.node_id,
                        answer_nodes=answer_nodes, relation=relation,
                    )
        trace.append("Retrieval: no matching 1-hop edges")
        return RetrievedContext(question, QueryMode.ONE_HOP, anchor=anchors[0].node_id, relation=relation)

    @staticmethod
    def _matches_filter(edge: EdgeRecord, tokens: set[str]) -> bool:
        haystack = " ".join([edge.source, edge.target, *map(str, edge.attributes.values())])
        return bool(tokens & set(normalize_key(haystack).split()))

    # -- multi-hop ------------------------------------------------------------ #
    def _multi_hop(
        self, question: str, mentions: list[Mention], max_hops: int, trace: list[str]
    ) -> RetrievedContext:
        risks = [m for m in mentions if self.kg.entity_type(m.node_id) is EntityType.RISK]
        start = (risks or mentions)[0].node_id
        others = [m.node_id for m in mentions if m.node_id != start]
        target = others[-1] if others else None
        trace.append(f"Traversal: start={start}, target={target or '(all downstream)'}, max_hops={max_hops}")

        try:
            tr = graph_search(self.kg, start, max_hops=max_hops, target=target)
        except EntityNotFoundError as exc:  # pragma: no cover - linker guarantees resolution
            trace.append(f"Traversal failed: {exc}")
            return RetrievedContext(question, QueryMode.NO_MATCH)

        trace.append(f"Traversal: enumerated {len(tr.paths)} simple paths, {len(tr.impacted())} nodes reachable")
        ctx = RetrievedContext(question, QueryMode.MULTI_HOP, traversal=tr, anchor=start)

        if target is None:
            # Report mode: the generator walks traversal.by_hop(); facts drive highlighting.
            ctx.facts = list({h.edge.key: h.edge for p in tr.paths for h in p.hops}.values())
            return ctx

        best = tr.target_path
        if best:
            alternatives = [p for p in tr.paths if p.end == target and p is not best]
            trace.append(
                f"Selected path (score {best.score:.3f}): {best.render()}"
                + (f"; {len(alternatives)} alternative path(s) ranked lower" if alternatives else "")
            )
            ctx.paths = [best]
            return ctx

        # No causal path within the horizon: explain why, then fall back to structure.
        notes = [f"No impact path from **{start}** to **{target}** within {max_hops} hops."]
        wider = graph_search(self.kg, start, max_hops=EXTENDED_HORIZON, target=target).target_path
        if wider:
            notes.append(
                f"A {wider.length}-hop impact path exists beyond the current horizon - increase max hops to "
                f"{wider.length} to analyse it: `{wider.render()}`"
            )
            trace.append(f"Horizon check: impact path exists at {wider.length} hops")
            return RetrievedContext(question, QueryMode.CONNECTION, traversal=tr, anchor=start, notes=notes)
        return self._connection(question, start, target, max_hops, trace, notes)

    def _connection(
        self,
        question: str,
        a: str,
        b: str,
        max_hops: int,
        trace: list[str],
        notes: list[str] | None = None,
    ) -> RetrievedContext:
        tr = graph_search(self.kg, a, max_hops=max_hops, target=b)
        if tr.target_path and not notes:
            trace.append(f"Selected impact path: {tr.target_path.render()}")
            return RetrievedContext(question, QueryMode.MULTI_HOP, paths=[tr.target_path], traversal=tr, anchor=a)

        path = structural_path(self.kg, a, b)
        notes = list(notes or [])
        if path:
            trace.append(f"Structural path (direction-agnostic): {path.render()}")
        else:
            notes.append(f"**{a}** and **{b}** are not connected in the current knowledge graph.")
            trace.append("Structural path: none (disconnected components)")
        return RetrievedContext(
            question, QueryMode.CONNECTION, paths=[path] if path else [], traversal=tr, anchor=a, notes=notes
        )


# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from extractor import bootstrap_knowledge_graph

    logging.basicConfig(level=logging.WARNING)
    kg, _ = bootstrap_knowledge_graph()
    pipeline = GraphRAGPipeline(kg)

    for q in [
        "Who supplies batteries to TechNova?",
        "How does the European strike impact the NovaPhone?",
        "What are the downstream impacts of the dockworker strike?",
        "Who competes with TechNova?",
    ]:
        r = pipeline.answer(q)
        print("=" * 100)
        print(f"Q: {q}   [{r.mode.value}, {r.latency_ms:.1f} ms]")
        for step in r.trace:
            print(f"   . {step}")
        for p in r.paths:
            print(f"   PATH: {p.render()}")
        print()
        print(r.answer)
    print("=" * 100)
    print("Exported:", kg.export_html("graph.html"))
