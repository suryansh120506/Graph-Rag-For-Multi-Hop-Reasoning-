"""
extractor.py
============

Knowledge extraction layer: raw text  ->  typed entities & relationships.

Design
------
The extraction pipeline is written against a tiny ``LLMClient`` protocol
(``complete(prompt) -> str``). The production contract is the same one you
would use with a hosted or local model: we send a schema-constrained prompt and
expect a JSON document back, which is then *validated* before it touches the
graph. ``MockLLMExtractor`` honours that contract deterministically with a
rule-based information-extraction engine, so the project runs offline with no
API keys. Swapping in a real model (OpenAI, Ollama, vLLM, ...) means writing
one class with a ``complete`` method.

What the mock "model" actually does (in order, per document):
  1. sentence segmentation;
  2. coreference resolution for demonstratives ("these batteries" -> the most
     recently mentioned component with the same head noun);
  3. subject-ellipsis resolution for coordinated clauses
     ("X is a rival to Y and sources Z from W" -> two clauses with subject X);
  4. priority-ordered relation patterns; a span consumed by a more specific
     pattern cannot be re-used by a more generic one;
  5. slot-based entity typing (the role an entity plays in a pattern decides
     whether it is a Company, Component, Product, Risk or Region);
  6. calibrated confidences - each inference step (coreference, ellipsis)
     discounts the confidence of the facts that depend on it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol

from graph_engine import (
    SEVERITY_WEIGHTS,
    EntityType,
    KnowledgeGraph,
    RelationType,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Seed corpus
# --------------------------------------------------------------------------- #
SEED_DOCUMENTS: dict[str, str] = {
    "Doc 1": (
        "ApexCorp supplies Lithium Batteries to TechNova. "
        "TechNova uses these batteries to manufacture the NovaPhone."
    ),
    "Doc 2": (
        "GlobalLogistics manages shipping for TechNova. "
        "A recent dockworker strike in Europe has severely delayed GlobalLogistics operations."
    ),
    "Doc 3": (
        "OmegaCorp is a direct competitor to TechNova and sources batteries "
        "from a different supplier, BetaTech."
    ),
}


# --------------------------------------------------------------------------- #
# LLM contract
# --------------------------------------------------------------------------- #
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        ...


EXTRACTION_PROMPT = """\
You are an information-extraction system for supply-chain risk analysis.
Extract every entity and relationship stated in the document.

Allowed entity types: {entity_types}
Allowed relation types: {relation_types}

Rules:
- Resolve pronouns and demonstratives ("these batteries") to the entity they refer to.
- Only extract facts that are explicitly stated. Do not speculate.
- Give each relation a confidence in [0, 1] and quote the supporting sentence as evidence.

Respond with JSON only, matching:
{{"entities": [{{"name": str, "type": str, "aliases": [str]}}],
  "relations": [{{"source": str, "relation": str, "target": str,
                 "confidence": float, "evidence": str, "attributes": {{}}}}]}}

<document id="{doc_id}">
{text}
</document>
"""


def build_extraction_prompt(doc_id: str, text: str) -> str:
    return EXTRACTION_PROMPT.format(
        entity_types=", ".join(t.value for t in EntityType if t is not EntityType.UNKNOWN),
        relation_types=", ".join(r.value for r in RelationType),
        doc_id=doc_id,
        text=text.strip(),
    )


# --------------------------------------------------------------------------- #
# Extraction result objects
# --------------------------------------------------------------------------- #
@dataclass
class ExtractedEntity:
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class ExtractedRelation:
    source: str
    relation: str
    target: str
    confidence: float
    evidence: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    doc_id: str
    text: str
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    rejected: list[str] = field(default_factory=list)

    def triples(self) -> list[dict[str, Any]]:
        return [
            {
                "doc": self.doc_id,
                "source": r.source,
                "relation": r.relation,
                "target": r.target,
                "confidence": round(r.confidence, 2),
            }
            for r in self.relations
        ]


# --------------------------------------------------------------------------- #
# Mock LLM: deterministic rule-based IE engine behind an LLM-shaped interface
# --------------------------------------------------------------------------- #
_PROPER = r"[A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*)*"
_NOUN = r"[A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,2}?"
_LOWER_NOUN = r"[a-z][a-z\-]*(?:\s+[a-z][a-z\-]*){0,2}?"
_ART = r"(?:(?:[Tt]he|[Aa]n?)\s+)?"

_DEMONYMS = {
    "Europe": "European",
    "Asia": "Asian",
    "Africa": "African",
    "North America": "North American",
    "South America": "South American",
    "America": "American",
    "China": "Chinese",
    "Taiwan": "Taiwanese",
    "Japan": "Japanese",
    "Germany": "German",
    "India": "Indian",
    "Mexico": "Mexican",
    "Canada": "Canadian",
}

_BASE_CONFIDENCE = 0.95
_COREF_DISCOUNT = 0.95
_ELLIPSIS_DISCOUNT = 0.95

_DOC_RE = re.compile(r'<document id="(?P<id>[^"]*)">\s*(?P<text>.*?)\s*</document>', re.S)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_DEMONSTRATIVE_RE = re.compile(r"\b(?:[Tt]hese|[Tt]hose|[Tt]his|[Tt]hat|[Tt]he same)\s+([a-z]+)\b")
_ELLIPSIS_RE = re.compile(
    rf"^(?P<subj>{_PROPER})\s+(?P<first>.+?),?\s+and\s+(?P<second>[a-z]+s\b.*)$"
)


@dataclass
class _Clause:
    text: str
    sentence: str
    confidence: float


@dataclass
class _Pattern:
    name: str
    regex: re.Pattern[str]
    emit: Callable[["_Emitter", re.Match[str]], None]


class _Emitter:
    """Accumulates entities/relations for one document, de-duplicating names."""

    def __init__(self) -> None:
        self.entities: dict[str, ExtractedEntity] = {}
        self.relations: list[ExtractedRelation] = []
        self.evidence = ""
        self.confidence = _BASE_CONFIDENCE

    def entity(self, raw: str, etype: EntityType, aliases: list[str] | None = None) -> str:
        name = _clean_name(raw)
        if etype is EntityType.COMPONENT:
            name = _titleize(name)
        ent = self.entities.setdefault(name, ExtractedEntity(name, etype.value, []))
        if ent.type == EntityType.UNKNOWN.value and etype is not EntityType.UNKNOWN:
            ent.type = etype.value
        for a in aliases or []:
            if a and a != name and a not in ent.aliases:
                ent.aliases.append(a)
        return name

    def relation(self, src: str, rel: RelationType, dst: str, **attributes: Any) -> None:
        for existing in self.relations:  # same fact restated within one document
            if (existing.source, existing.relation, existing.target) == (src, rel.value, dst):
                existing.confidence = max(existing.confidence, round(self.confidence, 4))
                existing.attributes.update(attributes)
                return
        self.relations.append(
            ExtractedRelation(
                source=src,
                relation=rel.value,
                target=dst,
                confidence=round(self.confidence, 4),
                evidence=self.evidence,
                attributes=attributes,
            )
        )


def _clean_name(raw: str) -> str:
    name = re.sub(r"^(?:the|a|an)\s+", "", raw.strip(), flags=re.I)
    return re.sub(r"'s$", "", name).strip(" ,.;:")


def _titleize(text: str) -> str:
    return " ".join(w if w[:1].isupper() else w.capitalize() for w in text.split())


def _head_noun(text: str) -> str:
    head = text.split()[-1].lower()
    if head.endswith("ies"):
        return head[:-3] + "y"
    return head[:-1] if head.endswith("s") else head


# ---- pattern emitters ------------------------------------------------------ #
def _emit_supply_to(e: _Emitter, m: re.Match[str]) -> None:
    supplier = e.entity(m["supplier"], EntityType.COMPANY)
    component = e.entity(m["component"], EntityType.COMPONENT)
    customer = e.entity(m["customer"], EntityType.COMPANY)
    e.relation(supplier, RelationType.SUPPLIES, customer, component=component)
    e.relation(supplier, RelationType.PRODUCES, component)
    e.relation(customer, RelationType.USES, component)


def _emit_sources_from(e: _Emitter, m: re.Match[str]) -> None:
    buyer = e.entity(m["buyer"], EntityType.COMPANY)
    component = e.entity(m["component"], EntityType.COMPONENT)
    supplier = e.entity(m["supplier"], EntityType.COMPANY)
    e.relation(supplier, RelationType.SUPPLIES, buyer, component=component)
    e.relation(supplier, RelationType.PRODUCES, component)
    e.relation(buyer, RelationType.USES, component)


def _emit_uses_to_make(e: _Emitter, m: re.Match[str]) -> None:
    maker = e.entity(m["maker"], EntityType.COMPANY)
    component = e.entity(m["component"], EntityType.COMPONENT)
    product = e.entity(m["product"], EntityType.PRODUCT)
    e.relation(maker, RelationType.USES, component)
    e.relation(maker, RelationType.MANUFACTURES, product)
    e.relation(component, RelationType.COMPONENT_OF, product)


def _emit_manufactures(e: _Emitter, m: re.Match[str]) -> None:
    maker = e.entity(m["maker"], EntityType.COMPANY)
    product = e.entity(m["product"], EntityType.PRODUCT)
    e.relation(maker, RelationType.MANUFACTURES, product)


def _emit_shipping(e: _Emitter, m: re.Match[str]) -> None:
    provider = e.entity(m["provider"], EntityType.COMPANY)
    client = e.entity(m["client"], EntityType.COMPANY)
    e.relation(provider, RelationType.MANAGES_SHIPPING_FOR, client, service=m["service"].lower())


def _emit_disruption(e: _Emitter, m: re.Match[str]) -> None:
    event, region_raw = m["event"].lower(), _clean_name(m["region"])
    demonym = _DEMONYMS.get(region_raw, region_raw)
    head = event.split()[-1]
    aliases = [event, head, f"{demonym} {head}", f"{region_raw} {head}", f"{event} in {region_raw}"]
    risk = e.entity(f"{demonym} {_titleize(event)}", EntityType.RISK, aliases)
    region = e.entity(region_raw, EntityType.REGION)
    victim = e.entity(m["victim"], EntityType.COMPANY)
    severity = (m["severity"] or "").lower() or None
    e.relation(
        risk,
        RelationType.DELAYS,
        victim,
        severity=severity,
        scope="operations",
        impact_weight=SEVERITY_WEIGHTS.get(severity or "", None),
    )
    e.relation(risk, RelationType.OCCURS_IN, region)


def _emit_competitor(e: _Emitter, m: re.Match[str]) -> None:
    a = e.entity(m["a"], EntityType.COMPANY)
    b = e.entity(m["b"], EntityType.COMPANY)
    e.relation(a, RelationType.COMPETES_WITH, b, qualifier=(m["qual"] or "").strip() or None)


def _emit_exposed(e: _Emitter, m: re.Match[str]) -> None:
    entity = e.entity(m["entity"], EntityType.UNKNOWN)
    risk = e.entity(_titleize(m["risk"]), EntityType.RISK)
    e.relation(entity, RelationType.EXPOSED_TO, risk)


# Ordered from most to least specific. Earlier patterns consume their span.
_PATTERNS: list[_Pattern] = [
    _Pattern(
        "supply_to",
        re.compile(
            rf"(?P<supplier>{_PROPER})\s+(?:supplies|provides|delivers)\s+{_ART}"
            rf"(?P<component>{_NOUN})\s+to\s+{_ART}(?P<customer>{_PROPER})"
        ),
        _emit_supply_to,
    ),
    _Pattern(
        "sources_from",
        re.compile(
            rf"(?P<buyer>{_PROPER})\s+(?:sources|buys|procures|purchases)\s+{_ART}"
            rf"(?P<component>{_NOUN})\s+from\s+{_ART}"
            rf"(?:(?:different|new|alternative|another|second)\s+)?(?:supplier|vendor)?,?\s*"
            rf"(?P<supplier>{_PROPER})"
        ),
        _emit_sources_from,
    ),
    _Pattern(
        "uses_to_make",
        re.compile(
            rf"(?P<maker>{_PROPER})\s+uses\s+{_ART}(?P<component>{_NOUN})\s+to\s+"
            rf"(?:manufacture|make|build|produce|assemble)\s+{_ART}(?P<product>{_PROPER})"
        ),
        _emit_uses_to_make,
    ),
    _Pattern(
        "disruption",
        re.compile(
            rf"{_ART}(?:(?:recent|major|ongoing|new|prolonged)\s+)?(?P<event>{_LOWER_NOUN})\s+in\s+"
            rf"(?P<region>{_PROPER})\s+(?:has|have|is|are)\s+"
            rf"(?:(?P<severity>severely|significantly|moderately|slightly)\s+)?"
            rf"(?:delayed|delaying|disrupted|disrupting|halted|halting)\s+"
            rf"(?P<victim>{_PROPER})(?:'s)?(?:\s+operations)?"
        ),
        _emit_disruption,
    ),
    _Pattern(
        "shipping",
        re.compile(
            rf"(?P<provider>{_PROPER})\s+(?:manages|handles|runs|provides)\s+"
            rf"(?P<service>shipping|logistics|freight|transportation)(?:\s+services)?\s+for\s+"
            rf"{_ART}(?P<client>{_PROPER})"
        ),
        _emit_shipping,
    ),
    _Pattern(
        "competitor",
        re.compile(
            rf"(?P<a>{_PROPER})\s+(?:is\s+(?:a|an|the)\s+(?P<qual>(?:direct|key|major|main|primary)\s+)?"
            rf"(?:competitor|rival)\s+(?:to|of)|competes\s+with)\s+{_ART}(?P<b>{_PROPER})"
        ),
        _emit_competitor,
    ),
    _Pattern(
        "manufactures",
        re.compile(
            rf"(?P<maker>{_PROPER})\s+(?:manufactures|makes|builds|assembles)\s+{_ART}(?P<product>{_PROPER})"
        ),
        _emit_manufactures,
    ),
    _Pattern(
        "exposed_to",
        re.compile(rf"(?P<entity>{_PROPER})\s+is\s+(?:directly\s+)?exposed\s+to\s+{_ART}(?P<risk>{_NOUN})[.,;]?$"),
        _emit_exposed,
    ),
]


class MockLLMExtractor:
    """Deterministic stand-in for an extraction LLM.

    Implements ``LLMClient.complete``: accepts the extraction prompt, returns
    a JSON string. It never sees anything other than the prompt, exactly like
    a remote model would.
    """

    model_name = "mock-ie-rules-v1"

    def complete(self, prompt: str) -> str:
        match = _DOC_RE.search(prompt)
        if not match:
            return json.dumps({"entities": [], "relations": []})
        emitter = _Emitter()
        for clause in self._clauses(match["text"]):
            self._apply_patterns(emitter, clause)
        return json.dumps(
            {
                "entities": [asdict(e) for e in emitter.entities.values()],
                "relations": [asdict(r) for r in emitter.relations],
            }
        )

    # -- pipeline stages ---------------------------------------------------- #
    def _clauses(self, text: str) -> list[_Clause]:
        clauses: list[_Clause] = []
        component_memory: dict[str, str] = {}  # head noun -> component name

        for sentence in filter(None, (s.strip() for s in _SENTENCE_RE.split(text))):
            resolved, confidence = self._resolve_coreference(sentence, component_memory)

            if m := _ELLIPSIS_RE.match(resolved):
                subj = m["subj"]
                clauses.append(_Clause(f"{subj} {m['first']}", sentence, confidence * _ELLIPSIS_DISCOUNT))
                clauses.append(_Clause(f"{subj} {m['second']}", sentence, confidence * _ELLIPSIS_DISCOUNT))
            else:
                clauses.append(_Clause(resolved, sentence, confidence))

            # Remember components introduced in this sentence for later coreference.
            for p in _PATTERNS:
                if "component" in p.regex.groupindex:
                    for pm in p.regex.finditer(resolved):
                        comp = _titleize(_clean_name(pm["component"]))
                        component_memory[_head_noun(comp)] = comp
        return clauses

    @staticmethod
    def _resolve_coreference(sentence: str, memory: dict[str, str]) -> tuple[str, float]:
        confidence = _BASE_CONFIDENCE
        out = sentence
        for m in _DEMONSTRATIVE_RE.finditer(sentence):
            antecedent = memory.get(_head_noun(m.group(1)))
            if antecedent:
                out = out.replace(m.group(0), antecedent, 1)
                confidence *= _COREF_DISCOUNT
                logger.debug("Coreference: '%s' -> '%s'", m.group(0), antecedent)
        return out, confidence

    @staticmethod
    def _apply_patterns(emitter: _Emitter, clause: _Clause) -> None:
        consumed: list[tuple[int, int]] = []
        emitter.evidence = clause.sentence
        emitter.confidence = clause.confidence
        for pattern in _PATTERNS:
            for m in pattern.regex.finditer(clause.text):
                if any(m.start() < end and start < m.end() for start, end in consumed):
                    continue
                consumed.append(m.span())
                pattern.emit(emitter, m)
                logger.debug("Pattern %s matched: %s", pattern.name, m.group(0))


# --------------------------------------------------------------------------- #
# Orchestration: prompt -> LLM -> validate -> write to graph
# --------------------------------------------------------------------------- #
class KnowledgeExtractor:
    """Runs documents through an ``LLMClient`` and loads validated facts."""

    def __init__(self, graph: KnowledgeGraph, llm: LLMClient | None = None) -> None:
        self.graph = graph
        self.llm: LLMClient = llm or MockLLMExtractor()
        self.documents: dict[str, str] = {}
        self.results: dict[str, ExtractionResult] = {}

    def next_doc_id(self) -> str:
        return f"Doc {len(self.documents) + 1}"

    def ingest(self, doc_id: str, text: str) -> ExtractionResult:
        if not text or not text.strip():
            raise ValueError("Document text is empty.")
        raw = self.llm.complete(build_extraction_prompt(doc_id, text))
        result = self._parse_and_validate(doc_id, text, raw)
        self._load(result)
        self.documents[doc_id] = text.strip()
        self.results[doc_id] = result
        logger.info(
            "Ingested %s: %d entities, %d relations (%d rejected)",
            doc_id,
            len(result.entities),
            len(result.relations),
            len(result.rejected),
        )
        return result

    def ingest_many(self, documents: Mapping[str, str]) -> list[ExtractionResult]:
        return [self.ingest(doc_id, text) for doc_id, text in documents.items()]

    # -- validation --------------------------------------------------------- #
    @staticmethod
    def _parse_and_validate(doc_id: str, text: str, raw: str) -> ExtractionResult:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON for {doc_id}: {exc}") from exc

        valid_types = {t.value for t in EntityType}
        valid_relations = {r.value for r in RelationType}
        rejected: list[str] = []

        entities: list[ExtractedEntity] = []
        for item in payload.get("entities", []):
            name = str(item.get("name", "")).strip()
            if not name:
                rejected.append(f"entity without name: {item}")
                continue
            etype = item.get("type") if item.get("type") in valid_types else EntityType.UNKNOWN.value
            entities.append(ExtractedEntity(name, etype, [str(a) for a in item.get("aliases", [])]))

        relations: list[ExtractedRelation] = []
        for item in payload.get("relations", []):
            rel = str(item.get("relation", "")).upper()
            src, dst = str(item.get("source", "")).strip(), str(item.get("target", "")).strip()
            if rel not in valid_relations:
                rejected.append(f"unknown relation '{rel}'")
                continue
            if not src or not dst or src == dst:
                rejected.append(f"invalid endpoints {src!r} -> {dst!r}")
                continue
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            relations.append(
                ExtractedRelation(
                    src, rel, dst, confidence, str(item.get("evidence", "")), dict(item.get("attributes") or {})
                )
            )

        for reason in rejected:
            logger.warning("%s: rejected extraction - %s", doc_id, reason)
        return ExtractionResult(doc_id, text, entities, relations, rejected)

    def _load(self, result: ExtractionResult) -> None:
        for ent in result.entities:
            self.graph.add_entity(ent.name, ent.type, aliases=ent.aliases, doc_id=result.doc_id)
        for rel in result.relations:
            attrs = {k: v for k, v in rel.attributes.items() if v is not None}
            impact_weight = attrs.pop("impact_weight", None)
            self.graph.add_relationship(
                rel.source,
                rel.relation,
                rel.target,
                confidence=rel.confidence,
                evidence=rel.evidence,
                doc_id=result.doc_id,
                impact_weight=impact_weight,
                **attrs,
            )


def bootstrap_knowledge_graph(
    documents: Mapping[str, str] = SEED_DOCUMENTS,
    llm: LLMClient | None = None,
) -> tuple[KnowledgeGraph, KnowledgeExtractor]:
    """Build a fresh graph and ingest the seed corpus (called on app start-up)."""
    graph = KnowledgeGraph()
    extractor = KnowledgeExtractor(graph, llm)
    extractor.ingest_many(documents)
    return graph, extractor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    kg, ex = bootstrap_knowledge_graph()
    for res in ex.results.values():
        print(f"\n=== {res.doc_id} ===")
        for t in res.triples():
            print(f"  {t['source']} -[{t['relation']}]-> {t['target']}  (conf={t['confidence']})")
    print("\nGraph stats:", kg.stats())
