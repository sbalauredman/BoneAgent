from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from boneagent.agents.base import Agent, AgentResponse, MessageBus
from boneagent.domain import AgentKind, Candidate
from boneagent.evidence.records import HarmonizedStudy
from boneagent.search.space import CandidateSpace


class EvidenceRepository(Protocol):
    def search(self, query: str, limit: int) -> list[HarmonizedStudy]: ...

    def identifiers(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class LiteratureRequest:
    query: str
    candidate_count: int
    evidence_limit: int = 1000


@dataclass(frozen=True)
class LiteratureResult:
    candidates: tuple[Candidate, ...]
    studies: tuple[HarmonizedStudy, ...]
    knowledge_edges: tuple[tuple[str, str, str], ...]


class MemoryEvidenceRepository:
    def __init__(self, studies: list[HarmonizedStudy]) -> None:
        self._studies = studies

    def search(self, query: str, limit: int) -> list[HarmonizedStudy]:
        terms = set(query.lower().split())
        scored = []
        for study in self._studies:
            haystack = f"{study.source.title} {study.source.source_name}".lower()
            score = sum(term in haystack for term in terms)
            scored.append((score, study))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [study for _, study in scored[:limit]]

    def identifiers(self) -> tuple[str, ...]:
        return tuple(study.source.source_id for study in self._studies)


class LiteratureAgent(Agent[LiteratureRequest, LiteratureResult]):
    kind = AgentKind.LITERATURE

    def __init__(
        self,
        bus: MessageBus,
        repository: EvidenceRepository,
        candidate_space: CandidateSpace,
    ) -> None:
        super().__init__(bus)
        self.repository = repository
        self.candidate_space = candidate_space

    def _edges(self, studies: list[HarmonizedStudy]) -> tuple[tuple[str, str, str], ...]:
        edges = []
        for study in studies:
            source = study.source.source_id
            for material in study.materials:
                phase = material.normalized_formula or material.text
                edges.append((source, "reports_material", phase))
            for endpoint in study.endpoints:
                edges.append((source, "reports_endpoint", endpoint.endpoint.value))
            for link in study.links:
                edges.append(
                    (
                        f"{source}:material:{link.material_index}",
                        "associated_with",
                        f"{source}:endpoint:{link.endpoint_index}",
                    )
                )
        return tuple(edges)

    def act(self, request: LiteratureRequest, cycle: int) -> AgentResponse[LiteratureResult]:
        started = time.perf_counter()
        studies = self.repository.search(request.query, request.evidence_limit)
        candidates = self.candidate_space.sample(request.candidate_count, cycle)
        result = LiteratureResult(tuple(candidates), tuple(studies), self._edges(studies))
        sources = tuple(study.source.source_id for study in studies)
        self.notify(AgentKind.ORCHESTRATOR, "literature_result", cycle, result, sources)
        return AgentResponse(result, sources, time.perf_counter() - started)
