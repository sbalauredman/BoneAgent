from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from boneagent.agents.base import MessageBus
from boneagent.agents.clinical import ClinicalFeedbackAgent
from boneagent.agents.evaluation import EvaluationAgent
from boneagent.agents.literature import LiteratureAgent, MemoryEvidenceRepository
from boneagent.agents.simulation import AnalyticPropertyBackend, SimulationAgent
from boneagent.agents.synthesis import SynthesisAgent
from boneagent.console.common import atomic_json, configure_logging, parser
from boneagent.engine.campaign import CampaignEngine
from boneagent.randomness import set_seed
from boneagent.science.pareto import ParetoArchive
from boneagent.science.scoring import CompositeScorer
from boneagent.search.space import CandidateSpace
from boneagent.settings import load_settings

logger = logging.getLogger(__name__)


def build_engine(config: Path) -> CampaignEngine:
    settings = load_settings(config)
    set_seed(settings.seed)
    bus = MessageBus()
    space = CandidateSpace(settings.seed)
    literature = LiteratureAgent(bus, MemoryEvidenceRepository([]), space)
    simulation = SimulationAgent(bus, AnalyticPropertyBackend(settings.seed))
    clinical = ClinicalFeedbackAgent(bus)
    evaluation = EvaluationAgent(
        bus,
        CompositeScorer(settings.score),
        ParetoArchive([]),
        settings.seed,
    )
    synthesis = SynthesisAgent(bus)
    return CampaignEngine(
        settings,
        literature,
        simulation,
        clinical,
        evaluation,
        synthesis,
        space,
    )


def main() -> None:
    arguments = parser("Run the BoneAgent closed-loop campaign").parse_args()
    configure_logging(arguments.verbose)
    engine = build_engine(arguments.config)
    summary = engine.run()
    destination = Path(engine.settings.output_directory) / "campaign_summary.json"
    payload = {
        "cycles": summary.cycles,
        "converged": summary.converged,
        "hypervolume_history": summary.hypervolume_history,
        "pareto_front": [
            {
                "identifier": item.candidate.identifier,
                "cbmd": item.cbmd,
                "objectives": item.objective_vector.tolist(),
                "composition": asdict(item.candidate.composition),
                "scaffold": asdict(item.candidate.scaffold),
                "processing": asdict(item.candidate.processing),
            }
            for item in summary.pareto_front
        ],
    }
    atomic_json(destination, payload)
    logger.info("campaign completed after %d cycles", summary.cycles)


if __name__ == "__main__":
    main()
