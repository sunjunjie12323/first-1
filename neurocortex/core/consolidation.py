from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from neurocortex.core.amygdala import Amygdala
from neurocortex.core.hippocampus import Hippocampus
from neurocortex.core.llm_engine import LLMEngine
from neurocortex.core.memory_trace import EpisodicTrace, MemoryPhase, SemanticSchema
from neurocortex.core.neocortex import Neocortex

logger = logging.getLogger(__name__)


class Consolidation:
    def __init__(
        self,
        hippocampus: Hippocampus,
        neocortex: Neocortex,
        amygdala: Amygdala,
        llm_engine: LLMEngine,
        consolidation_threshold: float = 0.3,
        max_consolidation_per_cycle: int = 50,
    ):
        self.hippocampus = hippocampus
        self.neocortex = neocortex
        self.amygdala = amygdala
        self.llm_engine = llm_engine
        self.consolidation_threshold = consolidation_threshold
        self.max_consolidation_per_cycle = max_consolidation_per_cycle

    async def consolidate(self, consolidation_gate: float = 1.0) -> Dict[str, int]:
        if consolidation_gate < 0.2:
            logger.info("Consolidation gate too low, skipping")
            return {"consolidated": 0, "forgotten": 0, "schemas_created": 0}

        candidates = self._get_consolidation_candidates()
        consolidated = 0
        forgotten = 0
        schemas_created = 0

        for trace in candidates[:self.max_consolidation_per_cycle]:
            if trace.memory_strength < self.consolidation_threshold:
                self._forget_trace(trace.trace_id)
                forgotten += 1
                continue

            await self._replay_and_consolidate(trace)
            consolidated += 1

            if trace.consolidation_level >= 0.7 and trace.phase != MemoryPhase.SEMANTIC:
                schema = await self._extract_semantic(trace)
                if schema is not None:
                    schemas_created += 1

        return {
            "consolidated": consolidated,
            "forgotten": forgotten,
            "schemas_created": schemas_created,
        }

    def _get_consolidation_candidates(self) -> List[EpisodicTrace]:
        traces = list(self.hippocampus.traces.values())
        traces.sort(key=lambda t: t.consolidation_level)
        return traces

    async def _replay_and_consolidate(self, trace: EpisodicTrace) -> None:
        trace.consolidation_level = min(1.0, trace.consolidation_level + 0.1)

        if trace.consolidation_level > 0.5:
            trace.phase = MemoryPhase.CONSOLIDATING
        if trace.consolidation_level > 0.8:
            trace.phase = MemoryPhase.SEMANTIC

        modified_decay = self.amygdala.modify_decay_rate(trace.decay_rate, trace.importance)
        trace.decay_rate = modified_decay

        logger.debug(f"Replayed trace {trace.trace_id}, consolidation={trace.consolidation_level:.2f}")

    async def _extract_semantic(self, trace: EpisodicTrace) -> Optional[SemanticSchema]:
        try:
            prompt = (
                f"Extract the core semantic gist from this episodic memory. "
                f"Provide only the gist as a single sentence, followed by key entities "
                f"separated by commas on a new line.\n\n"
                f"Memory: {trace.content}"
            )
            response = await self.llm_engine.generate(prompt, max_tokens=128, temperature=0.3)

            lines = response.strip().split("\n")
            gist = lines[0].strip() if lines else trace.content[:100]
            key_entities = []
            if len(lines) > 1:
                key_entities = [e.strip() for e in lines[1].split(",") if e.strip()]

            schema = self.neocortex.create_schema(
                gist=gist,
                embedding=trace.embedding.copy(),
                source_trace_id=trace.trace_id,
                key_entities=key_entities,
                initial_confidence=trace.consolidation_level * 0.5,
            )
            return schema
        except Exception as e:
            logger.error(f"Semantic extraction failed for trace {trace.trace_id}: {e}")
            return None

    def _forget_trace(self, trace_id: str) -> None:
        self.hippocampus.traces.pop(trace_id, None)
        self.hippocampus.association_graph.pop(trace_id, None)
        for neighbors in self.hippocampus.association_graph.values():
            neighbors.pop(trace_id, None)
        logger.debug(f"Forgot trace {trace_id}")
