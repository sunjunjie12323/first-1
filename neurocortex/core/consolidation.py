from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .hippocampus import Hippocampus
from .memory_trace import EpisodicTrace, MemoryPhase, SemanticSchema
from .neocortex import Neocortex

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """
    Replay-based memory consolidation engine.

    Implements the hippocampal-neocortical dialogue that occurs
    during sleep in the human brain. During "sleep" periods (idle
    time or scheduled consolidation), the system:

    1. SELECTS: Chooses recent episodic traces for replay,
       prioritized by importance and recency
    2. REPLAYS: Re-activates these traces through the LLM,
       which extracts semantic gist and relationships
    3. CONSOLIDATES: Creates or updates semantic schemas in the
       neocortex based on the extracted knowledge
    4. PRUNES: Applies active forgetting — decaying weak,
       unimportant memories and strengthening reinforced ones

    Innovation: Using the LLM as the "consolidation engine" is
    analogous to how the brain uses replay to extract patterns.
    The LLM naturally performs the abstraction and generalization
    that the neocortex performs slowly over time.

    This is fundamentally different from:
    - Periodic fine-tuning: modifies weights permanently
    - Simple summarization: doesn't create structured schemas
    - Vector DB compaction: doesn't extract semantic knowledge
    """

    def __init__(
        self,
        replay_batch_size: int = 5,
        consolidation_threshold: float = 0.3,
        forgetting_threshold: float = 0.05,
        replay_temperature: float = 0.3,
        max_consolidation_rounds: int = 3,
    ):
        self.replay_batch_size = replay_batch_size
        self.consolidation_threshold = consolidation_threshold
        self.forgetting_threshold = forgetting_threshold
        self.replay_temperature = replay_temperature
        self.max_consolidation_rounds = max_consolidation_rounds

        self._consolidation_count: int = 0

    async def consolidate(
        self,
        hippocampus: Hippocampus,
        neocortex: Neocortex,
        llm_engine=None,
        amygdala=None,
    ) -> Dict:
        """
        Perform one round of sleep-like consolidation.

        Returns a summary of what was consolidated.
        """
        self._consolidation_count += 1

        candidates = hippocampus.get_consolidation_candidates(min_age_hours=0.0)

        if not candidates:
            logger.info("Consolidation: no candidates found")
            return {"consolidated": 0, "schemas_created": 0, "pruned": 0}

        batch = candidates[: self.replay_batch_size]

        schemas_created = 0
        schemas_updated = 0
        consolidated_traces = 0

        for trace in batch:
            schema_result = await self._replay_and_consolidate(
                trace, hippocampus, neocortex, llm_engine, amygdala
            )

            if schema_result:
                if schema_result.consolidation_rounds == 1:
                    schemas_created += 1
                else:
                    schemas_updated += 1
                consolidated_traces += 1

        pruned = self._apply_forgetting(hippocampus, amygdala)

        result = {
            "round": self._consolidation_count,
            "candidates": len(candidates),
            "consolidated": consolidated_traces,
            "schemas_created": schemas_created,
            "schemas_updated": schemas_updated,
            "pruned": pruned,
        }

        logger.info(
            f"Consolidation round {self._consolidation_count}: "
            f"{result}"
        )

        return result

    async def _replay_and_consolidate(
        self,
        trace: EpisodicTrace,
        hippocampus: Hippocampus,
        neocortex: Neocortex,
        llm_engine=None,
        amygdala=None,
    ) -> Optional[SemanticSchema]:
        """
        Replay a single episodic trace and consolidate it into
        the neocortex. The LLM extracts semantic gist from the
        replayed experience.
        """
        associated_traces = []
        for assoc_id in trace.associations[:3]:
            assoc_trace = hippocampus.get_trace(assoc_id)
            if assoc_trace and not assoc_trace.is_decayed:
                associated_traces.append(assoc_trace)

        if llm_engine:
            gist, entities, concepts = await self._llm_extract_semantics(
                trace, associated_traces, llm_engine
            )
        else:
            gist = self._simple_extract_gist(trace, associated_traces)
            entities = self._simple_extract_entities(trace)
            concepts = []

        if not gist:
            return None

        embedding = trace.embedding
        if embedding is None:
            return None

        schema = neocortex.create_schema(
            gist=gist,
            embedding=embedding,
            source_traces=[trace.trace_id] + [t.trace_id for t in associated_traces],
            key_entities=entities,
            abstract_concepts=concepts,
            confidence=0.3 + trace.importance * 0.3,
        )

        consolidation_level = min(1.0, trace.consolidation_level + 0.3)
        hippocampus.mark_consolidated(trace.trace_id, consolidation_level)

        return schema

    async def _llm_extract_semantics(
        self,
        trace: EpisodicTrace,
        associated_traces: List[EpisodicTrace],
        llm_engine,
    ) -> Tuple[str, List[str], List[str]]:
        """
        Use the LLM to extract semantic knowledge from a replayed
        episodic trace. This is the core of the consolidation process.

        The LLM is asked to extract:
        1. The semantic gist (what was learned, not what was said)
        2. Key entities mentioned
        3. Abstract concepts involved
        """
        replay_context = f"原始经历：{trace.content}\n"
        if associated_traces:
            replay_context += "相关经历：\n"
            for at in associated_traces:
                replay_context += f"  - {at.content}\n"

        prompt = (
            "你正在从一段经历中提取语义知识，就像人脑在睡眠中整理白天的记忆一样。\n"
            "请从以下经历中提取：\n\n"
            f"{replay_context}\n"
            "请按以下格式输出：\n"
            "要点：[用一句话概括这段经历的核心含义，不是复述，而是提炼]\n"
            "实体：[列出涉及的人、物、地点等，用逗号分隔]\n"
            "概念：[列出涉及的抽象概念，如信任、时间、合作等，用逗号分隔]"
        )

        response = await llm_engine.generate(
            prompt=prompt,
            temperature=self.replay_temperature,
            max_tokens=300,
        )

        gist, entities, concepts = self._parse_extraction_response(response)

        if not gist:
            gist = trace.compressed_gist or self._simple_extract_gist(trace, [])

        return gist, entities, concepts

    def _parse_extraction_response(
        self, response: str
    ) -> Tuple[str, List[str], List[str]]:
        gist = ""
        entities = []
        concepts = []

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("要点：") or line.startswith("要点:"):
                gist = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("实体：") or line.startswith("实体:"):
                entity_str = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                entities = [e.strip() for e in entity_str.split(",") if e.strip()]
            elif line.startswith("概念：") or line.startswith("概念:"):
                concept_str = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                concepts = [c.strip() for c in concept_str.split(",") if c.strip()]

        return gist, entities, concepts

    def _apply_forgetting(
        self,
        hippocampus: Hippocampus,
        amygdala=None,
    ) -> int:
        """
        Apply active forgetting to the hippocampal store.
        Memories below the forgetting threshold are removed,
        but their consolidated knowledge persists in the neocortex.

        This is analogous to how humans forget episodic details
        while retaining semantic knowledge — you may not remember
        exactly when you learned something, but you still know it.
        """
        pruned = 0
        traces = hippocampus.get_all_traces()

        for trace in traces:
            if trace.is_decayed:
                if trace.consolidation_level >= 0.5:
                    trace.phase = MemoryPhase.CONSOLIDATING
                    continue
                hippocampus.remove_trace(trace.trace_id)
                pruned += 1
                continue

            if amygdala:
                decay_modifier = amygdala.compute_decay_modifier(trace)
                trace.decay_rate = 0.1 * decay_modifier

        return pruned

    def _simple_extract_gist(
        self,
        trace: EpisodicTrace,
        associated_traces: List[EpisodicTrace],
    ) -> str:
        content = trace.content
        if len(content) <= 100:
            return content
        sentences = content.replace("。", "。\n").split("\n")
        return sentences[0].strip()

    def _simple_extract_entities(self, trace: EpisodicTrace) -> List[str]:
        words = trace.content.split()
        entities = [w for w in words if len(w) >= 2 and not w.isascii()]
        return entities[:5]

    @property
    def consolidation_count(self) -> int:
        return self._consolidation_count
