from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .hippocampus import Hippocampus
from .memory_trace import EpisodicTrace, ReconstructedMemory, SemanticSchema
from .neocortex import Neocortex

logger = logging.getLogger(__name__)


class ReconstructiveRecall:
    """
    Reconstructive Recall - the CORE INNOVATION of NeuroCortex.

    Unlike RAG systems that retrieve exact documents and insert them
    into prompts, this module implements human-like reconstructive
    memory recall. In the human brain, recall is not playback of a
    recording — it is an active reconstruction process that:

    1. Activates fragmented cues from multiple memory traces
    2. Fills in gaps using semantic knowledge
    3. Integrates emotional context from the amygdala
    4. Produces a coherent but potentially distorted narrative

    Key properties of reconstructive recall (matching human memory):
    - CONTEXT-DEPENDENT: The same memory is reconstructed differently
      depending on the current context and cue
    - INTEGRATIVE: Multiple related traces are merged into a single
      narrative, just as humans blend related memories
    - DISTORTABLE: Like human memory, reconstruction can introduce
      distortions — this is a FEATURE, not a bug
    - ADAPTIVE: Emotional context modulates what is emphasized in
      the reconstruction

    This is fundamentally different from:
    - RAG: exact document retrieval → no reconstruction
    - Conversation history: chronological playback → no integration
    - Fine-tuning: weight modification → no dynamic recall
    """

    def __init__(
        self,
        cue_activation_threshold: float = 0.15,
        max_cue_traces: int = 7,
        max_cue_schemas: int = 3,
        reconstruction_temperature: float = 0.7,
        distortion_sensitivity: float = 0.3,
    ):
        self.cue_activation_threshold = cue_activation_threshold
        self.max_cue_traces = max_cue_traces
        self.max_cue_schemas = max_cue_schemas
        self.reconstruction_temperature = reconstruction_temperature
        self.distortion_sensitivity = distortion_sensitivity

    async def recall(
        self,
        query: str,
        query_embedding,
        hippocampus: Hippocampus,
        neocortex: Neocortex,
        llm_engine=None,
        emotional_valence: float = 0.0,
        current_context: str = "",
    ) -> ReconstructedMemory:
        """
        Perform reconstructive recall for a given query.

        The process mirrors human memory recall:
        1. CUE GENERATION: The query activates related episodic traces
           in the hippocampus (pattern completion)
        2. SPREADING ACTIVATION: Associated traces are activated
           through the association graph (like free association)
        3. SCHEMA ACTIVATION: Relevant semantic schemas are retrieved
           from the neocortex (provides general knowledge)
        4. FRAGMENT ASSEMBLY: Activated traces and schemas provide
           fragmented cues, not complete documents
        5. LLM RECONSTRUCTION: The LLM reconstructs a coherent
           narrative from these fragments — this is the key step
           that makes recall reconstructive rather than reproductive
        """
        # Step 1: Hippocampal cue-driven retrieval
        episodic_cues = hippocampus.retrieve_by_cue(
            query_embedding,
            top_k=self.max_cue_traces,
            min_strength=self.cue_activation_threshold,
        )

        # Step 2: Spreading activation through associations
        seed_ids = [trace.trace_id for trace, _ in episodic_cues]
        spread_activations = hippocampus.spread_activation(
            seed_ids, depth=2, decay=0.5
        )

        # Collect additionally activated traces
        spread_traces = []
        for trace_id, activation in spread_activations.items():
            if activation >= self.cue_activation_threshold * 0.5:
                trace = hippocampus.get_trace(trace_id)
                if trace and trace.trace_id not in seed_ids:
                    spread_traces.append((trace, activation))

        # Step 3: Neocortical schema retrieval
        schema_cues = neocortex.retrieve_relevant(
            query_embedding,
            top_k=self.max_cue_schemas,
        )

        # Step 4: Assemble fragments for reconstruction
        fragments = self._assemble_fragments(
            episodic_cues=episodic_cues,
            spread_traces=spread_traces,
            schema_cues=schema_cues,
            emotional_valence=emotional_valence,
        )

        # Step 5: Reconstruct using LLM
        if llm_engine and fragments:
            reconstructed_narrative = await self._llm_reconstruct(
                query=query,
                fragments=fragments,
                current_context=current_context,
                llm_engine=llm_engine,
            )
        else:
            reconstructed_narrative = self._fallback_reconstruct(
                query, fragments
            )

        # Compute confidence and distortion metrics
        confidence = self._compute_confidence(episodic_cues, schema_cues)
        distortion = self._estimate_distortion(
            episodic_cues, spread_traces, schema_cues
        )

        source_traces = [t.trace_id for t, _ in episodic_cues]
        source_traces += [t.trace_id for t, _ in spread_traces]
        source_schemas = [s.schema_id for s, _ in schema_cues]

        memory = ReconstructedMemory(
            query=query,
            reconstructed_narrative=reconstructed_narrative,
            source_traces=source_traces,
            source_schemas=source_schemas,
            confidence=confidence,
            distortion_score=distortion,
            emotional_tone=emotional_valence,
        )

        logger.info(
            f"Reconstructive recall: {len(source_traces)} traces, "
            f"{len(source_schemas)} schemas, confidence={confidence:.2f}, "
            f"distortion={distortion:.2f}"
        )

        return memory

    def _assemble_fragments(
        self,
        episodic_cues: List[Tuple[EpisodicTrace, float]],
        spread_traces: List[Tuple[EpisodicTrace, float]],
        schema_cues: List[Tuple[SemanticSchema, float]],
        emotional_valence: float,
    ) -> Dict[str, List]:
        """
        Assemble memory fragments from multiple brain regions.
        These are NOT complete documents — they are fragments,
        cues, and gists that will be reconstructed by the LLM.
        """
        fragments = {
            "episodic": [],
            "spread": [],
            "semantic": [],
        }

        for trace, activation in episodic_cues:
            importance_mod = 1.0 + trace.importance * 0.5
            emotional_mod = 1.0 + abs(trace.emotional_valence) * 0.3

            fragment_detail = self._determine_fragment_detail(
                activation * importance_mod * emotional_mod
            )

            if fragment_detail == "full":
                content = trace.content
            elif fragment_detail == "gist":
                content = trace.compressed_gist or self._extract_gist(trace.content)
            else:
                content = self._extract_keywords(trace.content)

            fragments["episodic"].append({
                "content": content,
                "detail_level": fragment_detail,
                "activation": float(activation),
                "importance": trace.importance,
                "emotional_valence": trace.emotional_valence,
                "source": trace.source,
                "timestamp": trace.timestamp.isoformat(),
            })

        for trace, activation in spread_traces:
            fragments["spread"].append({
                "content": self._extract_gist(trace.content),
                "detail_level": "gist",
                "activation": float(activation),
                "association": "indirect",
            })

        for schema, score in schema_cues:
            fragments["semantic"].append({
                "gist": schema.gist,
                "confidence": schema.confidence,
                "key_entities": schema.key_entities,
                "abstract_concepts": schema.abstract_concepts,
                "activation": float(score),
            })

        return fragments

    def _determine_fragment_detail(self, combined_strength: float) -> str:
        """
        Determine how much detail to include in a fragment.
        Strong activation → full detail (like vivid recall)
        Medium activation → gist only (like remembering the main point)
        Weak activation → keywords only (like having a vague feeling)
        This mirrors how human recall varies in vividness.
        """
        if combined_strength > 0.7:
            return "full"
        elif combined_strength > 0.3:
            return "gist"
        else:
            return "keyword"

    async def _llm_reconstruct(
        self,
        query: str,
        fragments: Dict[str, List],
        current_context: str,
        llm_engine,
    ) -> str:
        """
        Use the LLM to reconstruct a coherent memory narrative
        from fragmented cues. This is the key innovation step.

        The LLM receives FRAGMENTS, not complete documents, and
        must reconstruct what it remembers — just like a human
        reconstructing a memory from partial cues.
        """
        fragment_text = self._format_fragments_for_llm(fragments)

        prompt = (
            "你正在从自己的记忆中回忆信息。这不是检索文档，而是像人脑一样"
            "从记忆碎片中重建回忆。以下是你被激活的记忆片段：\n\n"
            f"{fragment_text}\n\n"
            f"当前对话语境：{current_context}\n\n"
            f"关于这个问题：{query}\n\n"
            "请基于这些记忆碎片重建你的回忆。注意：\n"
            "1. 你不是在复述原文，而是在回忆——像人一样，回忆可能不完整\n"
            "2. 将相关的记忆片段整合成一个连贯的叙述\n"
            "3. 如果某些细节模糊，用你的理解来填补，但要标注不确定的部分\n"
            "4. 情感色彩较强的记忆应该更突出\n"
            "5. 只输出你回忆起来的内容，不要添加你不知道的信息\n\n"
            "你的回忆："
        )

        response = await llm_engine.generate(
            prompt=prompt,
            temperature=self.reconstruction_temperature,
            max_tokens=500,
        )

        return response

    def _fallback_reconstruct(
        self, query: str, fragments: Dict[str, List]
    ) -> str:
        """Fallback reconstruction when LLM is not available."""
        parts = []

        for frag in fragments.get("episodic", []):
            parts.append(frag["content"])

        for frag in fragments.get("semantic", []):
            parts.append(frag["gist"])

        return "；".join(parts) if parts else "我对此没有清晰的记忆。"

    def _format_fragments_for_llm(self, fragments: Dict[str, List]) -> str:
        lines = []

        lines.append("【情景记忆片段】")
        for i, frag in enumerate(fragments.get("episodic", []), 1):
            detail = frag["detail_level"]
            lines.append(
                f"  片段{i}（{detail}级回忆，强度{frag['activation']:.2f}）："
                f"{frag['content']}"
            )
            if frag.get("emotional_valence", 0) != 0:
                lines.append(f"    情感标记：{frag['emotional_valence']:.1f}")

        if fragments.get("spread"):
            lines.append("\n【联想激活片段】")
            for i, frag in enumerate(fragments.get("spread", []), 1):
                lines.append(
                    f"  联想{i}（间接回忆，强度{frag['activation']:.2f}）："
                    f"{frag['content']}"
                )

        if fragments.get("semantic"):
            lines.append("\n【语义知识】")
            for i, frag in enumerate(fragments.get("semantic", []), 1):
                lines.append(
                    f"  知识{i}（置信度{frag['confidence']:.2f}）：{frag['gist']}"
                )
                if frag.get("key_entities"):
                    lines.append(f"    关键实体：{', '.join(frag['key_entities'])}")

        return "\n".join(lines)

    def _extract_gist(self, content: str, max_length: int = 100) -> str:
        if len(content) <= max_length:
            return content
        sentences = content.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n")
        gist = sentences[0]
        for s in sentences[1:]:
            if len(gist) + len(s) <= max_length:
                gist += s
            else:
                break
        return gist

    def _extract_keywords(self, content: str, max_words: int = 10) -> str:
        words = content.split()
        return " ".join(words[:max_words])

    def _compute_confidence(
        self,
        episodic_cues: List[Tuple[EpisodicTrace, float]],
        schema_cues: List[Tuple[SemanticSchema, float]],
    ) -> float:
        if not episodic_cues and not schema_cues:
            return 0.0

        ep_conf = 0.0
        if episodic_cues:
            ep_conf = np.mean([a for _, a in episodic_cues])

        sc_conf = 0.0
        if schema_cues:
            sc_conf = np.mean([s.confidence * a for s, a in schema_cues])

        if episodic_cues and schema_cues:
            return float(0.6 * ep_conf + 0.4 * sc_conf)
        elif episodic_cues:
            return float(ep_conf)
        else:
            return float(sc_conf)

    def _estimate_distortion(
        self,
        episodic_cues: List[Tuple[EpisodicTrace, float]],
        spread_traces: List[Tuple[EpisodicTrace, float]],
        schema_cues: List[Tuple[SemanticSchema, float]],
    ) -> float:
        """
        Estimate potential distortion in the reconstructed memory.
        Higher distortion occurs when:
        - Many spread activation traces are involved (more integration needed)
        - Low-activation cues are used (more gap-filling required)
        - Multiple schemas are merged (more generalization)
        This metric is itself a novel contribution — quantifying
        memory distortion in an artificial memory system.
        """
        distortion = 0.0

        if episodic_cues:
            avg_activation = np.mean([a for _, a in episodic_cues])
            distortion += (1.0 - avg_activation) * self.distortion_sensitivity

        distortion += len(spread_traces) * 0.05 * self.distortion_sensitivity

        if len(schema_cues) > 1:
            distortion += (len(schema_cues) - 1) * 0.1 * self.distortion_sensitivity

        return float(min(1.0, distortion))


