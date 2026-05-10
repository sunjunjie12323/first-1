from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hippocampus import Hippocampus
from .memory_trace import EpisodicTrace, ReconstructedMemory, SemanticSchema
from .neocortex import Neocortex

logger = logging.getLogger(__name__)


class ReconstructiveRecall:
    """
    INNOVATION 1: Fragment-Level Reconstructive Recall with Variable Detail

    This is the CORE innovation of NeuroCortex, formally defined as:

    R(q) = LLM(q, {frag(e_i)}, {schema_j})

    where frag(e_i) produces fragments at VARIABLE detail levels:
      - FULL: complete content (when activation * importance > alpha_full)
      - GIST: compressed summary (when activation * importance > alpha_gist)
      - KEYWORD: key words only (when activation * importance <= alpha_gist)

    Differentiation from existing work:
    - RAG: R_RAG(q) = {e_i | sim(q, e_i) > theta} (retrieval, no reconstruction)
    - True Memory (Adler & Zehavi, 2026): mentions "reconstructive recall" but
      implements verbatim preservation + retrieval pipeline, not fragment reconstruction
    - E-mem (Wang et al., ICML 2026): multi-agent context reasoning, not fragment
      assembly with variable detail levels
    - CA3Mem (Zhang et al., AAAI 2026): trajectory recombination for GVAs, not
      conversational memory fragment reconstruction

    The VARIABLE DETAIL LEVEL mechanism is entirely novel. It models the
    vividness gradient in human recall: strongly activated memories are recalled
    in full detail, while weakly activated ones produce only vague impressions.
    This directly implements Bartlett's (1932) reconstructive memory theory.
    """

    def __init__(
        self,
        cue_activation_threshold: float = 0.15,
        max_cue_traces: int = 7,
        max_cue_schemas: int = 3,
        reconstruction_temperature: float = 0.7,
        alpha_full: float = 0.7,
        alpha_gist: float = 0.3,
        distortion_weights: Optional[Dict[str, float]] = None,
    ):
        self.cue_activation_threshold = cue_activation_threshold
        self.max_cue_traces = max_cue_traces
        self.max_cue_schemas = max_cue_schemas
        self.reconstruction_temperature = reconstruction_temperature
        self.alpha_full = alpha_full
        self.alpha_gist = alpha_gist

        self.distortion_weights = distortion_weights or {
            "activation_gap": 0.30,
            "spread_integration": 0.20,
            "schema_generalization": 0.25,
            "consolidation_instability": 0.25,
        }

    async def recall(
        self,
        query: str,
        query_embedding,
        hippocampus: Hippocampus,
        neocortex: Neocortex,
        llm_engine=None,
        emotional_valence: float = 0.0,
        current_context: str = "",
        neuromodulatory_state=None,
    ) -> ReconstructedMemory:
        """
        Perform reconstructive recall following the formal definition:

        Step 1: A(q) = {e_i | sim(emb(q), emb(e_i)) * strength(e_i) > theta}
        Step 2: A'(q) = A(q) U spread(A(q))
        Step 3: detail(e_i) = f(alpha_i) where alpha_i = activation * importance * emotion
        Step 4: R(q) = LLM(q, {frag(e_i)}, {schema_j})
        Step 5: D(R(q)) = distortion metric
        """
        # Step 1: Cue-driven hippocampal retrieval
        episodic_cues = hippocampus.retrieve_by_cue(
            query_embedding,
            top_k=self.max_cue_traces,
            min_strength=self.cue_activation_threshold,
        )

        # Step 2: CA3 spreading activation
        seed_ids = [trace.trace_id for trace, _ in episodic_cues]
        spread_activations = hippocampus.spread_activation(
            seed_ids, depth=2, decay=0.5
        )

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

        # Step 4: Fragment assembly with VARIABLE DETAIL LEVELS (NOVEL)
        fragments = self._assemble_fragments(
            episodic_cues=episodic_cues,
            spread_traces=spread_traces,
            schema_cues=schema_cues,
            emotional_valence=emotional_valence,
            neuromodulatory_state=neuromodulatory_state,
        )

        # Step 5: LLM reconstruction from fragments (NOVEL)
        if llm_engine and fragments["episodic"]:
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

        # Step 6: Compute confidence and distortion (INNOVATION 4)
        confidence = self._compute_confidence(episodic_cues, schema_cues)
        distortion = self._compute_distortion(
            episodic_cues=episodic_cues,
            spread_traces=spread_traces,
            schema_cues=schema_cues,
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
        neuromodulatory_state=None,
    ) -> Dict[str, List]:
        """
        Assemble memory fragments with VARIABLE DETAIL LEVELS.

        This is the key novel mechanism: each activated trace produces
        a fragment at a detail level determined by its combined activation:

          alpha_i = activation_i * (1 + importance_i) * (1 + |emotion_i|)

          detail(e_i) = FULL    if alpha_i > alpha_full  (0.7)
          detail(e_i) = GIST    if alpha_i > alpha_gist  (0.3)
          detail(e_i) = KEYWORD otherwise

        This models the vividness gradient in human recall:
        - Vivid recall (flashbulb memory): full detail
        - Normal recall: gist only
        - Vague impression: just keywords
        """
        fragments = {
            "episodic": [],
            "spread": [],
            "semantic": [],
        }

        ne_gate = 1.0
        if neuromodulatory_state is not None:
            ne_gate = 0.5 + 0.5 * neuromodulatory_state.norepinephrine

        for trace, activation in episodic_cues:
            importance_mod = 1.0 + trace.importance * 0.5
            emotional_mod = 1.0 + abs(trace.emotional_valence) * 0.3
            precision_mod = ne_gate

            alpha = activation * importance_mod * emotional_mod * precision_mod

            fragment_detail = self._determine_fragment_detail(alpha)

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
                "alpha": float(alpha),
                "consolidation_level": trace.consolidation_level,
            })

        for trace, activation in spread_traces:
            alpha = activation * (1.0 + trace.importance * 0.3)
            detail = "gist" if alpha > self.alpha_gist else "keyword"
            content = self._extract_gist(trace.content) if detail == "gist" else self._extract_keywords(trace.content)

            fragments["spread"].append({
                "content": content,
                "detail_level": detail,
                "activation": float(activation),
                "alpha": float(alpha),
            })

        for schema, score in schema_cues:
            fragments["semantic"].append({
                "gist": schema.gist,
                "confidence": schema.confidence,
                "key_entities": schema.key_entities,
                "abstract_concepts": schema.abstract_concepts,
                "activation": float(score),
                "maturity": schema.maturity,
            })

        return fragments

    def _determine_fragment_detail(self, alpha: float) -> str:
        """
        Determine fragment detail level based on combined activation alpha.

        This implements the formal definition:
          detail = FULL    if alpha > alpha_full
          detail = GIST    if alpha > alpha_gist
          detail = KEYWORD otherwise

        The thresholds are neurobiologically motivated:
        - alpha_full (0.7): Strong activation + high importance + emotional
          significance → vivid, detailed recall (like flashbulb memories)
        - alpha_gist (0.3): Moderate activation → gist-level recall
          (like remembering the main point of a conversation)
        - Below alpha_gist: Weak activation → only vague keywords
          (like having a "feeling of knowing" without specific details)
        """
        if alpha > self.alpha_full:
            return "full"
        elif alpha > self.alpha_gist:
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
        LLM reconstruction from fragments.

        This is the key step that makes recall RECONSTRUCTIVE rather than
        REPRODUCTIVE. The LLM receives fragments (not complete documents)
        and must reconstruct a coherent narrative, just as humans reconstruct
        memories from partial cues.

        The reconstruction prompt explicitly instructs the LLM to:
        1. Not simply repeat the fragments verbatim
        2. Integrate related fragments into a coherent narrative
        3. Acknowledge uncertainty where details are missing
        4. Emphasize emotionally significant content
        5. Not fabricate information beyond what the fragments provide
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
            detail_desc = {"full": "完整回忆", "gist": "大致印象", "keyword": "模糊感觉"}
            lines.append(
                f"  片段{i}（{detail_desc.get(detail, detail)}，"
                f"强度{frag['activation']:.2f}，alpha={frag.get('alpha', 0):.2f}）："
                f"{frag['content']}"
            )
            if frag.get("emotional_valence", 0) != 0:
                lines.append(f"    情感标记：{frag['emotional_valence']:.1f}")

        if fragments.get("spread"):
            lines.append("\n【联想激活片段】")
            for i, frag in enumerate(fragments.get("spread", []), 1):
                lines.append(
                    f"  联想{i}（{frag['detail_level']}级回忆，"
                    f"强度{frag['activation']:.2f}）："
                    f"{frag['content']}"
                )

        if fragments.get("semantic"):
            lines.append("\n【语义知识】")
            for i, frag in enumerate(fragments.get("semantic", []), 1):
                lines.append(
                    f"  知识{i}（置信度{frag['confidence']:.2f}，"
                    f"成熟度{frag.get('maturity', 0):.2f}）：{frag['gist']}"
                )
                if frag.get("key_entities"):
                    lines.append(f"    关键实体：{', '.join(frag['key_entities'])}")

        return "\n".join(lines)

    def _compute_confidence(
        self,
        episodic_cues: List[Tuple[EpisodicTrace, float]],
        schema_cues: List[Tuple[SemanticSchema, float]],
    ) -> float:
        if not episodic_cues and not schema_cues:
            return 0.0

        ep_conf = 0.0
        if episodic_cues:
            ep_conf = float(np.mean([a for _, a in episodic_cues]))

        sc_conf = 0.0
        if schema_cues:
            sc_conf = float(np.mean([s.confidence * a for s, a in schema_cues]))

        if episodic_cues and schema_cues:
            return float(0.6 * ep_conf + 0.4 * sc_conf)
        elif episodic_cues:
            return float(ep_conf)
        else:
            return float(sc_conf)

    def _compute_distortion(
        self,
        episodic_cues: List[Tuple[EpisodicTrace, float]],
        spread_traces: List[Tuple[EpisodicTrace, float]],
        schema_cues: List[Tuple[SemanticSchema, float]],
    ) -> float:
        """
        INNOVATION 4: Memory Distortion Quantification

        Formal definition:
          D(R(q)) = w1 * (1 - mean_activation)   [activation gap: more gap-filling needed]
                  + w2 * |spread| * scale          [integration: more traces merged]
                  + w3 * (|S| - 1) * scale          [generalization: more schemas merged]
                  + w4 * (1 - mean_consolidation)    [instability: less consolidated = more distortion]

        This metric quantifies how much a reconstructed memory may differ
        from the original experience. Higher distortion occurs when:
        1. Source traces have low activation (more gap-filling by LLM)
        2. Many spread activation traces are integrated (more blending)
        3. Multiple schemas are merged (more generalization)
        4. Source traces are not well consolidated (less stable memories)

        No existing LLM memory system quantifies memory distortion.
        """
        w = self.distortion_weights
        distortion = 0.0

        if episodic_cues:
            avg_activation = float(np.mean([a for _, a in episodic_cues]))
            distortion += w["activation_gap"] * (1.0 - avg_activation)

            avg_consolidation = float(np.mean([t.consolidation_level for t, _ in episodic_cues]))
            distortion += w["consolidation_instability"] * (1.0 - avg_consolidation)

        distortion += w["spread_integration"] * min(1.0, len(spread_traces) * 0.1)

        if len(schema_cues) > 1:
            distortion += w["schema_generalization"] * min(1.0, (len(schema_cues) - 1) * 0.15)

        return float(min(1.0, distortion))

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
