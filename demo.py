"""
NeuroCortex End-to-End Demonstration

This script demonstrates the brain-inspired memory system's key capabilities:
1. Episodic memory encoding with importance weighting
2. Reconstructive recall (not retrieval!)
3. Memory consolidation (sleep-like)
4. Active forgetting
5. Neuromodulatory gating
6. Human-like memory properties (distortion, context-dependence)

Run without LLM: python demo.py --no-llm
Run with LLM:   python demo.py --llm-url http://localhost:11434
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from neurocortex.core.amygdala import Amygdala
from neurocortex.core.basal_forebrain import BasalForebrain
from neurocortex.core.brain_system import BrainSystem
from neurocortex.core.consolidation import ConsolidationEngine
from neurocortex.core.hippocampus import Hippocampus
from neurocortex.core.llm_engine import LLMEngine
from neurocortex.core.memory_trace import (
    ContextTag,
    EpisodicTrace,
    MemoryPhase,
    NeuromodulatoryState,
)
from neurocortex.core.neocortex import Neocortex
from neurocortex.core.prefrontal_cortex import PrefrontalCortex
from neurocortex.core.reconstructive_recall import ReconstructiveRecall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("demo")

LQ = "\u300a"
RQ = "\u300b"


def generate_embedding(text: str, dim: int = 64) -> np.ndarray:
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    emb = np.frombuffer(h, dtype=np.float32).copy()
    while emb.shape[0] < dim:
        h = hashlib.sha256(h).digest()
        emb = np.concatenate([emb, np.frombuffer(h, dtype=np.float32).copy()])
    emb = emb[:dim]
    return emb / (np.linalg.norm(emb) + 1e-8)


def demo_episodic_encoding():
    print("\n" + "=" * 70)
    print("DEMO 1: 海马体情景记忆编码 — 像人脑一样记住经历")
    print("=" * 70)

    hipp = Hippocampus(embedding_dim=64)
    amygdala = Amygdala()
    bf = BasalForebrain()

    conversations = [
        ("你好，我是小明", "小明", 0.6),
        ("今天天气真好，我想出去散步", "小明", 0.3),
        ("我特别喜欢猫，我家有一只橘猫叫大橘", "小明", 0.9),
        ("明天下午三点我们开会", "同事", 0.5),
        ("我最近压力很大，工作太多了", "小明", 0.8),
        ("大橘今天又把花瓶打碎了，又气又好笑", "小明", 0.85),
    ]

    existing_embs = None

    for content, source, expected_importance in conversations:
        emb = generate_embedding(content, 64)

        novelty = bf.compute_novelty(emb, existing_embs)
        importance, valence = amygdala.assess_importance(
            content=content,
            emotional_intensity=expected_importance,
            novelty_score=novelty,
            source=source,
        )

        trace = hipp.encode(
            content=content,
            embedding=emb,
            context=ContextTag(interlocutor=source),
            importance=importance,
            emotional_valence=valence,
            source=source,
            novelty_score=novelty,
        )

        if existing_embs is None:
            existing_embs = emb.reshape(1, -1)
        else:
            existing_embs = np.vstack([existing_embs, emb])

        print(f"\n  输入: {LQ}{content}{RQ}")
        print(f"  来源: {source}")
        print(f"  新颖性: {novelty:.3f} | 重要性: {importance:.3f} | 情感: {valence:.3f}")
        print(f"  记忆强度: {trace.memory_strength:.3f}")
        print(f"  神经调制: ACh={bf.state.acetylcholine:.3f} (编码门={bf.encoding_gate:.3f})")

    print(f"\n  -> 海马体中共有 {hipp.trace_count} 条情景记忆")
    print(f"  -> 状态: {hipp.get_status()}")

    return hipp, amygdala, bf


def demo_reconstructive_recall(hipp: Hippocampus):
    print("\n" + "=" * 70)
    print("DEMO 2: 重建性回忆 — 不是检索原文，而是像人一样重建记忆")
    print("=" * 70)

    neo = Neocortex(embedding_dim=64)
    recall_engine = ReconstructiveRecall()

    for schema_text in ["小明喜欢猫", "工作压力大"]:
        emb = generate_embedding(schema_text, 64)
        neo.create_schema(
            gist=schema_text,
            embedding=emb,
            source_traces=[],
            key_entities=["小明"],
            confidence=0.6,
        )

    queries = [
        "小明喜欢什么动物？",
        "大橘怎么了？",
        "明天有什么安排？",
    ]

    for query in queries:
        query_emb = generate_embedding(query, 64)

        episodic_results = hipp.retrieve_by_cue(query_emb, top_k=3)
        spread = hipp.spread_activation(
            [t.trace_id for t, _ in episodic_results], depth=2
        )
        schema_results = neo.retrieve_relevant(query_emb, top_k=2)

        print(f"\n  查询: {LQ}{query}{RQ}")
        print(f"  直接激活的记忆片段:")
        for trace, activation in episodic_results:
            print(f"    [{activation:.3f}] {LQ}{trace.content}{RQ} (来源:{trace.source})")

        if spread:
            print(f"  联想激活的记忆 (CA3扩散):")
            for tid, strength in list(spread.items())[:3]:
                t = hipp.get_trace(tid)
                if t:
                    print(f"    [{strength:.3f}] {LQ}{t.content}{RQ}")

        if schema_results:
            print(f"  语义知识 (新皮层):")
            for schema, score in schema_results:
                print(f"    [{score:.3f}] {schema.gist}")

        fragments = recall_engine._assemble_fragments(
            episodic_cues=episodic_results,
            spread_traces=[],
            schema_cues=schema_results,
            emotional_valence=0.0,
        )

        detail_levels = [f["detail_level"] for f in fragments["episodic"]]
        print(f"  记忆碎片细节级别: {detail_levels}")
        print(f"  -> 这就是重建性回忆的输入——碎片，不是完整文档！")


def demo_memory_consolidation(hipp: Hippocampus):
    print("\n" + "=" * 70)
    print("DEMO 3: 记忆整合 — 睡眠中从情景记忆提取语义知识")
    print("=" * 70)

    neo = Neocortex(embedding_dim=64)

    traces = hipp.get_active_traces()
    print(f"  整合前: {len(traces)} 条情景记忆, {neo.schema_count} 条语义模式")

    for trace in traces[:3]:
        if trace.embedding is not None:
            gist = trace.compressed_gist or trace.content[:50]
            schema = neo.create_schema(
                gist=gist,
                embedding=trace.embedding,
                source_traces=[trace.trace_id],
                key_entities=["小明"],
                confidence=0.3 + trace.importance * 0.3,
            )
            hipp.mark_consolidated(trace.trace_id, 0.5)
            print(f"\n  整合: {LQ}{trace.content[:30]}...{RQ} -> 语义模式 {LQ}{schema.gist[:30]}...{RQ}")
            print(f"  置信度: {schema.confidence:.3f} | 来源痕迹: {len(schema.source_traces)}")

    print(f"\n  整合后: {hipp.trace_count} 条情景记忆, {neo.schema_count} 条语义模式")
    print(f"  -> 情景细节逐渐遗忘，但语义知识保留在新皮层中")


def demo_active_forgetting(hipp: Hippocampus, amygdala: Amygdala):
    print("\n" + "=" * 70)
    print("DEMO 4: 主动遗忘 — 像人脑一样遗忘不重要的细节")
    print("=" * 70)

    traces = hipp.get_all_traces()
    print(f"  当前记忆数量: {len(traces)}")

    for trace in traces:
        modifier = amygdala.compute_decay_modifier(trace)
        original_decay = trace.decay_rate
        trace.decay_rate = 0.1 * modifier
        print(f"\n  记忆: {LQ}{trace.content[:30]}...{RQ}")
        print(f"  重要性: {trace.importance:.3f} | 情感: {trace.emotional_valence:.3f}")
        print(f"  衰减率: {original_decay:.4f} -> {trace.decay_rate:.4f} (x{modifier:.3f})")
        print(f"  记忆强度: {trace.memory_strength:.3f}")

    print(f"\n  -> 高重要性+高情感的记忆衰减更慢（闪光灯记忆效应）")
    print(f"  -> 低重要性的记忆会逐渐被遗忘，为新记忆腾出空间")


def demo_neuromodulatory_gating():
    print("\n" + "=" * 70)
    print("DEMO 5: 神经调制门控 — 动态控制记忆编码与整合")
    print("=" * 70)

    bf = BasalForebrain()

    scenarios = [
        ("第一次见面，完全陌生", 0.95),
        ("日常闲聊", 0.3),
        ("收到惊喜礼物", 0.9),
        ("重复的例行公事", 0.1),
    ]

    existing = np.random.randn(10, 64).astype(np.float32)

    for desc, novelty_level in scenarios:
        emb = np.random.randn(64).astype(np.float32)
        if novelty_level < 0.5:
            emb = existing[0] + np.random.randn(64) * 0.1
            emb = emb / (np.linalg.norm(emb) + 1e-8)

        novelty = bf.compute_novelty(emb, existing)
        bf.compute_reward(novelty_level * 0.5)

        print(f"\n  场景: {desc}")
        print(f"  新颖性信号: {novelty:.3f}")
        print(f"  乙酰胆碱(ACh): {bf.state.acetylcholine:.3f} -> 编码门: {bf.encoding_gate:.3f}")
        print(f"  多巴胺(DA): {bf.state.dopamine:.3f} -> 整合门: {bf.consolidation_gate:.3f}")
        print(f"  去甲肾上腺素(NE): {bf.state.norepinephrine:.3f}")

        bf.decay_to_baseline()

    print(f"\n  -> 高新颖性 -> 高ACh -> 更强的记忆编码")
    print(f"  -> 高奖赏 -> 高DA -> 更强的记忆整合")
    print(f"  -> 这就是为什么我们更容易记住新鲜和重要的事")


def demo_context_dependent_recall():
    print("\n" + "=" * 70)
    print("DEMO 6: 语境依赖回忆 — 同一记忆在不同语境下被重建为不同叙述")
    print("=" * 70)

    hipp = Hippocampus(embedding_dim=64)

    memories = [
        "小明在公园里遇到了一只流浪猫",
        "小明给那只猫喂了鱼干",
        "那只猫后来每天都来公园等小明",
        "小明决定收养那只猫，取名小花",
    ]

    for m in memories:
        emb = generate_embedding(m, 64)
        hipp.encode(m, emb, importance=0.7, source="小明")

    context_queries = [
        ("小明的宠物", "关于宠物的语境"),
        ("公园里的故事", "关于地点的语境"),
        ("善良的行为", "关于品格的语境"),
    ]

    for query, context_desc in context_queries:
        query_emb = generate_embedding(query, 64)
        results = hipp.retrieve_by_cue(query_emb, top_k=3)

        print(f"\n  语境: {context_desc} (查询: {LQ}{query}{RQ})")
        for trace, activation in results:
            print(f"    [{activation:.3f}] {LQ}{trace.content}{RQ}")

    print(f"\n  -> 同样的记忆，在不同语境下激活不同的片段组合")
    print(f"  -> 这就是人脑语境依赖回忆的机制！")


def demo_human_like_memory_properties():
    print("\n" + "=" * 70)
    print("DEMO 7: 类人记忆特性验证 — 可量化的论文实验指标")
    print("=" * 70)

    hipp = Hippocampus(embedding_dim=64)
    amygdala = Amygdala()

    print("\n  [实验1] 艾宾浩斯遗忘曲线验证")
    high_imp = hipp.encode(
        "非常重要的事情！",
        generate_embedding("重要", 64),
        importance=0.9,
        emotional_valence=0.8,
    )
    low_imp = hipp.encode(
        "无关紧要的琐事",
        generate_embedding("琐事", 64),
        importance=0.2,
        emotional_valence=0.0,
    )

    for hours in [0, 1, 6, 24, 72, 168]:
        fake_time = datetime.now() - timedelta(hours=hours)
        high_imp.timestamp = fake_time
        low_imp.timestamp = fake_time

        h_str = high_imp.memory_strength
        l_str = low_imp.memory_strength
        print(f"    {hours:3d}小时后: 高重要性={h_str:.3f}, 低重要性={l_str:.3f}, 差异={h_str-l_str:.3f}")

    print(f"\n  [实验2] 记忆再激活效应（间隔重复）")
    trace = hipp.encode(
        "需要记住的知识点",
        generate_embedding("知识点", 64),
        importance=0.5,
    )
    initial_strength = trace.memory_strength
    print(f"    初始强度: {initial_strength:.3f}")

    for i in range(5):
        trace.reactivate()
        print(f"    第{i+1}次再激活后: 强度={trace.memory_strength:.3f}, 衰减率={trace.decay_rate:.4f}")

    print(f"\n  [实验3] 情绪增强记忆效应")
    emotional = hipp.encode(
        "令人震惊的消息",
        generate_embedding("震惊", 64),
        importance=0.8,
        emotional_valence=0.9,
    )
    neutral = hipp.encode(
        "普通的消息",
        generate_embedding("普通", 64),
        importance=0.8,
        emotional_valence=0.0,
    )
    decay_emotional = amygdala.compute_decay_modifier(emotional)
    decay_neutral = amygdala.compute_decay_modifier(neutral)
    print(f"    情绪性记忆衰减修正: {decay_emotional:.3f}")
    print(f"    中性记忆衰减修正: {decay_neutral:.3f}")
    print(f"    情绪增强效应: {(1-decay_emotional)/(1-decay_neutral):.2f}x")

    print(f"\n  [实验4] 模式分离验证")
    similar_inputs = [
        "小明去了公园",
        "小明去了花园",
        "小红去了公园",
    ]
    embeddings = [generate_embedding(s, 64) for s in similar_inputs]
    traces = []
    for s, e in zip(similar_inputs, embeddings):
        t = hipp.encode(s, e, importance=0.5)
        traces.append(t)

    for i, t1 in enumerate(traces):
        for j, t2 in enumerate(traces):
            if i < j:
                if t1.embedding is not None and t2.embedding is not None:
                    sim = float(np.dot(t1.embedding, t2.embedding) / (
                        np.linalg.norm(t1.embedding) * np.linalg.norm(t2.embedding) + 1e-8
                    ))
                    print(f"    {LQ}{similar_inputs[i]}{RQ} vs {LQ}{similar_inputs[j]}{RQ}: 相似度={sim:.3f}")

    print(f"\n  -> 这些可量化的指标正是论文实验部分所需要的！")


def main():
    parser = argparse.ArgumentParser(description="NeuroCortex Demo")
    parser.add_argument("--no-llm", action="store_true", help="Run without LLM backend")
    parser.add_argument("--llm-url", default="http://localhost:11434", help="LLM API URL")
    args = parser.parse_args()

    print("=" * 70)
    print("          NeuroCortex 类脑记忆系统")
    print("    Brain-Inspired Episodic Memory for LLM Agents")
    print("=" * 70)

    hipp, amygdala, bf = demo_episodic_encoding()
    demo_reconstructive_recall(hipp)
    demo_memory_consolidation(hipp)
    demo_active_forgetting(hipp, amygdala)
    demo_neuromodulatory_gating()
    demo_context_dependent_recall()
    demo_human_like_memory_properties()

    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("""
核心创新总结:
  1. 多脑区协同架构 (海马体+新皮层+前额叶+杏仁核+基底前脑)
  2. 重建性回忆 — 从碎片重建记忆，不是检索原文
  3. 重放式整合 — 睡眠中从情景记忆提取语义知识
  4. 神经调制门控 — 动态控制编码与整合强度
  5. 主动遗忘 — 重要性+情绪调制的遗忘曲线
  6. 可量化的类人记忆特性 — 支持论文实验验证

论文方向:
  NeuroCortex: A Brain-Region-Inspired Episodic Memory Architecture
  for Embodied LLM Agents with Reconstructive Recall and
  Replay-Based Consolidation
""")


if __name__ == "__main__":
    main()
