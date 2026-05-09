"""
向量存储系统
实现语义记忆的向量检索
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class MemoryVector:
    """记忆向量"""
    id: str
    vector: np.ndarray
    content: str
    metadata: Dict
    importance: float
    created_at: float
    last_accessed: float
    access_count: int


class VectorStore:
    """
    向量存储系统

    功能：
    1. 文本向量化
    2. 相似度检索
    3. 记忆索引
    4. 自动衰减
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.vectors: Dict[str, MemoryVector] = {}
        self.embedding_model = None

        self._init_embedding_model()

    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✓ 加载 SentenceTransformer 嵌入模型")
        except ImportError:
            print("⚠️ 未安装 sentence-transformers，使用随机向量")
            self.embedding_model = None

    def embed_text(self, text: str) -> np.ndarray:
        """将文本转为向量"""
        if self.embedding_model:
            embedding = self.embedding_model.encode(text)
            return embedding
        else:
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.embedding_dim)

    def add(self, content: str, metadata: Optional[Dict] = None,
            importance: float = 1.0) -> str:
        """
        添加记忆

        Args:
            content: 记忆内容
            metadata: 元数据
            importance: 重要性 (0-1)

        Returns:
            记忆 ID
        """
        memory_id = f"mem_{len(self.vectors)}_{datetime.now().timestamp()}"

        vector = self.embed_text(content)

        self.vectors[memory_id] = MemoryVector(
            id=memory_id,
            vector=vector,
            content=content,
            metadata=metadata or {},
            importance=importance,
            created_at=datetime.now().timestamp(),
            last_accessed=datetime.now().timestamp(),
            access_count=0
        )

        return memory_id

    def search(self, query: str, top_k: int = 5,
              importance_weight: float = 0.3) -> List[Dict]:
        """
        语义检索

        Args:
            query: 查询文本
            top_k: 返回数量
            importance_weight: 重要性权重

        Returns:
            检索结果列表
        """
        if not self.vectors:
            return []

        query_vector = self.embed_text(query)

        results = []
        for memory_id, memory in self.vectors.items():
            semantic_sim = self._cosine_similarity(query_vector, memory.vector)

            recency = self._time_decay(memory.last_accessed)

            combined_score = (
                semantic_sim * (1 - importance_weight) +
                memory.importance * importance_weight * 0.3 +
                recency * importance_weight * 0.2
            )

            results.append({
                "id": memory_id,
                "content": memory.content,
                "metadata": memory.metadata,
                "score": combined_score,
                "semantic_similarity": semantic_sim,
                "importance": memory.importance
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        for result in results[:top_k]:
            memory = self.vectors.get(result["id"])
            if memory:
                memory.last_accessed = datetime.now().timestamp()
                memory.access_count += 1

        return results[:top_k]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _time_decay(self, timestamp: float, half_life: float = 86400) -> float:
        """时间衰减（默认半衰期1天）"""
        age = datetime.now().timestamp() - timestamp
        return np.exp(-0.693 * age / half_life)

    def consolidate(self, decay_rate: float = 0.95):
        """
        记忆巩固

        1. 强化重要记忆
        2. 衰减不重要的
        3. 清理过期记忆
        """
        current_time = datetime.now().timestamp()

        to_remove = []
        for memory_id, memory in self.vectors.items():
            memory.importance *= decay_rate

            if memory.importance < 0.05:
                to_remove.append(memory_id)
            elif current_time - memory.last_accessed > 30 * 86400:
                to_remove.append(memory_id)

        for memory_id in to_remove:
            del self.vectors[memory_id]

        print(f"✓ 巩固完成：保留 {len(self.vectors)} 条记忆，清理 {len(to_remove)} 条")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        importances = [m.importance for m in self.vectors.values()]
        access_counts = [m.access_count for m in self.vectors.values()]

        return {
            "total_memories": len(self.vectors),
            "avg_importance": np.mean(importances) if importances else 0,
            "avg_access_count": np.mean(access_counts) if access_counts else 0,
            "embedding_dim": self.embedding_dim
        }
