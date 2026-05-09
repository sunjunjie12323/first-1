"""
研究级嵌入模型
支持 sentence-transformers 真实语义嵌入
"""

import numpy as np
from typing import List, Union, Optional


class ResearchEmbedder:
    """研究级嵌入模型"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = None, fallback_dim: int = 384):
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.model = None
        self.embedding_dim = fallback_dim

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✓ 加载嵌入模型: {model_name} (dim={self.embedding_dim}, device={device})")
        except Exception as e:
            print(f"⚠️ 嵌入模型加载失败: {e}")
            print(f"  使用降级模式 (dim={fallback_dim})")
            self.model = None
            self.embedding_dim = fallback_dim

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """文本转向量"""
        if isinstance(text, str):
            text = [text]

        if self.model is not None:
            embeddings = self.model.encode(text, show_progress_bar=False, normalize_embeddings=True)
            if len(text) == 1:
                return embeddings[0]
            return embeddings
        else:
            return self._fallback_embed(text)

    def _fallback_embed(self, texts: List[str]) -> Union[np.ndarray, List[np.ndarray]]:
        """降级嵌入：基于词袋 + TF-IDF风格"""
        results = []
        for text in texts:
            vec = np.zeros(self.fallback_dim)
            words = text.lower().split()
            for i, word in enumerate(words[:self.fallback_dim]):
                idx = i % self.fallback_dim
                vec[idx] += hash(word) % 100 / 100.0

            # 添加字符级n-gram特征
            for i, ch in enumerate(text[:50]):
                idx = (i + len(words)) % self.fallback_dim
                vec[idx] += hash(ch) % 50 / 50.0

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            results.append(vec)

        if len(results) == 1:
            return results[0]
        return np.array(results)

    def similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的语义相似度"""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
