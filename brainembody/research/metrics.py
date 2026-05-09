"""
研究级评估框架
标准信息检索指标 + 统计显著性检验
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class RetrievalResult:
    """单次检索结果"""
    query_id: str
    retrieved_ids: List[str]
    relevant_ids: List[str]
    scores: List[float]


class MetricsCalculator:
    """
    标准信息检索指标计算器

    指标：
    - F1 Score
    - Recall@K
    - Precision@K
    - MRR (Mean Reciprocal Rank)
    - Hit Rate
    - NDCG@K
    """

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Precision@K"""
        if k == 0:
            return 0.0
        retrieved_at_k = retrieved[:k]
        hits = len(set(retrieved_at_k) & set(relevant))
        return hits / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Recall@K"""
        if not relevant:
            return 0.0
        retrieved_at_k = retrieved[:k]
        hits = len(set(retrieved_at_k) & set(relevant))
        return hits / len(relevant)

    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """F1@K"""
        p = MetricsCalculator.precision_at_k(retrieved, relevant, k)
        r = MetricsCalculator.recall_at_k(retrieved, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def mrr(retrieved: List[str], relevant: List[str]) -> float:
        """Mean Reciprocal Rank"""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def hit_rate(retrieved: List[str], relevant: List[str]) -> float:
        """Hit Rate"""
        return 1.0 if set(retrieved) & set(relevant) else 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """NDCG@K"""
        def dcg_at_k(rel_list, k):
            dcg = 0.0
            for i in range(min(k, len(rel_list))):
                dcg += (2 ** rel_list[i] - 1) / np.log2(i + 2)
            return dcg

        rel_scores = [1.0 if doc in relevant else 0.0 for doc in retrieved[:k]]
        ideal_scores = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)

        dcg = dcg_at_k(rel_scores, k)
        idcg = dcg_at_k(ideal_scores, k)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def compute_all_metrics(
        results: List[RetrievalResult],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """计算所有指标"""
        metrics = {}

        for k in k_values:
            precisions = []
            recalls = []
            f1s = []
            ndcgs = []

            for result in results:
                precisions.append(MetricsCalculator.precision_at_k(
                    result.retrieved_ids, result.relevant_ids, k))
                recalls.append(MetricsCalculator.recall_at_k(
                    result.retrieved_ids, result.relevant_ids, k))
                f1s.append(MetricsCalculator.f1_at_k(
                    result.retrieved_ids, result.relevant_ids, k))
                ndcgs.append(MetricsCalculator.ndcg_at_k(
                    result.retrieved_ids, result.relevant_ids, k))

            metrics[f"Precision@{k}"] = np.mean(precisions)
            metrics[f"Recall@{k}"] = np.mean(recalls)
            metrics[f"F1@{k}"] = np.mean(f1s)
            metrics[f"NDCG@{k}"] = np.mean(ndcgs)

        mrrs = [MetricsCalculator.mrr(r.retrieved_ids, r.relevant_ids) for r in results]
        hits = [MetricsCalculator.hit_rate(r.retrieved_ids, r.relevant_ids) for r in results]

        metrics["MRR"] = np.mean(mrrs)
        metrics["HitRate"] = np.mean(hits)

        return metrics


class SignificanceTester:
    """
    统计显著性检验

    方法：
    - Paired t-test
    - Wilcoxon signed-rank test
    - Cohen's d (效应量)
    - Bootstrap confidence interval
    """

    @staticmethod
    def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Dict:
        """配对t检验"""
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        return {
            "test": "paired_t_test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_005": p_value < 0.05,
            "significant_at_001": p_value < 0.01
        }

    @staticmethod
    def wilcoxon_test(scores_a: List[float], scores_b: List[float]) -> Dict:
        """Wilcoxon符号秩检验（非参数）"""
        try:
            stat, p_value = stats.wilcoxon(scores_a, scores_b)
        except ValueError:
            stat, p_value = 0.0, 1.0
        return {
            "test": "wilcoxon",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant_at_005": p_value < 0.05
        }

    @staticmethod
    def cohens_d(scores_a: List[float], scores_b: List[float]) -> Dict:
        """Cohen's d 效应量"""
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        std_a = np.std(scores_a, ddof=1)
        std_b = np.std(scores_b, ddof=1)

        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        if abs(d) < 0.2:
            magnitude = "negligible"
        elif abs(d) < 0.5:
            magnitude = "small"
        elif abs(d) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        return {
            "cohens_d": float(d),
            "magnitude": magnitude,
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "improvement": float((mean_a - mean_b) / mean_b * 100) if mean_b != 0 else 0
        }

    @staticmethod
    def bootstrap_ci(scores: List[float], n_bootstrap: int = 1000,
                     confidence: float = 0.95) -> Dict:
        """Bootstrap置信区间"""
        scores = np.array(scores)
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return {
            "mean": float(np.mean(scores)),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "confidence": confidence
        }

    @staticmethod
    def full_comparison(scores_a: List[float], scores_b: List[float],
                       name_a: str = "System A", name_b: str = "System B") -> Dict:
        """完整统计比较"""
        return {
            "system_a": name_a,
            "system_b": name_b,
            "n_samples": len(scores_a),
            "paired_t_test": SignificanceTester.paired_t_test(scores_a, scores_b),
            "wilcoxon": SignificanceTester.wilcoxon_test(scores_a, scores_b),
            "effect_size": SignificanceTester.cohens_d(scores_a, scores_b),
            "ci_a": SignificanceTester.bootstrap_ci(scores_a),
            "ci_b": SignificanceTester.bootstrap_ci(scores_b)
        }
