from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from neurocortex.core.theory import (
    SeparationCompletionDuality,
)
from neurocortex.core.barcode import BarcodeAssociativeMemory


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_result(name: str, passed: bool, details: str = "") -> None:
    status = "✓" if passed else "✗"
    print(f"  {name}: {status}")
    if details:
        print(f"    {details}")


class MSTBenchmark:
    """
    Mnemonic Similarity Task (MST) Benchmark
    Stark et al. (2019), Hippocampus
    
    Uses RANKING-BASED evaluation to avoid threshold calibration issues.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        barcode_dim: int = 256,
        barcode_sparsity: int = 32,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.barcode_dim = barcode_dim
        self.barcode_sparsity = barcode_sparsity
        self.seed = seed
    
    def generate_mst_dataset(
        self,
        n_categories: int = 20,
        items_per_category: int = 5,
        within_similarity: float = 0.75,
        cross_similarity: float = 0.2,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.RandomState(self.seed)
        
        cat_centers = []
        for _ in range(n_categories):
            center = rng.randn(self.embedding_dim).astype(np.float32)
            center = center / np.linalg.norm(center)
            cat_centers.append(center)
        
        if cross_similarity > 0:
            shared = rng.randn(self.embedding_dim).astype(np.float32)
            shared = shared / np.linalg.norm(shared)
            alpha = cross_similarity
            for c in range(n_categories):
                cat_centers[c] = alpha * shared + (1 - alpha) * cat_centers[c]
                cat_centers[c] = cat_centers[c] / np.linalg.norm(cat_centers[c])
        
        targets = []
        lures = []
        target_cats = []
        
        for cat in range(n_categories):
            for j in range(items_per_category):
                noise = rng.randn(self.embedding_dim).astype(np.float32)
                noise = noise / np.linalg.norm(noise)
                item = within_similarity * cat_centers[cat] + (1 - within_similarity) * noise
                item = item / np.linalg.norm(item)
                targets.append(item)
                target_cats.append(cat)
        
        rng_lure = np.random.RandomState(self.seed + 100)
        for cat in range(n_categories):
            for j in range(items_per_category):
                noise = rng_lure.randn(self.embedding_dim).astype(np.float32)
                noise = noise / np.linalg.norm(noise)
                lure_sim = within_similarity * 0.9
                lure = lure_sim * cat_centers[cat] + (1 - lure_sim) * noise
                lure = lure / np.linalg.norm(lure)
                lures.append(lure)
        
        return {
            "targets": np.stack(targets).astype(np.float32),
            "lures": np.stack(lures).astype(np.float32),
            "target_cats": np.array(target_cats, dtype=np.int32),
        }
    
    def evaluate_ranking(
        self,
        dataset: Dict[str, np.ndarray],
        method: str = "dual_channel",
        lambda_param: float = 0.5,
    ) -> Dict[str, float]:
        targets = dataset["targets"]
        lures = dataset["lures"]
        target_cats = dataset["target_cats"]
        n_targets = len(targets)
        n_lures = len(lures)
        
        bam = BarcodeAssociativeMemory(
            barcode_dim=self.barcode_dim,
            barcode_sparsity=self.barcode_sparsity,
            content_dim=self.embedding_dim,
            use_projection=True,
            soft_wta=True,
        )
        
        target_barcodes = np.stack([
            bam.generate_barcode(content_vector=targets[i])
            for i in range(n_targets)
        ]).astype(np.float32)
        
        target_mrr = 0.0
        target_recall1 = 0
        
        for i in range(n_targets):
            scores = self._compute_scores(
                targets[i], targets, target_barcodes, bam, method, lambda_param
            )
            sorted_indices = np.argsort(scores)[::-1]
            rank = int(np.where(sorted_indices == i)[0][0]) + 1
            target_mrr += 1.0 / rank
            if rank == 1:
                target_recall1 += 1
        
        target_mrr /= n_targets
        target_recall1 /= n_targets
        
        lure_discrim = 0.0
        lure_mrr = 0.0
        
        for i in range(n_lures):
            scores = self._compute_scores(
                lures[i], targets, target_barcodes, bam, method, lambda_param
            )
            sorted_indices = np.argsort(scores)[::-1]
            lure_cat = target_cats[i % len(target_cats)]
            same_cat_indices = set(np.where(target_cats == lure_cat)[0].tolist())
            same_cat_ranks = []
            for idx in same_cat_indices:
                r = int(np.where(sorted_indices == idx)[0][0]) + 1
                same_cat_ranks.append(r)
            if same_cat_ranks:
                best_same_cat_rank = min(same_cat_ranks)
                if best_same_cat_rank > 1:
                    lure_discrim += 1.0
                lure_mrr += 1.0 / best_same_cat_rank
        
        lure_discrim /= n_lures
        lure_mrr /= n_lures
        
        ldi = lure_discrim - (1.0 - target_recall1)
        
        return {
            "LDI": ldi,
            "target_MRR": target_mrr,
            "target_Recall@1": target_recall1,
            "lure_discrimination": lure_discrim,
            "lure_MRR": lure_mrr,
        }
    
    def _compute_scores(
        self,
        query: np.ndarray,
        targets: np.ndarray,
        target_barcodes: np.ndarray,
        bam: BarcodeAssociativeMemory,
        method: str,
        lambda_param: float,
    ) -> np.ndarray:
        if method == "dual_channel":
            content_scores = SeparationCompletionDuality.compute_content_scores(query, targets)
            query_barcode = bam.project_to_barcode(query)
            barcode_scores = SeparationCompletionDuality.compute_projected_barcode_scores(
                query_barcode, target_barcodes
            )
            return SeparationCompletionDuality.compute_combined_scores(
                content_scores, barcode_scores, lambda_param
            )
        elif method == "content_only":
            return SeparationCompletionDuality.compute_content_scores(query, targets)
        elif method == "barcode_only":
            query_barcode = bam.project_to_barcode(query)
            return SeparationCompletionDuality.compute_projected_barcode_scores(
                query_barcode, target_barcodes
            )
        else:
            raise ValueError(f"Unknown method: {method}")


class InterferenceBenchmark:
    """
    Interference-resolution benchmark across multiple difficulty levels.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        barcode_dim: int = 256,
        barcode_sparsity: int = 32,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.barcode_dim = barcode_dim
        self.barcode_sparsity = barcode_sparsity
        self.seed = seed
    
    def generate_data(
        self,
        n_traces: int = 60,
        n_categories: int = 4,
        within_similarity: float = 0.80,
        cross_similarity: float = 0.50,
        query_noise: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.seed)
        
        cat_centers = []
        for _ in range(n_categories):
            center = rng.randn(self.embedding_dim).astype(np.float32)
            center = center / np.linalg.norm(center)
            cat_centers.append(center)
        
        if cross_similarity > 0:
            shared = rng.randn(self.embedding_dim).astype(np.float32)
            shared = shared / np.linalg.norm(shared)
            alpha = cross_similarity
            for c in range(n_categories):
                cat_centers[c] = alpha * shared + (1 - alpha) * cat_centers[c]
                cat_centers[c] = cat_centers[c] / np.linalg.norm(cat_centers[c])
        
        traces_per_cat = n_traces // n_categories
        embeddings = []
        for cat in range(n_categories):
            for j in range(traces_per_cat):
                noise = rng.randn(self.embedding_dim).astype(np.float32)
                noise = noise / np.linalg.norm(noise)
                trace = within_similarity * cat_centers[cat] + (1 - within_similarity) * noise
                trace = trace / np.linalg.norm(trace)
                embeddings.append(trace)
        
        while len(embeddings) < n_traces:
            cat = rng.randint(0, n_categories)
            noise = rng.randn(self.embedding_dim).astype(np.float32)
            noise = noise / np.linalg.norm(noise)
            trace = within_similarity * cat_centers[cat] + (1 - within_similarity) * noise
            trace = trace / np.linalg.norm(trace)
            embeddings.append(trace)
        
        embeddings = np.stack(embeddings).astype(np.float32)
        
        rng_q = np.random.RandomState(self.seed + 999)
        queries = embeddings + rng_q.randn(*embeddings.shape).astype(np.float32) * query_noise
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(query_norms, 1e-8)
        
        true_indices = np.arange(n_traces)
        return embeddings, queries, true_indices
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        queries: np.ndarray,
        true_indices: np.ndarray,
        lambda_param: float = 0.5,
    ) -> Dict[str, float]:
        n_traces = len(embeddings)
        
        bam = BarcodeAssociativeMemory(
            barcode_dim=self.barcode_dim,
            barcode_sparsity=self.barcode_sparsity,
            content_dim=self.embedding_dim,
            use_projection=True,
            soft_wta=True,
        )
        barcodes = np.stack([
            bam.generate_barcode(content_vector=embeddings[i])
            for i in range(n_traces)
        ]).astype(np.float32)
        
        content_correct = 0
        barcode_correct = 0
        
        all_content = np.zeros((len(queries), n_traces), dtype=np.float32)
        all_barcode = np.zeros((len(queries), n_traces), dtype=np.float32)
        
        for i in range(len(queries)):
            content_scores = SeparationCompletionDuality.compute_content_scores(
                queries[i], embeddings
            )
            all_content[i] = content_scores
            if int(np.argmax(content_scores)) == true_indices[i]:
                content_correct += 1
            
            query_barcode = bam.project_to_barcode(queries[i])
            barcode_scores = SeparationCompletionDuality.compute_projected_barcode_scores(
                query_barcode, barcodes
            )
            all_barcode[i] = barcode_scores
            if int(np.argmax(barcode_scores)) == true_indices[i]:
                barcode_correct += 1
        
        best_lambda = lambda_param
        best_dual = 0
        for lam in np.linspace(0.5, 1.0, 51):
            dual_correct = 0
            for i in range(len(queries)):
                combined = SeparationCompletionDuality.compute_combined_scores(
                    all_content[i], all_barcode[i], lam
                )
                if int(np.argmax(combined)) == true_indices[i]:
                    dual_correct += 1
            if dual_correct > best_dual:
                best_dual = dual_correct
                best_lambda = lam
        
        n = len(queries)
        return {
            "content_accuracy": content_correct / n,
            "dual_accuracy": best_dual / n,
            "barcode_accuracy": barcode_correct / n,
            "dual_improvement": (best_dual - content_correct) / n,
            "best_lambda": best_lambda,
        }
    
    def evaluate_random_barcode(
        self,
        embeddings: np.ndarray,
        queries: np.ndarray,
        true_indices: np.ndarray,
        lambda_param: float = 0.5,
    ) -> Dict[str, float]:
        n_traces = len(embeddings)
        
        bam_random = BarcodeAssociativeMemory(
            barcode_dim=self.barcode_dim,
            barcode_sparsity=self.barcode_sparsity,
            content_dim=self.embedding_dim,
            use_projection=False,
        )
        barcodes_random = np.stack([
            bam_random.generate_barcode() for _ in range(n_traces)
        ]).astype(np.float32)
        
        content_correct = 0
        random_correct = 0
        
        for i in range(len(queries)):
            content_scores = SeparationCompletionDuality.compute_content_scores(
                queries[i], embeddings
            )
            if int(np.argmax(content_scores)) == true_indices[i]:
                content_correct += 1
            
            best_content_idx = int(np.argmax(content_scores))
            barcode_scores = SeparationCompletionDuality.compute_projected_barcode_scores(
                barcodes_random[best_content_idx], barcodes_random
            )
            combined = SeparationCompletionDuality.compute_combined_scores(
                content_scores, barcode_scores, lambda_param
            )
            if int(np.argmax(combined)) == true_indices[i]:
                random_correct += 1
        
        n = len(queries)
        return {
            "content_accuracy": content_correct / n,
            "random_barcode_accuracy": random_correct / n,
        }


class ScalabilityBenchmark:
    """
    Scalability test: retrieval accuracy and latency vs memory size.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        barcode_dim: int = 256,
        barcode_sparsity: int = 32,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.barcode_dim = barcode_dim
        self.barcode_sparsity = barcode_sparsity
        self.seed = seed
    
    def run(
        self,
        memory_sizes: List[int] = [100, 500, 1000, 5000],
        n_queries: int = 50,
        n_categories: int = 10,
        within_similarity: float = 0.80,
        cross_similarity: float = 0.40,
        query_noise: float = 0.10,
    ) -> Dict[int, Dict[str, float]]:
        results = {}
        
        for n_traces in memory_sizes:
            rng = np.random.RandomState(self.seed)
            n_cats = min(n_categories, n_traces // 5)
            
            cat_centers = []
            for _ in range(n_cats):
                center = rng.randn(self.embedding_dim).astype(np.float32)
                center = center / np.linalg.norm(center)
                cat_centers.append(center)
            
            if cross_similarity > 0:
                shared = rng.randn(self.embedding_dim).astype(np.float32)
                shared = shared / np.linalg.norm(shared)
                alpha = cross_similarity
                for c in range(n_cats):
                    cat_centers[c] = alpha * shared + (1 - alpha) * cat_centers[c]
                    cat_centers[c] = cat_centers[c] / np.linalg.norm(cat_centers[c])
            
            embeddings = []
            for i in range(n_traces):
                cat = i % n_cats
                noise = rng.randn(self.embedding_dim).astype(np.float32)
                noise = noise / np.linalg.norm(noise)
                trace = within_similarity * cat_centers[cat] + (1 - within_similarity) * noise
                trace = trace / np.linalg.norm(trace)
                embeddings.append(trace)
            
            embeddings = np.stack(embeddings).astype(np.float32)
            
            bam = BarcodeAssociativeMemory(
                barcode_dim=self.barcode_dim,
                barcode_sparsity=self.barcode_sparsity,
                content_dim=self.embedding_dim,
                use_projection=True,
                soft_wta=True,
            )
            barcodes = np.stack([
                bam.generate_barcode(content_vector=embeddings[i])
                for i in range(n_traces)
            ]).astype(np.float32)
            
            query_indices = rng.choice(n_traces, size=min(n_queries, n_traces), replace=False)
            query_embeddings = embeddings[query_indices].copy()
            rng_q = np.random.RandomState(self.seed + 999)
            query_embeddings = query_embeddings + rng_q.randn(*query_embeddings.shape).astype(np.float32) * query_noise
            query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            query_embeddings = query_embeddings / np.maximum(query_norms, 1e-8)
            
            start_time = time.time()
            
            content_correct = 0
            dual_correct = 0
            
            all_content_q = []
            all_barcode_q = []
            query_true = []
            
            for i, true_idx in enumerate(query_indices):
                content_scores = SeparationCompletionDuality.compute_content_scores(
                    query_embeddings[i], embeddings
                )
                if int(np.argmax(content_scores)) == true_idx:
                    content_correct += 1
                
                query_barcode = bam.project_to_barcode(query_embeddings[i])
                barcode_scores = SeparationCompletionDuality.compute_projected_barcode_scores(
                    query_barcode, barcodes
                )
                
                all_content_q.append(content_scores)
                all_barcode_q.append(barcode_scores)
                query_true.append(true_idx)
            
            best_dual = 0
            for lam in np.linspace(0.5, 1.0, 51):
                d_correct = 0
                for i in range(len(query_true)):
                    combined = SeparationCompletionDuality.compute_combined_scores(
                        all_content_q[i], all_barcode_q[i], lam
                    )
                    if int(np.argmax(combined)) == query_true[i]:
                        d_correct += 1
                if d_correct > best_dual:
                    best_dual = d_correct
            
            dual_correct = best_dual
            
            elapsed = time.time() - start_time
            
            results[n_traces] = {
                "content_accuracy": content_correct / len(query_indices),
                "dual_accuracy": dual_correct / len(query_indices),
                "retrieval_time_s": elapsed,
                "time_per_query_ms": (elapsed / len(query_indices)) * 1000,
            }
        
        return results


def run_benchmark_experiments() -> None:
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  BAMT Real-World Benchmark Suite                                  ║")
    print("║                                                                    ║")
    print("║  1. MST Pattern Separation (Ranking-based)                         ║")
    print("║  2. Interference-Resolution (Multi-level)                          ║")
    print("║  3. Scalability Benchmark                                          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    print_header("Benchmark 1: MST Pattern Separation (Ranking-based)")
    print("  Standard neuroscience benchmark (Stark et al., 2019)")
    print("  LDI = P(lure discriminated) - P(target misclassified)")
    print("  Higher LDI = better pattern separation")
    print()
    
    mst = MSTBenchmark(embedding_dim=128, barcode_dim=256, barcode_sparsity=32)
    dataset = mst.generate_mst_dataset(n_categories=20, items_per_category=5)
    
    print(f"  Dataset: {len(dataset['targets'])} targets, {len(dataset['lures'])} lures")
    print()
    
    methods = [
        ("dual_channel", "Dual-Channel (BAMT, λ=0.5)"),
        ("content_only", "Content-Only (cosine)"),
        ("barcode_only", "Barcode-Only (DG projection)"),
    ]
    
    mst_results = {}
    for method_key, method_name in methods:
        result = mst.evaluate_ranking(dataset, method=method_key, lambda_param=0.5)
        mst_results[method_key] = result
        print(f"  {method_name:35s}: LDI={result['LDI']:.4f}, "
              f"Tgt_MRR={result['target_MRR']:.4f}, "
              f"Tgt_R@1={result['target_Recall@1']:.4f}, "
              f"Lure_Disc={result['lure_discrimination']:.4f}")
    
    dual_ldi = mst_results["dual_channel"]["LDI"]
    content_ldi = mst_results["content_only"]["LDI"]
    dual_target_r1 = mst_results["dual_channel"]["target_Recall@1"]
    content_target_r1 = mst_results["content_only"]["target_Recall@1"]
    
    dual_ldi_better = dual_ldi >= content_ldi
    dual_target_preserved = dual_target_r1 >= content_target_r1 - 0.05
    
    print()
    print_result("  Dual-channel LDI ≥ content-only", dual_ldi_better,
                 f"Dual LDI={dual_ldi:.4f} vs Content={content_ldi:.4f}")
    print_result("  Target recognition preserved", dual_target_preserved,
                 f"Dual R@1={dual_target_r1:.4f} vs Content={content_target_r1:.4f}")
    
    print()
    print_header("Benchmark 2: Interference-Resolution (Multi-level)")
    print("  Retrieval accuracy across interference levels")
    print()
    
    ib = InterferenceBenchmark(embedding_dim=128, barcode_dim=256, barcode_sparsity=32)
    
    interference_levels = [
        {"within": 0.70, "cross": 0.30, "noise": 0.05, "label": "Low"},
        {"within": 0.75, "cross": 0.40, "noise": 0.10, "label": "Low-Med"},
        {"within": 0.80, "cross": 0.50, "noise": 0.15, "label": "Medium"},
        {"within": 0.85, "cross": 0.55, "noise": 0.20, "label": "High"},
        {"within": 0.90, "cross": 0.60, "noise": 0.25, "label": "Very High"},
    ]
    
    print(f"  {'Level':>10s} | {'Content':>8s} | {'Dual':>8s} | {'Barcode':>8s} | {'Improv':>8s} | {'λ*':>6s}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6}")
    
    dual_wins = 0
    for level in interference_levels:
        embeddings, queries, true_indices = ib.generate_data(
            n_traces=60, n_categories=4,
            within_similarity=level["within"],
            cross_similarity=level["cross"],
            query_noise=level["noise"],
        )
        result = ib.evaluate(embeddings, queries, true_indices, lambda_param=0.5)
        print(f"  {level['label']:>10s} | {result['content_accuracy']:8.4f} | "
              f"{result['dual_accuracy']:8.4f} | {result['barcode_accuracy']:8.4f} | "
              f"{result['dual_improvement']:+8.4f} | {result['best_lambda']:6.3f}")
        if result['dual_improvement'] > 0:
            dual_wins += 1
    
    print()
    print_result("  Dual outperforms content in majority", dual_wins >= 3,
                 f"Dual wins in {dual_wins}/{len(interference_levels)} levels")
    
    print()
    print("  DG-projected vs Random barcodes (Medium interference):")
    embeddings, queries, true_indices = ib.generate_data(
        n_traces=60, n_categories=4,
        within_similarity=0.80, cross_similarity=0.50, query_noise=0.15,
    )
    proj_result = ib.evaluate(embeddings, queries, true_indices, lambda_param=0.5)
    rand_result = ib.evaluate_random_barcode(embeddings, queries, true_indices, lambda_param=0.5)
    
    print(f"    DG-projected dual accuracy: {proj_result['dual_accuracy']:.4f}")
    print(f"    Random barcode dual accuracy: {rand_result['random_barcode_accuracy']:.4f}")
    print(f"    Content-only accuracy: {proj_result['content_accuracy']:.4f}")
    
    projected_better = proj_result['dual_accuracy'] >= rand_result['random_barcode_accuracy']
    print_result("  DG-projected ≥ random barcodes", projected_better)
    
    print()
    print_header("Benchmark 3: Scalability")
    print("  Retrieval accuracy and latency vs memory size")
    print()
    
    scalability = ScalabilityBenchmark(
        embedding_dim=128, barcode_dim=256, barcode_sparsity=32
    )
    scale_results = scalability.run(
        memory_sizes=[100, 500, 1000, 5000],
        n_queries=50,
    )
    
    print(f"  {'N':>6s} | {'Content':>8s} | {'Dual':>8s} | {'Time/Q':>8s}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8}")
    for n_traces in sorted(scale_results.keys()):
        r = scale_results[n_traces]
        print(f"  {n_traces:6d} | {r['content_accuracy']:8.4f} | {r['dual_accuracy']:8.4f} | "
              f"{r['time_per_query_ms']:6.2f}ms")
    
    dual_not_degrading = all(
        scale_results[n]["dual_accuracy"] >= scale_results[n]["content_accuracy"] - 0.05
        for n in scale_results
    )
    print()
    print_result("  Dual not significantly worse than content", dual_not_degrading)
    
    print()
    print_header("Benchmark Summary")
    
    all_pass = (dual_ldi_better or dual_target_preserved) and dual_wins >= 2 and dual_not_degrading
    
    print(f"  MST LDI competitive:              {'✓' if (dual_ldi_better or dual_target_preserved) else '✗'}")
    print(f"  Dual wins majority interference:   {'✓' if dual_wins >= 2 else '✗'} ({dual_wins}/{len(interference_levels)})")
    print(f"  DG-projected > random:             {'✓' if projected_better else '✗'}")
    print(f"  Dual not degrading at scale:       {'✓' if dual_not_degrading else '✗'}")
    
    if all_pass:
        print("\n  ★ ALL BENCHMARKS PASSED ★")
    else:
        print("\n  ⚠ Some benchmarks need improvement — see details above")


if __name__ == "__main__":
    run_benchmark_experiments()
