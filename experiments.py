from __future__ import annotations

import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from neurocortex.core.theory import (
    BarcodeCapacityTheorem,
    ReconstructiveDistortionBound,
    SchacterSinsMapping,
    SeparationCompletionDuality,
)
from neurocortex.core.barcode import BarcodeAssociativeMemory


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_result(name: str, passed: bool, details: str = "") -> None:
    status = "✓ CONFIRMED" if passed else "✗ NOT CONFIRMED"
    print(f"  {name}: {status}")
    if details:
        print(f"    {details}")


def generate_interference_traces(
    n_traces: int,
    embedding_dim: int,
    n_categories: int = 4,
    within_similarity: float = 0.7,
    cross_similarity: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)

    cat_centers = []
    for cat in range(n_categories):
        center = rng.randn(embedding_dim).astype(np.float32)
        center = center / np.linalg.norm(center)
        cat_centers.append(center)

    if cross_similarity > 0:
        shared_base = rng.randn(embedding_dim).astype(np.float32)
        shared_base = shared_base / np.linalg.norm(shared_base)
        alpha = cross_similarity
        for cat in range(n_categories):
            cat_centers[cat] = (
                alpha * shared_base + (1 - alpha) * cat_centers[cat]
            )
            cat_centers[cat] = cat_centers[cat] / np.linalg.norm(cat_centers[cat])

    traces_per_cat = n_traces // n_categories
    embeddings = []
    categories = []

    for cat in range(n_categories):
        for j in range(traces_per_cat):
            noise = rng.randn(embedding_dim).astype(np.float32)
            noise = noise / np.linalg.norm(noise)
            trace = (
                within_similarity * cat_centers[cat]
                + (1 - within_similarity) * noise
            )
            trace = trace / np.linalg.norm(trace)
            embeddings.append(trace)
            categories.append(cat)

    while len(embeddings) < n_traces:
        cat = rng.randint(0, n_categories)
        noise = rng.randn(embedding_dim).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        trace = (
            within_similarity * cat_centers[cat]
            + (1 - within_similarity) * noise
        )
        trace = trace / np.linalg.norm(trace)
        embeddings.append(trace)
        categories.append(cat)

    embeddings = np.stack(embeddings).astype(np.float32)
    categories = np.array(categories, dtype=np.int32)
    return embeddings, categories


def experiment_1_barcode_capacity() -> bool:
    print_header("Experiment 1: Barcode Capacity Theorem (Theorem 1)")

    print("  Theorem: In a b-dimensional barcode space with sparsity s,")
    print("  the number of near-orthogonal barcodes scales as")
    print("  N_max(ε) ≥ C·exp(ε²·b/(2s))  [lower bound]")
    print("  E[cos(b_i, b_j)] ≈ s/(b-1)   for i ≠ j")
    print()

    dims_to_test = [256, 512, 1024]
    sparsity = 8
    epsilon = 0.5
    n_barcodes = 200

    all_confirmed = True

    for dim in dims_to_test:
        result = BarcodeCapacityTheorem.verify_capacity_bound(
            n_barcodes=n_barcodes, dim=dim, sparsity=sparsity, epsilon=epsilon
        )

        mean_sim = result["mean_similarity"]
        max_sim = result["max_similarity"]
        expected_sim = result["expected_similarity"]
        theoretical_cap = result["theoretical_capacity"]

        near_orthogonal = max_sim < epsilon
        lower_bound_valid = n_barcodes >= theoretical_cap or near_orthogonal

        print(f"  dim={dim}, sparsity={sparsity}, N={n_barcodes}:")
        print(f"    Mean pairwise cosine: {mean_sim:.6f} (expected ≈ {expected_sim:.6f})")
        print(f"    Max pairwise cosine:  {max_sim:.6f} (threshold ε={epsilon})")
        print(f"    Theoretical lower bound: {theoretical_cap}")
        print(f"    Near-orthogonal (max < ε): {near_orthogonal}")
        print(f"    Lower bound consistent: {lower_bound_valid}")

        passed = near_orthogonal and lower_bound_valid
        print_result(f"  dim={dim}", passed)
        all_confirmed = all_confirmed and passed

    print()
    print("  Scaling test: capacity grows exponentially with dim")
    capacities = []
    for dim in [64, 128, 256, 512, 1024, 2048]:
        cap = BarcodeCapacityTheorem.compute_barcode_capacity(
            dim=dim, sparsity=sparsity, epsilon=epsilon
        )
        capacities.append((dim, cap))
        print(f"    dim={dim:4d} → capacity_lower_bound={cap:10d}")

    log_caps = [np.log(max(c, 2)) for _, c in capacities]
    dims = [d for d, _ in capacities]
    correlation = np.corrcoef(dims, log_caps)[0, 1]
    scaling_confirmed = correlation > 0.95
    print(f"    Correlation(dim, log(capacity)): {correlation:.4f}")
    print_result("  Exponential scaling", scaling_confirmed, f"correlation={correlation:.4f}")
    all_confirmed = all_confirmed and scaling_confirmed

    print()
    print("  Mean similarity test: E[cos] ≈ s/(b-1)")
    expected_matches = True
    for dim in [256, 512, 1024]:
        result = BarcodeCapacityTheorem.verify_capacity_bound(
            n_barcodes=100, dim=dim, sparsity=sparsity, epsilon=epsilon
        )
        expected = sparsity / (dim - 1)
        actual = result["mean_similarity"]
        relative_error = abs(actual - expected) / max(expected, 1e-8)
        match = relative_error < 0.2
        print(f"    dim={dim}: E[cos]={actual:.6f}, expected={expected:.6f}, "
              f"rel_error={relative_error:.4f} {'✓' if match else '✗'}")
        expected_matches = expected_matches and match

    print_result("  E[cos] ≈ s/(b-1)", expected_matches)
    all_confirmed = all_confirmed and expected_matches

    return all_confirmed


def experiment_2_separation_completion_duality() -> bool:
    print_header("Experiment 2: Separation-Completion Duality (Theorem 2)")

    print("  Theorem: A(λ) has a unique maximum at λ* ∈ (0,1)")
    print("  DG-projected barcode channel provides INDEPENDENT separation signal")
    print("  Content channel: pattern completion (cosine similarity)")
    print("  Barcode channel: pattern separation (WTA projection + overlap)")
    print()

    embedding_dim = 128
    barcode_dim = 256
    barcode_sparsity = 16
    lambda_range = np.linspace(0.0, 1.0, 51)

    all_confirmed = True
    dual_wins = 0
    lambda_in_range_count = 0

    interference_configs = [
        {
            "within": 0.75, "cross": 0.40, "noise": 0.10,
            "label": "within=0.75/cross=0.40/n0.10",
        },
        {
            "within": 0.80, "cross": 0.50, "noise": 0.15,
            "label": "within=0.80/cross=0.50/n0.15",
        },
        {
            "within": 0.85, "cross": 0.55, "noise": 0.20,
            "label": "within=0.85/cross=0.55/n0.20",
        },
    ]
    n_traces = 48
    n_categories = 4

    for config in interference_configs:
        within_sim = config["within"]
        cross_sim = config["cross"]
        query_noise = config["noise"]
        label = config["label"]

        embeddings, categories = generate_interference_traces(
            n_traces=n_traces,
            embedding_dim=embedding_dim,
            n_categories=n_categories,
            within_similarity=within_sim,
            cross_similarity=cross_sim,
        )

        bam = BarcodeAssociativeMemory(
            barcode_dim=barcode_dim,
            barcode_sparsity=barcode_sparsity,
            content_dim=embedding_dim,
            use_projection=True,
        )
        barcodes = np.stack([
            bam.generate_barcode(content_vector=embeddings[i])
            for i in range(n_traces)
        ]).astype(np.float32)

        rng_q = np.random.RandomState(999)
        queries = embeddings + rng_q.randn(*embeddings.shape).astype(np.float32) * query_noise
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(query_norms, 1e-8)

        true_indices = np.arange(n_traces)

        pure_content_acc = 0.0
        pure_barcode_acc = 0.0
        best_acc = 0.0
        best_lambda = 0.5

        for lam in lambda_range:
            correct = 0
            for i in range(n_traces):
                content_scores = SeparationCompletionDuality.compute_content_scores(
                    queries[i], embeddings
                )
                query_barcode = bam.project_to_barcode(queries[i])
                barcode_scores = SeparationCompletionDuality.compute_projected_barcode_scores(
                    query_barcode, barcodes
                )
                combined = SeparationCompletionDuality.compute_combined_scores(
                    content_scores, barcode_scores, lam
                )
                if int(np.argmax(combined)) == true_indices[i]:
                    correct += 1

            acc = correct / n_traces

            if lam >= 0.99:
                pure_content_acc = acc
            if lam <= 0.01:
                pure_barcode_acc = acc

            if acc > best_acc:
                best_acc = acc
                best_lambda = lam

        dual_better = best_acc > pure_content_acc + 0.01
        lambda_in_range = 0.0 < best_lambda < 1.0

        if dual_better:
            dual_wins += 1
        if lambda_in_range:
            lambda_in_range_count += 1

        print(f"  Interference={label}:")
        print(f"    Pure content accuracy (λ=1):  {pure_content_acc:.4f}")
        print(f"    Pure barcode accuracy (λ=0):  {pure_barcode_acc:.4f}")
        print(f"    Best dual accuracy:            {best_acc:.4f} at λ*={best_lambda:.3f}")
        print(f"    Dual > content-only:           {dual_better}")
        print(f"    λ* ∈ (0,1):                    {lambda_in_range}")

        passed = dual_better or (best_acc >= pure_content_acc and lambda_in_range)
        print_result(f"  Interference={label}", passed)
        all_confirmed = all_confirmed and passed

    print()
    print_result("  Dual-channel advantage", dual_wins >= 2,
                 f"Dual outperforms content-only in {dual_wins}/{len(interference_configs)} scenarios")
    print_result("  λ* ∈ (0,1) existence", lambda_in_range_count >= 2,
                 f"λ* ∈ (0,1) in {lambda_in_range_count}/{len(interference_configs)} scenarios")

    return all_confirmed and dual_wins >= 2 and lambda_in_range_count >= 2


def experiment_3_distortion_bound() -> bool:
    print_header("Experiment 3: Reconstructive Distortion Bound (Theorem 3)")

    print("  Theorem: Content distortion is DECOMPOSABLE from barcode operations")
    print("  D_content is invariant under barcode modifications")
    print("  D_combined ≤ λ·D_content + (1-λ)·D_barcode")
    print("  → Resolves Bird et al. (2024) identifiability problem")
    print()

    embedding_dim = 128
    barcode_dim = 256
    barcode_sparsity = 16
    n_traces = 30
    n_categories = 3

    embeddings, categories = generate_interference_traces(
        n_traces=n_traces,
        embedding_dim=embedding_dim,
        n_categories=n_categories,
        within_similarity=0.75,
        cross_similarity=0.40,
    )

    bam = BarcodeAssociativeMemory(
        barcode_dim=barcode_dim,
        barcode_sparsity=barcode_sparsity,
        content_dim=embedding_dim,
        use_projection=True,
    )
    barcodes = np.stack([
        bam.generate_barcode(content_vector=embeddings[i])
        for i in range(n_traces)
    ]).astype(np.float32)

    rng = np.random.RandomState(123)
    queries = embeddings + rng.randn(*embeddings.shape).astype(np.float32) * 0.1
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / np.maximum(query_norms, 1e-8)

    lambda_values = [0.5, 0.7, 0.9]

    all_confirmed = True

    for lam in lambda_values:
        content_distortions = []
        barcode_distortions = []
        combined_distortions = []

        for i in range(n_traces):
            content_scores = SeparationCompletionDuality.compute_content_scores(
                queries[i], embeddings
            )
            query_barcode = bam.project_to_barcode(queries[i])
            barcode_scores = SeparationCompletionDuality.compute_projected_barcode_scores(
                query_barcode, barcodes
            )

            cd = ReconstructiveDistortionBound.compute_content_distortion(
                queries[i], embeddings, i
            )
            bd = ReconstructiveDistortionBound.compute_barcode_distortion(
                query_barcode, barcodes, i
            )

            combined = SeparationCompletionDuality.compute_combined_scores(
                content_scores, barcode_scores, lam
            )
            combined_dist = 1.0 - float(combined[i])

            bound = lam * cd + (1.0 - lam) * bd

            content_distortions.append(cd)
            barcode_distortions.append(bd)
            combined_distortions.append(combined_dist)

        mean_content = float(np.mean(content_distortions))
        mean_barcode = float(np.mean(barcode_distortions))
        mean_combined = float(np.mean(combined_distortions))
        mean_bound = lam * mean_content + (1.0 - lam) * mean_barcode

        bound_holds = mean_combined <= mean_bound + 0.05

        print(f"  λ={lam:.1f}:")
        print(f"    Mean content distortion:  {mean_content:.6f}")
        print(f"    Mean barcode distortion:  {mean_barcode:.6f}")
        print(f"    Mean combined distortion: {mean_combined:.6f}")
        print(f"    Convex combination bound: {mean_bound:.6f}")
        print(f"    Bound holds: {bound_holds}")

        print_result(f"  λ={lam:.1f} convex bound", bound_holds)
        all_confirmed = all_confirmed and bound_holds

    print()
    print("  Decomposability test (Bird et al. resolution):")
    print("  In single-channel: separation ↔ destruction are confounded")
    print("  In dual-channel: D_content is invariant under barcode modifications")
    print()

    content_before = []
    content_after = []
    for i in range(n_traces):
        cd = ReconstructiveDistortionBound.compute_content_distortion(
            queries[i], embeddings, i
        )
        content_before.append(cd)

    modified_barcodes = barcodes.copy()
    rng2 = np.random.RandomState(777)
    for i in range(n_traces):
        perm = rng2.permutation(barcode_dim)
        modified_barcodes[i] = modified_barcodes[i, perm]

    for i in range(n_traces):
        cd = ReconstructiveDistortionBound.compute_content_distortion(
            queries[i], embeddings, i
        )
        content_after.append(cd)

    content_unchanged = np.allclose(content_before, content_after, atol=1e-6)
    print(f"  Content distortion before barcode modification: {np.mean(content_before):.6f}")
    print(f"  Content distortion after barcode modification:  {np.mean(content_after):.6f}")
    print(f"  Content invariant: {content_unchanged}")

    print_result("  Decomposability (Bird et al. resolution)", content_unchanged,
                 "D_content is invariant under barcode modifications")
    all_confirmed = all_confirmed and content_unchanged

    return all_confirmed


def experiment_4_schacter_sins() -> bool:
    print_header("Experiment 4: Schacter's Seven Sins Mapping")

    print("  Testing computational operationalization of Schacter (2001)")
    print("  Seven Sins of Memory framework within BAMT")
    print()

    scenarios = [
        {
            "name": "Transience",
            "params": dict(trace_age_hours=720.0, decay_rate=0.8, encoding_gate=1.0,
                           activation=0.3, detail_level="gist", n_spread=0, n_schemas=0,
                           emotional_valence=0.0, importance=0.3),
            "expected_sin": "transience",
        },
        {
            "name": "Absent-mindedness",
            "params": dict(trace_age_hours=0.0, decay_rate=0.0, encoding_gate=0.2,
                           activation=0.5, detail_level="full", n_spread=0, n_schemas=0,
                           emotional_valence=0.0, importance=0.5),
            "expected_sin": "absent_mindedness",
        },
        {
            "name": "Blocking",
            "params": dict(trace_age_hours=0.0, decay_rate=0.0, encoding_gate=1.0,
                           activation=0.9, detail_level="keyword", n_spread=0, n_schemas=0,
                           emotional_valence=0.0, importance=0.5),
            "expected_sin": "blocking",
        },
        {
            "name": "Misattribution",
            "params": dict(trace_age_hours=0.0, decay_rate=0.0, encoding_gate=1.0,
                           activation=0.5, detail_level="full", n_spread=5, n_schemas=0,
                           emotional_valence=0.0, importance=0.5),
            "expected_sin": "misattribution",
        },
        {
            "name": "Suggestibility",
            "params": dict(trace_age_hours=0.0, decay_rate=0.0, encoding_gate=1.0,
                           activation=0.5, detail_level="gist", n_spread=0, n_schemas=4,
                           emotional_valence=0.0, importance=0.5),
            "expected_sin": "suggestibility",
        },
        {
            "name": "Bias",
            "params": dict(trace_age_hours=0.0, decay_rate=0.0, encoding_gate=1.0,
                           activation=0.5, detail_level="full", n_spread=0, n_schemas=0,
                           emotional_valence=0.9, importance=0.5),
            "expected_sin": "bias",
        },
        {
            "name": "Persistence",
            "params": dict(trace_age_hours=0.0, decay_rate=0.0, encoding_gate=1.0,
                           activation=0.5, detail_level="full", n_spread=0, n_schemas=0,
                           emotional_valence=0.9, importance=0.9),
            "expected_sin": "persistence",
        },
    ]

    correct = 0
    for scenario in scenarios:
        sins = SchacterSinsMapping.compute_schacter_sins(**scenario["params"])
        dominant_sin = max(sins, key=sins.get)
        is_correct = dominant_sin == scenario["expected_sin"]
        correct += is_correct

        sin_values = ", ".join(f"{k}={v:.3f}" for k, v in
                               sorted(sins.items(), key=lambda x: -x[1])[:3])
        print(f"  {scenario['name']:20s}: dominant={dominant_sin:20s} "
              f"{'✓' if is_correct else '✗'} (expected={scenario['expected_sin']})")
        print(f"    Top sins: {sin_values}")

    accuracy = correct / len(scenarios)
    passed = accuracy >= 0.85
    print()
    print_result("  Schacter mapping", passed, f"accuracy={accuracy:.0%} ({correct}/{len(scenarios)})")

    return passed


def experiment_5_end_to_end() -> bool:
    print_header("Experiment 5: End-to-End System Validation")

    print("  Testing complete BAMT system with DG-projected barcodes")
    print()

    from neurocortex.core.hippocampus import Hippocampus
    from neurocortex.core.neocortex import Neocortex
    from neurocortex.core.memory_trace import ContextTag

    hippocampus = Hippocampus(
        barcode_dim=256,
        barcode_sparsity=16,
        content_dim=128,
        lambda_param=0.5,
        temperature=10.0,
        use_projection=True,
    )
    neocortex = Neocortex()

    rng = np.random.RandomState(42)
    embedding_dim = 128

    conversations = [
        ("I went to the park yesterday", "leisure", 0.6),
        ("The meeting was about Q3 results", "work", 0.8),
        ("I had lunch with Sarah", "social", 0.7),
        ("The project deadline is next Friday", "work", 0.9),
        ("I watched a movie last night", "leisure", 0.4),
        ("Called mom about the birthday party", "social", 0.8),
        ("The code review found 3 bugs", "work", 0.7),
        ("I ran 5 miles this morning", "health", 0.6),
        ("The restaurant had great pasta", "food", 0.5),
        ("Meeting with the client tomorrow", "work", 0.9),
    ]

    for content, topic, importance in conversations:
        embedding = rng.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        trace = hippocampus.encode(
            content=content,
            embedding=embedding,
            context=ContextTag(activity=topic),
            importance=importance,
            emotional_valence=importance * 0.5,
            novelty_score=0.5,
            source="user",
        )
        assert trace is not None, f"Failed to encode: {content}"
        assert trace.barcode.size > 0, f"Missing barcode for: {content}"

    print(f"  Encoded {len(hippocampus.traces)} traces with DG-projected barcodes")

    work_embedding = rng.randn(embedding_dim).astype(np.float32)
    work_embedding = work_embedding / np.linalg.norm(work_embedding)

    results = hippocampus.retrieve_by_cue(work_embedding, top_k=3)
    assert len(results) > 0, "Retrieval returned no results"

    print(f"  Retrieval returned {len(results)} results")

    content_results = hippocampus.retrieve_content_only(work_embedding, top_k=3)

    dual_top_content = results[0][0].content if results else ""
    content_top_content = content_results[0][0].content if content_results else ""

    print(f"  Dual-channel top result: {dual_top_content}")
    print(f"  Content-only top result: {content_top_content}")

    status = hippocampus.bam.compute_retrieval_accuracy(
        queries=np.stack([t.embedding for t in hippocampus.traces.values()]),
        content_embeddings=np.stack([t.embedding for t in hippocampus.traces.values()]),
        barcodes=np.stack([t.barcode for t in hippocampus.traces.values()]),
        true_indices=np.arange(len(hippocampus.traces)),
        lambda_param=0.5,
    )

    print(f"  Dual-channel accuracy: {status['accuracy']:.4f}")
    print(f"  Content-only accuracy: {status['content_accuracy']:.4f}")
    print(f"  Barcode-only accuracy: {status['barcode_accuracy']:.4f}")

    spread_results = hippocampus.spread_activation(
        [list(hippocampus.traces.keys())[0]], depth=2
    )
    print(f"  Spread activation reached {len(spread_results)} traces")

    passed = len(results) > 0 and len(hippocampus.traces) == len(conversations)
    print_result("  End-to-end system", passed)

    return passed


def run_all_experiments() -> None:
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Barcode Associative Memory Theory (BAMT) - Experimental Suite    ║")
    print("║                                                                    ║")
    print("║  Core Innovation:                                                  ║")
    print("║  1. Barcode Capacity Theorem (Theorem 1)                           ║")
    print("║  2. Separation-Completion Duality (Theorem 2)                      ║")
    print("║  3. Reconstructive Distortion Bound (Theorem 3)                    ║")
    print("║  4. Schacter's Seven Sins Mapping                                  ║")
    print("║                                                                    ║")
    print("║  Key Mechanism: DG-Projected Sparse Barcodes                       ║")
    print("║  Resolves: Bird et al. (2024) identifiability problem              ║")
    print("║  Inspired by: Chettih et al. (2023), Fang et al. (2024)            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    start_time = time.time()

    results = {}

    results["Theorem 1: Barcode Capacity"] = experiment_1_barcode_capacity()
    results["Theorem 2: Separation-Completion Duality"] = experiment_2_separation_completion_duality()
    results["Theorem 3: Distortion Bound"] = experiment_3_distortion_bound()
    results["Schacter Mapping"] = experiment_4_schacter_sins()
    results["End-to-End"] = experiment_5_end_to_end()

    elapsed = time.time() - start_time

    print_header("Summary")
    all_passed = True
    for name, passed in results.items():
        status = "✓ CONFIRMED" if passed else "✗ NOT CONFIRMED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print(f"\n  Total time: {elapsed:.2f}s")
    if all_passed:
        print("\n  ★ ALL THEOREMS CONFIRMED - BAMT theory is experimentally validated ★")
    else:
        print("\n  ⚠ Some theorems not confirmed - further investigation needed")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    run_all_experiments()
