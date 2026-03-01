import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from consensus_graph import (
    build_similarity_matrix,
    build_graph,
    detect_dominant_cluster,
    score_candidates,
)


def run_stage8(candidates: list, language: str = "en",
               similarity_threshold: float = 0.5,
               outlier_penalty: float = 0.3) -> dict:
    """
    Stage 8: Cross-Transcript Consensus Modeling.

    Algorithm:
        1. Build NxN pairwise Levenshtein similarity matrix
        2. Construct similarity graph (edges where sim >= threshold)
        3. Detect dominant cluster via connected-component analysis
           weighted by size × internal cohesion
        4. Score each candidate by average similarity to dominant cluster;
           penalise outliers that fall outside it

    Args:
        candidates:           list of candidate transcript strings (raw/light-norm)
        language:             ISO language code (informational only)
        similarity_threshold: minimum similarity [0,1] to form a graph edge
        outlier_penalty:      fraction [0,1] to reduce outlier scores

    Returns:
        {
            "scored_candidates":  list of dicts (sorted by consensus_score desc)
            "best_candidate":     dict for the top-scoring candidate
            "similarity_matrix":  NxN list[list[float]]
            "dominant_cluster":   set of node indices
            "cluster_count":      int
            "all_clusters":       list[list[int]]
        }
    """
    print("\n" + "=" * 60)
    print("STAGE 8: CROSS-TRANSCRIPT CONSENSUS MODELING")
    print("=" * 60)

    n = len(candidates)

    if n == 0:
        print("  ⚠️  No candidates to score.")
        return {
            "scored_candidates": [],
            "best_candidate":    None,
            "similarity_matrix": [],
            "dominant_cluster":  set(),
            "cluster_count":     0,
            "all_clusters":      [],
        }

    print(f"  Candidates : {n}")
    print(f"  Language   : {language}")
    print(f"  Threshold  : {similarity_threshold}")
    print(f"  Outlier ↓  : {outlier_penalty * 100:.0f}%")

    # ── Step 1: Similarity matrix ─────────────────────────
    print("\n  [1/4] Pairwise Levenshtein similarity matrix")
    sim_matrix = build_similarity_matrix(candidates)

    header = "       " + "  ".join(f"[{j}]" for j in range(n))
    print(f"    {header}")
    for i in range(n):
        row = "  ".join(f"{sim_matrix[i][j]:.3f}" for j in range(n))
        print(f"    [{i}]  {row}")

    # ── Step 2: Build graph ───────────────────────────────
    print(f"\n  [2/4] Similarity graph (threshold ≥ {similarity_threshold})")
    graph      = build_graph(sim_matrix, threshold=similarity_threshold)
    edge_count = sum(len(v) for v in graph.values()) // 2
    print(f"    Edges formed: {edge_count}")
    for node, neighbors in graph.items():
        print(f"    [{node}] → {neighbors}")

    # ── Step 3: Dominant cluster ──────────────────────────
    print("\n  [3/4] Detecting dominant cluster")
    dominant_cluster, all_components = detect_dominant_cluster(
        graph, sim_matrix, n
    )
    print(f"    Total clusters   : {len(all_components)}")
    for idx, comp in enumerate(all_components):
        tag = " ← dominant" if comp == dominant_cluster else ""
        print(f"    Cluster {idx}        : {sorted(comp)}{tag}")

    # ── Step 4: Score candidates ──────────────────────────
    print("\n  [4/4] Scoring candidates")
    scored = score_candidates(
        candidates, sim_matrix, dominant_cluster, n,
        outlier_penalty=outlier_penalty
    )

    for s in scored:
        tag = "✅ dominant" if s["in_dominant_cluster"] else "⚠️  outlier"
        print(
            f"    Candidate {s['index']} | "
            f"consensus={s['consensus_score']:.4f} | "
            f"cluster_sim={s['avg_cluster_sim']:.4f} | "
            f"global_avg={s['global_avg_sim']:.4f} | "
            f"{tag}"
        )

    best = scored[0]
    print(f"\n  Best candidate : #{best['index']} "
          f"(consensus_score={best['consensus_score']:.4f})")
    print(f"  Text           : \"{best['text'][:80]}\"")
    print("=" * 60)

    return {
        "scored_candidates": scored,
        "best_candidate":    best,
        "similarity_matrix": sim_matrix,
        "dominant_cluster":  dominant_cluster,
        "cluster_count":     len(all_components),
        "all_clusters":      [sorted(c) for c in all_components],
    }
