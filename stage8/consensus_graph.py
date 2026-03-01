import re
import unicodedata


# ── Text Cleaning ─────────────────────────────────────────

def clean_for_comparison(text):
    """
    Minimal cleaning before comparison.
    Removes punctuation and collapses whitespace,
    but preserves all letter-level differences (diacritics,
    Alef variants, Hamza forms) so Levenshtein can see them.
    Works for any script.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


# ── Levenshtein ───────────────────────────────────────────

def levenshtein_distance(a, b):
    """
    Standard character-level Levenshtein distance.
    Uses O(n) space optimized DP.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,               # deletion
                curr[j - 1] + 1,           # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


def levenshtein_ratio(a, b):
    """
    Normalized similarity in [0, 1].
    1.0 = identical strings, 0.0 = completely different.
    Cleaning is applied before comparison so only meaningful
    letter-level differences are measured.
    """
    a_clean = clean_for_comparison(a)
    b_clean = clean_for_comparison(b)
    max_len = max(len(a_clean), len(b_clean), 1)
    dist    = levenshtein_distance(a_clean, b_clean)
    return 1.0 - dist / max_len


# ── Similarity Matrix ─────────────────────────────────────

def build_similarity_matrix(candidates):
    """
    Builds an NxN pairwise Levenshtein similarity matrix.
    Diagonal = 1.0 (self-similarity).

    Returns:
        list[list[float]]  — NxN matrix
    """
    n      = len(candidates)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim          = levenshtein_ratio(candidates[i], candidates[j])
            matrix[i][j] = sim
            matrix[j][i] = sim

    return matrix


# ── Graph Construction ────────────────────────────────────

def build_graph(similarity_matrix, threshold=0.5):
    """
    Creates an adjacency list graph from the similarity matrix.
    An edge exists between i and j if similarity >= threshold.

    Returns:
        dict  {node_index: [neighbor_indices]}
    """
    n     = len(similarity_matrix)
    graph = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= threshold:
                graph[i].append(j)
                graph[j].append(i)

    return graph


# ── Cluster Detection ─────────────────────────────────────

def find_connected_components(graph, n):
    """
    BFS over the graph to find all connected components.

    Returns:
        list[set]  — each set is a cluster of node indices
    """
    visited    = [False] * n
    components = []

    for start in range(n):
        if not visited[start]:
            cluster = set()
            queue   = [start]
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                visited[node] = True
                cluster.add(node)
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        queue.append(neighbor)
            components.append(cluster)

    return components


def _component_cohesion(comp, similarity_matrix):
    """
    Average pairwise similarity within a component.
    Single-node components return 0.0.
    """
    members = list(comp)
    if len(members) < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            total += similarity_matrix[members[i]][members[j]]
            count += 1
    return total / count if count else 0.0


def detect_dominant_cluster(graph, similarity_matrix, n):
    """
    Identifies the dominant cluster — the connected component
    that is both large and internally cohesive.

    Scoring: size × cohesion  (favors large, tight clusters)

    Returns:
        (dominant_cluster: set, all_components: list[set])
    """
    components = find_connected_components(graph, n)

    if len(components) == 1:
        return components[0], components

    scored = [
        (comp, len(comp) * _component_cohesion(comp, similarity_matrix))
        for comp in components
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[0][0], components


# ── Candidate Scoring ─────────────────────────────────────

def score_candidates(candidates, similarity_matrix, dominant_cluster, n,
                     outlier_penalty=0.3):
    """
    Scores each candidate by its average similarity to the dominant cluster.
    Candidates outside the dominant cluster are penalised.

    Args:
        candidates:        list of original transcript strings
        similarity_matrix: NxN similarity matrix
        dominant_cluster:  set of node indices in dominant cluster
        n:                 number of candidates
        outlier_penalty:   fraction to reduce score for outliers (0 = none, 1 = full)

    Returns:
        list[dict] sorted by consensus_score descending
    """
    results = []

    for i in range(n):
        # Average similarity to dominant cluster (excluding self)
        cluster_others = [j for j in dominant_cluster if j != i]
        if cluster_others:
            avg_cluster_sim = (
                sum(similarity_matrix[i][j] for j in cluster_others)
                / len(cluster_others)
            )
        else:
            avg_cluster_sim = 1.0  # sole member

        # Global average similarity across all other candidates
        all_others  = [j for j in range(n) if j != i]
        global_avg  = (
            sum(similarity_matrix[i][j] for j in all_others) / len(all_others)
            if all_others else 1.0
        )

        in_dominant     = i in dominant_cluster
        consensus_score = (
            avg_cluster_sim
            if in_dominant
            else avg_cluster_sim * (1.0 - outlier_penalty)
        )

        results.append({
            "index":               i,
            "text":                candidates[i],
            "consensus_score":     round(consensus_score, 4),
            "avg_cluster_sim":     round(avg_cluster_sim, 4),
            "global_avg_sim":      round(global_avg, 4),
            "in_dominant_cluster": in_dominant,
        })

    results.sort(key=lambda x: x["consensus_score"], reverse=True)
    return results
