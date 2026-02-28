from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from utils import normalize_text
import numpy as np

# Load multilingual embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def score_candidates(internal_transcript, candidates):

    internal_norm = normalize_text(internal_transcript)
    candidates_norm = [normalize_text(c) for c in candidates]

    # Semantic similarity
    embeddings = model.encode([internal_norm] + candidates_norm)
    internal_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    semantic_scores = cosine_similarity(
        [internal_embedding], candidate_embeddings
    )[0]

    # Lexical similarity
    lexical_scores = [
        fuzz.token_sort_ratio(internal_norm, c)/100
        for c in candidates_norm
    ]

    # Final weighted score
    final_scores = []
    for s, l in zip(semantic_scores, lexical_scores):
        final = 0.6 * s + 0.4 * l
        final_scores.append(final)

    best_index = np.argmax(final_scores)

    return best_index, final_scores