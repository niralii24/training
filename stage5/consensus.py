from collections import Counter


def build_consensus(texts):

    token_lists = [t.split() for t in texts]
    max_len = max(len(t) for t in token_lists)

    result = []

    for i in range(max_len):
        tokens_at_i = [tl[i] for tl in token_lists if i < len(tl)]

        if not tokens_at_i:
            continue

        most_common = Counter(tokens_at_i).most_common(1)[0][0]
        result.append(most_common)

    return " ".join(result)