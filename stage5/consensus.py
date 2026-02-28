from collections import Counter


def build_consensus(texts):

    token_lists = [t.split() for t in texts]
    max_len = max(len(t) for t in token_lists)

    result = []

    for i in range(max_len):
        tokens = [tokens[i] for tokens in token_lists if i < len(tokens)]

        if not tokens:
            continue

        most_common = Counter(tokens).most_common(1)[0][0]
        result.append(most_common)

    return " ".join(result)