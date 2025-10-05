import argparse
import json
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

import math
import numpy as np
from scipy.stats import chi2_contingency


def load_formatted(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sequences_from_rollouts(data: Dict[str, Any]) -> List[List[int]]:
    rollouts = data.get("rollouts", [])
    sequences: List[List[int]] = []
    for r in rollouts:
        edges = r.get("edges", [])
        if not edges:
            continue
        seq: List[int] = [edges[0]["from_cluster"]]
        for e in edges:
            seq.append(e["to_cluster"])
        sequences.append(seq)
    return sequences


def build_transition_counts(sequences: List[List[int]]) -> Tuple[np.ndarray, Dict[int, int]]:
    unique_states: List[int] = sorted(list({s for seq in sequences for s in seq}))
    state_to_idx: Dict[int, int] = {s: i for i, s in enumerate(unique_states)}
    n = len(unique_states)
    counts = np.zeros((n, n), dtype=int)
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            i = state_to_idx[a]
            j = state_to_idx[b]
            counts[i, j] += 1
    return counts, state_to_idx


def normalize_rows(counts: np.ndarray) -> np.ndarray:
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0)
    return probs


def conditional_mutual_information(sequences: List[List[int]], k: int) -> float:
    """
    Compute I(X_{t+1}; X_{t-k} | X_t, X_{t-1}, ..., X_{t-k+1}).
    Tests whether state k steps back adds information about the next state
    beyond what the most recent k-1 states provide.
    """
    if k <= 0:
        return 0.0

    triples = Counter()              # (recent_hist, distant, future)
    pairs_hist_future = Counter()    # (recent_hist, future)
    pairs_hist_distant = Counter()   # (recent_hist, distant)
    singles_hist = Counter()         # (recent_hist,)
    total = 0

    for seq in sequences:
        if len(seq) < k + 2:
            continue
        for t in range(k, len(seq) - 1):
            distant_past = seq[t - k]
            recent_hist = tuple(seq[t - k + 1:t + 1])
            future = seq[t + 1]

            triples[(recent_hist, distant_past, future)] += 1
            pairs_hist_future[(recent_hist, future)] += 1
            pairs_hist_distant[(recent_hist, distant_past)] += 1
            singles_hist[recent_hist] += 1
            total += 1

    if total == 0:
        return 0.0

    mi = 0.0
    for (recent, distant, future), c_rdf in triples.items():
        p_rdf = c_rdf / total
        p_rd = pairs_hist_distant[(recent, distant)] / total
        p_rf = pairs_hist_future[(recent, future)] / total
        p_r = singles_hist[recent] / total
        denom = (p_rd * p_rf) / p_r if p_r > 0 else 0.0
        if p_rdf > 0 and denom > 0:
            mi += p_rdf * math.log(p_rdf / denom)
    return mi


def chi_squared_independence_by_context(sequences: List[List[int]], k: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {"k": k, "contexts": []}
    if k <= 0:
        return results
    context_to_pairs: Dict[Tuple[int, ...], List[Tuple[int, int]]] = defaultdict(list)
    for seq in sequences:
        for t in range(k, len(seq) - 1):
            hist = tuple(seq[t - k:t])
            x = seq[t]
            y = seq[t + 1]
            context_to_pairs[hist].append((x, y))
    for hist, pairs in context_to_pairs.items():
        if len(pairs) < 2:
            continue
        xs = sorted(list({x for x, _ in pairs}))
        ys = sorted(list({y for _, y in pairs}))
        xi = {x: i for i, x in enumerate(xs)}
        yi = {y: i for i, y in enumerate(ys)}
        table = np.zeros((len(xs), len(ys)), dtype=int)
        for x, y in pairs:
            table[xi[x], yi[y]] += 1
        if table.sum() == 0 or table.shape[0] < 1 or table.shape[1] < 1:
            continue
        chi2, p, dof, _ = chi2_contingency(table, correction=False)
        results["contexts"].append({
            "history": list(hist),
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "n": int(table.sum()),
        })
    # Aggregate statistics
    if results["contexts"]:
        p_values = [c["p_value"] for c in results["contexts"]]
        chi2_values = [c["chi2"] for c in results["contexts"]]
        results["aggregate"] = {
            "mean_p_value": float(np.mean(p_values)),
            "median_p_value": float(np.median(p_values)),
            "fraction_significant": float(np.mean([p < 0.05 for p in p_values])),
            "mean_chi2": float(np.mean(chi2_values)),
        }
        bonferroni_alpha = 0.05 / len(p_values)
        results["aggregate"]["fraction_significant_bonferroni"] = float(
            np.mean([p < bonferroni_alpha for p in p_values])
        )
    return results


    


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute Markovianity metrics from formatted sentence outputs")
    p.add_argument("--input", required=True, help="Path to formatted_sentence_outputs_*.json")
    p.add_argument("--max_k", type=int, default=3, help="Max history length k to evaluate")
    p.add_argument("--out", type=str, default="markov_metrics.json", help="Output JSON file for metrics")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    data = load_formatted(args.input)
    sequences = sequences_from_rollouts(data)
    counts, state_to_idx = build_transition_counts(sequences)
    probs = normalize_rows(counts)

    metrics: Dict[str, Any] = {
        "metadata": {
            "num_states": int(len(state_to_idx)),
        },
        "transitions": {
            "counts": counts.tolist(),
            "matrix": probs.tolist(),
        },
        "markov_tests": {
            "cmi": [],
            "chi2_by_context": [],
        },
    }
    for k in range(1, args.max_k + 1):
        mi_k = conditional_mutual_information(sequences, k)
        metrics["markov_tests"]["cmi"].append({"k": int(k), "value": float(mi_k)})
        chi_k = chi_squared_independence_by_context(sequences, k)
        metrics["markov_tests"]["chi2_by_context"].append(chi_k)

    

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()


