import json
import argparse
import re
from typing import Any, Dict, List, Tuple, Set
from .utils import load_rollouts_fields, load_clusters_json, write_json, extract_sentences


def build_sentence_cluster_index(
    clusters: List[Dict[str, Any]]
) -> Tuple[Dict[Tuple[str, int], List[Tuple[int, int]]], Set[int]]:
    """Return ((text, rollout_id)->list[(cluster_id, remaining_count)], set(cluster_ids)).

    remaining_count tracks multiplicity of identical sentences within the same rollout.
    """
    index: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    cluster_ids: Set[int] = set()
    for cluster in clusters:
        cid = int(cluster["cluster_id"])  # type: ignore[index]
        cluster_ids.add(cid)
        for item in cluster.get("sentences", []):
            text = str(item.get("text", "")).strip()
            rollout_ids: List[int] = [int(x) for x in item.get("rollout_ids", [])]
            for rid in rollout_ids:
                key = (text, rid)
                if key not in index:
                    index[key] = []
                # Find existing cluster bucket
                found = False
                for i, (ecid, cnt) in enumerate(index[key]):
                    if ecid == cid:
                        index[key][i] = (ecid, cnt + 1)
                        found = True
                        break
                if not found:
                    index[key].append((cid, 1))
    return index, cluster_ids


def map_rollout_sentences_to_clusters(
    sentences: List[str],
    rollout_id: int,
    index: Dict[Tuple[str, int], List[Tuple[int, int]]],
) -> List[int]:
    """Return ordered cluster ids for sentences; consumes multiplicity counts.

    If a sentence is not found in any cluster for this rollout, it is skipped.
    """
    sequence: List[int] = []
    for text in sentences:
        key = (text, rollout_id)
        if key not in index:
            continue
        buckets = index[key]
        # Pick first cluster with remaining count
        selected_idx = -1
        for i, (cid, cnt) in enumerate(buckets):
            if cnt > 0:
                selected_idx = i
                break
        if selected_idx == -1:
            continue
        cid, cnt = buckets[selected_idx]
        buckets[selected_idx] = (cid, cnt - 1)
        sequence.append(cid)
    return sequence


def build_pathways(
    rollouts_path: str,
    clusters_path: str,
) -> Dict[str, Any]:
    """Construct pathways, transitions, cluster stats, and metadata from inputs."""
    fields = load_rollouts_fields(rollouts_path)
    cots: List[str] = fields.get("cot", [])

    clusters = load_clusters_json(clusters_path)
    index, cluster_id_set = build_sentence_cluster_index(clusters)

    eligible_rollout_ids: Set[int] = set()
    for (text, rid) in index.keys():
        eligible_rollout_ids.add(rid)

    pathways: List[Dict[str, Any]] = []
    all_edges: Dict[Tuple[int, int], Set[int]] = {}
    cluster_total_visits: Dict[int, int] = {}
    cluster_as_start: Dict[int, int] = {}
    cluster_as_end: Dict[int, int] = {}
    cluster_positions: Dict[int, List[float]] = {}
    cluster_rollouts_containing: Dict[int, Set[int]] = {}

    n_rollouts = 0
    total_path_length = 0

    for rid in sorted(eligible_rollout_ids):
        if rid < 0 or rid >= len(cots):
            continue
        sentences = extract_sentences(cots[rid])
        seq = map_rollout_sentences_to_clusters(sentences, rid, index)
        if not seq:
            continue
        assert len(seq) == len(sentences)
        pathways.append({
            "rollout_id": rid,
            "cluster_sequence": seq,
            "sentence_texts": sentences,
        })

        n_rollouts += 1
        total_path_length += len(seq)

        # Transitions
        for a, b in zip(seq, seq[1:]):
            key = (a, b)
            if key not in all_edges:
                all_edges[key] = set()
            all_edges[key].add(rid)

        # Cluster stats
        for pos, cid in enumerate(seq):
            cluster_total_visits[cid] = cluster_total_visits.get(cid, 0) + 1
            if pos == 0:
                cluster_as_start[cid] = cluster_as_start.get(cid, 0) + 1
            if pos == len(seq) - 1:
                cluster_as_end[cid] = cluster_as_end.get(cid, 0) + 1
            denom = len(seq) - 1 if len(seq) > 1 else 1
            norm_pos = pos / denom
            if cid not in cluster_positions:
                cluster_positions[cid] = []
            cluster_positions[cid].append(norm_pos)
            if cid not in cluster_rollouts_containing:
                cluster_rollouts_containing[cid] = set()
            cluster_rollouts_containing[cid].add(rid)

    edges_output: List[Dict[str, Any]] = []
    total_transitions = 0
    for (a, b), rids in sorted(all_edges.items()):
        count = len(rids)
        total_transitions += count
        edges_output.append({
            "from_cluster": int(a),
            "to_cluster": int(b),
            "count": int(count),
            "rollout_ids": sorted(list(rids)),
        })

    cluster_stats: Dict[str, Any] = {}
    for cid in sorted(cluster_id_set):
        visits = cluster_total_visits.get(cid, 0)
        starts = cluster_as_start.get(cid, 0)
        ends = cluster_as_end.get(cid, 0)
        positions = cluster_positions.get(cid, [])
        mean_pos = sum(positions) / len(positions) if positions else 0.0
        contains = sorted(list(cluster_rollouts_containing.get(cid, set())))
        cluster_stats[str(int(cid))] = {
            "total_visits": int(visits),
            "as_start": int(starts),
            "as_end": int(ends),
            "mean_position": float(mean_pos),
            "rollouts_containing": contains,
        }

    avg_path_len = (total_path_length / n_rollouts) if n_rollouts > 0 else 0.0

    result = {
        "pathways": pathways,
        "transitions": {
            "edges": edges_output,
            "total_transitions": int(total_transitions),
        },
        "cluster_stats": cluster_stats,
        "metadata": {
            "n_rollouts": int(n_rollouts),
            "n_clusters": int(len(cluster_id_set)),
            "avg_pathway_length": float(avg_path_len),
        },
    }
    return result


def build_and_write_pathways(
    rollouts_path: str,
    clusters_path: str,
    out_path: str,
) -> None:
    """Convenience function to build and write pathways JSON."""
    data = build_pathways(rollouts_path, clusters_path)
    write_json(out_path, data)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build sentence pathways from rollouts and clusters JSON")
    p.add_argument("--rollouts", default="/Users/jennakainic/global-cot/src/rollouts.json", required=True, help="Path to rollouts.json (fields-oriented)")
    p.add_argument("--clusters", required=True, default="/Users/jennakainic/global-cot/src/clusters_0.72.json", help="Path to clusters.json (from export_clusters_to_json)")
    p.add_argument("--out", required=True, default="/Users/jennakainic/global-cot/src/pathways.json", help="Output pathways JSON path")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    build_and_write_pathways(args.rollouts, args.clusters, args.out)


if __name__ == "__main__":
    main()

