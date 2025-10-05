import json
import os
import re
from typing import Any, Dict, List, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
from .utils import load_responses_as_rollouts_fields, extract_sentences, load_clusters_json

from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def gather_cot_sentences(
    rollouts_path: str,
) -> Tuple[List[str], Dict[int, Set[int]]]:
    """Return (sentences, sentence_index_to_rollout_ids) for all responses.

    sentence_index_to_rollout_ids maps sentence index (in the returned sentences list)
    to a set of rollout row indices where it appears.
    """
    fields = load_responses_as_rollouts_fields(rollouts_path)
    cots: List[str] = fields.get("cot", [])

    sentences: List[str] = []
    sentence_index_to_rollout_ids: Dict[int, Set[int]] = {}

    for row_index, cot_text in enumerate(cots):
        cot_sentences = extract_sentences(cot_text)
        for s in cot_sentences:
            sentences.append(s)
            sentence_idx = len(sentences) - 1
            sentence_index_to_rollout_ids[sentence_idx] = {row_index}
    return sentences, sentence_index_to_rollout_ids


def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load a sentence-transformers model."""
    return SentenceTransformer(model_name)


def embed_sentences(embedder: SentenceTransformer, sentences: List[str]) -> np.ndarray:
    """Compute embeddings for sentences as a float32 array of shape (n, d)."""
    embeddings = embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def cluster_by_cosine_threshold(embeddings: np.ndarray, threshold: float, precomputed_sims: np.ndarray = None) -> List[int]:
    """Cluster by connecting pairs with cosine similarity >= threshold; return labels."""
    num_items = embeddings.shape[0]
    if num_items == 0:
        return []
    if num_items == 1:
        return [0]

    sims = precomputed_sims if precomputed_sims is not None else embeddings @ embeddings.T

    # Build adjacency list
    adjacency: List[List[int]] = [[] for _ in range(num_items)]
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if sims[i, j] >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # Connected components via DFS
    labels = [-1] * num_items
    current_label = 0
    for start in range(num_items):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = current_label
        while stack:
            node = stack.pop()
            for neigh in adjacency[node]:
                if labels[neigh] == -1:
                    labels[neigh] = current_label
                    stack.append(neigh)
        current_label += 1
    return labels


def cluster_by_cosine_threshold_from_sims(sims: np.ndarray, threshold: float) -> List[int]:
    """Cluster using a precomputed cosine similarity matrix and SciPy connected components."""
    n = sims.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]
    mask = sims >= threshold
    np.fill_diagonal(mask, True)
    graph = csr_matrix(mask.astype(np.uint8))
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    return labels.tolist()


def plot_clusters_vs_threshold(
    embeddings: np.ndarray,
    thresholds: List[float],
    out_path: str,
    n_jobs: int = -1,
    precomputed_sims: np.ndarray = None,
) -> None:
    """Plot number of clusters vs cosine similarity threshold and save to path."""
    if embeddings.shape[0] == 0:
        plt.figure(figsize=(6, 4))
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    sims = precomputed_sims if precomputed_sims is not None else embeddings @ embeddings.T

    def _num_clusters_for_threshold(t: float) -> Tuple[float, int]:
        labels = cluster_by_cosine_threshold_from_sims(sims, t)
        return t, len(set(labels))

    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_num_clusters_for_threshold)(t) for t in thresholds)
    results.sort(key=lambda x: x[0])
    x_vals = [t for t, _ in results]
    y_vals = [k for _, k in results]

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Number of clusters")
    plt.title("Clusters vs Cosine Threshold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_silhouette_vs_threshold(
    embeddings: np.ndarray,
    thresholds: List[float],
    out_path: str,
    n_jobs: int = -1,
    precomputed_sims: np.ndarray = None,
) -> None:
    """Plot silhouette score vs cosine similarity threshold and save to path.
    
    Only considers thresholds that produce between 2 and n-1 clusters.
    """
    if embeddings.shape[0] < 2:
        return

    sims = precomputed_sims if precomputed_sims is not None else embeddings @ embeddings.T
    distances = 1.0 - sims
    np.fill_diagonal(distances, 0.0)

    def _silhouette_for_threshold(t: float) -> Tuple[float, float]:
        labels = cluster_by_cosine_threshold_from_sims(sims, t)
        num_clusters = len(set(labels))
        if num_clusters < 2 or num_clusters >= embeddings.shape[0]:
            return (t, np.nan)
        score = silhouette_score(distances, labels, metric="precomputed")
        return (t, float(score))

    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_silhouette_for_threshold)(t) for t in thresholds)
    results = [(t, s) for (t, s) in results if not np.isnan(s)]
    results.sort(key=lambda x: x[0])
    x_vals = [t for t, _ in results]
    y_vals = [s for _, s in results]

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel("Cosine similarity threshold") 
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Score vs Cosine Threshold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def load_sentence_cluster_counts_from_json(clusters_dir: str) -> Dict[float, int]:
    """Load number of clusters per threshold from saved sentence clusters_*.json files."""
    import os
    import glob

    pattern = os.path.join(clusters_dir, "clusters_*.json")
    files = glob.glob(pattern)
    threshold_to_count: Dict[float, int] = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        # Expect filename like clusters_{threshold}.json
        parts = filename.replace(".json", "").split("_")
        if len(parts) >= 2:
            t_str = parts[-1]
            try:
                t_val = float(t_str)
            except ValueError:
                continue
            clusters = load_clusters_json(file_path)
            threshold_to_count[t_val] = len(clusters)
    return threshold_to_count


def plot_sentence_clusters_vs_threshold_from_json(
    clusters_dir: str,
    out_path: str,
) -> None:
    """Plot number of clusters vs threshold by reading saved sentence cluster JSONs."""
    counts = load_sentence_cluster_counts_from_json(clusters_dir)
    if not counts:
        return
    thresholds = sorted(counts.keys())
    num_clusters = [counts[t] for t in thresholds]
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, num_clusters, marker="o")
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Number of clusters")
    plt.title("Sentence Clusters vs Cosine Threshold (cached)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_cluster_centroid(embeddings: np.ndarray, member_indices: List[int]) -> np.ndarray:
    """Compute centroid as mean of member embeddings and L2-normalize it."""
    centroid = embeddings[member_indices].mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0.0:
        centroid = centroid / norm
    return centroid.astype(np.float32)


def select_representative_sentence(
    embeddings: np.ndarray,
    member_indices: List[int],
    centroid: np.ndarray,
    sentences: List[str],
) -> Tuple[int, str]:
    """Return (representative_index, representative_sentence) as the one closest to centroid."""
    member_embs = embeddings[member_indices]
    sims = member_embs @ centroid
    best_local_idx = int(np.argmax(sims))
    rep_idx = member_indices[best_local_idx]
    return rep_idx, sentences[rep_idx]


def export_clusters_to_json(
    out_path: str,
    labels: List[int],
    embeddings: np.ndarray,
    sentences: List[str],
    sentence_index_to_rollout_ids: Dict[int, Set[int]],
) -> None:
    """Write clusters summary to JSON: id, size, centroid, representative, and sentences list."""
    cluster_id_to_members: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        if lab not in cluster_id_to_members:
            cluster_id_to_members[lab] = []
        cluster_id_to_members[lab].append(idx)

    clusters_output: List[Dict[str, Any]] = []
    for cluster_id, member_indices in sorted(cluster_id_to_members.items()):
        centroid = compute_cluster_centroid(embeddings, member_indices)
        rep_idx, rep_sentence = select_representative_sentence(embeddings, member_indices, centroid, sentences)

        # Keep duplicates in order (multiplicity preserved)
        sentences_list: List[Dict[str, Any]] = []
        for s_idx in member_indices:
            sentences_list.append({
                "text": sentences[s_idx],
                "rollout_ids": sorted(list(sentence_index_to_rollout_ids.get(s_idx, set()))),
            })

        # Also aggregate identical texts to report multiplicity
        agg: Dict[str, Dict[str, Any]] = {}
        for s_idx in member_indices:
            txt = sentences[s_idx]
            if txt not in agg:
                agg[txt] = {"count": 0, "rollout_ids": set()}
            agg[txt]["count"] += 1
            agg[txt]["rollout_ids"].update(sentence_index_to_rollout_ids.get(s_idx, set()))
        unique_sentences: List[Dict[str, Any]] = []
        for txt, payload in agg.items():
            unique_sentences.append({
                "text": txt,
                "count": int(payload["count"]),
                "rollout_ids": sorted(list(payload["rollout_ids"]))
            })
        # Stable sort by descending count then text
        unique_sentences.sort(key=lambda d: (-d["count"], d["text"]))

        clusters_output.append({
            "cluster_id": int(cluster_id),
            "size": int(len(member_indices)),
            "centroid": centroid.astype(float).tolist(),
            "representative_sentence": rep_sentence,
            "sentences": sentences_list,
            "unique_sentences": unique_sentences,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"clusters": clusters_output}, f, ensure_ascii=False, indent=2)


def cluster_sentences(
    rollouts_path: str,
    embed_model: str,
    threshold: float,
    out_json_path: str,
) -> None:
    """End-to-end: load sentences, embed, cluster at threshold, export clusters JSON."""
    sentences, sent_idx_to_rollout_ids = gather_cot_sentences(rollouts_path)
    embedder = load_embedder(embed_model)
    embeddings = embed_sentences(embedder, sentences)
    labels = cluster_by_cosine_threshold(embeddings, threshold)
    export_clusters_to_json(out_json_path, labels, embeddings, sentences, sent_idx_to_rollout_ids)


