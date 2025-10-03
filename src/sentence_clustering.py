import json
import os
import re
from typing import Any, Dict, List, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
from .utils import load_rollouts_fields, load_responses_as_rollouts_fields, extract_sentences

from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score


def gather_cot_sentences_for_prompt(
    rollouts_path: str,
    prompt_text: str,
) -> Tuple[List[str], Dict[int, Set[int]]]:
    """Return (sentences, sentence_index_to_rollout_ids) for a specific prompt.

    sentence_index_to_rollout_ids maps sentence index (in the returned sentences list)
    to a set of rollout row indices where it appears.
    """
    fields = load_responses_as_rollouts_fields(rollouts_path)
    prompts: List[str] = fields.get("prompt", [])
    cots: List[str] = fields.get("cot", [])

    sentences: List[str] = []
    sentence_index_to_rollout_ids: Dict[int, Set[int]] = {}

    for row_index, (p_text, cot_text) in enumerate(zip(prompts, cots)):
        if p_text != prompt_text:
            continue
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


def cluster_by_cosine_threshold(embeddings: np.ndarray, threshold: float) -> List[int]:
    """Cluster by connecting pairs with cosine similarity >= threshold; return labels."""
    num_items = embeddings.shape[0]
    if num_items == 0:
        return []
    if num_items == 1:
        return [0]

    # Cosine similarities via dot product (embeddings are normalized)
    sims = embeddings @ embeddings.T

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


def plot_clusters_vs_threshold(
    embeddings: np.ndarray,
    thresholds: List[float],
    out_path: str,
) -> None:
    """Plot number of clusters vs cosine similarity threshold and save to path."""
    x_vals: List[float] = []
    y_vals: List[int] = []
    for t in thresholds:
        labels = cluster_by_cosine_threshold(embeddings, t)
        x_vals.append(t)
        y_vals.append(len(set(labels)))

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Number of clusters")
    plt.title("Clusters vs Cosine Threshold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def choose_threshold_by_silhouette(
    embeddings: np.ndarray,
    thresholds: List[float],
) -> Tuple[float, float]:
    """Return (best_threshold, best_silhouette) over thresholds using cosine metric.

    Only considers thresholds that produce between 2 and n-1 clusters.
    """
    if embeddings.shape[0] < 2:
        return thresholds[0] if thresholds else 0.0, 0.0

    best_t = thresholds[0] if thresholds else 0.0
    best_score = -1.0
    for t in thresholds:
        labels = cluster_by_cosine_threshold(embeddings, t)
        num_clusters = len(set(labels))
        if num_clusters < 2 or num_clusters >= embeddings.shape[0]:
            continue
        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


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


def cluster_sentences_for_prompt(
    rollouts_path: str,
    prompt_text: str,
    embed_model: str,
    threshold: float,
    out_json_path: str,
) -> None:
    """End-to-end: load sentences, embed, cluster at threshold, export clusters JSON."""
    sentences, sent_idx_to_rollout_ids = gather_cot_sentences_for_prompt(rollouts_path, prompt_text)
    embedder = load_embedder(embed_model)
    embeddings = embed_sentences(embedder, sentences)
    labels = cluster_by_cosine_threshold(embeddings, threshold)
    export_clusters_to_json(out_json_path, labels, embeddings, sentences, sent_idx_to_rollout_ids)


