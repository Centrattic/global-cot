import json
import os
import numpy as np
from typing import Any, Dict, List, Tuple, Set
from tqdm import tqdm

from .utils import load_json, write_json, extract_sentences


def load_activations_from_processed_responses(
    processed_responses_path: str,
    activation_cache_dir: str,
    layer: int,
) -> Tuple[List[str], np.ndarray, Dict[int, Set[int]], Set[int]]:
    """Load activations for all sentences from processed responses."""
    print(f"Loading activations for layer {layer}...")
    
    # Sentence to filter out
    filtered_sentence = "We need to find base-16 number 66666 (hex). Convert to decimal? 6*16^4 +6*16^3+6*16^2+6*16+6. Let's compute: 16^4=65536. times6=393216. 16^3=4096*6=24576. 16^2=256*6=1536. 16*6=96. plus6. Sum=393216+24576=417792. +1536=419328. +96=419424. +6=419430. So decimal 419430. Now binary length? Need floor(log2(n))+1. log2(419430). 2^19=524288. 2^18=262144. 419430 between."
    
    responses = load_json(processed_responses_path)
    all_sentences = []
    all_activations = []
    sentence_index_to_rollout_ids = {}
    excluded_rollouts = set()  # Track rollouts with sentences > 200 chars
    
    for response in tqdm(responses, desc="Loading responses"):
        response_index = response.get("index")
        sentences = response.get("sentences", [])
        
        if not response_index or not sentences:
            continue
            
        # Check if any sentence in this rollout is > 150 characters
        has_long_sentence = any(len(sentence) > 150 for sentence in sentences)
        if has_long_sentence:
            excluded_rollouts.add(response_index)
            print(f"Excluding rollout {response_index} due to sentence > 150 characters")
            continue
            
        activation_file = os.path.join(activation_cache_dir, str(layer), f"completion_{response_index}.npy")
        
        if not os.path.exists(activation_file):
            print(f"Warning: No activation file found for response {response_index}")
            continue
            
        # Load activations
        activation_data = np.load(activation_file, allow_pickle=True).item()
        
        for sentence_idx, sentence in enumerate(sentences):
            # Filter out the specific sentence
            if sentence == filtered_sentence:
                print(f"Filtering out sentence from response {response_index}")
                continue
                
            if sentence in activation_data:
                all_sentences.append(sentence)
                all_activations.append(activation_data[sentence])
                
                # Map sentence index to rollout IDs
                global_sentence_idx = len(all_sentences) - 1
                if global_sentence_idx not in sentence_index_to_rollout_ids:
                    sentence_index_to_rollout_ids[global_sentence_idx] = set()
                sentence_index_to_rollout_ids[global_sentence_idx].add(response_index)
    
    if not all_activations:
        print("No activations found!")
        return [], np.array([]), {}, excluded_rollouts
    
    activations = np.array(all_activations)
    print(f"Loaded {len(all_sentences)} sentences with activations (after filtering)")
    print(f"Excluded {len(excluded_rollouts)} rollouts due to long sentences")
    
    return all_sentences, activations, sentence_index_to_rollout_ids, excluded_rollouts


def cluster_activations_by_cosine_threshold(
    activations: np.ndarray,
    threshold: float
) -> List[int]:
    """Cluster activations by connecting pairs with cosine similarity >= threshold."""
    num_items = activations.shape[0]
    if num_items == 0:
        return []
    if num_items == 1:
        return [0]

    # Normalize for cosine similarity
    print("Normalizing activations...")
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    normalized_activations = activations / (norms + 1e-8)

    # Compute similarities
    print("Computing similarities...")
    similarities = normalized_activations @ normalized_activations.T
    
    # Build adjacency list
    print("Building adjacency list...")
    adjacency = [[] for _ in range(num_items)]
    for i in tqdm(range(num_items), desc="Building adjacency"):
        for j in range(i + 1, num_items):
            if similarities[i, j] >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # Find connected components using DFS
    print("Finding connected components...")
    visited = [False] * num_items
    labels = [-1] * num_items
    current_label = 0

    for start in tqdm(range(num_items), desc="DFS"):
        if visited[start]:
            continue
            
        # DFS
        stack = [start]
        visited[start] = True
        
        while stack:
            node = stack.pop()
            labels[node] = current_label
            
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        
        current_label += 1

    return labels


def compute_activation_cluster_centroid(activations: np.ndarray, member_indices: List[int]) -> np.ndarray:
    """Compute centroid as mean of member activations and L2-normalize it."""
    centroid = activations[member_indices].mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0.0:
        centroid = centroid / norm
    return centroid.astype(np.float32)


def select_representative_sentence_from_activations(
    activations: np.ndarray,
    member_indices: List[int],
    centroid: np.ndarray,
    sentences: List[str],
) -> Tuple[int, str]:
    """Return (representative_index, representative_sentence) as the one closest to centroid."""
    member_activations = activations[member_indices]
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(member_activations, axis=1, keepdims=True)
    normalized_activations = member_activations / (norms + 1e-8)
    
    sims = normalized_activations @ centroid
    best_local_idx = int(np.argmax(sims))
    rep_idx = member_indices[best_local_idx]
    return rep_idx, sentences[rep_idx]


def compute_mean_cosine_similarity_for_cluster(activations: np.ndarray, member_indices: List[int]) -> float:
    """Compute mean pairwise cosine similarity within a cluster."""
    if len(member_indices) <= 1:
        return 0.0
    
    member_activations = activations[member_indices]
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(member_activations, axis=1, keepdims=True)
    normalized_activations = member_activations / (norms + 1e-8)
    
    # Compute pairwise similarities
    similarities = normalized_activations @ normalized_activations.T
    
    # Get upper triangle (avoid duplicates and diagonal)
    n = len(member_indices)
    upper_triangle = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_triangle.append(similarities[i, j])
    
    return float(np.mean(upper_triangle))


def export_activation_clusters_to_flowchart_json(
    out_json_path: str,
    labels: List[int],
    activations: np.ndarray,
    sentences: List[str],
    sentence_index_to_rollout_ids: Dict[int, Set[int]],
    layer: int,
    threshold: float,
    processed_responses_path: str,
    excluded_rollouts: Set[int]
):
    """Export activation clusters to flowchart JSON format."""
    print("Exporting clusters to flowchart JSON...")
    
    # Load processed responses to get response content and correctness
    responses = load_json(processed_responses_path)
    response_index_to_content = {}
    response_index_to_correctness = {}
    for response in responses:
        response_index = response.get("index")
        response_index_to_content[response_index] = response.get("response_content", "")
        response_index_to_correctness[response_index] = response.get("correctness", False)
    
    # Group sentences by cluster
    cluster_id_to_members: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        if lab not in cluster_id_to_members:
            cluster_id_to_members[lab] = []
        cluster_id_to_members[lab].append(idx)

    print("Building nodes...")
    nodes = []
    for cluster_id, member_indices in tqdm(sorted(cluster_id_to_members.items()), desc="Processing clusters"):
        centroid = compute_activation_cluster_centroid(activations, member_indices)
        rep_idx, rep_sentence = select_representative_sentence_from_activations(
            activations, member_indices, centroid, sentences)
        
        # Compute mean cosine similarity within cluster
        mean_similarity = compute_mean_cosine_similarity_for_cluster(activations, member_indices)
        
        # Aggregate sentences by text
        sentence_aggregates: Dict[str, Dict[str, Any]] = {}
        for s_idx in member_indices:
            txt = sentences[s_idx]
            if txt not in sentence_aggregates:
                sentence_aggregates[txt] = {"count": 0, "rollout_ids": set()}
            sentence_aggregates[txt]["count"] += 1
            sentence_aggregates[txt]["rollout_ids"].update(
                sentence_index_to_rollout_ids.get(s_idx, set()))
        
        # Convert to list format
        sentences_list = []
        for txt, payload in sentence_aggregates.items():
            sentences_list.append({
                "text": txt,
                "count": int(payload["count"]),
                "rollout_ids": sorted(list(payload["rollout_ids"]))
            })
        
        # Sort by count (descending) then text
        sentences_list.sort(key=lambda d: (-d["count"], d["text"]))

        nodes.append({
            "cluster_id": str(cluster_id),
            "freq": len(member_indices),
            "representative_sentence": rep_sentence,
            "mean_similarity": float(mean_similarity),
            "sentences": sentences_list
        })

    # Add response nodes
    print("Building response nodes...")
    response_cluster_id_to_content = {}
    for response_index, response_content in response_index_to_content.items():
        if response_content:  # Only add if response content exists
            response_cluster_id = f"response-{response_content}"
            if response_cluster_id not in response_cluster_id_to_content:
                response_cluster_id_to_content[response_cluster_id] = response_content
                nodes.append({
                    "cluster_id": response_cluster_id,
                    "freq": 1,  # Each response appears once
                    "representative_sentence": response_content,
                    "mean_similarity": 0.0,  # No similarity computation for response nodes
                    "sentences": [{
                        "text": response_content,
                        "count": 1,
                        "rollout_ids": [response_index]
                    }]
                })

    print("Building rollouts...")
    rollout_sentences: Dict[int, List[str]] = {}
    for s_idx, response_indices in sentence_index_to_rollout_ids.items():
        sentence = sentences[s_idx]
        for response_index in response_indices:
            if response_index not in rollout_sentences:
                rollout_sentences[response_index] = []
            rollout_sentences[response_index].append(sentence)

    # Build sentence to cluster mapping
    sentence_to_cluster = {}
    for cluster_id, member_indices in cluster_id_to_members.items():
        for s_idx in member_indices:
            sentence_to_cluster[sentences[s_idx]] = cluster_id

    # Get all response indices from processed_responses.json, excluding those with long sentences
    all_response_indices = set(response_index_to_content.keys()) - excluded_rollouts
    
    rollouts = {}
    for response_index in all_response_indices:
        edges = []
        rollout_sentences_list = rollout_sentences.get(response_index, [])
        
        # Create edges between consecutive sentences
        for i in range(len(rollout_sentences_list) - 1):
            current_sentence = rollout_sentences_list[i]
            next_sentence = rollout_sentences_list[i + 1]
            
            if current_sentence in sentence_to_cluster and next_sentence in sentence_to_cluster:
                current_cluster = sentence_to_cluster[current_sentence]
                next_cluster = sentence_to_cluster[next_sentence]
                
                edges.append({
                    "node_a": str(current_cluster),
                    "node_b": str(next_cluster)
                })
        
        # Add final edge from last sentence to response node
        if rollout_sentences_list and response_index in response_index_to_content:
            last_sentence = rollout_sentences_list[-1]
            response_content = response_index_to_content[response_index]
            
            if last_sentence in sentence_to_cluster and response_content:
                last_cluster = sentence_to_cluster[last_sentence]
                response_cluster_id = f"response-{response_content}"
                
                edges.append({
                    "node_a": str(last_cluster),
                    "node_b": response_cluster_id
                })
        
        # Include all rollouts, even if they have no edges (empty list)
        correctness = response_index_to_correctness.get(response_index, False)
        rollouts[str(response_index)] = {
            "edges": edges,
            "correctness": correctness
        }

    flowchart_data = {
        "nodes": nodes,
        "rollouts": rollouts
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    write_json(out_json_path, flowchart_data)
    print(f"Exported flowchart to {out_json_path}")


def cluster_activations(
    processed_responses_path: str,
    activation_cache_dir: str,
    layer: int,
    threshold: float,
    out_json_path: str
):
    """Main function to cluster activations and export flowchart."""
    print(f"Starting activation clustering for layer {layer} with threshold {threshold}")
    
    sentences, activations, sent_idx_to_rollout_ids, excluded_rollouts = load_activations_from_processed_responses(
        processed_responses_path, activation_cache_dir, layer)
    
    print(f"Clustering {len(sentences)} sentences with threshold {threshold}...")
    labels = cluster_activations_by_cosine_threshold(activations, threshold)
    print(f"Found {len(set(labels))} clusters")
    
    # Export full flowchart JSON
    export_activation_clusters_to_flowchart_json(
        out_json_path, labels, activations, sentences, sent_idx_to_rollout_ids, layer, threshold, processed_responses_path, excluded_rollouts)


def explore_activation_thresholds(
    processed_responses_path: str,
    activation_cache_dir: str,
    layer: int,
    thresholds: List[float],
    out_dir: str,
) -> None:
    """Explore different thresholds for activation clustering."""
    print(f"Exploring activation thresholds for layer {layer}")
    
    sentences, activations, sentence_index_to_rollout_ids, excluded_rollouts = load_activations_from_processed_responses(
        processed_responses_path, activation_cache_dir, layer)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Compute similarity statistics
    print("Computing similarity statistics...")
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    normalized_activations = activations / (norms + 1e-8)
    similarities = normalized_activations @ normalized_activations.T
    
    # Remove diagonal (self-similarities)
    np.fill_diagonal(similarities, 0)
    
    # Get upper triangle (avoid duplicates)
    upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
    
    stats = {
        "layer": layer,
        "mean": float(np.mean(upper_triangle)),
        "std": float(np.std(upper_triangle)),
        "min": float(np.min(upper_triangle)),
        "max": float(np.max(upper_triangle)),
        "n_sentences": len(activations)
    }
    
    stats_file = os.path.join(out_dir, f"activation_similarity_stats_layer_{layer}.json")
    write_json(stats_file, stats)
    
    print(f"Activation similarity stats for layer {layer}:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Min: {stats['min']:.3f}")
    print(f"  Max: {stats['max']:.3f}")
    
    # Debug: Check if we have rollout mappings
    print(f"Loaded {len(sentence_index_to_rollout_ids)} sentence-to-rollout mappings")
    if sentence_index_to_rollout_ids:
        sample_idx = next(iter(sentence_index_to_rollout_ids))
        sample_rollouts = sentence_index_to_rollout_ids[sample_idx]
        print(f"Sample: sentence {sample_idx} -> rollouts {list(sample_rollouts)}")
    
    # Compute clusters for each threshold
    for threshold in tqdm(thresholds, desc="Computing clusters for thresholds"):
        print(f"Computing clusters for threshold {threshold}")
        labels = cluster_activations_by_cosine_threshold(activations, threshold)
        n_clusters = len(set(labels))
        print(f"Found {n_clusters} clusters")
        
        # Export flowchart for this threshold
        out_json_path = os.path.join(out_dir, f"flowchart_layer_{layer}_threshold_{threshold}.json")
        export_activation_clusters_to_flowchart_json(
            out_json_path, labels, activations, sentences, sentence_index_to_rollout_ids, layer, threshold, processed_responses_path, excluded_rollouts)


if __name__ == "__main__":
    # Example usage
    processed_responses_path = "processed_responses.json"
    activation_cache_dir = "activation_cache"
    
    # Single threshold clustering
    layer = 17
    # threshold = 0.2
    # out_json_path = f"flowcharts/flowchart_layer_{layer}_threshold_{threshold}.json"
    
    # cluster_activations(
    #     processed_responses_path, activation_cache_dir, layer, threshold, out_json_path)
    
    # 0.7, 0.9: 1 cluster
    # 0.95: 22 clusters, but most of the sentences are in one cluster! Need higher threshold
    # 0.96: 116 cluster -- pretty good!
    # 0.975: 1242 clusters
    # 0.99: 13131 clusters
    # 0.995: 21746 clusters

    thresholds = [0.98, 0.985, 0.99]
    explore_activation_thresholds(
        processed_responses_path, activation_cache_dir, layer, thresholds, "flowcharts")