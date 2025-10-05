#%%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sentence_clustering import (
    gather_cot_sentences,
    load_embedder,
    embed_sentences,
    plot_clusters_vs_threshold,
    plot_silhouette_vs_threshold,
    cluster_by_cosine_threshold,
    cluster_sentences,
    plot_sentence_clusters_vs_threshold_from_json,
)
from src.utils import load_responses_as_rollouts_fields

#%% Minimal example scaffold
print("Setting params...")
# Accept either a folder of response JSONs or a single composite JSON file
rollouts_path = "/Users/jennakainic/global-cot/processed_responses.json"
out_path = "/Users/jennakainic/global-cot/clusters"
thresholds = [0.4 * 0.025*i for i in range(20)]

#%%
print("Gathering sentences...")
sentences, _ = gather_cot_sentences(rollouts_path)

#%%
print("Embedding sentences...")
E = embed_sentences(load_embedder("sentence-transformers/paraphrase-mpnet-base-v2"), sentences)


#%%
print("Computing pairwise cosine similarities distribution and caching sims...")
import numpy as np
import matplotlib.pyplot as plt

# Compute pairwise cosine similarities
sims = E @ E.T

# Cache sims to disk for reuse
os.makedirs(out_path, exist_ok=True)
np.save(os.path.join(out_path, "sentence_sims.npy"), sims)

# Get upper triangle values (excluding diagonal)
triu_indices = np.triu_indices(len(E), k=1)
sim_values = sims[triu_indices]

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(sim_values, bins=65, density=True)
plt.xlabel("Cosine similarity")
plt.ylabel("Density")
plt.title("Distribution of Pairwise Cosine Similarities")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_path+"/similarity_distribution.png", dpi=150)
plt.close()

print(f"Mean similarity: {sim_values.mean():.3f}")
print(f"Std similarity: {sim_values.std():.3f}")
print(f"Min similarity: {sim_values.min():.3f}")
print(f"Max similarity: {sim_values.max():.3f}")

#%%
print("Computing random vector cosine similarities distribution...")
dim = E.shape[1]
n = len(E)
rand_E = np.random.randn(n, dim)
rand_E /= np.linalg.norm(rand_E, axis=1, keepdims=True)
rand_sims = rand_E @ rand_E.T
rand_sim_values = rand_sims[triu_indices]

plt.figure(figsize=(8, 5))
plt.hist(rand_sim_values, bins=65, density=True)
plt.xlabel("Cosine similarity")
plt.ylabel("Density")
plt.title("Distribution of Pairwise Cosine Similarities (Random Vectors)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_path+"/random_similarity_distribution.png", dpi=150)
plt.close()

print(f"Random mean similarity: {rand_sim_values.mean():.3f}")
print(f"Random std similarity: {rand_sim_values.std():.3f}")
print(f"Random min similarity: {rand_sim_values.min():.3f}")
print(f"Random max similarity: {rand_sim_values.max():.3f}")

#%%
print("Plotting clusters vs threshold (reusing cached sims)...")
plot_clusters_vs_threshold(E, thresholds, out_path+"/clusters_vs_threshold.png", precomputed_sims=sims)

#%%
print("Plotting silhouette vs threshold (reusing cached sims)...")
plot_silhouette_vs_threshold(E, thresholds, out_path+"/silhouette_vs_threshold.png", precomputed_sims=sims)
# %%
len(sentences)
#%% Generate clusterings for candidate thresholds - EXAMPLE
print("Generating both clusterings...")
thresholds_to_compare = [0.8, 0.85]

for t in thresholds_to_compare:
    print(f"Threshold: {t}")
    
    labels = cluster_by_cosine_threshold(E, t)
    n_clusters = len(set(labels))
    
    print(f"Number of clusters: {n_clusters}")
    
    # Show representative sentences
    from collections import defaultdict
    clusters_dict = defaultdict(list)
    for i, label in enumerate(labels):
        clusters_dict[label].append(sentences[i])
    
    for cluster_id in sorted(clusters_dict.keys()):
        sents = clusters_dict[cluster_id]
        print(f"\nCluster {cluster_id} ({len(sents)} sentences):")
        for sent in sents:
            print(f"  {sent}")
# %%
# OPTIONAL: export clusters for multiple thresholds and plot from cached JSONs
export_thresholds = []  # e.g., [0.75, 0.8, 0.85]
for t in export_thresholds:
    print(f"Exporting clusters JSON for threshold {t}...")
    cluster_sentences(rollouts_path, "sentence-transformers/paraphrase-mpnet-base-v2", t, out_path+f"/clusters_{t}.json")

if export_thresholds:
    print("Plotting clusters vs threshold from cached JSONs...")
    plot_sentence_clusters_vs_threshold_from_json(out_path, out_path+"/clusters_vs_threshold_cached.png")
# %%