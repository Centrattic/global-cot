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
)
from src.utils import load_responses_as_rollouts_fields

#%% Minimal example scaffold
print("Setting params...")
rollouts_path = "/Users/jennakainic/global-cot/responses"
out_path = "/Users/jennakainic/global-cot/clusters"
thresholds = [0.7 + 0.05*i for i in range(6)]

#%%
print("Gathering sentences...")
sentences, _ = gather_cot_sentences(rollouts_path)

#%%
print("Embedding sentences...")
E = embed_sentences(load_embedder("sentence-transformers/all-MiniLM-L6-v2"), sentences)

#%%
print("Plotting clusters vs threshold...")
plot_clusters_vs_threshold(E, thresholds, out_path+"/clusters_vs_threshold.png")

#%%
print("Plotting silhouette vs threshold...")
plot_silhouette_vs_threshold(E, thresholds, out_path+"/silhouette_vs_threshold.png")
# %%
#%% Generate clusterings for candidate thresholds - EXAMPLE
print("Generating both clusterings...")
thresholds_to_compare = [0.675, 0.85]

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
# EXAMPLE - create and export clusters to json
threshold = 0.72
cluster_sentences(rollouts_path, "sentence-transformers/all-MiniLM-L6-v2", threshold, out_path+f"/clusters_{threshold}.json")
# %%