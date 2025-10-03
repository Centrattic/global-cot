#%%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sentence_clustering import (
    gather_cot_sentences_for_prompt,
    load_embedder,
    embed_sentences,
    plot_clusters_vs_threshold,
    choose_threshold_by_silhouette,
    cluster_by_cosine_threshold,
    cluster_sentences_for_prompt,
)
from src.utils import load_rollouts_fields, load_prompts_json

#%% Minimal example scaffold
print("Setting params...")
rollouts_path = "/Users/jennakainic/global-cot/src/rollouts.json"
prompts, _ = load_prompts_json("/Users/jennakainic/global-cot/src/prompts.json")
prompt_text = prompts[0]
out_plot = "/Users/jennakainic/global-cot/src/clusters_vs_threshold.png"
thresholds = [0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775,]

#%%
print("Gathering sentences...")
sentences, _ = gather_cot_sentences_for_prompt(rollouts_path, prompt_text)

#%%
print("Embedding sentences...")
E = embed_sentences(load_embedder("sentence-transformers/all-MiniLM-L6-v2"), sentences)

#%%
print("Plotting clusters vs threshold...")
plot_clusters_vs_threshold(E, thresholds, out_plot)

#%%
print("Choosing threshold by silhouette...")
best_t, sil = choose_threshold_by_silhouette(E, thresholds)
print(f"Best threshold: {best_t}, silhouette: {sil}")
# %%
#%% Generate clusterings for candidate thresholds - EXAMPLE
print("Generating both clusterings...")
thresholds_to_compare = [0.65, 0.72]

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
cluster_sentences_for_prompt(rollouts_path, prompt_text, "sentence-transformers/all-MiniLM-L6-v2", 0.72, "/Users/jennakainic/global-cot/src/clusters_0.72.json")
# %%