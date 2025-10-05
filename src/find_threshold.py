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
    plot_sentence_clusters_vs_threshold_from_json,
    export_clusters_to_json,
)
import argparse
import time

def _embed(rollouts_path: str, model: str, batch_size: int, cache_dir: str = None):
    t0 = time.time()
    print("Gathering sentences...")
    sentences, sentence_index_to_rollout_ids = gather_cot_sentences(rollouts_path)
    print(f"  -> {time.time()-t0:.2f}s, {len(sentences)} sentences")
    embedder = load_embedder(model)
    cache_path = os.path.join(cache_dir, "embeddings_cache.npz") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        import numpy as np
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['sentences'].tolist(), data['sentence_index_to_rollout_ids'].item(), data['embeddings']
    t0 = time.time()
    print("Embedding sentences...")
    embeddings = embed_sentences(
        embedder,
        sentences,
        batch_size=batch_size,
    )
    print(f"  -> {time.time()-t0:.2f}s")
    if cache_path:
        import numpy as np
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(
            cache_path,
            sentences=sentences,
            sentence_index_to_rollout_ids=sentence_index_to_rollout_ids,
            embeddings=embeddings,
        )
    return sentences, sentence_index_to_rollout_ids, embeddings


def run_explore(rollouts_path: str, out_dir: str, model: str, thresholds, batch_size: int, cache_dir: str = None) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    sentences, _, E = _embed(rollouts_path, model, batch_size, cache_dir)

    os.makedirs(out_dir, exist_ok=True)
    # Only save if not already in cache_dir
    emb_path = os.path.join(out_dir, "sentence_embeddings.npy")
    if not os.path.exists(emb_path):
        np.save(emb_path, E)

    sims_path = os.path.join(out_dir, "sentence_sims.npy")
    if os.path.exists(sims_path):
        print(f"Loading cached similarity matrix from {sims_path}")
        sims = np.load(sims_path).astype(np.float32)
    else:
        print("Computing pairwise cosine similarities and caching sims...")
        sims = E @ E.T
        np.save(sims_path, sims.astype(np.float16))
        sims = sims.astype(np.float32)

    triu_indices = np.triu_indices(len(E), k=1)
    sim_values = sims[triu_indices]

    plt.figure(figsize=(8, 5))
    plt.hist(sim_values, bins=65, density=True)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Distribution of Pairwise Cosine Similarities")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir+"/similarity_distribution.png", dpi=150)
    plt.close()

    print(f"Mean similarity: {sim_values.mean():.3f}")
    print(f"Std similarity: {sim_values.std():.3f}")
    print(f"Min similarity: {sim_values.min():.3f}")
    print(f"Max similarity: {sim_values.max():.3f}")

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
    plt.savefig(out_dir+"/random_similarity_distribution.png", dpi=150)
    plt.close()

    print(f"Random mean similarity: {rand_sim_values.mean():.3f}")
    print(f"Random std similarity: {rand_sim_values.std():.3f}")
    print(f"Random min similarity: {rand_sim_values.min():.3f}")
    print(f"Random max similarity: {rand_sim_values.max():.3f}")

    print("Plotting clusters vs threshold (reusing cached sims)...")
    plot_clusters_vs_threshold(E, thresholds, out_dir+"/clusters_vs_threshold.png", precomputed_sims=sims)

    print("Plotting silhouette vs threshold (reusing cached sims)...")
    plot_silhouette_vs_threshold(E, thresholds, out_dir+"/silhouette_vs_threshold.png", precomputed_sims=sims)


def run_cluster(rollouts_path: str, out_json_path: str, model: str, threshold: float, batch_size: int, cache_dir: str = None) -> None:
    sentences, sentence_index_to_rollout_ids, E = _embed(rollouts_path, model, batch_size, cache_dir)
    labels = cluster_by_cosine_threshold(E, threshold)
    export_clusters_to_json(out_json_path, labels, E, sentences, sentence_index_to_rollout_ids)


def _parse_args():
    parser = argparse.ArgumentParser(description="Explore thresholds or cluster sentences.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--rollouts-path", type=str, default="/Users/jennakainic/global-cot/processed_responses.json")
        p.add_argument("--out-dir", type=str, default="/Users/jennakainic/global-cot/clusters")
        p.add_argument("--embedding-model", type=str, default="sentence-transformers/paraphrase-mpnet-base-v2")
        p.add_argument("--batch-size", type=int, default=2048)
        p.add_argument("--cache-dir", type=str, default="", help="Directory to cache/reuse embeddings .npz")

    explore = subparsers.add_parser("explore", help="Make plots for threshold exploration")
    add_common(explore)
    explore.add_argument("--thresholds", type=str, default="", help="Comma-separated list, e.g. 0.7,0.75,0.8")
    explore.add_argument("--min-threshold", type=float, default=0.4)
    explore.add_argument("--max-threshold", type=float, default=0.9)
    explore.add_argument("--num-points", type=int, default=10)

    cluster = subparsers.add_parser("cluster", help="Cluster at a specified threshold and write JSON")
    add_common(cluster)
    cluster.add_argument("--threshold", type=float, required=True)
    cluster.add_argument("--out-json", type=str, default="", help="Output JSON path; defaults to out-dir/clusters_{threshold}.json")

    return parser.parse_args()


def main():
    args = _parse_args()
    if args.command == "explore":
        if args.thresholds:
            thresholds = [float(x) for x in args.thresholds.split(",") if x]
        else:
            if args.num_points <= 1:
                thresholds = [float(args.min_threshold)]
            else:
                step = (args.max_threshold - args.min_threshold) / float(args.num_points - 1)
                thresholds = [args.min_threshold + step * i for i in range(args.num_points)]
        run_explore(
            args.rollouts_path,
            args.out_dir,
            args.embedding_model,
            thresholds,
            args.batch_size,
            args.cache_dir or None,
        )
    elif args.command == "cluster":
        out_json = args.out_json if args.out_json else os.path.join(args.out_dir, f"clusters_{args.threshold}.json")
        os.makedirs(args.out_dir, exist_ok=True)
        run_cluster(
            args.rollouts_path,
            out_json,
            args.embedding_model,
            float(args.threshold),
            args.batch_size,
            args.cache_dir or None,
        )

if __name__ == "__main__":
    main()