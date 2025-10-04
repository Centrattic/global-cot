import argparse
import json


def save_clustered_pathways_json(
        clusters,  # List[Dict] with cluster info (e.g., id, freq, sentences, etc.)
        pathways,  # List[Dict] with pathway info (e.g., rollout_id, cluster_sequence, etc.)
        edges,
        out_path: str,
        transitions_edges=None,  # Optional[List[Dict]] from pathways['edges'] or ['transitions']['edges']
):
    """
    Save a JSON file with nodes (clusters) and rollouts (pathways as edge lists).
    """
    # Format nodes
    nodes = []
    for c in clusters:
        node = {
            "cluster_id": str(c.get("cluster_id")),
            "freq": c.get("size"),
            "representative_sentence": c.get("representative_sentence"),
            "sentences": c.get("unique_sentences"),
        }
        nodes.append(node)

    edges = []
    for e in edges:
        edge = {
            "from_cluster": e.get("from_cluster"),
            "to_cluster": e.get("to_cluster"),
            "freq": e.get("count")
        }
        edges.append(edge)

    # Format rollouts as pathways with edges
    rollouts = []
    for p in pathways:
        cluster_seq = p.get("cluster_sequence", [])
        edges = []
        for i in range(len(cluster_seq) - 1):
            edge = {
                "from_cluster": cluster_seq[i],
                "to_cluster": cluster_seq[i + 1],
                "freq":
                1  # TODO CHECK THINKING HERE -- difference between 1 rollout and aggregate
            }
            edges.append(edge)
        rollout = {"rollout_id": p.get("rollout_id"), "edges": edges}
        rollouts.append(rollout)

    out = {"nodes": nodes, "edges": edges, "rollouts": rollouts}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Format clustered pathways and save as JSON.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Threshold for clustering and pathways identification")
    parser.add_argument("--out",
                        type=str,
                        default="formatted_sentence_outputs_0.8.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    clusters_path = f"/Users/jennakainic/global-cot/clusters/clusters_{args.threshold}.json"
    pathways_path = f"/Users/jennakainic/global-cot/pathways/pathways_{args.threshold}.json"

    with open(clusters_path, "r") as f:
        clusters = json.load(f)
    with open(pathways_path, "r") as f:
        pathways = json.load(f)
    # If clusters/pathways are wrapped in a dict, extract the lists
    if isinstance(clusters, dict) and "clusters" in clusters:
        clusters = clusters["clusters"]
    if isinstance(pathways, dict) and "pathways" in pathways:
        edges = pathways["edges"]
        pathways = pathways["pathways"]

    save_clustered_pathways_json(clusters, pathways, edges, args.out)


if __name__ == "__main__":
    main()
