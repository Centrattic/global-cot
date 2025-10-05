export interface Node {
    cluster_id: string;
    freq: number;
    representative_sentence: string;
    mean_similarity: number;
    sentences: Sentence[];
}

export interface Sentence {
    text: string;
    count: number;
    rollout_ids: number[];
}

export interface Edge {
    node_a: string;
    node_b: string;
}

export interface Rollout {
    index: string;
    edges: Edge[];
}

export interface FlowchartData {
    nodes: Node[];
    rollouts: { [key: string]: Edge[] } | Rollout[];
}