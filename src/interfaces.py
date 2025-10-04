class Edge:
    def __init__(self, from_cluster: int, to_cluster: int, freq: int):
        self.from_cluster = from_cluster
        self.to_cluster = to_cluster
        self.freq = freq
    
    def print(self):
        print(f"Edge: {self.from_cluster} -> {self.to_cluster} (freq: {self.freq})")

class Node:
    def __init__(self, cluster_id: str, freq: int, sentences: List[str], 
                 activations: Optional[List[List[float]]] = None,
                 sentence_embeddings: Optional[List[List[float]]] = None,
                 mean_cosine_similarity: float = 0.0):
        self.cluster_id = cluster_id
        self.freq = freq # note: not the same as len(sentences) because could exist with multiplicity
        self.sentences = sentences
        self.activations = activations
        self.sentence_embeddings = sentence_embeddings
        self.mean_cosine_similarity = self._compute_mean_cosine_similarity(sentence_embeddings)

    def _compute_mean_cosine_similarity(self, sentence_embeddings: Optional[List[List[float]]]) -> float:
        if sentence_embeddings is None or len(sentence_embeddings) <= 1:
            return 0.0
            
        n = len(sentence_embeddings)
        total_sim = 0.0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                dot = sum(a*b for a,b in zip(sentence_embeddings[i], sentence_embeddings[j]))
                norm_i = sum(x*x for x in sentence_embeddings[i]) ** 0.5
                norm_j = sum(x*x for x in sentence_embeddings[j]) ** 0.5
                sim = dot / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0.0
                total_sim += sim
                count += 1
        return total_sim / count if count > 0 else 0.0
        self.index = cluster_id # Using cluster_id as index since they appear to be integers
    
    def print(self):
        print(f"Node {self.index}:")
        print(f"  Frequency: {self.freq}")
        print(f"  # Sentences: {len(self.sentences)}")
        print(f"  Mean Cosine Similarity: {self.mean_cosine_similarity:.3f}")
        print(f"  Has Activations: {self.activations is not None}")
        print(f"  Has Embeddings: {self.sentence_embeddings is not None}")

    class AnswerNode(Node):
        def __init__(self, true_answer: int, candidate_answers: List[str]):
            super().__init(f"Answer: {true_answer}", 0, candidate_answers) # freq is meaningless here
    
    class PromptNode(Node):
        def __init__(self, prompt: str):
            super().__init("prompt", 0, [prompt]) # freq is meaningless here

    # TODO Do we want a pathway class?