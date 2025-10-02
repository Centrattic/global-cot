# global-cot

https://github.com/interp-reasoning/thought-anchors

The plan:

1. Start with 1 prompt, and generate like 1000 rollouts
2. Embed all generated sentences with sentence transformer (https://huggingface.co/sentence-transformers)
3. Set some embedding cosine similarity threshold for sentences being "equivalent"/discussing equivalent concept.  (Manually label some sentences ourselves and use that to estimate threshold? Start with arbitrary threshold).
4. Find thought anchors by looking at sentence-level attention matrices (set some attention score threshold)
5. Build a tool to visualize flowcharts for thought anchors. A node i
- Cluster thought anchors by the embedding similarity threshold, every cluster is its own node
- Connect nodes a,b with edge (a,b) if there are rollouts such that a is before b

Challenge: analyzing thought anchors semantically is hard


Problem description: 

Most work studying chain-of-thought has focused on single rollouts or local neighborhoods. This is useful, but I'm excited about zooming out. Can we talk about the structure of computation across many random seeds for a fixed prompt?
Research Questions:
Are there certain key thought anchors that frequently arise across different rollouts?
If we cluster all generated sentences by embedding, is there roughly a finite range of concepts that get covered, or is it messier than that?

Could we imagine a massive flowchart, where you move between classes of sentences with certain probabilities?

How often do you see a genuinely new type of reasoning structure in a rollout?
Meta: I have no idea if this is possible, but it would be very cool if someone tried.

