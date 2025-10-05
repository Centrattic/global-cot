# Flowchart Visualizer

A clean, fast flowchart visualizer built with Next.js and D3.js that can be deployed on Vercel.

## Features

- **Simple rollout selection**: Enter rollout IDs (e.g., `1, 2-5, 10`) to visualize specific reasoning paths
- **Fast edge lookup**: Direct lookup from JSON dictionary for instant performance
- **Clean visualization**: Each node in its own layer with edges connecting them
- **Interactive**: Click nodes to see details, zoom and pan the visualization
- **Vercel ready**: Optimized for deployment on Vercel

## How it Works

1. **Select a file** from the dropdown
2. **Enter rollout IDs** in the input field (supports ranges like `1-5` and comma separation)
3. **See the visualization** - only nodes and edges from selected rollouts are shown

## Data Format

The tool expects JSON files with this structure:

```json
{
  "nodes": [
    {
      "cluster_id": "0",
      "freq": 1250,
      "representative_sentence": "The answer is 19 bits.",
      "mean_similarity": 0.85,
      "sentences": [...]
    }
  ],
  "rollouts": {
    "1": [{"node_a": "0", "node_b": "1"}],
    "2": [{"node_a": "1", "node_b": "2"}]
  }
}
```

## Development

1. Install dependencies:
```bash
npm install
```

2. The app automatically reads from the `../flowcharts/` directory (no copying needed)

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000)

5. For public access with ngrok:
```bash
npm run ngrok
```

## Deployment on Vercel

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Vercel will automatically deploy the Next.js application

The app will be available at your Vercel URL.

## Performance

- **Fast edge lookup**: O(1) lookup from rollout ID to edges
- **Limited rendering**: Only renders nodes and edges for selected rollouts
- **Efficient D3.js**: Optimized SVG rendering with zoom/pan
- **No Python backend**: Pure JavaScript/TypeScript for maximum speed
