# Flowchart Visualizer - Python Flask Version

A simple, elegant Python web application for visualizing flowchart data with interactive rollouts.

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
python app.py
```

3. **Access the application:**
   - Local: http://localhost:5000
   - Public (via ngrok): The terminal will show the public URL

## Features

- ✅ **File Selection**: Choose from available JSON files in dropdown
- ✅ **Interactive Nodes**: Click nodes to see all sentences in cluster
- ✅ **Rollout Visualization**: Select individual rollouts to highlight paths
- ✅ **Adaptive Layout**: Grid layout for files without edges, hierarchical for files with edges
- ✅ **Public Access**: ngrok integration for sharing with others
- ✅ **Clean Design**: Minimalist, elegant interface

## Supported File Formats

The visualizer handles both JSON formats:

### Format 1 (Small files with edges):
```json
{
  "nodes": [...],
  "edges": [...],
  "rollouts": [...]
}
```

### Format 2 (Large files without edges):
```json
{
  "nodes": [...],
  "rollouts": []
}
```

## File Structure

```
deployment/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Main HTML template with D3.js
└── README_PYTHON.md   # This file
```

## Usage

1. **Select a File**: Use the dropdown to choose a flowchart file
2. **Explore Nodes**: Click any node to see detailed sentence information
3. **View Rollouts**: Use the sidebar to select individual reasoning paths
4. **Share**: The ngrok URL allows others to access your visualization

## Dependencies

- Flask 3.0.0 - Web framework
- pyngrok 7.0.0 - Public tunnel for sharing

## Notes

- The app automatically detects files in `../flowcharts/` and `../` directories
- ngrok creates a public tunnel so you can share the URL with others
- All styling is inline CSS for simplicity
- D3.js is loaded from CDN for visualization
