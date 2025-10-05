from flask import Flask, render_template, request, jsonify
import json
import os
from typing import Dict, List, Any
from pyngrok import ngrok
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Configuration
FLOWCHART_DIR = "../flowcharts"
DATA_DIR = ".."
PUBLIC_DATA_DIR = "public/data"

def get_flowchart_files() -> List[Dict[str, str]]:
    """Get list of available flowchart files"""
    files = []
    
    # Add main flowchart files
    main_files = [
        "formatted_sentence_outputs_0.8.json",
    ]
    
    for file in main_files:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            files.append({
                "name": file,
                "path": file_path
            })
    
    # Add flowchart directory files
    if os.path.exists(FLOWCHART_DIR):
        for file in os.listdir(FLOWCHART_DIR):
            if file.endswith('.json'):
                files.append({
                    "name": file,
                    "path": os.path.join(FLOWCHART_DIR, file)
                })
    
    # Add public data directory files
    if os.path.exists(PUBLIC_DATA_DIR):
        for file in os.listdir(PUBLIC_DATA_DIR):
            if file.endswith('.json'):
                files.append({
                    "name": file,
                    "path": os.path.join(PUBLIC_DATA_DIR, file)
                })
    
    return files

def load_flowchart_data(file_path: str) -> Dict[str, Any]:
    """Load flowchart data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

@app.route('/')
def index():
    """Main page"""
    files = get_flowchart_files()
    return render_template('index.html', files=files)

@app.route('/api/files')
def api_files():
    """API endpoint to get available files"""
    files = get_flowchart_files()
    return jsonify(files)

@app.route('/api/flowchart/<path:filename>')
def api_flowchart(filename):
    """API endpoint to get flowchart data"""
    files = get_flowchart_files()
    file_data = None
    
    for file_info in files:
        if file_info['name'] == filename:
            file_data = load_flowchart_data(file_info['path'])
            break
    
    if file_data is None:
        return jsonify({"error": "File not found"}), 404
    
    return jsonify(file_data)

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"Local server: http://localhost:5000")
    print(f"Public URL: {public_url}")
    print(f"Share this URL for public access: {public_url}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
