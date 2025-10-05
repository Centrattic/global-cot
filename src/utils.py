import json
from typing import Any, Dict, List, Tuple
import re
import os
import glob


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_responses_from_folder(responses_folder: str) -> List[Dict[str, Any]]:
    """Load all response files from the responses folder and return as list."""
    responses = []
    pattern = os.path.join(responses_folder, "*.json")

    for file_path in sorted(glob.glob(pattern)):
        response_data = load_json(file_path)
        if isinstance(response_data, dict):
            responses.append(response_data)
    
    return responses


def load_responses_from_composite(responses_file: str) -> List[Dict[str, Any]]:
    """Load responses from a single composite JSON file (list of dicts)."""
    data = load_json(responses_file)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def load_responses_as_rollouts_fields(responses_folder: str) -> Dict[str, List[Any]]:
    """Convert responses at path (folder or composite file) to rollouts fields format."""
    if os.path.isdir(responses_folder):
        responses = load_responses_from_folder(responses_folder)
    elif os.path.isfile(responses_folder):
        responses = load_responses_from_composite(responses_folder)
    else:
        responses = []
    
    # Initialize fields
    fields = {
        "cot": [],
        "response_content": [],
        "sentences": [],
        "index": [],
        "seed": []
    }
    
    # Sort responses by best-available index to maintain order
    def _sort_key(x: Dict[str, Any]) -> Any:
        if "processed_index" in x:
            return x.get("processed_index", 0)
        if "index" in x:
            return x.get("index", 0)
        return x.get("response_index", 0)

    responses.sort(key=_sort_key)

    for response in responses:
        fields["cot"].append(response.get("cot_content", ""))
        fields["response_content"].append(
            response.get("processed_response_content", response.get("response_content", ""))
        )
        fields["sentences"].append(response.get("sentences", []))
        fields["index"].append(
            response.get(
                "processed_index",
                response.get("index", response.get("response_index", 0)),
            )
        )
        fields["seed"].append(response.get("seed", 0))

    return fields


def load_clusters_json(clusters_path: str) -> List[Dict[str, Any]]:
    data = load_json(clusters_path)
    if isinstance(data, dict):
        clusters = data.get("clusters", [])
        if isinstance(clusters, list):
            return clusters
    return []


def extract_sentences(text: str) -> List[str]:
    if not text:
        return []

    # Find all sentence boundaries (punctuation followed by space or end of string)
    boundaries = []
    for i, char in enumerate(text):
        if char in '.!?' and (i + 1 >= len(text) or text[i + 1].isspace()):
            boundaries.append(i + 1)

    # Split text at boundaries
    sentences = []
    start = 0
    for boundary in boundaries:
        sentence = text[start:boundary].strip()
        if sentence:
            sentences.append(sentence)
        start = boundary

    # Handle remaining text (if no punctuation at end)
    if start < len(text):
        remaining = text[start:].strip()
        if remaining:
            sentences.append(remaining)

    # Filter out empty sentences
    sentences = [s for s in sentences if s.strip()]

    # Merge short sentences (<=3 words) with previous sentence
    final_sentences = []
    for sentence in sentences:
        word_count = len(sentence.split())
        if word_count <= 3 and final_sentences:
            final_sentences[-1] += " " + sentence
        else:
            final_sentences.append(sentence)

    return final_sentences
