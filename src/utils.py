import json
from typing import Any, Dict, List, Tuple
import re


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_rollouts_fields(rollouts_path: str) -> Dict[str, List[Any]]:
    data = load_json(rollouts_path)
    if isinstance(data, dict) and "fields" in data:
        return data["fields"]
    return data


def load_clusters_json(clusters_path: str) -> List[Dict[str, Any]]:
    data = load_json(clusters_path)
    if isinstance(data, dict):
        clusters = data.get("clusters", [])
        if isinstance(clusters, list):
            return clusters
    return []


def load_prompts_json(prompts_path: str) -> Tuple[List[str], List[str]]:
    prompts: List[str] = []
    answers: List[str] = []
    data = load_json(prompts_path)
    if isinstance(data, dict):
        parr = data.get("prompts", [])
        aarr = data.get("answers", [])
        if isinstance(parr, list):
            for item in parr:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        prompts.append(text)
        if isinstance(aarr, list):
            for item in aarr:
                if isinstance(item, str):
                    answers.append(item.strip())
    return prompts, answers


def extract_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r'[.!?]+\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


