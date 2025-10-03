import os
import csv
import json
import re
import argparse
from typing import List, Tuple, Dict, Any

# Repo rule: avoid try/except and getattr/hasattr. Keep comments sparse; prefer docstrings.

from tqdm import tqdm
from openai import OpenAI


def load_env(path: str) -> Dict[str, str]:
    """Load KEY=VALUE lines from a .env file into os.environ and return them."""
    env: Dict[str, str] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                env[key] = value
    return env


def read_prompts(json_path: str) -> Tuple[List[str], List[str]]:
    """Read prompts and answers from JSON {"prompts": [...], "answers": [...]}"""
    prompts: List[str] = []
    answers: List[str] = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
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


def split_cot_and_response(text: str) -> Tuple[str, str]:
    """Split a model output into (cot, response).

    Heuristics:
    - If <thought>...</thought> is present, CoT is the inner text, response is the remainder after </thought>.
    - Otherwise, attempt to split on a final answer cue; if none, put all text into response.
    """
    start_tag = "<thought>"
    end_tag = "</thought>"
    if start_tag in text and end_tag in text:
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag, start)
        cot = text[start:end].strip()
        response = (text[end + len(end_tag):]).strip()
        return cot, response

    final_markers = ["Final Answer:", "Answer:", "Response:"]
    for marker in final_markers:
        idx = text.rfind(marker)
        if idx != -1:
            cot = text[:idx].strip()
            response = text[idx + len(marker):].strip()
            return cot, response

    return "", text.strip()


def normalize_answer(text: str) -> str:
    """Lowercase, trim, collapse spaces, strip surrounding punctuation."""
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,!?:;\n\t\r\f\v")
    return s


def analyze_thought_tags(text: str) -> Tuple[bool, int, int]:
    """Return (has_thought, thought_start_index, thought_end_index) for <thought> tags in text."""
    start_tag = "<thought>"
    end_tag = "</thought>"
    if start_tag in text and end_tag in text:
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag, start_idx) + len(end_tag)
        return True, start_idx, end_idx
    return False, -1, -1


def count_sentences(text: str) -> int:
    """Count sentences by ., !, or ? delimiters."""
    normalized = re.sub(r"[.!?]+", ".", text)
    matches = re.findall(r"[.!?]", normalized)
    return len(matches)


def build_openrouter_client(api_key: str) -> OpenAI:
    """Create an OpenAI client configured for OpenRouter."""
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def openrouter_chat(client: OpenAI, model: str, prompt: str, temperature: float, max_tokens: int, effort: str, seed: int = 42) -> str:
    """Call OpenRouter chat completions and return assistant content as text."""
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        extra_headers={
            "HTTP-Referer": "https://local",
            "X-Title": "global-cot-rollouts",
        },
        #extra_body={
        #    "include_reasoning": True,
        #    "reasoning": {"effort": effort, "exclude": False},
        #    "seed": seed,
        #},
    )
    content = completion.choices[0].message.content
    return content if content is not None else ""


def generate_rollouts(prompts: List[str], client: OpenAI, model: str, num_rollouts: int, temperature: float, max_tokens: int, effort: str, answers: List[str]) -> List[Dict[str, str]]:
    """Generate rollouts for each prompt and return rows for CSV writing."""
    rows: List[Dict[str, Any]] = []
    total = len(prompts) * num_rollouts
    pbar = tqdm(total=total, desc="Generating rollouts", unit="rollout")
    for i, prompt in enumerate(prompts):
        for _ in range(num_rollouts):
            full_text = openrouter_chat(client, model, prompt, temperature, max_tokens, effort)


            has_thought, thought_start, thought_end = analyze_thought_tags(full_text)
            cot, response = split_cot_and_response(full_text)
            cot_sentence_count = count_sentences(cot)
            response_sentence_count = count_sentences(response)
            combined_sentence_count = cot_sentence_count + response_sentence_count
            expected = answers[i] if i < len(answers) else ""
            is_correct = False
            if expected:
                expected_norm = normalize_answer(expected)
                response_norm = normalize_answer(response)
                is_correct = expected_norm in response_norm
            rows.append({
                "prompt": prompt,
                "cot": cot,
                "response": response,
                "sentence_count": combined_sentence_count,
                "has_thought": has_thought,
                "is_correct": is_correct,
                "thought_start": thought_start,
                "thought_end": thought_end,
                "raw": full_text,
            })
            pbar.update(1)
    pbar.close()
    return rows


def write_json(rows: List[Dict[str, Any]], out_path: str) -> None:
    """Write fields-oriented JSON: keys are column names, values are lists."""
    if rows:
        keys = list(rows[0].keys())
    else:
        keys = [
            "prompt",
            "cot",
            "response",
            "sentence_count",
            "has_thought",
            "is_correct",
            "thought_start",
            "thought_end",
            "raw",
        ]
    fields: Dict[str, List[Any]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            fields[k].append(row.get(k))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate CoT rollouts via OpenRouter and save to JSON")
    p.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts.json"), help="Path to input JSON with key 'prompts' (array of strings)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "rollouts.json"), help="Output JSON path")
    p.add_argument("--num", type=int, default=1, help="Number of rollouts per prompt")
    p.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--max_tokens", type=int, default=1440, help="Max tokens per completion")
    p.add_argument("--model", default="openai/gpt-oss-20b", help="OpenRouter model slug")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high"], help="Reasoning effort hint")
    p.add_argument("--dotenv", default=os.path.join(os.path.dirname(__file__), ".env"), help="Path to .env containing OPENROUTER_KEY")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    load_env(args.dotenv)
    # Fallback to project root .env if key not present after initial load
    has_key = (os.environ.get("OPENROUTER_API_KEY") is not None) or (os.environ.get("OPENROUTER_KEY") is not None)
    if not has_key:
        project_root_env = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))
        load_env(project_root_env)

    api_key = os.environ.get("OPENROUTER_KEY")

    client = build_openrouter_client(api_key)
    prompts, answers = read_prompts(args.prompts)
    rows = generate_rollouts(prompts, client, args.model, args.num, args.temp, args.max_tokens, args.effort, answers)
    write_json(rows, args.out)


if __name__ == "__main__":
    main()


