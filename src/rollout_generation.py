import os
import csv
import re
import argparse
from typing import List, Tuple, Dict

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


def read_prompts(csv_path: str) -> List[str]:
    """Read a single column named 'prompt' from a CSV file and return non-empty prompts."""
    prompts: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("prompt")
            if value is None:
                continue
            text = value.strip()
            if text:
                prompts.append(text)
    return prompts


def split_cot_and_response(text: str) -> Tuple[str, str]:
    """Split a model output into (cot, response).

    Heuristics:
    - If <think>...</think> is present, CoT is the inner text, response is the remainder after </think>.
    - Otherwise, attempt to split on a final answer cue; if none, put all text into response.
    """
    start_tag = "<think>"
    end_tag = "</think>"
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


def analyze_think_tags(text: str) -> Tuple[bool, int, int]:
    """Return (has_think, think_start_index, think_end_index) for <think> tags in text."""
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
        extra_body={
            "include_reasoning": True,
            "reasoning": {"effort": effort, "exclude": False},
            "seed": seed,
        },
    )
    content = completion.choices[0].message.content
    return content if content is not None else ""


def generate_rollouts(prompts: List[str], client: OpenAI, model: str, num_rollouts: int, temperature: float, max_tokens: int, effort: str) -> List[Dict[str, str]]:
    """Generate rollouts for each prompt and return rows for CSV writing."""
    rows: List[Dict[str, str]] = []
    total = len(prompts) * num_rollouts
    pbar = tqdm(total=total, desc="Generating rollouts", unit="rollout")
    for prompt in prompts:
        for _ in range(num_rollouts):
            full_text = openrouter_chat(client, model, prompt, temperature, max_tokens, effort)
            has_think, think_start, think_end = analyze_think_tags(full_text)
            cot, response = split_cot_and_response(full_text)
            cot_sentence_count = count_sentences(cot)
            response_sentence_count = count_sentences(response)
            combined_sentence_count = cot_sentence_count + response_sentence_count
            rows.append({
                "prompt": prompt,
                "cot": cot,
                "response": response,
                "sentence_count": str(combined_sentence_count),
                "has_think": str(has_think),
                "think_start": str(think_start),
                "think_end": str(think_end),
                "raw": full_text,
            })
            pbar.update(1)
    pbar.close()
    return rows


def write_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    """Write rows to CSV with required columns."""
    fieldnames = [
        "prompt",
        "cot",
        "response",
        "sentence_count",
        "has_think",
        "think_start",
        "think_end",
        "raw",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate CoT rollouts via OpenRouter and save to CSV")
    p.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompt.csv"), help="Path to input CSV with a 'prompt' column")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "rollouts.csv"), help="Output CSV path")
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
    api_key = os.environ.get("OPENROUTER_KEY", "")
    client = build_openrouter_client(api_key)
    prompts = read_prompts(args.prompts)
    rows = generate_rollouts(prompts, client, args.model, args.num, args.temp, args.max_tokens, args.effort)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()


