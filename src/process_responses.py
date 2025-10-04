#!/usr/bin/env python3
"""
Script to process JSON response files and create a consolidated processed_responses.json file.
"""

import json
import os
import glob
from pathlib import Path


def is_single_int(value):
    """Check if a string represents a single integer."""
    try:
        int(value.strip())
        return True
    except ValueError:
        return False


def extract_processed_response_content(response_content, stats=None):
    """
    Extract processed_response_content based on the specified rules:
    1. If response_content is a single int, use it
    2. If there is only one integer in the content (even with other text), use that integer
    3. If "19" is found in response_content, use "19"
    4. Otherwise, return empty string
    """
    if not response_content or not response_content.strip():
        if stats:
            stats["empty_content"] += 1
        return ""
    
    import re
    
    # Check if it's a single integer (no other characters)
    if is_single_int(response_content):
        if stats:
            stats["single_int"] += 1
        return response_content.strip()
    
    # Find all integers in the response content
    integers = re.findall(r'\d+', response_content)
    
    # If there's exactly one integer, use it
    if len(integers) == 1:
        if stats:
            stats["one_int_with_text"] += 1
        return integers[0]
    
    # If there are multiple integers, check if "19" is among them
    if "19" in integers:
        if stats:
            stats["multiple_ints_with_19"] += 1
        return "19"
    
    # Otherwise return empty string
    if stats:
        stats["no_match"] += 1
    return ""


def process_responses():
    """Process all JSON response files and create consolidated output."""
    responses_dir = Path("/home/ubuntu/riya-probing/global-cot/responses")
    
    # Find all completion_*.json files
    json_files = sorted(glob.glob(str(responses_dir / "completion_*.json")))
    
    processed_responses = []
    processed_index = 0
    
    # Initialize statistics tracking
    stats = {
        "empty_content": 0,
        "single_int": 0,
        "one_int_with_text": 0,
        "multiple_ints_with_19": 0,
        "no_match": 0,
        "filtered_out": 0,
        "processed": 0
    }
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                response_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {json_file}")
                continue
        
        # Extract processed response content first
        processed_response_content = extract_processed_response_content(
            response_data.get("response_content", ""), stats
        )
        
        # Check if this response should be filtered out
        response_content = response_data.get("response_content", "")
        if not response_content.strip() or not processed_response_content:
            stats["filtered_out"] += 1
            print(f"Filtering out {json_file} - empty response content")
            continue
        
        # Determine correctness (assuming "19" is the correct answer)
        correctness = processed_response_content == "19"
        
        # Create processed entry
        processed_entry = {
            "response_index": response_data.get("index", 0),
            "processed_index": processed_index,
            "cot_content": response_data.get("cot_content", ""),
            "response_content": response_data.get("response_content", ""),
            "processed_response_content": processed_response_content,
            "correctness": correctness,
            "sentences": response_data.get("sentences", []),
            "seed": response_data.get("seed", 0)
        }
        
        processed_responses.append(processed_entry)
        processed_index += 1
        stats["processed"] += 1
        
        print(f"Processed {json_file}: index {response_data.get('index', 0)} -> processed_index {processed_index - 1}")
    
    # Write consolidated output
    output_file = "/home/ubuntu/riya-probing/global-cot/processed_responses.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_responses, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete!")
    print(f"Total input files: {len(json_files)}")
    print(f"Total processed entries: {len(processed_responses)}")
    print(f"Output written to: {output_file}")
    
    # Print some statistics
    correct_count = sum(1 for entry in processed_responses if entry["correctness"])
    print(f"Correct responses: {correct_count}/{len(processed_responses)} ({correct_count/len(processed_responses)*100:.1f}%)")
    
    # Print processing stage statistics
    print(f"\nResponse Processing Stage Statistics:")
    print(f"  Empty content: {stats['empty_content']}")
    print(f"  Single integer: {stats['single_int']}")
    print(f"  One integer with text: {stats['one_int_with_text']}")
    print(f"  Multiple integers with '19': {stats['multiple_ints_with_19']}")
    print(f"  No match: {stats['no_match']}")
    print(f"  Total filtered out: {stats['filtered_out']}")
    print(f"  Total processed: {stats['processed']}")


if __name__ == "__main__":
    process_responses()
