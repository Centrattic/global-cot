from __future__ import annotations
import unsloth
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
from src.utils import extract_sentences
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
    ReasoningEffort
)
from tqdm import tqdm


class ActivationExtractor:
    def __init__(self, model_name: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit", cache_dir: Path = Path("activation_cache")):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load model and tokenizer using Unsloth FastLanguageModel
        print("Loading model with Unsloth FastLanguageModel")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        
        # Enable inference mode for faster generation
        FastLanguageModel.for_inference(self.model)
        print("Model loaded successfully!")
        
        # Configure tokenizer
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        print("Tokenizer configured successfully!")
        
        # Load harmony encoding
        print("Loading harmony encoding...")
        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        print("Harmony encoding loaded successfully!")
        
        # Track completion index
        self.completion_index = 0
        
        # Initialize layer folders (extract from specific layers)
        self.layers = [7, 13, 17]
        for layer in self.layers:
            layer_dir = self.cache_dir / str(layer)
            layer_dir.mkdir(exist_ok=True)
        
        # Initialize responses folder
        self.responses_dir = self.cache_dir / "responses"
        self.responses_dir.mkdir(exist_ok=True)
    
    def extract_activations(self, prompt: str, max_tokens: int = 512, seed: int = 0) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract activations for chain-of-thought tokens only."""

        # print("[DEBUG] SystemContent.new()")
        # print(dir(SystemContent.new()))

        low_reasoning_system_prompt = SystemContent.new()
        low_reasoning_system_prompt.reasoning_effort = ReasoningEffort.LOW

        # Create conversation
        convo = Conversation.from_messages([
            Message.from_role_and_content(
                Role.SYSTEM,
                low_reasoning_system_prompt,
            ),
            Message.from_role_and_content(Role.USER, prompt),
        ])
        
        # Get tokens for completion
        tokens = self.enc.render_conversation_for_completion(convo, Role.ASSISTANT)
        
        # Generate response with deterministic seed
        torch.manual_seed(seed)
        
        # Convert tokens to tensor
        input_ids = torch.tensor([tokens], device=self.model.device)
        
        # Create attention mask (all tokens are valid)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate response using Unsloth's optimized generation with streaming
        print("Generating response...")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True,
                streamer=self._create_streamer()
            )
        
        # Extract only the new tokens
        completion_tokens = outputs[0][len(input_ids[0]):].tolist()
        
        # Debug: print the raw completion
        print(f"\n[DEBUG] Raw completion tokens: {completion_tokens[:20]}...")
        print(f"[DEBUG] Raw completion text: {self.tokenizer.decode(completion_tokens)}")
        
        # Parse chain of thought and response from completion text
        completion_text = self.tokenizer.decode(completion_tokens)
        
        # Extract chain of thought between "<|channel|>analysis<|message|>" and "<|end|>"
        cot_start = "<|channel|>analysis<|message|>"
        cot_end = "<|end|>"
        
        cot_content = ""
        if cot_start in completion_text and cot_end in completion_text:
            start_idx = completion_text.find(cot_start) + len(cot_start)
            end_idx = completion_text.find(cot_end, start_idx)
            if start_idx < end_idx:
                cot_content = completion_text[start_idx:end_idx].strip()
        
        # Extract response between "<|start|>assistant<|channel|>final<|message|>" and "<|return|>"
        response_start = "<|start|>assistant<|channel|>final<|message|>"
        response_end = "<|return|>"
        
        response_content = ""
        if response_start in completion_text and response_end in completion_text:
            start_idx = completion_text.find(response_start) + len(response_start)
            end_idx = completion_text.find(response_end, start_idx)
            if start_idx < end_idx:
                response_content = completion_text[start_idx:end_idx].strip()
        
        if not cot_content:
            print("No chain of thought found in response")
            print(f"Completion text: {completion_text[:500]}...")
            return {}
        
        # Tokenize chain of thought using harmony encoding
        cot_tokens = self.enc.encode(cot_content)
        
        # Extract activations for each layer
        layer_activations = {}
        for layer in self.layers:
            activations = self._extract_layer_activations(cot_tokens, layer)
            if activations is not None:
                layer_activations[layer] = activations
        
        # Store activations
        self._store_activations(layer_activations, cot_content, response_content, cot_tokens, seed)
        
        return layer_activations
    
    def _create_streamer(self):
        """Create a text streamer to show generation in real-time."""
        from transformers import TextStreamer
        
        class ConsoleStreamer(TextStreamer):
            def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
                super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
                self.generated_text = ""
            
            def put(self, value):
                if value is not None:
                    # Handle different input types
                    if hasattr(value, 'cpu'):  # It's a tensor
                        # Convert tensor to list of token IDs
                        token_ids = value.tolist()
                    elif isinstance(value, list):  # It's already a list
                        token_ids = value
                    else:  # It's a string
                        self.generated_text += value
                        print(value, end="", flush=True)
                        return
                    
                    # Flatten the token_ids if it's nested (batch dimension)
                    if isinstance(token_ids[0], list):
                        token_ids = token_ids[0]  # Take first batch
                    
                    # Debug: print token IDs to see what we're getting
                    # print(f"\n[DEBUG] Token IDs: {token_ids[:10]}...")  # First 10 tokens
                    
                    # Decode the token IDs to text
                    text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    # print(f"[DEBUG] Decoded text: {text[:100]}...")  # First 100 chars
                    # Only print new tokens (not the prompt)
                    if hasattr(self, '_last_text'):
                        new_text = text[len(self._last_text):]
                        if new_text:
                            self.generated_text += new_text
                            print(new_text, end="", flush=True)
                    else:
                        self.generated_text = text
                        print(text, end="", flush=True)
                    self._last_text = text
            
            def end(self):
                print()  # New line when generation ends
        
        return ConsoleStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    def _extract_layer_activations(self, tokens: List[int], layer: int) -> Optional[np.ndarray]:
        """Extract activations for a specific layer in MoE architecture using Unsloth."""
        try:
            # Convert tokens to tensor
            input_ids = torch.tensor([tokens], device=self.model.device)
            
            activations = None
            
            def hook_fn(module, input, output):
                nonlocal activations
                if isinstance(output, tuple):
                    activations = output[0].detach().cpu().to(torch.float32).numpy()
                else:
                    activations = output.detach().cpu().to(torch.float32).numpy()
            
            # For GPT-OSS MoE architecture, hook the transformer block output
            # The model structure should be: model.model.layers[layer_index]
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                target_layer = self.model.model.layers[layer - 1]  # 0-indexed
                
                # Hook the entire transformer block output (after MoE processing)
                handle = target_layer.register_forward_hook(hook_fn)
                
                # Forward pass
                with torch.no_grad():
                    _ = self.model(input_ids)
                
                handle.remove()
                
                return activations
            else:
                print(f"Could not find layer {layer} in model structure")
                print(f"Model attributes: {dir(self.model)}")
                if hasattr(self.model, 'model'):
                    print(f"Model.model attributes: {dir(self.model.model)}")
                return None
                
        except Exception as e:
            print(f"Error extracting activations for layer {layer}: {e}")
            return None
    
    def _store_activations(self, layer_activations: Dict[int, Dict[str, np.ndarray]], cot_content: str, response_content: str, cot_tokens: List[int], seed: int):
        """Store activations with sentence-level aggregation."""
        
        # Split into sentences
        sentences = self._split_into_sentences(cot_content)
        
        # Store completion metadata
        completion_data = {
            "index": self.completion_index,
            "cot_content": cot_content,
            "response_content": response_content,
            "sentences": sentences,
            "seed": seed
        }
        
        # Store completion metadata in separate file
        completion_file = self.responses_dir / f"completion_{self.completion_index}.json"
        with open(completion_file, 'w') as f:
            json.dump(completion_data, f, indent=2)
        
        # Store activations for each layer
        for layer, activations in layer_activations.items():
            if activations is None:
                continue
                
            layer_dir = self.cache_dir / str(layer)
            
            # Aggregate activations by sentence
            sentence_activations = self._aggregate_by_sentences(activations, sentences, cot_content, cot_tokens)
            
            # Save to .npy file
            npy_path = layer_dir / f"completion_{self.completion_index}.npy"
            np.save(npy_path, sentence_activations)
        
        self.completion_index += 1
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving punctuation and merging short sentences."""
        return extract_sentences(text)
    
    def _aggregate_by_sentences(self, activations: np.ndarray, sentences: List[str], cot_content: str, cot_tokens: List[int]) -> Dict[str, np.ndarray]:
        """Aggregate activations by sentences using sequential token indexing."""
        if len(activations.shape) != 3:  # Should be (batch, seq, hidden)
            print(f"Unexpected activation shape: {activations.shape}")
            return {}
        
        # Remove batch dimension
        activations = activations[0]  # Shape: (seq, hidden)
        
        sentence_activations = {}
        current_index = 0
        
        for sentence in sentences:
            # Tokenize the sentence to get its length
            sentence_tokens = self.enc.encode(sentence)
            sentence_length = len(sentence_tokens)
            
            # Extract activations for this sentence's tokens
            sentence_activations_tokens = activations[current_index:current_index + sentence_length]
            
            # Aggregate across tokens in the sentence (mean pooling)
            sentence_activation = np.mean(sentence_activations_tokens, axis=0)
            sentence_activations[sentence] = sentence_activation
            
            # Move to next sentence
            current_index += sentence_length
        
        return sentence_activations
    
    def load_activations(self, layer: int, completion_index: int) -> Optional[Dict[str, np.ndarray]]:
        """Load activations for a specific layer and completion."""
        layer_dir = self.cache_dir / str(layer)
        npy_path = layer_dir / f"completion_{completion_index}.npy"
        
        if npy_path.exists():
            return np.load(npy_path, allow_pickle=True).item()
        return None
    
    def debug_activation_format(self, layer: int, completion_index: int):
        """Debug function to inspect .npy file format and structure."""
        layer_dir = self.cache_dir / str(layer)
        npy_path = layer_dir / f"completion_{completion_index}.npy"
        
        if not npy_path.exists():
            print(f"File not found: {npy_path}")
            return
        
        print(f"\n=== DEBUG: Activation file {npy_path} ===")
        
        # Load the .npy file
        data = np.load(npy_path, allow_pickle=True).item()
        
        print(f"Data type: {type(data)}")
        print(f"Number of sentences: {len(data)}")
        
        # Print structure without activation vectors
        for i, (sentence, activation) in enumerate(data.items()):
            print(f"\nSentence {i+1}:")
            print(f"  Text: {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
            print(f"  Activation shape: {activation.shape}")
            print(f"  Activation dtype: {activation.dtype}")
            print(f"  Activation min: {activation.min():.6f}")
            print(f"  Activation max: {activation.max():.6f}")
            print(f"  Activation mean: {activation.mean():.6f}")
            print(f"  Activation std: {activation.std():.6f}")
            
            # Show first few values (but not the whole vector)
            if len(activation) > 10:
                print(f"  First 10 values: {activation[:10]}")
            else:
                print(f"  All values: {activation}")
        
        print(f"\n=== END DEBUG ===\n")
    
    def get_completion_info(self, completion_index: int) -> Optional[Dict[str, Any]]:
        """Get completion information for a specific completion."""
        completion_file = self.responses_dir / f"completion_{completion_index}.json"
        if completion_file.exists():
            with open(completion_file, 'r') as f:
                return json.load(f)
        return None


def main():
    """Example usage with command line arguments for parallelization."""
    parser = argparse.ArgumentParser(description='Extract activations from GPT-OSS-20B model')
    parser.add_argument('--start', type=int, default=0, help='Start index for completion range (default: 0)')
    parser.add_argument('--end', type=int, default=1, help='End index for completion range (default: 1)')
    parser.add_argument('--prompt', type=str, default=None, help='Custom prompt (optional)')
    
    args = parser.parse_args()
    
    extractor = ActivationExtractor()
    
    # Example prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "Solve this problem step by step: \n When the base-16 number 66666 is written in base 2, how many base-2 digits (bits) does it have? \n Think carefully but respond with only the answer."
    
    # Process completions in the specified range
    idx_start = args.start
    idx_end = args.end
    num_completions = idx_end - idx_start
    
    print(f"Processing completions {idx_start} to {idx_end-1} (total: {num_completions})")
    print(f"Using prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
    # Process completions sequentially
    for i in tqdm(range(idx_start, idx_end)):
        print(f"\n--- Processing completion {i} (seed={i}) ---")
        activations = extractor.extract_activations(prompt, seed=i)
        print(f"Extracted activations for {len(activations)} layers")
        
        # Debug activation format for first completion in range
        if i == idx_start:
            print("\n=== DEBUGGING ACTIVATION FORMAT ===")
            for layer in [7, 13, 17]:
                if layer in activations:
                    extractor.debug_activation_format(layer, i)
            print("=== END DEBUGGING ===\n")

    print(f"\nCompleted {num_completions} extractions (indices {idx_start}-{idx_end-1}). Check activation_cache/ for stored activations.")


if __name__ == "__main__":
    main()
