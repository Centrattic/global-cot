from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple, Iterable, Dict, Optional, Any, Union
import json

import numpy as np
import zipfile
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ActivationManager:
    """Individual activation storage with per-activation lengths.

	For each (layer, component) we store:
	1. **`<hook>.json`** - Metadata file with hash-to-index mapping and individual lengths
	2. **`<hook>.npz`** - Compressed numpy file storing individual activations with their actual lengths

	This approach eliminates padding and allows each activation to be stored at its natural size.
	"""

    @classmethod
    def create_readonly(
        cls,
        model_name: str,
        d_model: int,
        cache_dir: Path,
        device: str = "cuda:0",
    ):
        """
		Create a read-only ActivationManager that can load existing caches without the full model.
		This is useful for evaluation when activations are already cached.
		"""
        instance = cls.__new__(cls)
        instance.model = None  # No model needed for reading
        instance.tokenizer = None  # No tokenizer needed for reading
        instance.model_name = model_name
        instance.device = device  # Device to use if we lazily load
        instance.d_model = d_model
        instance.cache_dir = cache_dir
        instance.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Cache for metadata to avoid reloading
        instance._metadata_cache = {}
        instance._metadata_mtimes = {}

        print(f"[INFO] Created read-only ActivationManager for {model_name} (no tokenizer needed)")
        return instance

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        d_model: int,
        cache_dir: Path,
    ):
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        assert self.tokenizer is not None, "Tokenizer must be provided with the model"
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        self.device = device
        # Best-effort record of model_name for potential debug
        try:
            self.model_name = getattr(
                getattr(
                    model,
                    'config',
                    None,
                ),
                'name_or_path',
                None,
            )
        except Exception:
            self.model_name = None
        self.d_model = d_model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # public entrypoint #
    # We dont want to extract activations on the fly, since its extremely slow for our probes.
    def get_activations_for_texts(
        self,
        texts: Iterable[str],
        layer: int,
        component: str,
        format_type: str,
        activation_type: str = "full",
        question_texts: Optional[Iterable[str]] = None,
        sysprompts: Optional[Iterable[str]] = None,
        return_newly_added_count: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        """
		Get activations for texts, with caching and aggregation support.
		
		Args:
			texts: Input texts (prompts)
			layer: Layer number
			component: Component name (e.g., 'resid_post')
			activation_type: Type of activation to return:
				- "full": Full activations (padded to max length)
				- "linear_mean", "linear_max", "linear_last": Linear aggregations
				- "sae_mean", "sae_max", "sae_last": SAE aggregations
				- "act_sim_mean", "act_sim_max", "act_sim_last", "act_sim_softmax": Activation similarity aggregations
			question_texts: Optional list of question texts (for on-policy experiments)
			sysprompts: Optional list of system prompts (for on-policy experiments)
			format_type: Format for activation extraction:
				- "qr": On-policy - format as prompt+question using chat template, extract question tokens
				- "r": Off-policy instruct - format prompt as user prompt using chat template, extract prompt tokens
				- "r-no-it": Off-policy non-instruct - use prompt text directly without chat template
			return_newly_added_count: If True, also return the number of newly computed activations
		
		Returns:
			Activations array (no masks returned), or tuple of (activations, newly_added_count) if return_newly_added_count=True
		"""
        texts = list(texts)
        metadata_path, activations_path = self._paths(
            layer,
            component,
        )

        # Check which texts need extraction using lightweight metadata (avoid loading arrays)
        existing_keys = set()
        metadata = self._load_metadata(
            layer,
            component,
        )
        if metadata is not None and isinstance(
                metadata,
                dict,
        ) and "hash_to_idx" in metadata:
            existing_keys = set(metadata["hash_to_idx"].keys())

        missing_texts: List[str] = []
        missing_question_texts: List[str] = []
        missing_sysprompts: List[str] = []
        for i, text in enumerate(texts):
            text_hash = self._hash(text)
            if text_hash not in existing_keys:
                missing_texts.append(text)
                # For on-policy, we need the corresponding question and system prompt for formatting
                if question_texts is not None and i < len(question_texts):
                    missing_question_texts.append(question_texts[i])
                if sysprompts is not None and i < len(sysprompts):
                    missing_sysprompts.append(sysprompts[i])

        newly_added_count = len(missing_texts)
        print(f"[DEBUG] Newly added count: {newly_added_count}")

        # Extract missing activations
        if missing_texts:
            loaded_here = False
            if self.model is None or self.tokenizer is None:
                # Lazily load the model to compute missing activations
                print("[DEBUG] No model/tokenizer in ActivationManager; lazily loading to compute missing activations…")
                self._lazy_load_model()
                loaded_here = True
            print(f"[DEBUG] Extracting activations for {len(missing_texts)} missing texts")
            print(f"[DEBUG] Memory before extraction: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            try:
                new_activations = self._extract_activations(
                    missing_texts,
                    layer,
                    component,
                    format_type,
                    missing_question_texts if question_texts is not None else None,
                    missing_sysprompts if sysprompts is not None else None,
                )
                print(f"[DEBUG] Memory after extraction: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"[DEBUG] Extracted {len(new_activations)} new activations")

                # Save new activations (load existing arrays only now)
                print(f"[DEBUG] Saving full activations...")
                existing_arrays: Dict[str, np.ndarray] = {}
                if activations_path.exists():
                    try:
                        existing_arrays = dict(np.load(
                            activations_path,
                            allow_pickle=True,
                        ))
                    except zipfile.BadZipFile:
                        print(f"[DEBUG] Corrupted activations file detected during save (BadZipFile): {activations_path}. Overwriting.")
                        existing_arrays = {}
                    except Exception as e:
                        print(f"[DEBUG] Failed to read existing activations from {activations_path}: {e}. Overwriting.")
                        existing_arrays = {}
                all_activations = {**existing_arrays, **new_activations}
                np.savez(
                    activations_path,
                    **all_activations,
                )
                print(f"[DEBUG] Saved full activations: {len(all_activations)} total entries")
                # Update metadata to reflect new entries
                self._update_metadata(
                    layer,
                    component,
                    new_activations,
                )
            except Exception as e:
                print(f"[DEBUG] Error during activation extraction: {e}")
                raise

            # Compute and save aggregated activations for new activations
            print(f"[DEBUG] Computing aggregated activations for {len(new_activations)} new activations")
            print(f"[DEBUG] Memory before aggregation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

            self._compute_and_save_aggregations(
                layer,
                component,
                new_activations,
            )

            # If we loaded the model in this call, unload to free memory
            if loaded_here:
                try:
                    print("[DEBUG] Unloading lazily loaded model/tokenizer…")
                    del self.model
                    del self.tokenizer
                    self.model = None
                    self.tokenizer = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as _e:
                    print(f"[DEBUG] Failed to unload lazily loaded model: {_e}")

        # We might need to compute aggregations for missing aggregated files anyway
        else:
            missing_aggregations = self._check_missing_aggregations(
                layer,
                component,
            )
            if missing_aggregations:
                print(f"[DEBUG] Computing missing aggregated files...")
                self._compute_and_save_aggregations(
                    layer,
                    component,
                )

        # Return appropriate activations based on type
        result = self._get_activations_for_return(
            texts,
            activation_type,
            layer,
            component,
        )
        print(f"[DEBUG] Returning {result.shape} activations")

        if return_newly_added_count:
            return result, newly_added_count
        else:
            return result

    def _lazy_load_model(self):
        """Load a causal LM and tokenizer using saved model_name for on-demand extraction."""
        if getattr(
                self,
                'model_name',
                None,
        ) is None:
            raise ValueError("Cannot lazily load model: model_name is unknown")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            model_name = self.model_name
            device_map = getattr(
                self,
                'device',
                'cuda:0',
            )
            if model_name == "meta-llama/Llama-3.3-70B-Instruct":
                # Use 8-bit quantization for Llama 70B
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_compute_dtype=torch.float16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    quantization_config=bnb_config,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Align tokenizer sides
            self.tokenizer.padding_side = "right"
            self.tokenizer.truncation_side = "right"
        except Exception as e:
            raise RuntimeError(f"Failed to lazily load model '{self.model_name}': {e}")

    def _extract_activations(
        self,
        texts: List[str],
        layer: int,
        component: str,
        format_type: str,
        question_texts: Optional[List[str]] = None,
        sysprompts: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
		Core activation extraction logic based on format type.
		
		Args:
			texts: Input texts (prompts)
			layer: Layer number
			component: Component name
			format_type: Format type ("qr", "r", "r-no-it")
			question_texts: Optional question texts for on-policy
			sysprompts: Optional system prompts for on-policy
			
		Returns:
			Dictionary mapping text hashes to activations
		"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("ActivationManager is read-only. A full model is required to compute activations.")

        hook_name = f"blocks.{layer}.hook_{component}"
        new_entries: Dict[str, np.ndarray] = {}

        self.model.eval()
        # Don't call .to() for quantized models as they're already on the correct device
        if not hasattr(self.model, 'is_loaded_in_8bit') and not hasattr(self.model, 'is_loaded_in_4bit') and not hasattr(self.model, 'quantization_config'):
            self.model.to(self.device)

        with torch.no_grad():
            for i in tqdm(
                    range(len(texts)),
                    desc=f"Extracting activations for {format_type}",
            ):
                text = texts[i]

                if format_type == "qr" and question_texts is not None:
                    # On-policy: format as prompt+question using chat template, extract response tokens
                    question = question_texts[i]
                    # Use actual system prompt from dataset if available, otherwise fallback
                    system_prompt = "You are a helpful assistant."
                    if sysprompts is not None and i < len(sysprompts):
                        system_prompt = sysprompts[i] if sysprompts[i].strip() else "You are a helpful assistant."
                    
                    formatted = self.tokenizer.apply_chat_template(
                        [{"role": "system", "content": system_prompt}, 
                         {"role": "user", "content": question}, 
                         {"role": "assistant", "content": text}
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                    # Tokenize and get activations
                    enc = self.tokenizer(
                        formatted,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                        add_special_tokens=True,
                    )
                    input_ids = enc["input_ids"].to(self.device)

                    # Get activations and extract only response tokens (not question tokens)
                    response_enc = self.tokenizer(
                     text,  # This is the response
                     return_tensors="pt",
                     padding=False,
                     truncation=False,
                     add_special_tokens=True,
                    )
                    response_ids = response_enc["input_ids"][0]

                    # Find response token positions in full sequence
                    full_ids = input_ids[0]
                    # Align devices to avoid cuda/cpu mismatch during comparison
                    if response_ids.device != full_ids.device:
                        response_ids = response_ids.to(full_ids.device)
                    response_start = None
                    for j in range(len(full_ids) - len(response_ids) + 1):
                        if torch.equal(
                                full_ids[j:j + len(response_ids)],
                                response_ids,
                        ):
                            response_start = j
                            break

                    if response_start is None:
                        # Fallback: use all tokens
                        activations = self._get_activations_for_input_ids(
                            input_ids,
                            hook_name,
                        )
                    else:
                        activations = self._get_activations_for_input_ids(
                            input_ids,
                            hook_name,
                        )
                        activations = activations[:, response_start:response_start + len(response_ids), :]

                elif format_type == "r":
                    # Off-policy instruct: format prompt as user prompt using chat template, extract prompt tokens
                    formatted = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                    enc = self.tokenizer(
                        formatted,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                        add_special_tokens=True,
                    )
                    input_ids = enc["input_ids"].to(self.device)
                    activations = self._get_activations_for_input_ids(
                        input_ids,
                        hook_name,
                    )

                elif format_type == "r-no-it":
                    # Off-policy non-instruct: use prompt text directly without chat template
                    enc = self.tokenizer(
                        text,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                        add_special_tokens=True,
                    )
                    input_ids = enc["input_ids"].to(self.device)
                    activations = self._get_activations_for_input_ids(
                        input_ids,
                        hook_name,
                    )

                else:
                    raise ValueError(f"Invalid format_type: {format_type}")

                # Store activation with its hash
                text_hash = self._hash(text)
                if torch.is_tensor(activations):
                    new_entries[text_hash] = activations.cpu().numpy()
                else:
                    new_entries[text_hash] = activations

        return new_entries

    def _get_activations_for_input_ids(
        self,
        input_ids: torch.Tensor,
        hook_name: str,
    ) -> torch.Tensor:
        """Helper to get activations for input_ids with hook."""
        # Parse hook_name to get layer and component
        # hook_name format: "blocks.{layer}.hook_{component}"
        parts = hook_name.split(".")
        if len(parts) != 3 or parts[0] != "blocks" or not parts[1].isdigit() or not parts[2].startswith("hook_"):
            raise ValueError(f"Invalid hook_name format: {hook_name}. Expected: blocks.{{layer}}.hook_{{component}}")

        layer = int(parts[1])
        component = parts[2].replace(
            "hook_",
            "",
        )

        # Use the existing _run_and_capture method which properly handles hook registration
        return self._run_and_capture(
            layer,
            component,
            input_ids,
            attention_mask=None,
        )

    def _get_activations_for_return(
        self,
        texts: List[str],
        activation_type: str,
        layer: int,
        component: str,
    ) -> np.ndarray:
        """
		Load and return appropriate activations based on activation_type.
		
		Args:
			texts: List of input texts
			activation_type: Type of activation to return
			layer: Layer number
			component: Component name
			
		Returns:
			Appropriate activations array
		"""
        if activation_type == "full":
            # Load full activations from saved file
            _, activations_path = self._paths(
                layer,
                component,
            )

            if not activations_path.exists():
                raise ValueError(
                    f"Full activations file {activations_path} does not exist. Run extract_activations_for_dataset first."
                )

            print(f"[DEBUG] Loading full activations from {activations_path}")
            existing = np.load(
                activations_path,
                allow_pickle=True,
                mmap_mode='r',
            )

            print(f"[DEBUG] Existing: {len(existing.files)}")

            # Load activations in the order of texts
            activations_list = []
            for text in texts:
                text_hash = self._hash(text)
                if text_hash not in getattr(
                        existing,
                        "files",
                    [],
                ):
                    raise ValueError(f"Text not found in full activations: {text[:50]}...")
                arr = np.asarray(existing[text_hash])
                # Unsqueeze 2D arrays (S, D) to (1, S, D)
                if len(arr.shape) == 2:
                    arr = arr[None, ...]
                activations_list.append(arr)

            # Estimate memory requirement before processing
            total_elements = sum(
                act.size for act in activations_list\
            )
            estimated_memory_gb = total_elements * 4 / (1024**3)  # 4 bytes per float32
            print(f"[DEBUG] Processing {len(activations_list)} full activations")
            print(f"[DEBUG] Estimated memory requirement: {estimated_memory_gb:.2f} GB")

            print(f"[WARNING] Large memory requirement ({estimated_memory_gb:.2f} GB).")

            # Pad to max length and stack - optimized for memory usage
            max_len = max(
                act.shape[1] for act in activations_list\
            )
            print(f"[DEBUG] Max sequence length: {max_len}")

            # Pre-allocate the final array to avoid multiple copies
            total_batch_size = sum(
                act.shape[0] for act in activations_list\
            )
            print(f"[DEBUG] Shape of first activation: {activations_list[0].shape}")
            final_shape = (total_batch_size, max_len, activations_list[0].shape[2])
            print(f"[DEBUG] Final array shape: {final_shape}")

            result = np.zeros(
                final_shape,
                dtype=activations_list[0].dtype,
            )

            # Fill the result array directly
            start_idx = 0
            for act in activations_list:
                batch_size = act.shape[0]
                end_idx = start_idx + batch_size

                if act.shape[1] < max_len:
                    # Pad with zeros
                    result[start_idx:end_idx, :act.shape[1], :] = act
                    # The rest is already zero from np.zeros
                else:
                    result[start_idx:end_idx, :, :] = act

                start_idx = end_idx

            return result

        else:
            # For aggregated types, load from saved aggregation files
            # Extract the aggregation method from the activation_type
            # Use the suffix after the last underscore so that
            #   - "linear_mean" -> "mean"
            #   - "sae_max" -> "max"
            #   - "act_sim_mean" -> "mean" (not "sim_mean")
            aggregation = activation_type.rsplit(
                "_",
                1,
            )[1]
            _, activations_path = self._paths(
                layer,
                component,
            )
            aggregated_path = activations_path.parent / f"{activations_path.stem}_aggregated_{aggregation}.npz"

            if not aggregated_path.exists():
                raise ValueError(
                    f"Aggregated activations file {aggregated_path} does not exist. Run extract_activations_for_dataset first."
                )

            print(f"[DEBUG] Loading {activation_type} aggregated activations from {aggregated_path}")
            try:
                aggregated_data = np.load(
                    aggregated_path,
                    mmap_mode='r',
                )
            except zipfile.BadZipFile:
                raise RuntimeError(f"Aggregated activations file is corrupted (BadZipFile): {aggregated_path}. Delete it to regenerate.")

            # Load in the order of texts
            aggregated_list = []
            for text in texts:
                text_hash = self._hash(text)
                if text_hash not in aggregated_data:
                    raise ValueError(f"Text not found in aggregated activations: {text[:50]}...")
                vec = np.asarray(aggregated_data[text_hash]).ravel()
                if vec.shape[0] != self.d_model:
                    raise ValueError(f"Aggregated vector has shape {vec.shape}, expected ({self.d_model},).")
                aggregated_list.append(vec)

            result = np.stack(
                aggregated_list,
                axis=0,
            )
            print(f"[DEBUG] Loaded {activation_type} aggregated activations: {result.shape}")
            return result

    def _compute_and_save_aggregations(
        self,
        layer: int,
        component: str,
        new_activations: Dict[str, np.ndarray] = None,
    ):
        """Compute and save aggregated activations. If new_activations is None, compute for all existing activations."""
        _, activations_path = self._paths(
            layer,
            component,
        )
        print(f"[DEBUG] Activations path: {activations_path}")

        # Load activations to process
        if new_activations is None:
            print(f"[DEBUG] No new activations provided, loading all existing activations...")
            if not activations_path.exists():
                print(f"[DEBUG] No full activations file found, skipping aggregation")
                return
            activations_to_process = dict(np.load(
                activations_path,
                allow_pickle=True,
                mmap_mode='r',
            ))
        else:
            print(f"[DEBUG] Starting aggregation computation for {len(new_activations)} new activations")
            activations_to_process = new_activations

        from scipy.special import softmax

        # Define aggregation methods
        aggregation_methods = ["mean", "max", "last", "softmax"]
        print(f"[DEBUG] Computing aggregations: {aggregation_methods}")

        for i, aggregation in enumerate(aggregation_methods):
            print(f"[DEBUG] Processing aggregation {i+1}/{len(aggregation_methods)}: {aggregation}")
            aggregated_path = activations_path.parent / f"{activations_path.stem}_aggregated_{aggregation}.npz"
            print(f"[DEBUG] Aggregated path: {aggregated_path}")

            # Load existing aggregated activations
            existing_aggregated: Dict[str, np.ndarray] = {}
            if aggregated_path.exists():
                print(f"[DEBUG] Loading existing {aggregation} aggregated activations...")
                try:
                    existing_aggregated = dict(np.load(
                        aggregated_path,
                        allow_pickle=True,
                    ))
                    print(
                        f"[DEBUG] Loaded existing {aggregation} aggregated activations: {len(existing_aggregated)} entries"
                    )
                except zipfile.BadZipFile:
                    print(f"[DEBUG] Corrupted aggregated file detected (BadZipFile): {aggregated_path}. Rebuilding from scratch.")
                    existing_aggregated = {}
            else:
                print(f"[DEBUG] No existing {aggregation} aggregated file found, creating new one")

            for j, (text_hash, activation) in enumerate(activations_to_process.items()):
                if j % 100 == 0:  # Progress update every 100 activations
                    print(f"[DEBUG] Processing activation {j+1}/{len(activations_to_process)} for {aggregation}")

                try:
                    # Debug: print the activation shape before aggregation when it looks suspicious
                    _act = np.asarray(activation)
                    if not (_act.ndim == 2 and _act.shape[1] == self.d_model):
                        print(
                            f"[DEBUG] Pre-aggregation shape anomaly (idx={j}) for {aggregation}: shape={_act.shape}, ndim={_act.ndim}, expected=(*, {self.d_model})"
                        )
                        # If we have a leading singleton dimension, squeeze it away
                        # Should fix this earlier, but whatever
                        if _act.ndim == 3 and _act.shape[0] == 1:
                            activation = np.squeeze(
                                _act,
                                axis=0,
                            )
                            print(f"[DEBUG] Squeezed leading singleton dim: new shape={activation.shape}")
                    if aggregation == "mean":
                        aggregated = np.mean(
                            activation,
                            axis=0,
                        )
                    elif aggregation == "max":
                        aggregated = np.max(
                            activation,
                            axis=0,
                        )
                    elif aggregation == "last":
                        aggregated = activation[-1]
                    elif aggregation == "softmax":
                        weights = softmax(
                            activation,
                            axis=0,
                        )
                        aggregated = np.sum(
                            weights * activation,
                            axis=0,
                        )

                    # Validate that aggregated is a 1D vector of length d_model before saving
                    aggregated = np.asarray(aggregated)
                    if not (aggregated.ndim == 1 and aggregated.shape[0] == self.d_model):
                        raise ValueError(f"Aggregated vector has shape {aggregated.shape}, expected ({self.d_model},).")

                    existing_aggregated[text_hash] = aggregated
                except Exception as e:
                    print(f"[DEBUG] Error processing activation {j} for {aggregation}: {e}")
                    print(f"[DEBUG] Activation shape: {activation.shape}")
                    raise ValueError(f"Error processing activation {j} for {aggregation}: {e}")

            # Save updated aggregated activations
            print(f"[DEBUG] Saving {aggregation} aggregated activations...")
            try:
                np.savez(
                    aggregated_path,
                    **existing_aggregated,
                )
                print(f"[DEBUG] Saved {aggregation} aggregated activations: {len(existing_aggregated)} total entries")
            except Exception as e:
                print(f"[DEBUG] Error saving {aggregation} aggregated activations: {e}")
                print(f"[DEBUG] File path: {aggregated_path}")
                print(f"[DEBUG] Number of entries: {len(existing_aggregated)}")
                raise

        print(f"[DEBUG] Completed all aggregation computations")

    def count_missing_texts(
        self,
        texts: Iterable[str],
        layer: int,
        component: str,
    ) -> int:
        """Count how many of the provided texts are missing in the full activations cache.

        This method only inspects metadata and does not attempt to compute activations.
        It works in both read-only and full modes.
        """
        texts = list(texts)
        metadata = self._load_metadata(
            layer,
            component,
        )
        existing_keys = set()
        if metadata is not None and isinstance(
                metadata,
                dict,
        ) and "hash_to_idx" in metadata:
            existing_keys = set(metadata["hash_to_idx"].keys())

        missing = 0
        for text in texts:
            if self._hash(text) not in existing_keys:
                missing += 1
        return missing

    def ensure_aggregations_exist(
        self,
        layer: int,
        component: str,
    ) -> None:
        """Ensure aggregated activation files exist; compute them if missing.

        This does not require a model; it derives aggregations from saved full activations.
        """
        try:
            if self._check_missing_aggregations(
                    layer,
                    component,
            ):
                print(f"[DEBUG] Aggregations missing for layer={layer}, component={component}; computing…")
                self._compute_and_save_aggregations(
                    layer,
                    component,
                )
        except Exception as e:
            print(f"[DEBUG] Failed to ensure aggregations exist for layer={layer}, component={component}: {e}")
            raise

    def _check_missing_aggregations(
        self,
        layer: int,
        component: str,
    ) -> bool:
        """Check if any aggregated files are missing."""
        _, activations_path = self._paths(
            layer,
            component,
        )

        # Check if full activations exist
        if not activations_path.exists():
            return False

        # Define aggregation methods
        aggregation_methods = ["mean", "max", "last", "softmax"]

        for aggregation in aggregation_methods:
            aggregated_path = activations_path.parent / f"{activations_path.stem}_aggregated_{aggregation}.npz"
            if not aggregated_path.exists():
                print(f"[DEBUG] Missing aggregated file for {aggregation}")
                return True

        return False

    def get_actual_max_len(
        self,
        layer: int,
        component: str,
    ) -> Optional[int]:
        """Get the actual maximum token length from activations if available."""
        metadata_path, activations_path = self._paths(
            layer,
            component,
        )

        if not activations_path.exists():
            return None

        # Load activations and find max length
        try:
            all_activations = dict(np.load(
                activations_path,
                allow_pickle=True,
                mmap_mode='r',
            ))
        except zipfile.BadZipFile:
            print(f"[DEBUG] Detected corrupted activations file (BadZipFile): {activations_path}. Ignoring for max_len.")
            return None
        except Exception as e:
            print(f"[DEBUG] Failed to read activations for max_len from {activations_path}: {e}")
            return None
        if not all_activations:
            return None

        max_len = max(
            activation.shape[0] for activation in all_activations.values()\
        )
        return max_len

    # internals #
    def _paths(
        self,
        layer: int,
        component: str,
    ) -> Tuple[Path, Path]:
        hook = f"blocks.{layer}.hook_{component}".replace(
            ".",
            "‑",
        )
        metadata = self.cache_dir / f"{hook}.json"
        activations = self.cache_dir / f"{hook}.npz"
        return metadata, activations

    def _load_metadata(
        self,
        layer: int,
        component: str,
    ) -> Optional[Dict[str, Any]]:
        """Load metadata JSON for a given layer/component (no caching)."""
        metadata_path, _ = self._paths(
            layer,
            component,
        )
        if not metadata_path.exists():
            return None
        try:
            with open(
                    metadata_path,
                    "r",
            ) as f:
                return json.load(f)
        except Exception as e:
            print(f"[DEBUG] Failed to load metadata from {metadata_path}: {e}")
            return None

    def _update_metadata(
        self,
        layer: int,
        component: str,
        new_entries: Dict[str, np.ndarray],
    ):
        """Update metadata JSON with new hashes and lengths."""
        metadata_path, _ = self._paths(
            layer,
            component,
        )
        data = {
            "hash_to_idx": {},
            "lengths": {},
        }
        if metadata_path.exists():
            try:
                with open(
                        metadata_path,
                        "r",
                ) as f:
                    data = json.load(f)
            except Exception:
                pass
        if "hash_to_idx" not in data:
            data["hash_to_idx"] = {}
        if "lengths" not in data:
            data["lengths"] = {}
        # Append new items; use next index after current max
        next_index = 0
        if data["hash_to_idx"]:
            try:
                next_index = 1 + max(
                    int(
                        i) for i in data["hash_to_idx"].values()\
                )
            except Exception:
                next_index = len(data["hash_to_idx"])  # fallback
        for h, arr in new_entries.items():
            if h not in data["hash_to_idx"]:
                data["hash_to_idx"][h] = next_index
                next_index += 1
            # arr shape can be (S, D) or (B, S, D). Store sequence length S.
            seq_len = int(arr.shape[-2]) if arr.ndim >= 2 else int(arr.shape[0])
            data["lengths"][h] = seq_len
        with open(
                metadata_path,
                "w",
        ) as f:
            json.dump(
                data,
                f,
            )

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha1(prompt.encode()).hexdigest()

    def clear_metadata_cache(self):
        """Clear activation cache to free memory."""
        import gc
        gc.collect()

    # Backwards-compatible alias used by Dataset
    def clear_activation_cache(self):
        self.clear_metadata_cache()

    # HF hooking utilities #
    def _get_block_module(
        self,
        layer_index: int,
    ) -> Any:
        """
		Try to resolve a transformer block module across popular architectures.
		Returns the module whose forward output is the per-token hidden states at that layer.
		"""
        m = self.model
        # For llama/gemma: model.model.layers
        return m.model.layers[layer_index]
        raise ValueError(f"Could not locate transformer block module for layer {layer_index}")

    def _run_and_capture(
        self,
        layer: int,
        component: str,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
		Run the model and capture per-token activations at the specified layer/component.
		Only 'resid_post' is currently supported and maps to the block output hidden states.
		Returns tensor of shape [B, S, D].
		"""
        if component != "resid_post":
            raise NotImplementedError(
                f"Component '{component}' is not supported without TransformerLens. Supported: 'resid_post'."
            )

        block_module = self._get_block_module(layer)
        captured: Dict[str, torch.Tensor] = {"x": None}

        def hook_fn(
            module,
            inputs,
            output,
        ):  # type: ignore[override]
            hidden = output[0] if isinstance(
                output,
                (tuple, list),
            ) else output
            captured["x"] = hidden

        handle = block_module.register_forward_hook(hook_fn)
        try:
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
            )
        finally:
            handle.remove()

        if captured["x"] is None:
            raise RuntimeError("Failed to capture activations; hook did not run.")

        print(f"[DEBUG] Captured activation shape: {captured['x'].shape}")
        return captured["x"]