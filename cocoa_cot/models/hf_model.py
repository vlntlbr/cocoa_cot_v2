"""
HuggingFace white-box model wrapper for CoCoA-CoT.

Provides:
- Greedy decoding with token log-probabilities and entropies
- Stochastic sampling (temperature / top-k / top-p)
- Hidden state extraction for CoCoA-CoT Light
- Batched generation

Token log-probabilities are in *natural log* throughout.  Token entropies
are H(p_t) = -Σ_v p_v log p_v computed from the full vocabulary distribution.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cocoa_cot.models.base import BaseModel, GenerationOutput
from cocoa_cot.parsing.chain_parser import ChainParser

logger = logging.getLogger(__name__)


CACHE_FORMAT_VERSION = "hfmodel_decode_v2"


class HFModel(BaseModel):
    """White-box HuggingFace model wrapper.

    Args:
        model_name: HuggingFace model identifier.
        device: ``"cuda"``, ``"cpu"``, or ``"auto"``.
        dtype: ``"bfloat16"``, ``"float16"``, or ``"float32"``.
        parser: :class:`~cocoa_cot.parsing.ChainParser` instance for extracting
            chain and answer from raw generations.
        max_new_tokens: Default maximum generation length.
        cache_dir: Directory for caching generation outputs.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        parser: Optional[ChainParser] = None,
        max_new_tokens: int = 512,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.parser = parser or ChainParser()
        self.max_new_tokens = max_new_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._model = None
        self._tokenizer = None
        self._hidden_state_hook: Optional[torch.Tensor] = None
        self._hook_handle = None

    # ── Lazy loading ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load model and tokenizer (called lazily on first use)."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model: %s (dtype=%s, device=%s)", self.model_name, self.dtype, self.device)

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device_map = "auto" if self.device == "auto" else None
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device_map is None:
            self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("Model loaded successfully.")

    # ── Public generation API ─────────────────────────────────────────────────

    def generate_greedy(self, prompt: str) -> GenerationOutput:
        """Greedy-decode a single output with full token statistics.

        Args:
            prompt: Input prompt string.

        Returns:
            :class:`GenerationOutput` with token log-probs and entropies.
        """
        cache_key = self._cache_key(prompt, "greedy")
        if cached := self._load_cache(cache_key):
            return cached

        self._load()
        output = self._generate_single(
            prompt,
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
        )
        self._save_cache(cache_key, output)
        return output

    def generate_sample(
        self,
        prompt: str,
        M: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> list[GenerationOutput]:
        """Generate M stochastic samples.

        Args:
            prompt: Input prompt string.
            M: Number of samples.
            temperature: Sampling temperature.
            top_k: Top-k parameter.
            top_p: Nucleus sampling parameter.

        Returns:
            List of M :class:`GenerationOutput` objects.
        """
        cfg_str = f"sample_M{M}_t{temperature}_k{top_k}_p{top_p}"
        cache_key = self._cache_key(prompt, cfg_str)
        if cached := self._load_cache(cache_key):
            return cached

        self._load()
        outputs = []
        for i in range(M):
            try:
                out = self._generate_single(
                    prompt,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                outputs.append(out)
            except Exception as e:
                logger.warning("Sample %d failed: %s. Using greedy fallback.", i+1, str(e))
                outputs.append(self.generate_greedy(prompt))

        self._save_cache(cache_key, outputs)
        return outputs

    def get_hidden_states(
        self, prompt: str, layer_idx: int
    ) -> np.ndarray:
        """Extract hidden states at a specific transformer layer.

        Args:
            prompt: Input prompt (the greedy-decoded text is used as context).
            layer_idx: 0-indexed transformer layer.

        Returns:
            Array of shape ``(seq_len, d_model)`` on CPU.
        """
        self._load()
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        hidden_states_capture: list[torch.Tensor] = []

        def _hook(module: torch.nn.Module, inp: tuple, out: tuple) -> None:
            # out is typically (hidden_state, ...) or just hidden_state
            hs = out[0] if isinstance(out, tuple) else out
            hidden_states_capture.append(hs.detach().cpu().float())

        # Register hook on the target layer
        layers = self._get_layers()
        if layer_idx >= len(layers):
            layer_idx = len(layers) // 2
            logger.warning("layer_idx out of range; using layer %d", layer_idx)

        handle = layers[layer_idx].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                self._model(**inputs, output_hidden_states=False)
        finally:
            handle.remove()

        if hidden_states_capture:
            hs = hidden_states_capture[0].squeeze(0).numpy()  # (seq_len, d)
            return hs
        return np.zeros((1, self._model.config.hidden_size))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_single(
        self,
        prompt: str,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GenerationOutput:
        """Core generation method used by both greedy and sample variants."""
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            return_offsets_mapping=False,
        )
        attention_mask = inputs["attention_mask"].to(self._model.device)
        input_ids = inputs["input_ids"].to(self._model.device)
        input_length = input_ids.shape[1]

        gen_kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self._tokenizer.pad_token_id,
            repetition_penalty=1.1,
            max_time=60.0,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p
        else:
            # Some model configs carry sampling defaults; clear them for greedy decode.
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None

        with torch.no_grad():
            try:
                gen_out = self._model.generate(**gen_kwargs)
            except Exception as e:
                logger.warning("Generation failed: %s. Returning greedy fallback.", str(e))
                # Fallback to greedy if sampling fails
                gen_kwargs_greedy = {k: v for k, v in gen_kwargs.items() if k not in ["temperature", "top_k", "top_p", "max_time"]}
                gen_kwargs_greedy["do_sample"] = False
                gen_out = self._model.generate(**gen_kwargs_greedy)

        # Extract generated token IDs (excluding input)
        generated_ids = gen_out.sequences[0][input_length:].tolist()

        # scores: tuple of (vocab_size,) tensors, one per generated token
        # Compute log-probs and entropies
        token_logprobs: list[float] = []
        token_entropies: list[float] = []

        for t, score in enumerate(gen_out.scores):
            # score: (1, vocab_size) in logit space
            logits = score[0]  # (vocab_size,)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            # Log prob of chosen token
            chosen_id = generated_ids[t]
            token_logprobs.append(log_probs[chosen_id].item())

            # Token entropy H(p_t) = -Σ_v p_v log p_v
            entropy = -(probs * log_probs).sum().item()
            token_entropies.append(entropy)

        # Decode full text.
        full_text = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # GPT-2/LLaMA BPE tokenizers encode spaces as Ġ (U+0120) and newlines
        # as Ċ (U+010A). decode() does not always strip these, so we do it here.
        full_text = full_text.replace('Ġ', ' ').replace('ĊĊ', '\n\n').replace('Ċ', '\n')

        # Parse chain and answer
        chain_text, answer_text = self.parser.parse(full_text, format="auto")

        # Map answer tokens using offset alignment
        answer_token_logprobs, answer_token_entropies = self._extract_answer_token_stats(
            prompt=prompt,
            full_text=full_text,
            answer_text=answer_text,
            generated_ids=generated_ids,
            token_logprobs=token_logprobs,
            token_entropies=token_entropies,
        )

        return GenerationOutput(
            text=full_text,
            token_ids=generated_ids,
            token_logprobs=token_logprobs,
            token_entropies=token_entropies,
            chain_text=chain_text,
            answer_text=answer_text,
            answer_token_logprobs=answer_token_logprobs,
            answer_token_entropies=answer_token_entropies,
        )

    def _extract_answer_token_stats(
        self,
        prompt: str,
        full_text: str,
        answer_text: str,
        generated_ids: list[int],
        token_logprobs: list[float],
        token_entropies: list[float],
    ) -> tuple[list[float], list[float]]:
        """Identify answer token positions and return their statistics.

        Uses the tokenizer's offset_mapping to robustly map character offsets
        of the answer span to token positions, avoiding subword mismatches.
        """
        if not answer_text:
            return [], []

        # Find character offsets of answer in the full generated text
        answer_start = full_text.rfind(answer_text)
        if answer_start == -1:
            return token_logprobs, token_entropies  # fallback: all tokens

        answer_end = answer_start + len(answer_text)

        # Tokenize the generated text with offset mapping
        enc = self._tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"]  # list of (start, end) per token

        # Find which tokens fall within the answer span
        answer_indices = [
            i
            for i, (s, e) in enumerate(offsets)
            if s >= answer_start and e <= answer_end and s < e
        ]

        if not answer_indices:
            # Fallback: use last third of tokens as rough answer proxy
            n = len(token_logprobs)
            answer_indices = list(range(max(0, 2 * n // 3), n))

        # Clip to generated token range (offset indices may include prompt tokens)
        # The generated tokens start at index 0 in token_logprobs/entropies
        # but the offset_mapping covers the full text starting at char 0
        # We need the offsets of the generated part only
        prompt_enc = self._tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True
        )
        prompt_token_len = len(prompt_enc["input_ids"])

        # Adjust: answer_indices are into full_text tokens; shift by -prompt_token_len
        # (generated tokens start at prompt_token_len in the offset mapping)
        gen_answer_indices = [
            i - prompt_token_len
            for i in answer_indices
            if i >= prompt_token_len and (i - prompt_token_len) < len(token_logprobs)
        ]

        if not gen_answer_indices:
            n = len(token_logprobs)
            gen_answer_indices = list(range(max(0, 2 * n // 3), n))

        return (
            [token_logprobs[i] for i in gen_answer_indices],
            [token_entropies[i] for i in gen_answer_indices],
        )

    def _get_layers(self) -> torch.nn.ModuleList:
        """Return the list of transformer layers."""
        model = self._model
        # Try common attribute paths
        for attr in ("model.layers", "transformer.h", "gpt_neox.layers", "layers"):
            obj = model
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                return obj
        raise AttributeError(f"Could not find transformer layers in {type(model).__name__}")

    # ── Caching ───────────────────────────────────────────────────────────────

    def _cache_key(self, prompt: str, tag: str) -> str:
        """Generate a deterministic cache key."""
        content = f"{CACHE_FORMAT_VERSION}|{self.model_name}|{tag}|{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[object]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                logger.warning("Cache load failed for %s: %s", path, exc)
        return None

    def _save_cache(self, key: str, obj: object) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        except Exception as exc:
            logger.warning("Cache save failed for %s: %s", path, exc)
