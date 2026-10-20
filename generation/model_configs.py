#!/usr/bin/env python3
"""
model_configs.py

Model registry for the six open-source models evaluated in the paper.
Each model config defines: HF model ID, chat/prompt template,
tokenizer settings, and 4-bit NF4 quantization settings.
"""

import torch


_PRIME = "1. INTERPRETATION\ninitialUnderstandingTarget: "


def build_prompt_llama2(system: str, user_msg: str) -> str:
    """LLaMA 2 inference prompt — primed with template start."""
    return (
        f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
        f"{user_msg} [/INST] {_PRIME}"
    )


def build_prompt_llama3(system: str, user_msg: str) -> str:
    """LLaMA 3 inference prompt — primed with template start."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{_PRIME}"
    )


def build_prompt_apply_template(system: str, user_msg: str,
                                tokenizer=None) -> str:
    """Inference prompt using apply_chat_template — primed with template start."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    if tokenizer is not None:
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return result + _PRIME
    # Fallback: generic ChatML
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{_PRIME}"
    )


def build_prompt_qwen3(system: str, user_msg: str, tokenizer=None) -> str:
    """Qwen3 inference prompt — thinking disabled, primed with template start."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    if tokenizer is not None:
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        return result + _PRIME
    return build_prompt_apply_template(system, user_msg)


# ── Model Registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "llama2": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "display_name": "LLaMA 2 7B Chat",
        "prompt_builder": build_prompt_llama2,
        "needs_tokenizer_for_template": False,
        # Tokenizer
        "tokenizer_kwargs": {
            "use_fast": False,
        },
        "padding_side": "right",
        "trust_remote_code": False,
        # Quantization
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        },
    },

    "llama3": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "display_name": "LLaMA 3.1 8B Instruct",
        "prompt_builder": build_prompt_llama3,
        "needs_tokenizer_for_template": False,
        "tokenizer_kwargs": {
            "use_fast": True,
        },
        "padding_side": "right",
        "trust_remote_code": False,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        },
    },

    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "display_name": "Qwen 2.5 7B Instruct",
        "prompt_builder": build_prompt_apply_template,
        "needs_tokenizer_for_template": True,
        "tokenizer_kwargs": {
            "use_fast": True,
        },
        "padding_side": "right",
        "trust_remote_code": True,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        },
    },

    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "display_name": "Mistral 7B Instruct v0.3",
        "prompt_builder": build_prompt_apply_template,
        "needs_tokenizer_for_template": True,
        "tokenizer_kwargs": {
            "use_fast": True,
        },
        "padding_side": "right",
        "trust_remote_code": False,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        },
    },

    "olmo": {
        "model_id": "allenai/OLMo-2-1124-7B-Instruct",
        "display_name": "OLMo 2 7B Instruct",
        "prompt_builder": build_prompt_apply_template,
        "needs_tokenizer_for_template": True,
        "tokenizer_kwargs": {
            "use_fast": True,
        },
        "padding_side": "right",
        "trust_remote_code": True,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        },
    },

    "qwen3": {
        "model_id": "Qwen/Qwen3-8B",
        "display_name": "Qwen3 8B",
        "prompt_builder": build_prompt_qwen3,
        "needs_tokenizer_for_template": True,
        "tokenizer_kwargs": {"use_fast": True},
        "padding_side": "right",
        "trust_remote_code": True,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        },
    },
}


def get_model_config(model_name: str) -> dict:
    """Get model config by short name. Raises KeyError if not found."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    return MODEL_REGISTRY[model_name]


def list_models() -> list:
    """Return list of available model short names."""
    return list(MODEL_REGISTRY.keys())
