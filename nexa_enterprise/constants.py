import os
from pathlib import Path
from enum import Enum

# Paths for caching, model hub, and tokens
NEXA_CACHE_ROOT = Path(os.getenv("NEXA_CACHE_ROOT") or "~/.cache/nexa").expanduser()
NEXA_TOKEN_PATH = NEXA_CACHE_ROOT / "token"
NEXA_MODELS_HUB_DIR = NEXA_CACHE_ROOT / "hub"
NEXA_MODEL_EVAL_RESULTS_PATH = NEXA_CACHE_ROOT / "eval"
NEXA_MODELS_HUB_OFFICIAL_DIR = NEXA_MODELS_HUB_DIR / "official"
NEXA_MODELS_HUB_HF_DIR = NEXA_MODELS_HUB_DIR / "huggingface"
NEXA_MODELS_HUB_MS_DIR = NEXA_MODELS_HUB_DIR / "modelscope"
NEXA_MODEL_LIST_PATH = NEXA_MODELS_HUB_DIR / "model_list.json"

# URLs and buckets
NEXA_API_URL = "https://model-hub-backend.nexa4ai.com"
NEXA_OFFICIAL_BUCKET = "https://public-storage.nexa4ai.com/"

# Nexa logo
NEXA_LOGO = """
      _|    _|  _|_|_|  _|    _|    _|_|      _|_|    _|_|_|_|
      _|_|  _|  _|       _|  _|   _|    _|  _|    _|     _|
      _|_|_|_|  _|_|_|     _|     _|_|_|_|  _|_|_|_|     _|
      _|  _|_|  _|        _| _|   _|    _|  _|    _|     _|
      _|    _|  _|_|_|  _|    _|  _|    _|  _|    _|  _|_|_|_|
"""

# Model producer info
PRODUCER_INFO = {
    "producer_version": "0.0.0",
    "doc_string": "Model exported by Nexa.ai",
}


class ModelType(Enum):
    NLP = "NLP"
    COMPUTER_VISION = "Computer Vision"
    AUDIO = "Audio"
    TTS = "TTS"
    MULTIMODAL = "Multimodal"
    TEXT_EMBEDDING = "Text Embedding"
    AUDIOLM = "AudioLM"


NEXA_RUN_MODEL_MAP_TEXT = {
    "octopus-v2": "Octopus-v2:q4_0",
    "octopus-v4": "Octopus-v4:q4_0",
    "gpt2": "gpt2:q4_0",
    "tinyllama": "TinyLlama-1.1B-Chat-v1.0:fp16",
    "llama2": "Llama-2-7b-chat:q4_0",
    "llama3": "Meta-Llama-3-8B-Instruct:q4_0",
    "llama3.1": "Meta-Llama-3.1-8B-Instruct:q4_0",
    "llama3.2": "Llama3.2-3B-Instruct:q4_0",
    "gemma": "gemma-1.1-2b-instruct:q4_0",
    "gemma2": "gemma-2-2b-instruct:q4_0",
    "qwen1.5": "Qwen1.5-7B-Instruct:q4_0",
    "qwen2": "Qwen2-1.5B-Instruct:q4_0",
    "qwen2.5": "Qwen2.5-1.5B-Instruct:q4_0",
    "mistral": "Mistral-7B-Instruct-v0.3:q4_0",
    "codegemma": "codegemma-2b:q4_0",
    "codellama": "CodeLlama-7b-Instruct:q4_0",
    "codeqwen": "Qwen2.5-Coder-3B-Instruct:q4_0",
    "mathqwen": "Qwen2.5-Math-1.5B-Instruct:q4_0",
    "deepseek-coder": "deepseek-coder-1.3b-instruct:q4_0",
    "dolphin-mistral": "dolphin-2.8-mistral-7b:q4_0",
    "phi2": "Phi-2:q4_0",
    "phi3": "Phi-3-mini-128k-instruct:q4_0",
    "phi3.5": "Phi-3.5-mini-instruct:q4_0",
    "llama2-uncensored": "Llama2-7b-chat-uncensored:q4_0",
    "llama3-uncensored": "Llama3-8B-Lexi-Uncensored:q4_K_M",
    "openelm": "OpenELM-3B:q4_K_M",
}


NEXA_RUN_MODEL_MAP_FUNCTION_CALLING = {
    "llama2-function-calling": "Llama2-7b-function-calling:q4_K_M",
    "Llama2-7b-function-calling:fp16": "Llama2-7b-function-calling:fp16",
    "Llama2-7b-function-calling:q2_K": "Llama2-7b-function-calling:q2_K",
    "Llama2-7b-function-calling:q3_K_L": "Llama2-7b-function-calling:q3_K_L",
    "Llama2-7b-function-calling:q3_K_M": "Llama2-7b-function-calling:q3_K_M",
    "Llama2-7b-function-calling:q3_K_S": "Llama2-7b-function-calling:q3_K_S",
    "Llama2-7b-function-calling:q4_K_M": "Llama2-7b-function-calling:q4_K_M",
    "Llama2-7b-function-calling:q4_K_S": "Llama2-7b-function-calling:q4_K_S",
    "Llama2-7b-function-calling:q5_K_M": "Llama2-7b-function-calling:q5_K_M",
    "Llama2-7b-function-calling:q5_K_S": "Llama2-7b-function-calling:q5_K_S",
    "Llama2-7b-function-calling:q6_K": "Llama2-7b-function-calling:q6_K",
    "Llama2-7b-function-calling:q8_0": "Llama2-7b-function-calling:q8_0",
}


NEXA_RUN_MODEL_MAP_TEXT_EMBEDDING = {
    "mxbai": "mxbai-embed-large-v1:fp16",
    "mxbai-embed-large-v1": "mxbai-embed-large-v1:fp16",
    "mxbai-embed-large-v1:fp16": "mxbai-embed-large-v1:fp16",
    "nomic": "nomic-embed-text-v1.5:fp16",
    "nomic-embed-text-v1.5": "nomic-embed-text-v1.5:fp16",
    "nomic-embed-text-v1.5:fp16": "nomic-embed-text-v1.5:fp16",
    "all-MiniLM": "all-MiniLM-L6-v2:fp16",
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2:fp16",
    "all-MiniLM-L6-v2:fp16": "all-MiniLM-L6-v2:fp16",
    "all-MiniLM-L12-v2": "all-MiniLM-L12-v2:fp16",
    "all-MiniLM-L12-v2:fp16": "all-MiniLM-L12-v2:fp16",
}

NEXA_RUN_MODEL_MAP = {
    **NEXA_RUN_MODEL_MAP_TEXT,
    **NEXA_RUN_MODEL_MAP_FUNCTION_CALLING,
    **NEXA_RUN_MODEL_MAP_TEXT_EMBEDDING,
}

NEXA_RUN_CHAT_TEMPLATE_MAP = {
    "llama2": "llama-2",
    "llama-2-7b-chat": "llama-2",
    "llama3": "llama-3",
    "meta-llama-3-8b-instruct": "llama-3",
    "llama3.1": "llama-3",
    "meta-llama-3.1-8b-instruct": "llama-3",
    "llama3.2": "llama-3",
    "llama3.2-1b-instruct": "llama-3",
    "llama3.2-3b-instruct": "llama-3",
    "gemma": "gemma",
    "gemma-1.1-2b-instruct": "gemma",
    "gemma-1.1-7b-instruct": "gemma",
    "gemma-2b-instruct": "gemma",
    "gemma-7b-instruct": "gemma",
    "gemma-2-2b-instruct": "gemma",
    "gemma-2-9b-instruct": "gemma",
    "qwen1.5": "qwen",
    "qwen1.5-7b-instruct": "qwen",
    "codeqwen1.5-7b-instruct": "qwen",
    "qwen2": "qwen",
    "qwen2.5": "qwen",
    "qwen2-0.5b-instruct": "qwen",
    "qwen2-1.5b-instruct": "qwen",
    "qwen2-7b-instruct": "qwen",
    "qwen2.5-0.5b-instruct": "qwen",
    "qwen2.5-1.5b-instruct": "qwen",
    "qwen2.5-3b-instruct": "qwen",
    "qwen2.5-7b-instruct": "qwen",
    "qwen2.5-coder-0.5b-instruct": "qwen",
    "qwen2.5-coder-1.5b-instruct": "qwen",
    "qwen2.5-coder-3b-instruct": "qwen",
    "qwen2.5-coder-7b-instruct": "qwen",
    "qwen2.5-math-1.5b-instruct": "qwen",
    "qwen2.5-math-7b-instruct": "qwen",
    "mistral": "mistral-instruct",
    "mistral-7b-instruct-v0.3": "mistral-instruct",
    "mistral-7b-instruct-v0.2": "mistral-instruct",
}

NEXA_RUN_COMPLETION_TEMPLATE_MAP = {
    "octopus-v2": "Below is the query from the users, please call the correct function and generate the parameters to call the function.\n\nQuery: {input} \n\nResponse:",
    "octopus-v4": "<|system|>You are a router. Below is the query from the users, please call the correct function and generate the parameters to call the function.<|end|><|user|>{input}<|end|><|assistant|>",
}

EXIT_COMMANDS = ["/exit", "/quit", "/bye"]
EXIT_REMINDER = f"Please use Ctrl + d or one of {EXIT_COMMANDS} to exit.\n"

NEXA_STOP_WORDS_MAP = {"octopus-v2": ["<nexa_end>"], "octopus-v4": ["<nexa_end>"]}

DEFAULT_TEXT_GEN_PARAMS = {
    "temperature": 0.7,
    "max_new_tokens": 2048,
    "nctx": 2048,
    "top_k": 50,
    "top_p": 1.0,
}

DEFAULT_IMG_GEN_PARAMS = {
    "num_inference_steps": 20,
    "height": 512,
    "width": 512,
    "guidance_scale": 7.5,
    "output_path": "generated_images/image.png",
    "random_seed": 0,
}

DEFAULT_IMG_GEN_PARAMS_LCM = {
    "num_inference_steps": 4,
    "height": 512,
    "width": 512,
    "guidance_scale": 1.0,
    "output_path": "generated_images/image.png",
    "random_seed": 0,
}

DEFAULT_IMG_GEN_PARAMS_TURBO = {
    "num_inference_steps": 5,
    "height": 512,
    "width": 512,
    "guidance_scale": 5.0,
    "output_path": "generated_images/image.png",
    "random_seed": 0,
}

DEFAULT_VOICE_GEN_PARAMS = {
    "output_dir": "transcriptions",
    "beam_size": 5,
    "language": None,
    "task": "transcribe",
    "temperature": 0.0,
    "compute_type": "default",
}

# key is the repo name on Nexa model hub, NOT model abbreviation
# For example : https://nexa.ai/NexaAI/Octo-omni-vision/gguf-fp16/readme
# We need to register key : Octo-omni-vision
NEXA_OFFICIAL_MODELS_TYPE = {
    "gemma-2b": ModelType.NLP,
    "Llama-2-7b-chat": ModelType.NLP,
    "Llama-2-7b": ModelType.NLP,
    "Meta-Llama-3-8B-Instruct": ModelType.NLP,
    "Meta-Llama-3.1-8B-Instruct": ModelType.NLP,
    "Llama3.2-3B-Instruct": ModelType.NLP,
    "Llama3.2-1B-Instruct": ModelType.NLP,
    "Mistral-7B-Instruct-v0.3": ModelType.NLP,
    "Mistral-7B-Instruct-v0.2": ModelType.NLP,
    "Phi-3-mini-128k-instruct": ModelType.NLP,
    "Phi-3-mini-4k-instruct": ModelType.NLP,
    "Phi-3.5-mini-instruct": ModelType.NLP,
    "CodeQwen1.5-7B-Instruct": ModelType.NLP,
    "Qwen2-0.5B-Instruct": ModelType.NLP,
    "Qwen2-1.5B-Instruct": ModelType.NLP,
    "Qwen2-7B-Instruct": ModelType.NLP,
    "codegemma-2b": ModelType.NLP,
    "gemma-1.1-2b-instruct": ModelType.NLP,
    "gemma-2b-instruct": ModelType.NLP,
    "gemma-2-9b-instruct": ModelType.NLP,
    "gemma-1.1-7b-instruct": ModelType.NLP,
    "gemma-7b-instruct": ModelType.NLP,
    "gemma-7b": ModelType.NLP,
    "Qwen2-1.5B": ModelType.NLP,
    "Qwen2.5-0.5B-Instruct": ModelType.NLP,
    "Qwen2.5-1.5B-Instruct": ModelType.NLP,
    "Qwen2.5-3B-Instruct": ModelType.NLP,
    "Qwen2.5-Coder-0.5B-Instruct": ModelType.NLP,
    "Qwen2.5-Coder-1.5B-Instruct": ModelType.NLP,
    "Qwen2.5-Coder-3B-Instruct": ModelType.NLP,
    "Qwen2.5-Coder-7B-Instruct": ModelType.NLP,
    "Qwen2.5-Math-1.5B-Instruct": ModelType.NLP,
    "Qwen2.5-Math-7B-Instruct": ModelType.NLP,
    "codegemma-7b": ModelType.NLP,
    "TinyLlama-1.1B-Chat-v1.0": ModelType.NLP,
    "CodeLlama-7b-Instruct": ModelType.NLP,
    "gpt2": ModelType.NLP,
    "CodeLlama-7b": ModelType.NLP,
    "CodeLlama-7b-Python": ModelType.NLP,
    "Qwen1.5-7B-Instruct": ModelType.NLP,
    "Qwen1.5-7B": ModelType.NLP,
    "Phi-2": ModelType.NLP,
    "deepseek-coder-1.3b-instruct": ModelType.NLP,
    "deepseek-coder-1.3b-base": ModelType.NLP,
    "deepseek-coder-6.7b-instruct": ModelType.NLP,
    "dolphin-2.8-mistral-7b": ModelType.NLP,
    "gemma-2-2b-instruct": ModelType.NLP,
    "Octopus-v2": ModelType.NLP,
    "Octopus-v4": ModelType.NLP,
    "Octo-planner": ModelType.NLP,
    "deepseek-coder-6.7b-base": ModelType.NLP,
    "Llama2-7b-chat-uncensored": ModelType.NLP,
    "Llama3-8B-Lexi-Uncensored": ModelType.NLP,
    "Llama2-7b-function-calling": ModelType.NLP,
    "OpenELM-1_1B": ModelType.NLP,
    "OpenELM-3B": ModelType.NLP,
    "AMD-Llama-135m": ModelType.NLP,
    "lcm-dreamshaper-v7": ModelType.COMPUTER_VISION,
    "stable-diffusion-v1-5": ModelType.COMPUTER_VISION,
    "stable-diffusion-v1-4": ModelType.COMPUTER_VISION,
    "stable-diffusion-v2-1": ModelType.COMPUTER_VISION,
    "stable-diffusion-3-medium": ModelType.COMPUTER_VISION,
    "sdxl-turbo": ModelType.COMPUTER_VISION,
    "hassaku-hentai-model-v13-LCM": ModelType.COMPUTER_VISION,
    "anything-v30-LCM": ModelType.COMPUTER_VISION,
    "FLUX.1-schnell": ModelType.COMPUTER_VISION,
    "Phi-3-vision-128k-instruct": ModelType.MULTIMODAL,
    "omnivision-preview": ModelType.MULTIMODAL,
    "omnivision": ModelType.MULTIMODAL,
    "omnivision-ocr": ModelType.MULTIMODAL,
    "nanoLLaVA": ModelType.MULTIMODAL,
    "llava-v1.6-mistral-7b": ModelType.MULTIMODAL,
    "llava-v1.6-vicuna-7b": ModelType.MULTIMODAL,
    "llava-phi-3-mini": ModelType.MULTIMODAL,
    "llava-llama-3-8b-v1.1": ModelType.MULTIMODAL,
    "omniaudio": ModelType.AUDIOLM,
    "Qwen2-Audio-7.8B-Instruct": ModelType.AUDIOLM,
    "faster-whisper-tiny.en": ModelType.AUDIO,
    "faster-whisper-tiny": ModelType.AUDIO,
    "faster-whisper-small.en": ModelType.AUDIO,
    "faster-whisper-small": ModelType.AUDIO,
    "faster-whisper-medium.en": ModelType.AUDIO,
    "faster-whisper-medium": ModelType.AUDIO,
    "faster-whisper-base.en": ModelType.AUDIO,
    "faster-whisper-base": ModelType.AUDIO,
    "faster-whisper-large-v3": ModelType.AUDIO,
    "faster-whisper-large-v3-turbo": ModelType.AUDIO,
    "whisper-tiny.en": ModelType.AUDIO,
    "whisper-tiny": ModelType.AUDIO,
    "whisper-small.en": ModelType.AUDIO,
    "whisper-small": ModelType.AUDIO,
    "whisper-base.en": ModelType.AUDIO,
    "whisper-base": ModelType.AUDIO,
    "bark": ModelType.TTS,
    "bark-small": ModelType.TTS,
    "mxbai-embed-large-v1": ModelType.TEXT_EMBEDDING,
    "nomic-embed-text-v1.5": ModelType.TEXT_EMBEDDING,
    "all-MiniLM-L6-v2": ModelType.TEXT_EMBEDDING,
    "all-MiniLM-L12-v2": ModelType.TEXT_EMBEDDING,
}