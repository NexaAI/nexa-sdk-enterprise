from nexa_enterprise.gguf.llama.llama_cache import LlamaDiskCache
from typing import Any, Dict

def run_inference_with_disk_cache(
    model: Any, 
    cache_prompt: str,
    total_prompt: str,
    use_cache: bool = True, 
    cache_read_dir: str = "llama.cache",
    cache_save_dir: str | None = "llama.cache",
    **kwargs: Dict[str, Any]
) -> str:
    """
    Runs inference using a disk cache to store and retrieve model states.

    Parameters:
    - model: The model object that supports caching and inference.
    - cache_prompt: The prompt used to generate a cache key.
    - total_prompt: The full prompt for generating output.
    - use_cache: Flag to determine if caching should be used.
    - cache_read_dir: Directory where cache files are read from.
    - cache_save_dir: Directory where cache files are saved to.
    - kwargs: Additional parameters for model inference.

    Returns:
    - The generated text output as a string.
    """
    temperature = kwargs.get('temperature', 0.7)
    max_tokens = kwargs.get('max_tokens', 2048)
    top_p = kwargs.get('top_p', 0.8)
    top_k = kwargs.get('top_k', 50)
    repeat_penalty = kwargs.get('repeat_penalty', 1.0)

    if use_cache:
        # Initialize disk cache with specified read directory
        cache_read_context = LlamaDiskCache(cache_dir=cache_read_dir)
        model.set_cache(cache_read_context)
        prompt_tokens = model.tokenize(cache_prompt.encode("utf-8"))
        try:
            cached_state = cache_read_context[prompt_tokens]
            model.load_state(cached_state)
        except KeyError:
            print("Cache miss: Key not found. Proceeding without cache.")
            model.reset()
            model.set_cache(None)
        
        # Initialize disk cache with specified save directory only if provided
        if cache_save_dir:
            cache_save_context = LlamaDiskCache(cache_dir=cache_save_dir)
            model.set_cache(cache_save_context)
    else:
        model.reset()
        model.set_cache(None)

    output = model(
        total_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        stream=True,
    )

    # Process streaming output
    result = ""
    for item in output:
        chunk = item['choices'][0]['text']
        print(chunk, end='', flush=True)
        result += chunk
    return result