import ollama
import torch
from typing import Dict
from omegaconf import DictConfig


@torch.no_grad()
def call_llm(
    prompt: str,
    backend: str,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    tokenizer=None,
    model=None,
) -> str:
    """
    calling llm
    Args:
        prompt:
        backend:
        model_name:
        temperature:
        max_new_tokens:
        tokenizer:
        model:

    Returns:

    """
    if backend == "hf":
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

    # ollama
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": max_new_tokens,
        },
    )
    return response["response"].strip()


_FEWSHOT_CACHE: Dict[str, str] = {}


def generate_few_shots(
    q: str,
    k: int,
    cfg: DictConfig,
) -> str:
    """
    Generate k few-shot QA examples using a separate LLM.
    Args:
        q:
        k:
        cfg:

    Returns:

    """

    if cfg.shots.cache and q in _FEWSHOT_CACHE:
        return _FEWSHOT_CACHE[q]

    system_prompt = f"""
                        Generate {k} high-quality factual question-answer examples.
                        Keep answers concise, neutral, and correct.
                        Format strictly as:
                        
                        Q: ...
                        A: ...
                        
                        Do NOT include explanations or extra text.
                    """

    full_prompt = f"{system_prompt}\nTopic example: {q}"

    text = call_llm(
        prompt=full_prompt,
        backend=cfg.shots.backend,
        model_name=cfg.shots.model,
        temperature=cfg.shots.temperature,
        max_new_tokens=cfg.shots.max_new_tokens,
    )

    few_shot_block = text.strip() + f"\n\nQ: {q}\nA:"

    if cfg.shots.cache:
        _FEWSHOT_CACHE[q] = few_shot_block

    return few_shot_block
