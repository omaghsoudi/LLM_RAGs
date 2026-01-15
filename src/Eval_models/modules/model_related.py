import torch
import ollama

from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------


@torch.no_grad()
def generate(prompt: str, cfg, tokenizer=None, model=None) -> str:
    """
    generate the prediction
    Args:
        prompt:
        cfg:
        tokenizer:
        model:

    Returns:

    """
    if cfg.models.backend == "hf":
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.models.max_new_tokens,
            temperature=cfg.models.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(prompt) :].strip() or "<EMPTY>"

    response = ollama.generate(
        model=cfg.models.name,
        prompt=prompt,
        options={
            "temperature": cfg.models.temperature,
            "num_predict": cfg.models.max_new_tokens,
        },
    )
    return response["response"].strip() or "<EMPTY>"


# ------------------------------------------------------------------
# Model Loader
# ------------------------------------------------------------------


def load_model(cfg):
    """
    load the model
    Args:
        cfg:

    Returns:

    """
    if cfg.models.backend == "hf":
        tokenizer = AutoTokenizer.from_pretrained(cfg.models.name)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.models.name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return tokenizer, model
    return None, None
