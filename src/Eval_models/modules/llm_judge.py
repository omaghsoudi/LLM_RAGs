# modules/llm_judge.py
from __future__ import annotations

import re
from typing import List

import torch
import ollama

# ------------------------------------------------------------------
# LLM Judge
# ------------------------------------------------------------------


@torch.no_grad()
def judge_correctness(
    question: str,
    answer: str,
    references: List[str],
    cfg,
    tokenizer=None,
    model=None,
) -> float:
    """
    Uses an LLM to judge whether an answer is correct.
    Returns a score in [0, 1].

    Args:
        question:
        answer:
        references:
        cfg:
        tokenizer:
        model:

    Returns:

    """

    if answer == "<EMPTY>":
        return 0.0

    ref_block = "\n".join(f"- {r}" for r in references)

    prompt = f"""
                You are a strict factual evaluator.
                
                Question:
                {question}
                
                Reference answers:
                {ref_block}
                
                Model answer:
                {answer}
                
                Task:
                Assign a correctness score between 0 and 1.
                
                Scoring rules:
                - 1.0 = fully correct and consistent with references
                - 0.5 = partially correct or incomplete
                - 0.0 = incorrect or contradicts references
                
                Respond with ONLY a number between 0 and 1.
            """

    # ---------------- HF backend ----------------
    if cfg.judge.backend == "hf":
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ---------------- Ollama backend ----------------
    else:
        response = ollama.generate(
            model=cfg.judge.model,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "num_predict": 10,
            },
        )
        text = response["response"]

    # ---------------- Parse numeric score ----------------
    match = re.search(r"(0(\.\d+)?|1(\.0+)?)", text)
    if not match:
        return 0.0

    score = float(match.group(1))
    return max(0.0, min(1.0, score))
