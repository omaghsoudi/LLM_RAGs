# coding=utf-8
from __future__ import annotations

import re
from typing import List, Dict

import torch
import ollama


# ------------------------------------------------------------------
# Claim Decomposition
# ------------------------------------------------------------------


@torch.no_grad()
def decompose_claims(
    answer: str,
    cfg,
    tokenizer=None,
    model=None,
) -> List[str]:
    """
    Decompose an answer into atomic factual claims.
    """

    if answer == "<EMPTY>":
        return []

    prompt = f"""
                Break the following answer into a list of atomic factual claims.
                
                Rules:
                - One fact per line
                - No explanations
                - No duplicates
                - Use declarative sentences
                
                Answer:
                {answer}
                
                Claims:
            """

    if cfg.claims.backend == "hf":
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    else:
        response = ollama.generate(
            model=cfg.claims.model,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "num_predict": 128,
            },
        )
        text = response["response"]

    claims = [c.strip("-• ").strip() for c in text.splitlines() if len(c.strip()) > 5]

    return list(dict.fromkeys(claims))  # deduplicate


# ------------------------------------------------------------------
# Claim Verification
# ------------------------------------------------------------------


@torch.no_grad()
def verify_claim(
    claim: str,
    references: List[str],
    cfg,
    tokenizer=None,
    model=None,
) -> float:
    """
    Verify a single claim against references.
    Returns support score in [0, 1].
    """

    ref_block = "\n".join(f"- {r}" for r in references)

    prompt = f"""
                Determine whether the following claim is supported by the references.
                
                Claim:
                {claim}
                
                References:
                {ref_block}
                
                Scoring:
                - 1.0 = fully supported
                - 0.5 = partially supported or implied
                - 0.0 = not supported or contradicted
                
                Respond with ONLY a number.
            """

    if cfg.claims.backend == "hf":
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    else:
        response = ollama.generate(
            model=cfg.claims.model,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "num_predict": 10,
            },
        )
        text = response["response"]

    match = re.search(r"(0(\.\d+)?|1(\.0+)?)", text)
    if not match:
        return 0.0

    score = float(match.group(1))
    return max(0.0, min(1.0, score))


# ------------------------------------------------------------------
# End-to-end Claim Evaluation
# ------------------------------------------------------------------


def claim_support_metrics(
    answer: str,
    references: List[str],
    cfg,
    tokenizer=None,
    model=None,
) -> Dict[str, float]:
    """
    Returns:
    - claim_support_rate
    - hallucination_rate
    - num_claims
    """

    claims = decompose_claims(answer, cfg, tokenizer, model)

    if not claims:
        return {
            "claim_support_rate": 0.0,
            "hallucination_rate": 1.0,
            "num_claims": 0,
        }

    scores = [verify_claim(c, references, cfg, tokenizer, model) for c in claims]

    support_rate = float(sum(scores) / len(scores))
    hallucination_rate = 1.0 - support_rate

    return {
        "claim_support_rate": round(support_rate, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "num_claims": len(claims),
    }
