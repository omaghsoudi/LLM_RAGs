import os
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from itertools import combinations
from sentence_transformers import SentenceTransformer

from sacrebleu import BLEU
from rouge import Rouge
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

bleu_scorer = BLEU(effective_order=True)
rouge_scorer = Rouge()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
METRIC_KEYS = ["bleu", "rouge", "os_sim", "bertscore"]


def agreement_score(outputs: List[str]) -> float:
    valid = [o for o in outputs if o != "<EMPTY>"]
    if len(valid) < 2:
        return 0.0
    embs = embedder.encode(valid, normalize_embeddings=True)
    sims = [
        float(np.dot(embs[i], embs[j]))
        for i, j in combinations(range(len(embs)), 2)
    ]
    return float(np.mean(sims))

def compute_metrics(output: str, refs: List[str]) -> Dict[str, float]:
    if output == "<EMPTY>":
        return {}

    bleu = bleu_scorer.sentence_score(output, refs).score / 100
    rouge = np.mean([
        rouge_scorer.get_scores(output, r)[0]["rouge-l"]["f"]
        for r in refs
    ])

    emb_out = embedder.encode([output], normalize_embeddings=True)
    emb_ref = embedder.encode(refs, normalize_embeddings=True)
    os_sim = cosine_similarity(emb_out, emb_ref).max()

    _, _, F1 = bert_score([output] * len(refs), refs, lang="en")
    bert_f1 = float(F1.mean())

    return {
        "bleu": round(bleu, 3),
        "rouge": round(rouge, 3),
        "os_sim": round(os_sim, 3),
        "bertscore": round(bert_f1, 3),
    }

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        k: round(float(np.mean([m.get(k, 0.0) for m in metrics_list])), 3)
        for k in METRIC_KEYS
    }

def plot_metrics_per_sample(metrics_list: List[Dict[str, float]], cfg, output_dir: str, additional_metrics_list: List[str] = []):
    if not cfg.plotting.enabled or len(metrics_list) < 2:
        return

    all_metrics_list = METRIC_KEYS + additional_metrics_list
    os.makedirs(output_dir, exist_ok=True)

    x = range(1, len(metrics_list) + 1)

    plt.figure(figsize=(8, 5))

    for key in all_metrics_list:
        values = [m.get(key, 0.0) for m in metrics_list]
        plt.plot(x, values, marker="o", label=key.upper())

    plt.ylim(0, 1)
    plt.xlabel("Sample")
    plt.ylabel("Score")
    plt.title(f"{cfg.models.name} — Metrics per Sample")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if cfg.plotting.save_figures:
        plt.savefig(
            os.path.join(output_dir, "metrics_per_sample.png"),
            dpi=150,
        )

    plt.show()

def plot_aggregated(metrics: Dict[str, float], cfg, output_dir: str, additional_metrics_list: List[str] = []):
    if not cfg.plotting.enabled:
        return

    all_metrics_list = METRIC_KEYS + additional_metrics_list
    values = [metrics[k] for k in all_metrics_list]

    plt.figure(figsize=(7, 4))
    plt.bar(all_metrics_list, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{cfg.models.name} — Aggregated Metrics")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if cfg.plotting.save_figures:
        plt.savefig(
            os.path.join(output_dir, "aggregated_metrics.png"),
            dpi=150,
        )

    plt.show()
