# coding=utf-8
from __future__ import annotations

import os
import random

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
import torch
import nltk
from nltk.corpus import stopwords

from Common_modules.initialize import setup_logger

from Eval_models.modules.few_shots import generate_few_shots
from Eval_models.modules.evaluation import (
    aggregate_metrics,
    compute_metrics,
    agreement_score,
    plot_metrics_per_sample,
    plot_aggregated,
)
from Eval_models.modules.model_related import generate, load_model
from Eval_models.modules.llm_judge import judge_correctness
from Eval_models.modules.claim_verification import claim_support_metrics


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------------------
# Hydra Entrypoint
# ------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path="configs", config_name="llm_gen_eval")
def main(cfg: DictConfig):
    logger = setup_logger(cfg.logging.file)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    tokenizer, model = load_model(cfg)

    question = "Is eating watermelon seeds dangerous?"
    references = [
        "Watermelon seeds are safe to eat.",
        "They are not harmful and contain nutrients.",
    ]

    prompt = generate_few_shots(
        question,
        cfg.dataset.few_shot_k,
        cfg,
    )

    output_dir = cfg.output_dir

    if cfg.models.self_consistency_samples > 1:
        outputs = [
            generate(prompt, cfg, tokenizer, model)
            for _ in range(cfg.models.self_consistency_samples)
        ]

        metrics_list = [compute_metrics(o, references) for o in outputs]
        aggregated = aggregate_metrics(metrics_list)
        aggregated["agreement"] = round(agreement_score(outputs), 3)

        judge_scores = [
            judge_correctness(
                question,
                o,
                references,
                cfg,
                tokenizer,
                model,
            )
            for o in outputs
        ]
        aggregated["llm_correctness"] = round(float(np.mean(judge_scores)), 3)

        claim_metrics_list = [
            claim_support_metrics(
                o,
                references,
                cfg,
                tokenizer,
                model,
            )
            for o in outputs
        ]
        aggregated["claim_support_rate"] = round(
            float(np.mean([m["claim_support_rate"] for m in claim_metrics_list])), 3
        )
        aggregated["hallucination_rate"] = round(
            float(np.mean([m["hallucination_rate"] for m in claim_metrics_list])), 3
        )

        logger.info("Aggregated metrics: %s", aggregated)

        plot_metrics_per_sample(
            metrics_list,
            cfg,
            output_dir,
            additional_metrics_list=[
                "llm_correctness",
                "claim_support_rate",
                "hallucination_rate",
            ],
        )
        plot_aggregated(
            aggregated,
            cfg,
            output_dir,
            additional_metrics_list=[
                "llm_correctness",
                "claim_support_rate",
                "hallucination_rate",
            ],
        )

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"))
        aggregated_df = pd.DataFrame.from_dict(aggregated, orient="index")
        aggregated_df.to_csv(os.path.join(output_dir, "aggregated_metrics.csv"))

    else:
        output = generate(prompt, cfg, tokenizer, model)
        metrics = compute_metrics(output, references)

        judge_score = judge_correctness(
            question,
            output,
            references,
            cfg,
            tokenizer,
            model,
        )
        metrics["llm_correctness"] = round(judge_score, 3)

        claim_metrics = claim_support_metrics(
            output,
            references,
            cfg,
            tokenizer,
            model,
        )
        metrics.update(claim_metrics)

        logger.info("Metrics: %s", metrics)
        plot_aggregated(
            metrics,
            cfg,
            output_dir,
            additional_metrics_list=[
                "llm_correctness",
                "claim_support_rate",
                "hallucination_rate",
            ],
        )

        metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        metrics_df.to_csv(os.path.join(output_dir, "aggregated_metrics.csv"))


if __name__ == "__main__":
    main()
