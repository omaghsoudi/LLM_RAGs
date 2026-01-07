from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# from Agentic_AI.modules.multimodal_model_langgraph import MultimodalRAG
from Agentic_AI.modules.multimodal_model import MultimodalRAG

from Common_modules.initialize import setup_logger


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="run_multimodal_rag",
)
def main(cfg: DictConfig):

    # --------------------------------------------------
    # Setup
    # --------------------------------------------------
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(cfg.logging.file)

    logger.info("🚀 Starting Multimodal RAG Demo")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Working directory: {os.getcwd()}")

    # --------------------------------------------------
    # Initialize RAG
    # --------------------------------------------------
    rag = MultimodalRAG(
        chroma_dir=cfg.rag.chroma.chroma_dir,
        chroma_collection_name=cfg.rag.chroma.collection_name,
        k=cfg.rag.chroma.k,
        load_vector=cfg.rag.chroma.load_vector,
        local_llm_model=cfg.rag.llm.local_model,
        temperature=cfg.rag.llm.temperature,
        embed_model=cfg.rag.embeddings.model,
        speech_model=cfg.rag.speech.model,
        image_captioning_model=cfg.rag.vision.image_captioning_model,
        model_id_text_to_image=cfg.rag.vision.text_to_image_model,
        hf_token=cfg.rag.auth.hf_token,
        logger=logger,
    )

    # --------------------------------------------------
    # Execute Runs
    # --------------------------------------------------
    logger.info("*"*100)
    for run_cfg in cfg.runs:
        logger.info(f"\n🔹 RUN: {run_cfg.name}")
        logger.info(f"Comment: {run_cfg.comment}")

        # Input resolution
        input_modality = run_cfg.input.modality
        input_data = (
            run_cfg.input.data
            if "data" in run_cfg.input
            else run_cfg.input.file
        )

        # Output resolution
        output_modality = run_cfg.output.modality
        output_file = run_cfg.output.file

        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

        result = rag.run(
            input_data=input_data,
            input_modality=input_modality,
            output_modality=output_modality,
            file_path=output_file,
        )

        logger.info(f"Output saved to: {output_file}")
        try:
            # This is for LongGraphResult object
            logger.info(f"Input answer: {result['parsed_input']}")
            logger.info(f"Result answer: {result['answer']}")
            logger.info(f"Detailed Processes: {result['plan']}, {result['retrieved_docs']}")
            logger.info(f"Result preview: confidence score={result['confidence']}; haulucination={result['hallucination_detected']}; retry count={result['retry_count']}")
            logger.info(f"Result preview: used fallback={result['used_fallback']}")
        except Exception:
            # This is for older versions
            result = result.__dict__
            logger.info(f"Input answer: {result['parsed_input']}")
            logger.info(f"Result answer: {result['answer']}")
            logger.info(f"Detailed Processes: {result['plan']}, {result['retrieved_docs']}")
            logger.info(f"Result preview: confidence score={result['confidence']}; haulucination={result['hallucination_detected']}; retry count={result['retry_count']}")
        logger.info("-"*100)

    logger.info("\n✅ All multimodal RAG runs completed successfully!")


if __name__ == "__main__":
    main()
