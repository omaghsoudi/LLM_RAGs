# run_multimodal_rag_demo.py
import os
from pathlib import Path
from Agentic_AI.modules.multimodal_model import MultimodalRAG
from common_modules.initialize import setup_logger


# --------------------------------------------------
# Paths
# --------------------------------------------------
ASSETS_DIR = "/home/omid/github/LLM_RAGs/datasets/multimodal_rag"
OUTPUT_DIR = "./outputs/multimodal_rag"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"{OUTPUT_DIR}/multimodal_rag_demo.log")

    # log working directory
    logger.info(f"Working directory: {os.getcwd()}")

    logger.info("\n🚀 Initializing Multimodal RAG System...")
    rag = MultimodalRAG(chroma_dir  = "./chroma_db", chroma_collection_name = "multimodal_rag", logger=logger)

    # --------------------------------------------------
    # TEXT → TEXT
    # --------------------------------------------------
    logger.info("\n🔹 TEXT → TEXT")
    result = rag.run(
        input_data="Are watermelon seeds dangerous?",
        input_modality="text",
        output_modality="text",
    )
    logger.info(f"Result: {result}")
    assert isinstance(result, str) and len(result) > 10

    # --------------------------------------------------
    # IMAGE → TEXT
    # --------------------------------------------------
    logger.info("\n🔹 IMAGE → TEXT")
    image_path = os.path.join(ASSETS_DIR, "watermelon.jpg")
    result = rag.run(
        input_data=image_path,
        input_modality="image",
        output_modality="text",
    )
    logger.info(f"Result: {result}")
    assert isinstance(result, str)

    # --------------------------------------------------
    # AUDIO → TEXT
    # --------------------------------------------------
    logger.info("\n🔹 AUDIO → TEXT")
    audio_path = os.path.join(ASSETS_DIR, "question.wav")
    result = rag.run(
        input_data=audio_path,
        input_modality="audio",
        output_modality="text",
    )
    logger.info(f"Result: {result}")
    assert isinstance(result, str)

    # --------------------------------------------------
    # TEXT → IMAGE
    # --------------------------------------------------
    logger.info("\n🔹 TEXT → IMAGE")
    output_path = rag.run(
        input_data="Explain why watermelon seeds are safe to eat",
        input_modality="text",
        output_modality="image",
    )
    logger.info(f"Image saved at: {output_path}")
    assert os.path.exists(output_path)

    # --------------------------------------------------
    # TEXT → AUDIO
    # --------------------------------------------------
    logger.info("\n🔹 TEXT → AUDIO")
    output_path = rag.run(
        input_data="Are watermelon seeds harmful?",
        input_modality="text",
        output_modality="audio",
    )
    logger.info(f"Audio saved at: {output_path}")
    assert os.path.exists(output_path)

    # --------------------------------------------------
    # IMAGE → AUDIO
    # --------------------------------------------------
    logger.info("\n🔹 IMAGE → AUDIO")
    output_path = rag.run(
        input_data=image_path,
        input_modality="image",
        output_modality="audio",
    )
    logger.info(f"Audio saved at: {output_path}")
    assert os.path.exists(output_path)

    # --------------------------------------------------
    # AUDIO → IMAGE
    # --------------------------------------------------
    logger.info("\n🔹 AUDIO → IMAGE")
    output_path = rag.run(
        input_data=audio_path,
        input_modality="audio",
        output_modality="image",
    )
    logger.info(f"Image saved at: {output_path}")
    assert os.path.exists(output_path)

    logger.info("\n✅ All multimodal RAG demos completed successfully!")

if __name__ == "__main__":
    main()
