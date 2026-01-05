# run_multimodal_rag_demo.py
import os
from Agentic_AI.modules.multimodal_model import MultimodalRAG

# --------------------------------------------------
# Paths
# --------------------------------------------------
ASSETS_DIR = "/datasets/multimodal_rag"
OUTPUT_DIR = "../RAGs/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("\n🚀 Initializing Multimodal RAG System...")
    rag = MultimodalRAG(chroma_dir  = "./chroma_db", chroma_collection_name = "multimodal_rag")

    # --------------------------------------------------
    # TEXT → TEXT
    # --------------------------------------------------
    print("\n🔹 TEXT → TEXT")
    result = rag.run(
        input_data="Are watermelon seeds dangerous?",
        input_modality="text",
        output_modality="text",
    )
    print("Result:", result)
    assert isinstance(result, str) and len(result) > 10

    # --------------------------------------------------
    # IMAGE → TEXT
    # --------------------------------------------------
    print("\n🔹 IMAGE → TEXT")
    image_path = os.path.join(ASSETS_DIR, "watermelon.jpg")
    result = rag.run(
        input_data=image_path,
        input_modality="image",
        output_modality="text",
    )
    print("Result:", result)
    assert isinstance(result, str)

    # --------------------------------------------------
    # AUDIO → TEXT
    # --------------------------------------------------
    print("\n🔹 AUDIO → TEXT")
    audio_path = os.path.join(ASSETS_DIR, "question.wav")
    result = rag.run(
        input_data=audio_path,
        input_modality="audio",
        output_modality="text",
    )
    print("Result:", result)
    assert isinstance(result, str)

    # --------------------------------------------------
    # TEXT → IMAGE
    # --------------------------------------------------
    print("\n🔹 TEXT → IMAGE")
    output_path = rag.run(
        input_data="Explain why watermelon seeds are safe to eat",
        input_modality="text",
        output_modality="image",
    )
    print("Image saved at:", output_path)
    assert os.path.exists(output_path)

    # --------------------------------------------------
    # TEXT → AUDIO
    # --------------------------------------------------
    print("\n🔹 TEXT → AUDIO")
    output_path = rag.run(
        input_data="Are watermelon seeds harmful?",
        input_modality="text",
        output_modality="audio",
    )
    print("Audio saved at:", output_path)
    assert os.path.exists(output_path)

    # --------------------------------------------------
    # IMAGE → AUDIO
    # --------------------------------------------------
    print("\n🔹 IMAGE → AUDIO")
    output_path = rag.run(
        input_data=image_path,
        input_modality="image",
        output_modality="audio",
    )
    print("Audio saved at:", output_path)
    assert os.path.exists(output_path)

    # --------------------------------------------------
    # AUDIO → IMAGE
    # --------------------------------------------------
    print("\n🔹 AUDIO → IMAGE")
    output_path = rag.run(
        input_data=audio_path,
        input_modality="audio",
        output_modality="image",
    )
    print("Image saved at:", output_path)
    assert os.path.exists(output_path)

    print("\n✅ All multimodal RAG demos completed successfully!")


if __name__ == "__main__":
    main()
