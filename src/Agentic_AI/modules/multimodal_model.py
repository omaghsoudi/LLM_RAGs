from __future__ import annotations

import os
import requests
from io import BytesIO
from typing import Literal

import torch
from PIL import Image
from gtts import gTTS

from transformers import pipeline

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
)

from common_modules.initialize import setup_logger




# =========================================================
# FLUX.1 TEXT → IMAGE (HF INFERENCE API)
# =========================================================

class FluxTextToImage:
    """
    FLUX.1 via Hugging Face Router API (text-to-image)
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        hf_token: str | None = None,
    ):
        self.model_id = model_id
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        self.headers = {
            "Authorization": f"Bearer {hf_token or os.getenv('HF_TOKEN')}",
            "Content-Type": "application/json",
        }

        if not self.headers["Authorization"] or self.headers["Authorization"] == "Bearer None":
            raise ValueError("HF_TOKEN must be set for FLUX image generation")

    def generate(
        self,
        prompt: str,
        out_path: str = "output.png",
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
    ) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "width": width,
                "height": height,
                "num_inference_steps": steps,
            },
            "options": {
                "wait_for_model": True
            }
        }

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=300,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"FLUX image generation failed "
                f"({response.status_code}): {response.text}"
            )

        image = Image.open(BytesIO(response.content))
        image.save(out_path)
        return out_path




# =========================================================
# MULTIMODAL PROCESSORS
# =========================================================

class MultimodalProcessor:
    def __init__(
            self,
            image_captioning_model,
            speech_model,
            device,
        ):
        # 🎧 ASR — Whisper via HF Transformers
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=speech_model,
            device=0 if device == "cuda" else -1,
        )

        # 🖼️ Image captioning (BLIP — safetensors)
        self.blip_processor = BlipProcessor.from_pretrained(
            image_captioning_model,
            use_fast=True
        )

        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            image_captioning_model,
            torch_dtype=torch.float32,
            use_safetensors=True,
        ).to(device)

    # -------- INPUT --------
    def input_to_text(
        self,
        input_data: str,
        modality: Literal["text", "image", "audio"],
        device
    ) -> str:

        if modality == "text":
            return input_data

        if modality == "image":
            image = Image.open(input_data).convert("RGB")
            inputs = self.blip_processor(image, return_tensors="pt").to(device)
            output_ids = self.blip_model.generate(**inputs)
            return self.blip_processor.decode(
                output_ids[0],
                skip_special_tokens=True
            )

        if modality == "audio":
            result = self.asr(input_data)
            return result["text"]

        raise ValueError(f"Unsupported modality: {modality}")

    # -------- OUTPUT --------
    def text_to_voice(self, text: str, out_path: str = "output.mp3") -> str:
        tts = gTTS(text)
        tts.save(out_path)
        return out_path


# =========================================================
# LOAD PREBUILT CHROMA VECTOR STORE
# =========================================================

def load_vectorstore(chroma_dir, chroma_collection_name, embed_model, logger) -> Chroma:
    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(
            f"Chroma DB not found at: {chroma_dir}"
        )

    embeddings = HuggingFaceEmbeddings(model_name=embed_model)

    logger.info("Loading existing Chroma vector store")

    return Chroma(
        collection_name=chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

# =========================================================
# BUILD VECTOR STORE (CHROMA)
# =========================================================

def build_vectorstore(chroma_dir, chroma_collection_name, embed_model, logger) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)

    # Same pattern as rag_chroma.py
    vectordb = Chroma(
        collection_name=chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

    # Only seed if empty (important for persistence)
    if vectordb._collection.count() == 0:
        logger.info("Seeding Chroma vector store")

        docs = [
            Document(
                page_content="Watermelon seeds are safe to eat and contain nutrients.",
                metadata={"type": "text"},
            ),
            Document(
                page_content="Seeds of watermelon are edible and commonly roasted as snacks.",
                metadata={"type": "text"},
            ),
        ]

        vectordb.add_documents(docs)
        vectordb.persist()

    return vectordb


# =========================================================
# MULTIMODAL RAG PIPELINE
# =========================================================

class MultimodalRAG:
    def __init__(
            self,
            chroma_dir,
            chroma_collection_name,
            logger=None,
            device=None,
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            k=3,
            temperature=0.2,
            local_llm_model="llama3",
            image_captioning_model="Salesforce/blip-image-captioning-base",
            speech_model="openai/whisper-tiny",
            model_id_text_to_image = "black-forest-labs/FLUX.1-schnell",
            hf_token = None,
    ):
        self.model_id_text_to_image = model_id_text_to_image
        self.hf_token = hf_token
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if logger is None:
            logger = setup_logger(__name__)

        self.processor = MultimodalProcessor(
            image_captioning_model,
            speech_model,
            self.device
        )
        # self.vectorstore = load_vectorstore(chroma_dir, chroma_collection_name, embed_model, logger)
        self.vectorstore = build_vectorstore(chroma_dir, chroma_collection_name, embed_model, logger)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        # 🦙 LOCAL LLM (NO OPENAI)
        self.llm = Ollama(
            model=local_llm_model,
            temperature=temperature,
        )

    def run(
        self,
        input_data: str,
        input_modality: Literal["text", "image", "audio"],
        output_modality: Literal["text", "image", "audio"],
    ):
        # 1️⃣ Normalize input → text
        query_text = self.processor.input_to_text(
            input_data,
            input_modality,
            self.device
        )

        # 2️⃣ Retrieve context (RAG)
        docs = self.retriever.invoke(query_text)
        context = "\n".join(d.page_content for d in docs)

        # 3️⃣ LLM reasoning (LOCAL)
        prompt = f"""
                    You are a helpful assistant.
                    Answer the question using ONLY the context.
                    
                    Context:
                    {context}
                    
                    Question:
                    {query_text}
                """

        answer = self.llm.invoke(prompt)

        # 4️⃣ Output modality
        if output_modality == "text":
            return answer
        
        if output_modality == "image":
            # place holde r for text-to-image generation
            flux = FluxTextToImage(
                model_id = self.model_id_text_to_image,
                hf_token = self.hf_token,
            )
            out_path = "output.png"
            flux.generate(
                prompt=answer,
                out_path=out_path,
            )
            return out_path
            
        if output_modality == "audio":
            return self.processor.text_to_voice(answer)

        raise ValueError(f"Unsupported output modality: {output_modality}")
