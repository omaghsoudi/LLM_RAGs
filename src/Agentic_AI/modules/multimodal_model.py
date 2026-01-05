from __future__ import annotations

import os
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

logger = setup_logger(__name__)

# =========================================================
# CONFIG
# =========================================================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM_MODEL = "llama3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# MULTIMODAL PROCESSORS
# =========================================================

class MultimodalProcessor:
    def __init__(self):
        # 🎧 ASR — Whisper via HF Transformers
        self.asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=0 if DEVICE == "cuda" else -1,
        )

        # 🖼️ Image captioning (BLIP — safetensors)
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_fast=True
        )

        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float32,
            use_safetensors=True,
        ).to(DEVICE)

    # -------- INPUT --------
    def input_to_text(
        self,
        input_data: str,
        modality: Literal["text", "image", "audio"]
    ) -> str:

        if modality == "text":
            return input_data

        if modality == "image":
            image = Image.open(input_data).convert("RGB")
            inputs = self.blip_processor(image, return_tensors="pt").to(DEVICE)
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

def load_vectorstore(chroma_dir, chroma_collection_name) -> Chroma:
    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(
            f"Chroma DB not found at: {chroma_dir}"
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    logger.info("Loading existing Chroma vector store")

    return Chroma(
        collection_name=chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

# =========================================================
# BUILD VECTOR STORE (CHROMA)
# =========================================================

def build_vectorstore(chroma_dir, chroma_collection_name) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

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
    def __init__(self, chroma_dir, chroma_collection_name):
        self.processor = MultimodalProcessor()
        # self.vectorstore = load_vectorstore(chroma_dir, chroma_collection_name)
        self.vectorstore = build_vectorstore(chroma_dir, chroma_collection_name)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        # 🦙 LOCAL LLM (NO OPENAI)
        self.llm = Ollama(
            model=LOCAL_LLM_MODEL,
            temperature=0.2,
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
            input_modality
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

        if output_modality == "audio":
            return self.processor.text_to_voice(answer)

        raise ValueError(f"Unsupported output modality: {output_modality}")
