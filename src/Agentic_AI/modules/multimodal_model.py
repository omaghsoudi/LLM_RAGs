from __future__ import annotations

import os
import requests
from io import BytesIO

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

from Common_modules.initialize import setup_logger

from dataclasses import dataclass, field
from typing import List, Optional
import json
import time


# =========================================================
# AGENT STATE (ADDITIVE ONLY)
# =========================================================


@dataclass
class AgentState:
    raw_input: str
    parsed_input: str
    input_modality: str
    output_modality: str

    plan: List[str] = field(default_factory=list)
    retrieved_docs: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    confidence: Optional[float] = None

    hallucination_detected: bool = False
    used_fallback: bool = False
    retry_count: int = 0


# =========================================================
# FLUX.1 TEXT → IMAGE
# =========================================================


class FluxTextToImage:
    def __init__(
        self,
        model_id="black-forest-labs/FLUX.1-schnell",
        api_url="https://router.huggingface.co/hf-inference/models",
        hf_token=None,
    ):
        hf_token = hf_token or os.getenv("HF_TOKEN")
        self.api_url = f"{api_url}/{model_id}"
        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        }

        if not hf_token:
            raise ValueError("HF_TOKEN must be set")

    def generate(self, prompt, out_path="output.png", width=1024, height=1024, steps=4):
        payload = {
            "inputs": prompt,
            "parameters": {
                "width": width,
                "height": height,
                "num_inference_steps": steps,
            },
            "options": {"wait_for_model": True},
        }

        r = requests.post(self.api_url, headers=self.headers, json=payload, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(r.text)

        image = Image.open(BytesIO(r.content))
        image.save(out_path)
        return image


# =========================================================
# MULTIMODAL PROCESSOR
# =========================================================


class MultimodalProcessor:
    def __init__(self, image_captioning_model, speech_model, device):
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=speech_model,
            device=0 if device == "cuda" else -1,
        )

        self.blip_processor = BlipProcessor.from_pretrained(
            image_captioning_model, use_fast=True
        )

        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            image_captioning_model,
            torch_dtype=torch.float32,
            use_safetensors=True,
        ).to(device)

    def input_to_text(self, input_data, modality, device):
        if modality == "text":
            return input_data

        if modality == "image":
            image = Image.open(input_data).convert("RGB")
            inputs = self.blip_processor(image, return_tensors="pt").to(device)
            output_ids = self.blip_model.generate(**inputs)
            return self.blip_processor.decode(output_ids[0], skip_special_tokens=True)

        if modality == "audio":
            return self.asr(input_data)["text"]

        raise ValueError("Unsupported modality")

    def text_to_voice(self, text, out_path):
        gTTS(text).save(out_path)
        return out_path


# =========================================================
# VECTOR STORE (RESTORED)
# =========================================================


def load_vectorstore(chroma_dir, chroma_collection_name, embed_model, logger) -> Chroma:
    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(f"Chroma DB not found at: {chroma_dir}")

    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    logger.info("Loading existing Chroma vector store")

    return Chroma(
        collection_name=chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )


def build_vectorstore(
    chroma_dir, chroma_collection_name, embed_model, logger
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)

    vectordb = Chroma(
        collection_name=chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

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
# MULTIMODAL RAG (FILE SAVING RESTORED)
# =========================================================


class MultimodalRAG:
    def __init__(
        self,
        chroma_dir,
        chroma_collection_name,
        k=3,
        logger=None,
        device=None,
        hf_token=None,
        temperature=0.2,
        load_vector=False,
        local_llm_model="llama3",
        speech_model="openai/whisper-tiny",
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        model_id_text_to_image="black-forest-labs/FLUX.1-schnell",
        image_captioning_model="Salesforce/blip-image-captioning-base",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or setup_logger(__name__)

        self.processor = MultimodalProcessor(
            image_captioning_model, speech_model, self.device
        )

        # 🔒 EXACT ORIGINAL LOGIC
        if load_vector:
            self.vectorstore = load_vectorstore(
                chroma_dir, chroma_collection_name, embed_model, self.logger
            )
        else:
            self.vectorstore = build_vectorstore(
                chroma_dir, chroma_collection_name, embed_model, self.logger
            )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        self.llm = Ollama(model=local_llm_model, temperature=temperature)

        self.text_to_image = FluxTextToImage(
            model_id=model_id_text_to_image, hf_token=hf_token
        )

    # =====================================================
    # RUN (FILES SAVED)
    # =====================================================

    def run(self, input_data, input_modality, output_modality, file_path=None):
        query = self.processor.input_to_text(input_data, input_modality, self.device)

        state = AgentState(
            raw_input=input_data,
            parsed_input=query,
            input_modality=input_modality,
            output_modality=output_modality,
        )

        state.plan = self._plan_steps(query)

        docs, _ = self._retrieve_with_confidence(query)
        state.retrieved_docs = [d.page_content for d in docs]
        context = "\n".join(state.retrieved_docs)

        answer = self._retry_with_backoff(
            lambda: self.llm.invoke(self._build_prompt(context, query)),
            state,
        )

        answer, confidence = self._self_critique(answer, context)

        if self._detect_hallucination(answer, context):
            state.hallucination_detected = True
            answer = self._retry_with_backoff(
                lambda: self.llm.invoke(self._build_prompt(context, query)),
                state,
            )

        state.answer = answer
        state.confidence = confidence

        # =================================================
        # ✅ OUTPUT + FILE SAVING (RESTORED)
        # =================================================

        if output_modality == "text":
            if file_path:
                with open(file_path, "w") as f:
                    f.write(state.answer)
            return state

        if output_modality == "image":
            file_path = file_path or "generated_image.png"
            self.text_to_image.generate(state.answer, out_path=file_path)
            return state

        if output_modality == "audio":
            file_path = file_path or "generated_audio.mp3"
            self.processor.text_to_voice(state.answer, file_path)
            return state

        raise ValueError("Unsupported output modality")

    # =====================================================
    # HELPERS
    # =====================================================

    def _retry_with_backoff(self, fn, state, max_retries=3):
        for i in range(max_retries):
            try:
                return fn()
            except Exception as e:
                state.retry_count += 1
                time.sleep(2**i)
                self.logger.warning(f"Retry {i + 1}: {e}")
        raise RuntimeError("Failed after retries")

    def _build_prompt(self, context, query):
        return f"""
        Use ONLY the context below.

        Context:
        {context}

        Question:
        {query}
        """

    def _plan_steps(self, query):
        try:
            return json.loads(
                self.llm.invoke(f"Return reasoning steps as JSON list:\n{query}")
            )
        except Exception:
            return ["retrieve context", "answer question"]

    def _retrieve_with_confidence(self, query):
        docs = self.retriever.invoke(query)
        return docs, min(1.0, len(docs) / 2)

    def _self_critique(self, answer, context):
        try:
            result = json.loads(
                self.llm.invoke(
                    f"""
                    Context:
                    {context}
                    Answer:
                    {answer}
                    Return JSON {{ "improved_answer": "...", "confidence": 0-1 }}
                    """
                )
            )
            return result["improved_answer"], float(result["confidence"])
        except Exception:
            return answer, 0.6

    def _detect_hallucination(self, answer, context):
        try:
            result = json.loads(
                self.llm.invoke(
                    f"""
                    Context:
                    {context}
                    Answer:
                    {answer}
                    Return JSON {{ "hallucination": true/false }}
                    """
                )
            )
            return result.get("hallucination", False)
        except Exception:
            return False
