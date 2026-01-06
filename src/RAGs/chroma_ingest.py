#!/usr/bin/python3

import os
import shutil
import hydra
from omegaconf import DictConfig, OmegaConf

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from Common_modules.initialize import setup_logger

from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


@hydra.main(
    config_path="configs",
    config_name="chroma_ingest",
    version_base="1.1.0"
)
def main(cfg: DictConfig):
    logger = setup_logger(path_to_logger=cfg.logging.file)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # --------------------------------------------------
    # Reset database (optional)
    # --------------------------------------------------
    if cfg.database.reset:
        logger.info("✨ Clearing database")
        clear_database(cfg.database.persist_path)

    # --------------------------------------------------
    # Load, split, and ingest documents
    # --------------------------------------------------
    documents = load_documents(cfg.data.path)
    chunks = split_documents(documents, cfg.text_splitter, logger)
    add_to_chroma(chunks, cfg.database.persist_path, logger)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_documents(data_path: str):
    loader = PyPDFDirectoryLoader(data_path)
    return loader.load()


def split_documents(
    documents: list[Document],
    splitter_cfg: DictConfig,
    logger
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=splitter_cfg.chunk_size,
        chunk_overlap=splitter_cfg.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def add_to_chroma(
    chunks: list[Document],
    persist_path: str,
    logger
):
    db = Chroma(
        persist_directory=persist_path,
        embedding_function=get_embedding_function(),
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [
        chunk for chunk in chunks_with_ids
        if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        logger.info(f"👉 Adding new documents: {len(new_chunks)}")
        new_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
        db.persist()
    else:
        logger.info("✅ No new documents to add")


def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def clear_database(persist_path: str):
    if os.path.exists(persist_path):
        shutil.rmtree(persist_path)


if __name__ == "__main__":
    main()
