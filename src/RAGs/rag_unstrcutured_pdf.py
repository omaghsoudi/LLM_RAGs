#!/usr/bin/python3

from omegaconf import DictConfig, OmegaConf
import hydra

from common_modules.initialize import setup_logger

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


@hydra.main(
    config_path="configs",
    config_name="rag_unstructured",
    version_base="1.1.0"
)
def run(cfg: DictConfig):
    logger = setup_logger(path_to_logger=cfg.logging.file)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # --------------------------------------------------
    # Load unstructured documents
    # --------------------------------------------------
    documents = []
    for path in cfg.documents.paths:
        loader = UnstructuredFileLoader(path)
        documents.extend(loader.load())

    # --------------------------------------------------
    # Split documents
    # --------------------------------------------------
    text_splitter = CharacterTextSplitter(
        separator=cfg.text_splitter.separator,
        chunk_size=cfg.text_splitter.chunk_size,
        chunk_overlap=cfg.text_splitter.chunk_overlap
    )
    splits = text_splitter.split_documents(documents)

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------
    embeddings = HuggingFaceEmbeddings()

    # --------------------------------------------------
    # Vector store
    # --------------------------------------------------
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": cfg.retriever.k}
    )

    # --------------------------------------------------
    # LLM (Ollama)
    # --------------------------------------------------
    llm = Ollama(
        model=cfg.llm.model,
        temperature=cfg.llm.temperature
    )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                cfg.prompt.system
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}"
            )
        ]
    )

    # --------------------------------------------------
    # LCEL RAG Chain
    # --------------------------------------------------
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------
    # Run questions
    # --------------------------------------------------
    for question in cfg.questions:
        logger.info(f"Question: {question}")
        answer = rag_chain.invoke(question)
        logger.info("Answer:")
        logger.info(answer)


if __name__ == "__main__":
    run()
