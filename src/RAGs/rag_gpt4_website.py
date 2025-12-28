#!/usr/bin/python3

from omegaconf import DictConfig, OmegaConf
import hydra
import os
import pprint
from dotenv import load_dotenv

from RAGs.modules.initialize import setup_logger

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


@hydra.main(
    config_path="configs",
    config_name="rag_website",
    version_base="1.1.0"
)
def run(cfg: DictConfig):
    logger = setup_logger(path_to_logger=cfg.logging.file)

    # --------------------------------------------------
    # Environment
    # --------------------------------------------------
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # --------------------------------------------------
    # Load website
    # --------------------------------------------------
    loader = WebBaseLoader(cfg.website.url)
    docs = loader.load()

    # --------------------------------------------------
    # Split documents
    # --------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.text_splitter.chunk_size,
        chunk_overlap=cfg.text_splitter.chunk_overlap
    )
    splits = text_splitter.split_documents(docs)

    # --------------------------------------------------
    # Embeddings + Vector Store
    # --------------------------------------------------
    embeddings = OpenAIEmbeddings(
        model=cfg.embeddings.model
    )

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": cfg.retriever.k}
    )

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------
    llm = ChatOpenAI(
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
    # Queries
    # --------------------------------------------------
    pp = pprint.PrettyPrinter(indent=4)

    for question in cfg.questions:
        result = rag_chain.invoke(question)
        logger.info("Question:")
        logger.info(question)

        logger.info("Answer:")
        logger.info(pp.pformat(result))


if __name__ == "__main__":
    run()
