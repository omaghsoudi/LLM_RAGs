#!/usr/bin/python3

from omegaconf import DictConfig
import hydra
from pathlib import Path
from omid_llm.modules.initialize import setup_logger
from omegaconf import OmegaConf

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter


@hydra.main(
    config_path="configs",
    config_name="rag_unstructured",
    version_base="1.1.0"
)
def run(cfg: DictConfig):
    Path("./logs").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(path_to_logger=f"./logs/run_hydra.log")

    logger.info(OmegaConf.to_yaml(cfg))

    loader = UnstructuredFileLoader(cfg["unstructured_pdf_file"])
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()

    db = FAISS.from_documents(texts, embeddings)

    llm = Ollama(model="llama3")

    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever()
    )

    question = "Can you please summarize the document"
    result = chain.invoke({"query": question})

    print(result['result'])

if __name__ == "__main__":
    run()