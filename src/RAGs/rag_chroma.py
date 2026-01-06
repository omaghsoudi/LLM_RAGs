#!/usr/bin/env python3

from omegaconf import DictConfig, OmegaConf
import hydra

from Common_modules.initialize import setup_logger

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama


from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings



PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


@hydra.main(
    config_path="configs",
    config_name="rag_chroma",
    version_base="1.1"
)
def run(cfg: DictConfig):
    logger = setup_logger(path_to_logger=cfg.logging.file)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------
    embedding_function = get_embedding_function()

    # --------------------------------------------------
    # Vector DB
    # --------------------------------------------------
    db = Chroma(
        persist_directory=cfg.chroma.persist_directory,
        embedding_function=embedding_function
    )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------
    prompt_template = ChatPromptTemplate.from_template(
        cfg.prompt.template
    )

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------
    llm = Ollama(
        model=cfg.llm.model
    )

    # --------------------------------------------------
    # Queries
    # --------------------------------------------------
    for query_text in cfg.queries:
        logger.info(f"\nQUESTION: {query_text}")

        results = db.similarity_search_with_score(
            query_text,
            k=cfg.retriever.k
        )

        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _ in results]
        )

        prompt = prompt_template.format(
            context=context_text,
            question=query_text
        )

        response_text = llm.invoke(prompt)

        sources = [
            doc.metadata.get("id", None)
            for doc, _ in results
        ]

        logger.info("ANSWER:")
        logger.info(response_text)
        logger.info("SOURCES:")
        logger.info(sources)


if __name__ == "__main__":
    run()
