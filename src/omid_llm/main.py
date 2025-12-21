#!/usr/bin/python3

from omegaconf import DictConfig
import hydra
from pathlib import Path
from omid_llm.modules.initialize import setup_logger

@hydra.main(
    config_path="../omid_llm/configs",
    config_name="run_config",
    version_base="0.1.1"
)
def run(cfg: DictConfig):
    Path("./logs").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(path_to_logger=f"./logs/run_hydra.log")
    hydra.utils.instantiate(cfg.module_1.test, logger=logger)

if __name__ == "__main__":
    run()