# **📘 LLM_RAGs**
LLM_RAGs is a reproducible research repository for building, training, and deploying retrieval-augmented generation (RAG) systems and custom LLM training/finetuning from scratch — including:
🧠 Training GPT-like language models from scratch
🔍 Building a RAG pipeline with vector databases and embeddings
🛠 End-to-end scripts, configs, and examples
This repo is designed for experimentation and education, especially for practitioners building custom RAG systems and training LLMs with PyTorch.


## 🚀 Features
### 🧠 LLM Training
Train GPT-like transformers from scratch using PyTorch:
Custom GPT models inspired by “Build a Large Language Model From Scratch”
Instruction finetuning support
Hydra-based training configs
Evaluation and generation utilities
Tokenized training & evaluation
### 🔎 RAG (Retrieval-Augmented Generation)
Index and query documents with vector embeddings:
Ingest document collections
Build vector store (Chroma / FAISS / others)
Query + generate responses using embeddings + LLM
Modular and reusable pipeline
### 📁 Repository Structure
```
LLM_RAGs/
├── README.md
├── LICENSE.txt
├── init.sh
├── pyproject.toml
├── requirements.txt
├── datasets/
│   └── (various dataset files and subfolders)
├── install/
│   └── (installation helper scripts)
└── src/
    ├── __init__.py
    ├── common_modules/
    │   └── (various dataset files and subfolders)
    ├── LLMs_training/
    │   └── (various dataset files and subfolders)
    └── RAGs/
        └── (various dataset files and subfolders)
    
```

###  Examples to tun the codes
```
python gpt_train_finetune_instructions.py
python train_instruction_finetune.py \
    --data_file data/instruction-data.json \
    --config configs/finetune.yaml
python chroma_ingest.py 
```
## ⚙️ Installation
easiest way to install the package:
```bash
bash <(curl -fsSL https://raw.githubusercontent.com/omaghsoudi/LLM_RAGs/main/init.sh)
```
```bash
📌 Clone
git clone https://github.com/omaghsoudi/LLM_RAGs.git
cd LLM_RAGs
poetry install
poetry shell
```

## 📜 License
This project is licensed under the Apache 2.0 License.
🙌 Contributions
Contributions welcome via PRs and issues!
📬 Contact
Created by Omid Haji — happy to help on community discussions and questions.