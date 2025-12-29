import os
import re
import json
import torch
import tiktoken
from tqdm import tqdm
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

from modules.models import (
    GPTModel,
    generate_text_simple,
    update_gpt_model_config,
    generate
)
from modules.weights import load_weights_into_gpt
from modules.plots import plot_losses_train_val
from modules.collate_functions import custom_collate_fn

from common_modules.initialize import setup_logger
from common_modules.losses import calc_loss_loader, calc_loss_batch, calc_loss_loader_v2
from common_modules.tokens_helpers import token_ids_to_text, text_to_token_ids
from common_modules.prompts import InstructionDataset, format_input
from common_modules.utils import download_and_load_file


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


# ---------------------------------------------------------
# Text generation helper
# ---------------------------------------------------------
def generate_and_print_sample(model, tokenizer, device, start_context, logger):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]

    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model, encoded, max_new_tokens=50, context_size=context_size
        )

    logger.info(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    cfg,
    tokenizer,
    logger
):
    train_losses, val_losses, tokens_seen = [], [], []
    total_tokens = 0
    global_step = 0

    for epoch in range(cfg.training.num_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()

            total_tokens += x.numel()
            global_step += 1

            if global_step % cfg.training.eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, cfg.training.eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen.append(total_tokens)

                logger.info(
                    f"Epoch {epoch+1} | Step {global_step:06d} | "
                    f"Train {train_loss:.3f} | Val {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, cfg.training.start_context, logger
        )

    return train_losses, val_losses, tokens_seen


# ---------------------------------------------------------
# Hydra main
# ---------------------------------------------------------
@hydra.main(version_base="1.3", config_path="configs", config_name="gpt_train_finetune_instructions")
def main(cfg: DictConfig):
    logger = setup_logger(path_to_logger=cfg.logging.file)

    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg, gpt_hf = update_gpt_model_config(cfg)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.device.seed)

    if cfg.device.use_cuda == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    logger.info(f"device: {device}")



    # ----------------------------
    # Load data
    # ----------------------------
    data = download_and_load_file(cfg.data.local_file, cfg.data.url)
    train_portion = int(len(data) * cfg.data.train_ratio)
    test_portion = int(len(data) * (cfg.data.test_ratio))

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    if cfg.test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    logger.info(f"Training set length: {len(train_data)}")
    logger.info(f"Validation set length: {len(val_data)}")
    logger.info(f"Test set length: {len(test_data)}")
    logger.info(50*"-")


    tokenizer = tiktoken.get_encoding(cfg.tiktoken.model_name)

    # ----------------------------
    # Dataloaders
    # ----------------------------
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.data.num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers
    )


    # ----------------------------
    # Model
    # ----------------------------
    model = GPTModel(cfg.model)
    model.to(device)

    if gpt_hf:
        logger.info(f"Loading pretrained model")
        model = load_weights_into_gpt(model, gpt_hf, cfg.model.n_layers)

    logger.info(f"model is loaded")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )


    logger.info(f"Start training")


    #######################################
    # Finetuning the model
    #######################################
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader_v2(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader_v2(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)


    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        cfg,
        tokenizer,
        logger
    )


    epochs_tensor = torch.linspace(0, cfg.training.num_epochs, len(train_losses))
    plot_losses_train_val(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50 * "-")

    #######################################
    # Saving results
    #######################################
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=cfg.model.context_length,
            context_size=cfg.model.context_length,
            eos_id=cfg.model.vocab_size
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = "instruction-data-with-response-standalone.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Responses saved as {test_data_path}")

    file_name = f"{re.sub(r'[ ()]', '', cfg.model.model_name)}-sft-standalone.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")


if __name__ == "__main__":
    main()
