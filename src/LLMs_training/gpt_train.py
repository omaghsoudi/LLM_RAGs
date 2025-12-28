import os
import requests
import torch
import tiktoken
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

from modules.models import (
    GPTModel,
    create_dataloader_v1,
    generate_text_simple,
)

from common_modules.initialize import setup_logger

# ---------------------------------------------------------
# Token helpers
# ---------------------------------------------------------
def text_to_token_ids(text, tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


# ---------------------------------------------------------
# Loss + evaluation helpers
# ---------------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")

    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total_loss = 0.0

    for i, (x, y) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(x, y, model, device).item()

    return total_loss / num_batches


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
# Plot
# ---------------------------------------------------------
def plot_losses(tokens_seen, train_losses, val_losses, out_path="loss.pdf"):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Eval step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path)


# ---------------------------------------------------------
# Hydra main
# ---------------------------------------------------------
@hydra.main(version_base="1.3", config_path="configs", config_name="gpt_train")
def main(cfg: DictConfig):

    logger = setup_logger(path_to_logger=cfg.logging.file)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.manual_seed(cfg.device.seed)

    if cfg.device.use_cuda == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    logger.info(f"device: {device}")

    # ----------------------------
    # Load data
    # ----------------------------
    if not os.path.exists(cfg.data.local_file):
        r = requests.get(cfg.data.url, timeout=30)
        r.raise_for_status()
        with open(cfg.data.local_file, "w", encoding="utf-8") as f:
            f.write(r.text)

    with open(cfg.data.local_file, "r", encoding="utf-8") as f:
        text_data = f.read()

    split_idx = int(cfg.data.train_ratio * len(text_data))

    # ----------------------------
    # Dataloaders
    # ----------------------------
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=cfg.training.batch_size,
        max_length=cfg.model.context_length,
        stride=cfg.model.context_length,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.data.num_workers,
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=cfg.training.batch_size,
        max_length=cfg.model.context_length,
        stride=cfg.model.context_length,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
    )

    # ----------------------------
    # Model
    # ----------------------------
    model = GPTModel(cfg.model)
    model.to(device)

    logger.info(f"model is loaded")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    logger.info(f"Start training")

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

    plot_losses(tokens_seen, train_losses, val_losses, out_path=f"{output_dir}/loss.pdf")

    torch.save(model.state_dict(), f"{output_dir}/model.pth")

    logger.info(f"model.pth is saved: {output_dir}/model.pth")

    logger.info(f"DONE")


if __name__ == "__main__":
    main()
