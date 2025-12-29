import torch

# ---------------------------------------------------------
# Loss helpers
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