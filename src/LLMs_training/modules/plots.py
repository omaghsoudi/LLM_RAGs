
import matplotlib.pyplot as plt


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


def plot_losses_train_val(epochs_seen, tokens_seen, train_losses, val_losses, out_path="loss.pdf"):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    print(f"Plot saved as {out_path}")
    plt.savefig(out_path)
    # plt.show()