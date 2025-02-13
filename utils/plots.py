import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path=None):
    """
    Plots training and validation losses.
    """
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()



def plot_lrs(steps, lrs, save_path=None):
    """
    Plots the learning rate schedule over training steps.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(steps, lrs)
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
