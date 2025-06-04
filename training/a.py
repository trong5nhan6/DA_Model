import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

def plot_dann_history(history, title=None):
    """
    Plot training history with legends below each plot, and increased figure size.
    Args:
        history: Dictionary containing training history
        title: Title for the plot (optional)
    """
    epochs = history['epoch']
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # ⬅️ tăng chiều rộng và cao

    # Plot losses
    axs[0].plot(epochs, history['train_cls_loss'], marker='o',
                color='blue', label='Classification Loss')
    axs[0].plot(epochs, history['domain_loss'], marker='o',
                color='orange', label='Domain Loss')
    axs[0].set_title("Losses")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    # Plot accuracies
    axs[1].plot(epochs, [a * 100 for a in history['train_acc']],
                marker='o', color='green', label='Train Accuracy')
    axs[1].plot(epochs, [a * 100 for a in history['test_acc']],
                marker='o', color='red', label='Test Accuracy')
    axs[1].plot(epochs, [a * 100 for a in history['target_acc']],
                marker='o', color='purple', label='Target Accuracy')
    axs[1].set_title("Accuracies (%)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    # Super title and layout
    if title is not None:
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    else:
        plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.show()



def test_plot_dann_history():
    import numpy as np

    # Giả lập dữ liệu lịch sử training
    epochs = list(range(1, 21))
    history = {
        'epoch': epochs,
        'train_cls_loss': np.linspace(0.25, 0.05, 20),
        'domain_loss': np.clip(np.random.normal(loc=0.7, scale=0.05, size=20), 0.6, 0.85),
        'train_acc': np.clip(np.random.normal(loc=0.95, scale=0.01, size=20), 0.9, 1.0),
        'test_acc': np.clip(np.random.normal(loc=0.93, scale=0.03, size=20), 0.8, 0.98),
        'target_acc': np.clip(np.random.normal(loc=0.55, scale=0.15, size=20), 0.3, 0.75),
    }

    # Gọi hàm vẽ
    plot_dann_history(history, title="resnet18 dense-moe dann blood 224x224")


# Gọi hàm test
test_plot_dann_history()
