import torch
import torch.nn as nn
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model performance on a dataset
    Args:
        model: Model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Computing device (CPU/GPU)
    Returns:
        Accuracy rate on the dataset
    """
    model.eval()
    correct = 0
    total = 0

    data_iter = iter(dataloader)
    for _ in range(len(dataloader)):
        xs, ys = next(data_iter)
        xs, ys = xs.to(device), ys.to(device)

        logits, _ = model(xs, alpha=0.0)
        preds = logits.argmax(dim=1)

        correct += (preds == ys).sum().item()
        total += ys.size(0)

    return correct / total


def train_dann(model, source_loader, target_loader, source_test_loader, target_test_loader,
               device, epochs=10, log_fn=None):
    """
    Train DANN (Domain Adaptation Neural Network) model
    Args:
        model: DANN model
        source_loader: DataLoader for source data
        target_loader: DataLoader for target data
        source_test_loader: DataLoader for source test data
        target_test_loader: DataLoader for target test data
        device: Computing device (CPU/GPU)
        epochs: Number of training epochs
        log_fn: Callback function for logging (optional)
    Returns:
        Training history containing metrics
    """
    # Initialize training history dictionary
    history = {
        'epoch': [],
        'train_cls_loss': [],
        'domain_loss': [],
        'train_acc': [],
        'test_acc': [],
        'target_acc': []
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_cls_loss = 0.0
        total_dom_loss = 0.0
        total_samples = 0
        n_batches = min(len(source_loader), len(target_loader))
        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)

        for _ in range(n_batches):
            # Get batch data from source and target
            xs, ys = next(src_iter)
            xt, _ = next(tgt_iter)
            xs, ys = xs.to(device, non_blocking=True), ys.to(
                device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)

            # Combine source and target data
            x_combined = torch.cat([xs, xt], dim=0)
            y_domain = torch.cat([
                torch.zeros(xs.size(0), dtype=torch.long),  # Source domain = 0
                torch.ones(xt.size(0), dtype=torch.long)   # Target domain = 1
            ]).to(device, non_blocking=True)

            # Forward pass
            y_cls, y_dom = model(x_combined, alpha=1.0)
            y_cls_src = y_cls[:xs.size(0)]

            # Calculate losses
            loss_cls = criterion(y_cls_src, ys)  # Classification loss
            loss_dom = criterion(y_dom, y_domain)  # Domain adaptation loss
            loss = loss_cls + loss_dom

            # Backward pass and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            total_cls_loss += loss_cls.item() * xs.size(0)
            total_dom_loss += loss_dom.item() * x_combined.size(0)
            total_samples += xs.size(0)

        # Calculate average losses
        avg_cls_loss = total_cls_loss / total_samples
        avg_dom_loss = total_dom_loss / (2 * total_samples)

        # Evaluate model on datasets
        train_acc = evaluate(model, source_loader, device)
        test_acc = evaluate(model, source_test_loader, device)
        target_acc = evaluate(model, target_test_loader, device)

        # Print results
        print(f"[Epoch {epoch+1:02d}] "
              f"ClsLoss: {avg_cls_loss:.4f} | DomLoss: {avg_dom_loss:.4f} | "
              f"TrainAcc: {train_acc*100:.2f}% | TestAcc: {test_acc*100:.2f}% | TargetAcc: {target_acc*100:.2f}%")

        if log_fn:
            history = results(history, epoch+1, avg_cls_loss,
                              avg_dom_loss, train_acc, test_acc, target_acc)
    return history


def results(history, epoch, cls_loss, dom_loss, train_acc, test_acc, target_acc):
    """
    Update training history
    """
    history['epoch'].append(epoch)
    history['train_cls_loss'].append(cls_loss)
    history['domain_loss'].append(dom_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    history['target_acc'].append(target_acc)
    return history


def plot_dann_history(history):
    """
    Plot training history
    Args:
        history: Dictionary containing training history
    """
    epochs = history['epoch']

    plt.figure(figsize=(10, 4))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_cls_loss'], marker='o',
             color='blue', label='Classification Loss')
    plt.plot(epochs, history['domain_loss'], marker='o',
             color='orange', label='Domain Loss')
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history['train_acc']],
             marker='o', color='green', label='Train Accuracy')
    plt.plot(epochs, [a * 100 for a in history['test_acc']],
             marker='o', color='red', label='Test Accuracy')
    plt.plot(epochs, [a * 100 for a in history['target_acc']],
             marker='o', color='purple', label='Target Accuracy')
    plt.title("Accuracies (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()
