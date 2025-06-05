import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def compute_grl_lambda(current_epoch, total_epochs):
    """
    Compute the Gradient Reversal Layer (GRL) lambda value based on training progress
    Args:
        current_epoch: Current training epoch
        total_epochs: Total number of training epochs
    Returns:
        Lambda value that gradually increases from 0 to 1 during training
    """
    p = current_epoch / total_epochs
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Simple mixup loss function
    Args:
        criterion: Base loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixup lambda
    Returns:
        loss: Combined loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    data_iter = iter(dataloader)
    for _ in range(len(dataloader)):
        xs, ys = next(data_iter)
        xs, ys = xs.to(device), ys.to(device)

        # Forward pass without gradient computation
        logits, _ = model(xs, alpha=0.0)
        preds = logits.argmax(dim=1)

        # Calculate accuracy
        correct += (preds == ys).sum().item()
        total += ys.size(0)

    return correct / total


def train_dann(model, source_loader, target_loader, source_test_loader, target_test_loader,
               device, epochs=10, lr=1e-3, step_size=5, gamma=0.5, beta=0.8,
               log_fn=None, auxiliary_loss=False, use_mixup=False, mixup_fn=None, mixup_alpha=0.4):
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
        lr: Learning rate
        step_size: Step size for learning rate scheduler
        gamma: Learning rate decay factor
        beta: Weight for domain adaptation loss
        log_fn: Callback function for logging (optional)
        auxiliary_loss: Whether to use auxiliary loss
        use_mixup: Whether to use mixup augmentation
        mixup_fn: Mixup function to use when use_mixup is True
        mixup_alpha: Alpha parameter for mixup (default: 0.4)
    Returns:
        Training history containing metrics
    """
    # Initialize training history dictionary to track metrics
    history = {
        'epoch': [],
        'train_cls_loss': [],
        'domain_loss': [],
        'train_acc': [],
        'test_acc': [],
        'target_acc': []
    }

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_cls_loss = 0.0
        total_dom_loss = 0.0
        total_samples = 0

        # Get minimum number of batches between source and target
        n_batches = min(len(source_loader), len(target_loader))
        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)

        # Compute GRL lambda for current epoch
        grl_lambda = compute_grl_lambda(epoch, epochs)

        for _ in range(n_batches):
            # Get batch data from source and target
            xs, ys = next(src_iter)
            xt, _ = next(tgt_iter)
            xs, ys = xs.to(device, non_blocking=True), ys.to(
                device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)

            # Apply mixup if enabled
            if use_mixup and mixup_fn is not None:
                # Apply mixup to source data
                xs, y_a, y_b, lam = mixup_fn(xs, ys, device, mixup_alpha)
                # Update ys to be the mixed labels
                ys = (y_a, y_b, lam)

            # Combine source and target data
            x_combined = torch.cat([xs, xt], dim=0)
            y_domain = torch.cat([
                torch.zeros(xs.size(0), dtype=torch.long),  # Source domain = 0
                torch.ones(xt.size(0), dtype=torch.long)   # Target domain = 1
            ]).to(device, non_blocking=True)

            # Forward pass
            y_cls, y_dom = model(x_combined, alpha=grl_lambda)
            y_cls_src = y_cls[:xs.size(0)]

            # Calculate classification and domain adaptation losses
            if use_mixup:
                # Mixup loss calculation using mixup_criterion
                y_a, y_b, lam = ys
                loss_cls = mixup_criterion(criterion, y_cls_src, y_a, y_b, lam)
            else:
                loss_cls = criterion(y_cls_src, ys)  # Classification loss

            loss_dom = criterion(y_dom, y_domain)  # Domain adaptation loss

            if auxiliary_loss:
                loss = loss_cls + loss_dom*beta + model.label_classifier.auxiliary_loss + \
                    model.domain_classifier.auxiliary_loss
            else:
                loss = loss_cls + loss_dom*beta

            # Backward pass and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            total_cls_loss += loss_cls.item() * xs.size(0)
            total_dom_loss += loss_dom.item() * x_combined.size(0)
            total_samples += xs.size(0)

        scheduler.step()

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
