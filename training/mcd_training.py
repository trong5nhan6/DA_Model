import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def classifier_discrepancy(p1, p2):
    return torch.mean(torch.abs(F.softmax(p1, dim=1) - F.softmax(p2, dim=1)))


def results(history, epoch, cls_loss, dom_loss, train_acc, test_acc, target_acc):
    """
    Update training history
    """
    history['epoch'].append(epoch)
    history['train_cls_loss'].append(cls_loss)
    history['disc_loss'].append(dom_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    history['target_acc'].append(target_acc)
    return history


def plot_mcd_history(history, title=None):
    """
    Plot training history with legends below each subplot.
    Args:
        history: Dictionary containing training history
        title: Title for the plot (optional)
    """
    epochs = history['epoch']
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Mở rộng khổ biểu đồ

    # Plot losses
    axs[0].plot(epochs, history['train_cls_loss'], marker='o',
                color='blue', label='Cls Loss')
    axs[0].plot(epochs, history['disc_loss'], marker='o',
                color='orange', label='Discrepancy Loss')
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


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for xs, ys in dataloader:
        xs, ys = xs.to(device), ys.to(device)
        out1, out2 = model(xs)
        preds = (out1 + out2).argmax(dim=1)
        correct += (preds == ys).sum().item()
        total += ys.size(0)
    return correct / total


def train_mcd(model, source_loader, target_loader, source_test_loader, target_test_loader,
              device, epochs=20, lr=1e-3, step_size=5, gamma=0.5, beta=1.0, log_fn=None,
              k=1, auxiliary_loss=False):

    # Optimizers for feature extractor and individual classifiers
    optimizer_f = torch.optim.Adam(model.feature_extractor.parameters(), lr=lr)
    optimizer_c1 = torch.optim.Adam(model.classifier1.parameters(), lr=lr)
    optimizer_c2 = torch.optim.Adam(model.classifier2.parameters(), lr=lr)

    # Learning rate schedulers
    scheduler_f = torch.optim.lr_scheduler.StepLR(
        optimizer_f, step_size, gamma)
    scheduler_c1 = torch.optim.lr_scheduler.StepLR(
        optimizer_c1, step_size, gamma)
    scheduler_c2 = torch.optim.lr_scheduler.StepLR(
        optimizer_c2, step_size, gamma)

    criterion = nn.CrossEntropyLoss()

    history = {
        'epoch': [], 'train_cls_loss': [], 'disc_loss': [],
        'train_acc': [], 'test_acc': [], 'target_acc': []
    }

    for epoch in range(epochs):
        model.train()
        total_cls_loss, total_disc_loss = 0.0, 0.0
        total_samples = 0
        n_batches = min(len(source_loader), len(target_loader))
        src_iter, tgt_iter = iter(source_loader), iter(target_loader)

        for _ in range(n_batches):
            xs, ys = next(src_iter)
            xt, _ = next(tgt_iter)
            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            # -------------------------
            # Step 1: Train on source (classification loss)
            # -------------------------
            optimizer_f.zero_grad()
            optimizer_c1.zero_grad()
            optimizer_c2.zero_grad()
            out1, out2 = model(xs)
            loss1 = criterion(out1, ys)
            loss2 = criterion(out2, ys)

            if auxiliary_loss:
                loss = loss1 + loss2 + model.classifier1.auxiliary_loss + \
                    model.classifier2.auxiliary_loss
            else:
                loss = loss1 + loss2

            loss.backward()
            optimizer_f.step()
            optimizer_c1.step()
            optimizer_c2.step()

            # -------------------------
            # Step 2: Maximize discrepancy on target
            # -------------------------
            optimizer_c1.zero_grad()
            optimizer_c2.zero_grad()
            out1, out2 = model(xt)
            loss_dis = classifier_discrepancy(out1, out2)
            (-loss_dis).backward()
            optimizer_c1.step()
            optimizer_c2.step()

            # -------------------------
            # Step 3: Minimize discrepancy by updating feature extractor
            # -------------------------
            for _ in range(k):
                optimizer_f.zero_grad()
                out1, out2 = model(xt)
                loss_dis = classifier_discrepancy(out1, out2)
                loss_dis.backward()
                optimizer_f.step()

            total_cls_loss += (loss1.item() + loss2.item()) * xs.size(0)
            total_disc_loss += loss_dis.item() * xt.size(0)
            total_samples += xs.size(0)

        # Step learning rate
        scheduler_f.step()
        scheduler_c1.step()
        scheduler_c2.step()

        avg_cls_loss = total_cls_loss / total_samples
        avg_disc_loss = total_disc_loss / total_samples

        train_acc = evaluate(model, source_loader, device)
        test_acc = evaluate(model, source_test_loader, device)
        target_acc = evaluate(model, target_test_loader, device)

        print(f"[Epoch {epoch+1:02d}] ClsLoss: {avg_cls_loss:.4f} | DiscLoss: {avg_disc_loss:.4f} | "
              f"TrainAcc: {train_acc*100:.2f}% | TestAcc: {test_acc*100:.2f}% | TargetAcc: {target_acc*100:.2f}%")

        if log_fn:
            history = results(history, epoch+1, avg_cls_loss,
                              avg_disc_loss, train_acc, test_acc, target_acc)

    return history
