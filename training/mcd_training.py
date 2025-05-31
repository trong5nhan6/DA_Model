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
    epochs = history['epoch']
    plt.figure(figsize=(10, 4))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_cls_loss'], label='Cls Loss')
    plt.plot(epochs, history['disc_loss'], label='Discrepancy Loss')
    plt.legend()
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history['train_acc']], label='Train')
    plt.plot(epochs, [a * 100 for a in history['test_acc']], label='Test')
    plt.plot(epochs, [a * 100 for a in history['target_acc']], label='Target')
    plt.legend()
    plt.title('Accuracies (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    # Only set suptitle if title is not None
    if title is not None:
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

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
              device, epochs=20, lr=1e-3, step_size=5, gamma=0.5, beta=1.0, log_fn=None, k=4, auxiliary_loss=False):

    optimizer_f = torch.optim.Adam(model.feature_extractor.parameters(), lr=lr)
    optimizer_c = torch.optim.Adam(
        list(model.classifier1.parameters()) + list(model.classifier2.parameters()), lr=lr)
    scheduler_f = torch.optim.lr_scheduler.StepLR(
        optimizer_f, step_size, gamma)
    scheduler_c = torch.optim.lr_scheduler.StepLR(
        optimizer_c, step_size, gamma)

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
            optimizer_c.zero_grad()
            out1, out2 = model(xs)
            loss1 = criterion(out1, ys)
            loss2 = criterion(out2, ys)

            if auxiliary_loss:
                loss = loss1 + loss2 + model.classifier1.auxiliary_loss + model.classifier2.auxiliary_loss
            else:
                loss = loss1 + loss2
                
            loss.backward()
            optimizer_f.step()
            optimizer_c.step()

            # -------------------------
            # Step 2: Maximize discrepancy on target
            # -------------------------
            for _ in range(1):  # T thường = 1
                optimizer_c.zero_grad()
                out1, out2 = model(xt)
                loss_dis = classifier_discrepancy(out1, out2)
                (-loss_dis).backward()
                optimizer_c.step()

            # -------------------------
            # Step 3: Minimize discrepancy by updating feature extractor
            # -------------------------
            for _ in range(k):  # S thường = 1
                optimizer_f.zero_grad()
                out1, out2 = model(xt)
                loss_dis = classifier_discrepancy(out1, out2)
                loss_dis.backward()
                optimizer_f.step()

            total_cls_loss += (loss1.item() + loss2.item()) * xs.size(0)
            total_disc_loss += loss_dis.item() * xt.size(0)
            total_samples += xs.size(0)

        scheduler_f.step()
        scheduler_c.step()

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
