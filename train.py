# ==========================================
# Project: Image Tampering Detection Pipeline
# Owner & Lead Developer: Vaishnav Anand
# ==========================================
# train.py — Enhanced with full metrics:
#   Accuracy, F1, Precision, Recall,
#   Confusion Matrix, AUC-ROC + saved plots
# ==========================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for all OSes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# ──────────────────────────────────────────────
# 0.  Output directory for all saved plots
# ──────────────────────────────────────────────
PLOTS_DIR = 'training_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 1.  Model Definition
# ──────────────────────────────────────────────
def get_model(device):
    print("Loading pre-trained ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace head: 2 classes — Authentic (0) / Tampered (1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model.to(device)


# ──────────────────────────────────────────────
# 2.  Helper: run one full epoch, return metrics
# ──────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, phase):
    """
    Returns:
        epoch_loss   – float
        all_labels   – list[int]  ground-truth class indices
        all_preds    – list[int]  predicted class indices
        all_probs    – list[float] softmax probability of class-1 (Tampered)
    """
    is_train = (phase == 'train')
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss   += loss.item() * inputs.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].detach().cpu().numpy().tolist())  # P(Tampered)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, all_labels, all_preds, all_probs


# ──────────────────────────────────────────────
# 3.  Compute & pretty-print all metrics
# ──────────────────────────────────────────────
def compute_metrics(labels, preds, probs, phase, epoch):
    acc       = accuracy_score(labels, preds)
    f1        = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)

    # AUC-ROC — only meaningful if both classes are present
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float('nan')

    print(f"  [{phase.upper()}]  "
          f"Acc={acc:.4f}  F1={f1:.4f}  "
          f"Prec={precision:.4f}  Recall={recall:.4f}  "
          f"AUC-ROC={auc:.4f}")

    return dict(acc=acc, f1=f1, precision=precision,
                recall=recall, auc=auc)


# ──────────────────────────────────────────────
# 4.  Save all per-epoch curve plots
# ──────────────────────────────────────────────
def save_training_curves(history, num_epochs):
    """history keys: train_loss, val_loss, train_acc, val_acc,
                     train_f1, val_f1, train_auc, val_auc"""
    epochs = range(1, num_epochs + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics — Image Tampering Detector', fontsize=14, fontweight='bold')

    configs = [
        ('train_loss',  'val_loss',  'Loss',    'Loss'),
        ('train_acc',   'val_acc',   'Accuracy','Accuracy'),
        ('train_f1',    'val_f1',    'F1-Score','F1-Score'),
        ('train_auc',   'val_auc',   'AUC-ROC', 'AUC-ROC'),
    ]
    for ax, (tk, vk, ylabel, title) in zip(axes.flat, configs):
        ax.plot(epochs, history[tk], 'b-o', label='Train', linewidth=2, markersize=5)
        ax.plot(epochs, history[vk], 'r-s', label='Val',   linewidth=2, markersize=5)
        ax.set_xlabel('Epoch');  ax.set_ylabel(ylabel)
        ax.set_title(title);     ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved training curves → {path}")


# ──────────────────────────────────────────────
# 5.  Save Confusion Matrix (final val epoch)
# ──────────────────────────────────────────────
def save_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix — Validation Set', fontweight='bold')
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved confusion matrix → {path}")


# ──────────────────────────────────────────────
# 6.  Save AUC-ROC Curve (final val epoch)
# ──────────────────────────────────────────────
def save_roc_curve(labels, probs):
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc          = roc_auc_score(labels, probs)
    except ValueError:
        print("  ⚠️  Could not plot ROC curve (single class in val set).")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Validation Set', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, 'roc_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved ROC curve → {path}")


# ──────────────────────────────────────────────
# 7.  Save Per-class Precision / Recall bar plot
# ──────────────────────────────────────────────
def save_precision_recall_bar(labels, preds, class_names):
    prec_per_class   = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
    f1_per_class     = f1_score(labels, preds, average=None, zero_division=0)

    x     = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, prec_per_class,   width, label='Precision', color='steelblue')
    ax.bar(x,         recall_per_class, width, label='Recall',    color='coral')
    ax.bar(x + width, f1_per_class,     width, label='F1-Score',  color='seagreen')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics — Validation Set', fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, 'per_class_metrics.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved per-class metrics bar chart → {path}")


# ──────────────────────────────────────────────
# 8.  Main Training Loop
# ──────────────────────────────────────────────
def train_model(model, dataloaders, device, num_epochs=5,
                class_names=('Authentic', 'Tampered')):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )

    history = {k: [] for k in
               ['train_loss', 'val_loss',
                'train_acc',  'val_acc',
                'train_f1',   'val_f1',
                'train_auc',  'val_auc']}

    final_val_labels = final_val_preds = final_val_probs = None

    for epoch in range(num_epochs):
        print(f'\n{"="*50}')
        print(f'  Epoch {epoch+1}/{num_epochs}')
        print(f'{"="*50}')

        for phase in ['train', 'val']:
            loss, labels, preds, probs = run_epoch(
                model, dataloaders[phase],
                criterion, optimizer, device, phase
            )
            m = compute_metrics(labels, preds, probs, phase, epoch + 1)

            history[f'{phase}_loss'].append(loss)
            history[f'{phase}_acc'].append(m['acc'])
            history[f'{phase}_f1'].append(m['f1'])
            history[f'{phase}_auc'].append(m['auc'])

            print(f"  [{phase.upper()}] Loss: {loss:.4f}")

            # Keep the last validation epoch data for final plots
            if phase == 'val':
                final_val_labels = labels
                final_val_preds  = preds
                final_val_probs  = probs

    # ── Final diagnostic plots ──────────────────
    print("\nGenerating and saving all metric plots...")
    save_training_curves(history, num_epochs)
    save_confusion_matrix(final_val_labels, final_val_preds, list(class_names))
    save_roc_curve(final_val_labels, final_val_probs)
    save_precision_recall_bar(final_val_labels, final_val_preds, list(class_names))

    return model


# ──────────────────────────────────────────────
# 9.  Entry Point
# ──────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    if device.type == 'cpu':
        print("WARNING: Still using CPU! Check your PyTorch installation.")

    data_dir = r'D:\Research\hackathon\dataset'

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ]),
    }

    print("Loading datasets...")
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32,
                      shuffle=(x == 'train'), num_workers=2)
        for x in ['train', 'val']
    }

    class_names = image_datasets['train'].classes
    print(f"Classes mapped: {image_datasets['train'].class_to_idx}")

    model = get_model(device)
    print("Starting Training...\n")

    model = train_model(model, dataloaders, device,
                        num_epochs=10, class_names=class_names)

    save_path = 'tampering_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Training Complete! Weights saved → {save_path}")
    print(f"✅ All metric plots saved in → ./{PLOTS_DIR}/")
