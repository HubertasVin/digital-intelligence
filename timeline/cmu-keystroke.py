import argparse
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CSVKeystrokeDataset(Dataset):
    def __init__(self, csv_path, session):
        df = pd.read_csv(csv_path)
        if session is not None:
            df = df[df["sessionIndex"] == session]
        feats = df.drop(["subject", "sessionIndex", "rep"], axis=1).values
        self.scaler = StandardScaler().fit(feats)
        X = self.scaler.transform(feats)
        y = df["subject"].astype("category").cat.codes.values.copy()
        self.samples = torch.from_numpy(X).float().unsqueeze(1)
        self.labels = torch.from_numpy(y).long()
        self.num_classes = len(df["subject"].unique())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class Residual1DCNN(nn.Module):
    def __init__(self, num_classes=30, dropout=0.1, activation="relu"):
        super().__init__()
        Act = {"relu": nn.ReLU, "elu": nn.ELU}[activation]

        def block(in_ch, out_ch, kernel=3, stride=1):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, stride, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                Act(),
                nn.Conv1d(out_ch, out_ch, kernel, 1, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                Act(),
            )

        self.layer1 = block(1, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.layer2 = block(32, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.layer3 = block(64, 128)
        self.pool3 = nn.MaxPool1d(2)
        self.layer4 = block(128, 256)
        self.pool4 = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))
        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return running_loss / n, correct / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            running_loss += criterion(logits, y).item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return running_loss / n, correct / n


def main(args):
    set_seed(SEED)

    if os.path.isfile(args.data_dir):
        ds = CSVKeystrokeDataset(args.data_dir, args.session)
    else:
        print("File not found!")
        return

    num_classes = getattr(ds, "num_classes", 30)

    n = len(ds)
    val = int(n * args.val_split)
    test = int(n * args.test_split)
    train = n - val - test
    train_ds, val_ds, test_ds = random_split(ds, [train, val, test])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Residual1DCNN(
        num_classes=num_classes, dropout=args.dropout, activation=args.activation
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = eval_epoch(model, val_loader, criterion, device)
        print(
            f"Epocha {epoch}: mokymo tikslumas {ta:.3f} | Validacijos tikslumas {va:.3f}"
        )
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))
    tl, ta = eval_epoch(model, test_loader, criterion, device)
    print(f"Testavimo paklaida: {tl:.4f} Tikslumas {ta:.4f}")

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    cm_full = confusion_matrix(all_labels, all_preds)
    num_classes = cm_full.shape[0]

    TP = np.diag(cm_full)
    FP = cm_full.sum(axis=0) - TP
    FN = cm_full.sum(axis=1) - TP
    TN = cm_full.sum() - (TP + FP + FN)

    print("Per-class confusion counts:")
    for i in range(num_classes):
        print(
            f" Class {i:2d}: TP={TP[i]:5d}  FP={FP[i]:5d}  FN={FN[i]:5d}  TN={TN[i]:5d}"
        )

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=range(num_classes),
            target_names=[f"Class {i}" for i in range(num_classes)],
            zero_division=0,
        )
    )

    total_samples = cm_full.sum()
    TP = TP.sum()
    FP = FP.sum()
    FN = FN.sum()
    TN = total_samples - (TP + FP)

    global_cm = np.array([[TP, FP], [FN, TN]])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=global_cm,
        display_labels=["Predicted Correct", "Predicted Incorrect"],
    )
    plt.figure()
    disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
    plt.title("Global Confusion Matrix")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "global_confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_full)
    disp.plot(cmap="Blues", include_values=False)
    plt.title("Daugiaklasė klasifikavimo lentelė")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "multiclass_confusion_matrix.png")
    plt.close()

    true_positives = np.sum(np.diag(cm_full))
    n_samples = np.sum(cm_full)
    false_negatives = np.sum(cm_full) - true_positives
    false_positives = false_negatives
    accuracy = true_positives / n_samples
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    print(f"Tikslumas: {accuracy:.4f}")
    print(f"Precizija: {precision:.4f}")
    print(f"Atšaukimas: {recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Klaviatūros duomenų direktorija")
    parser.add_argument("--session", type=int, default=1, help="Sesijos numeris (1–8)")
    parser.add_argument("--output_dir", type=str, default="ts_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", choices=["relu", "elu"], default="relu")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
