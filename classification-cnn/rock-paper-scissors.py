import argparse
import os
import random
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import kagglehub

SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_rps_dataset(dest_dir):
    dest = Path(dest_dir)
    if (dest / "rock").exists() and (dest / "paper").exists() and (dest / "scissors").exists():
        return

    print("Downloading Rock–Paper–Scissors dataset...")
    dataset_path = Path(kagglehub.dataset_download("drgfreeman/rockpaperscissors"))

    if dataset_path.suffix == ".zip":
        with zipfile.ZipFile(dataset_path) as archive:
            archive.extractall(dest)
    elif dataset_path.is_dir() and set(["rock","paper","scissors"]).issubset(
            {p.name for p in dataset_path.iterdir() if p.is_dir()}):
        for cls in ["rock", "paper", "scissors"]:
            shutil.copytree(dataset_path/cls, dest/cls, dirs_exist_ok=True)
    else:
        extracted = dataset_path / "rock_paper_scissors"
        for split in ["train", "test", "validation"]:
            split_dir = extracted / split
            if not split_dir.exists():
                continue
            for class_dir in split_dir.iterdir():
                target = dest / class_dir.name
                target.mkdir(parents=True, exist_ok=True)
                for img in class_dir.glob("*.png"):
                    shutil.move(str(img), target / img.name)

def get_dataloaders(data_dir, img_size, batch_size, val_split, test_split):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)
    class_names = full_dataset.classes

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    test_size = int(test_split * len(full_dataset))
    val_size  = int(val_split  * len(full_dataset))
    train_size = len(full_dataset) - val_size - test_size

    train_indices = indices[:train_size]
    val_indices   = indices[train_size:train_size+val_size]
    test_indices  = indices[train_size+val_size:]

    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    
    eval_dataset = datasets.ImageFolder(root=data_dir, transform=transform_eval)
    val_ds  = torch.utils.data.Subset(eval_dataset, val_indices)
    test_ds = torch.utils.data.Subset(eval_dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3, img_size=224, dropout=0.25, activation="relu", batchnorm=True, **kwargs):
        super().__init__()
        act = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}[activation]

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
            act(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            act(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
            act(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 256),
            act(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class VGGStyleCNN(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3, activation="relu", batchnorm=True, **kwargs):
        super().__init__()
        act = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}[activation]
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, "M"]
        layers, in_ch = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.extend([
                    nn.Conv2d(in_ch, v, 3, padding=1),
                    nn.BatchNorm2d(v) if batchnorm else nn.Identity(),
                    act(),
                ])
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            act(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(self.features(x))
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, batchnorm=True, activation="relu"):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}[activation]
        self.act = act_cls()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch) if batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch) if batchnorm else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch) if batchnorm else nn.Identity(),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act(out)


class MiniResNet(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3, batchnorm=True, activation="relu", **kwargs):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}[activation]
        self.act = act_cls()

        self.in_ch = 32
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(32) if batchnorm else nn.Identity()

        self.layer1 = self._make_layer(32, 2, 1, batchnorm, activation)
        self.layer2 = self._make_layer(64, 2, 2, batchnorm, activation)
        self.layer3 = self._make_layer(128, 2, 2, batchnorm, activation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(128, num_classes)

    def _make_layer(self, out_ch, blocks, stride, batchnorm, activation):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBlock(self.in_ch, out_ch, s, batchnorm, activation))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        return self.fc(out)


MODELS = {"simple": SimpleCNN, "vgg": VGGStyleCNN, "resnet": MiniResNet}

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir):
    epochs = range(1, len(train_losses) + 1)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label="train")
    plt.plot(epochs, val_accs, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "accuracy_curve.png")
    plt.close()

def plot_confusion_matrix(model, loader, classes, device, out_dir):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "confusion_matrix.png")
    plt.close()

def main(args):
    set_seed(SEED)

    download_rps_dataset(args.data_dir)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.val_split, args.test_split
    )

    device = get_device()
    model = MODELS[args.model](num_classes=len(class_names), dropout=args.dropout, 
                               activation=args.activation, batchnorm=args.batchnorm)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam" else \
                optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3) if args.scheduler else None

    best_loss = float('inf')
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    tolerated_epochs = 0

    if args.train:
        for epoch in range(args.epochs):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            train_accs.append(tr_acc)
            val_accs.append(val_acc)

            if scheduler:
                scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
                tolerated_epochs = 0
            else:
                tolerated_epochs += 1
                if tolerated_epochs >= 10:
                    print("Early stopping: no improvement for 10 epochs.")
                    break

            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Train loss {tr_loss:.4f}; Acc {tr_acc:.4f} | "
                  f"Val loss {val_loss:.4f}; Acc {val_acc:.4f}")

        plot_curves(train_losses, val_losses, train_accs, val_accs, args.output_dir)

    model.load_state_dict(
        torch.load(
            os.path.join(args.output_dir, "model.pt"),
            map_location=get_device()
        )
    )
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f}; Accuracy {test_acc:.4f}")

    y_true_all, y_pred_all = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            y_true_all.extend(y.numpy())
            y_pred_all.extend(preds)

    # print("Tikroji klasė ir prognozė")
    # for i in range(30):
    #     true_name = class_names[y_true_all[i]]
    #     pred_name = class_names[y_pred_all[i]]
    #     print(f"{i+1} & {true_name} & {pred_name}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_confusion_matrix(model, test_loader, class_names, device, args.output_dir)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CNN for Rock‑Paper‑Scissors classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save outputs")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="simple", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "elu"], default="relu")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--scheduler", action="store_true", help="Use ReduceLROnPlateau scheduler")
    parser.add_argument("--batchnorm", action="store_true", help="Enable Batch Normalization")
    parser.add_argument("--train", type=str2bool, default=True, help="Train the model")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
