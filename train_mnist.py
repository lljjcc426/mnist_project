# train_mnist.py  (fixed)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.fc1   = nn.Linear(64*7*7, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg["lr"]

def main():
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # —— 关键修改：去掉 verbose 参数 ——
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs("runs_mnist", exist_ok=True)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = correct / total

        # 使用 1-acc 做“更好就更小”的监控量
        prev_lr = get_lr(optimizer)
        scheduler.step(1.0 - acc)
        now_lr = get_lr(optimizer)

        print(f"Epoch {epoch:02d}: train_loss={total_loss/len(train_set):.4f}, val_acc={acc*100:.2f}%, lr={now_lr:.6f}")
        if now_lr < prev_lr:
            print(f"  -> 学习率降低: {prev_lr:.6f} -> {now_lr:.6f}")

        if acc > best_acc:
            best_acc = acc
            state = {
                "model": model.state_dict(),
                "classes": list(range(10)),
                "normalize_mean": (0.1307,),
                "normalize_std":  (0.3081,)
            }
            torch.save(state, "runs_mnist/best_model.pt")
            print(f"  -> 保存最优模型：runs_mnist/best_model.pt (acc={best_acc*100:.2f}%)")

    print("训练完成。最佳验证准确率：{:.2f}%".format(best_acc*100))

if __name__ == "__main__":
    main()
