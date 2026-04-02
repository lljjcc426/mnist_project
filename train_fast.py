# train_fast.py  —— 目标：≤10分钟收敛到 val_acc >= 98%
import os, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ===== 1) 轻量高效的 CNN（足够达到 98%+） =====
class FastCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28 -> 14
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14 -> 7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*7, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.head(self.conv(x))

# ===== 2) 训练主流程（OneCycleLR + 早停 + AMP/GPU 加速） =====
def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    print(f"[Device] {device}")

    # 让 CPU 也尽可能快
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    if is_cuda:
        torch.backends.cudnn.benchmark = True

    # 轻量增强（不影响分布、但能稳一点）
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST("./data", train=True,  download=True, transform=train_tf)
    test_set  = datasets.MNIST("./data", train=False, download=True, transform=test_tf)

    # 批量大小：GPU 大、CPU 中等；尽量用较多 workers（Windows 也可以）
    bs = 512 if is_cuda else 256
    nw = max(2, (os.cpu_count() or 4) // 2)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=is_cuda, persistent_workers=True)
    test_loader  = DataLoader(test_set,  batch_size=bs*2, shuffle=False,
                              num_workers=nw, pin_memory=is_cuda, persistent_workers=True)

    model = FastCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    # OneCycleLR：快速拉到峰值再退火，通常 3~6 个 epoch 就到 98%+
    max_epochs = 8 if is_cuda else 10  # CPU 稍微多给一点轮数
    steps_per_epoch = math.ceil(len(train_set)/bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01 if is_cuda else 0.006,
                                                steps_per_epoch=steps_per_epoch, epochs=max_epochs)
    # 轻微 label smoothing，稳定边界
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)  # GPU 用混合精度
    best_acc = 0.0
    os.makedirs("runs_mnist", exist_ok=True)

    target_acc = 0.98   # 达到就早停
    time_budget = 10*60 # 10 分钟

    for epoch in range(1, max_epochs+1):
        model.train()
        run_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=is_cuda)
            labels = labels.to(device, non_blocking=is_cuda)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=is_cuda):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * imgs.size(0)

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for imgs, labels in test_loader:
                imgs = imgs.to(device, non_blocking=is_cuda)
                labels = labels.to(device, non_blocking=is_cuda)
                logits = model(imgs)
                pred = logits.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        tr_loss = run_loss / len(train_set)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{max_epochs}  loss={tr_loss:.4f}  val_acc={acc*100:.2f}%  "
              f"lr={sched.get_last_lr()[0]:.5f}  time={elapsed:.1f}s")

        # 保存更优模型
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(),
                        "normalize_mean": (0.1307,), "normalize_std": (0.3081,)},
                       "runs_mnist/best_model.pt")
            print(f"  -> 保存最优：{best_acc*100:.2f}%  @ runs_mnist/best_model.pt")

        # 早停条件：到 98% 立刻退出；或超时保护
        if acc >= target_acc:
            print(f"达到 {target_acc*100:.0f}% 目标，提前结束训练。"); break
        if elapsed > time_budget:
            print("达到 10 分钟时间预算，提前停止。"); break

    print(f"完成。最佳验证准确率：{best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
