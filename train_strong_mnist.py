import os, math, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.short = (nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_ch)) if stride!=1 or in_ch!=out_ch else nn.Identity())
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        x = self.short(x)
        return F.relu(x + y, inplace=True)

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = BasicBlock(32, 64, stride=2)   # 28->14
        self.layer2 = BasicBlock(64, 128, stride=2)  # 14->7
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    print("[Device]", device)
    if is_cuda:
        torch.backends.cudnn.benchmark = True

    mean, std = (0.1307,), (0.3081,)
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.12,0.12), scale=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = datasets.MNIST("./data", train=True,  download=True, transform=train_tf)
    test_set  = datasets.MNIST("./data", train=False, download=True, transform=test_tf)

    bs = 512 if is_cuda else 256
    nw = max(2, (os.cpu_count() or 4)//2)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=is_cuda, persistent_workers=True)
    test_loader  = DataLoader(test_set,  batch_size=bs*2, shuffle=False,
                              num_workers=nw, pin_memory=is_cuda, persistent_workers=True)

    model = SmallResNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    max_epochs = 10 if is_cuda else 12
    steps_per_epoch = math.ceil(len(train_set)/bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01 if is_cuda else 0.006,
                                                steps_per_epoch=steps_per_epoch, epochs=max_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)

    best = 0.0
    target = 0.988  # 达到就早停
    os.makedirs("runs_mnist", exist_ok=True)

    t0 = time.time()
    for ep in range(1, max_epochs+1):
        model.train(); run=0.0
        for x,y in train_loader:
            x,y = x.to(device, non_blocking=is_cuda), y.to(device, non_blocking=is_cuda)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=is_cuda):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            sched.step()
            run += loss.item() * x.size(0)

        model.eval(); correct=0; total=0
        with torch.inference_mode():
            for x,y in test_loader:
                x,y = x.to(device, non_blocking=is_cuda), y.to(device, non_blocking=is_cuda)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item(); total += y.size(0)
        acc = correct/total
        print(f"Epoch {ep:02d}/{max_epochs}  loss={run/len(train_set):.4f}  val_acc={acc*100:.2f}%  "
              f"lr={sched.get_last_lr()[0]:.5f}  time={time.time()-t0:.1f}s")

        if acc>best:
            best = acc
            torch.save({
                "arch": "SmallResNet",
                "normalize_mean": mean,
                "normalize_std":  std,
                "model": model.state_dict()
            }, "runs_mnist/best_model.pt")
            print(f"  -> 保存最优: {best*100:.2f}%")

        if acc >= target:
            print(f"达到 {target*100:.2f}% 目标，提前结束。"); break

    print(f"完成。最佳验证准确率：{best*100:.2f}%")

if __name__ == "__main__":
    main()
