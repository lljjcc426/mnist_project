import os, time
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch, torch.nn as nn, torch.nn.functional as F

MODEL_PATH = r"runs_mnist\best_model.pt"

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
        self.layer1 = BasicBlock(32, 64, stride=2)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); return self.head(x)

class App:
    def __init__(self, root, model_path=MODEL_PATH):
        self.root = root
        self.root.title("MNIST 手写识别（离屏绘制稳态版）")
        self.size = 320
        self.brush = 14
        self.bg = 0    # 黑底
        self.fg = 255  # 白色笔迹（与 MNIST 一致）
        # 画布 + 离屏图像
        self.canvas = tk.Canvas(root, width=self.size, height=self.size, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=5, padx=10, pady=10)
        self.img = Image.new("L", (self.size, self.size), color=self.bg)
        self.draw = ImageDraw.Draw(self.img)
        self.last = None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "last", None))

        # 按钮区
        tk.Button(root, text="清空", width=10, command=self.clear).grid(row=1, column=0, pady=6)
        tk.Button(root, text="识别", width=10, command=self.predict).grid(row=1, column=1, pady=6)
        tk.Button(root, text="保存样本", width=10, command=self.save_sample).grid(row=1, column=2, pady=6)
        tk.Button(root, text="退出", width=10, command=root.quit).grid(row=1, column=3, pady=6)

        self.result = tk.StringVar(value="结果：-")
        tk.Label(root, textvariable=self.result, font=("Microsoft YaHei", 16)).grid(row=1, column=4, padx=8)

        # 模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.mean, self.std = self.load_model(model_path)
        self.model.to(self.device).eval()

    # 载入模型
    def load_model(self, path):
        if not os.path.exists(path):
            messagebox.showerror("错误", f"未找到模型：{path}\n请先运行训练脚本。")
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu")
        arch = state.get("arch", "SmallResNet")
        if arch != "SmallResNet":
            messagebox.showwarning("提示", f"检测到 arch={arch}，已按 SmallResNet 加载（请用配套训练脚本）。")
        model = SmallResNet()
        model.load_state_dict(state["model"])
        mean = tuple(state.get("normalize_mean", (0.1307,)))
        std  = tuple(state.get("normalize_std",  (0.3081,)))
        print(f"[Loader] arch={arch}  path={path}")
        return model, mean, std

    # 画线
    def paint(self, e):
        if self.last is None:
            self.last = (e.x, e.y); return
        x0,y0 = self.last; x1,y1 = e.x, e.y
        self.canvas.create_line(x0, y0, x1, y1, fill="white", width=self.brush, capstyle=tk.ROUND, smooth=True)
        self.draw.line((x0, y0, x1, y1), fill=self.fg, width=self.brush, joint="curve")
        self.last = (x1, y1)

    def clear(self):
        self.canvas.delete("all")
        self.img.paste(self.bg, [0,0,self.size,self.size])
        self.result.set("结果：-")

    def _prep_mnist(self, img_gray: Image.Image):
        arr = np.array(img_gray, dtype=np.uint8)
        ys, xs = np.where(arr > 10)   # 前景阈值
        if len(xs)==0 or len(ys)==0:
            return None, None
        x0,x1,y0,y1 = xs.min(), xs.max(), ys.min(), ys.max()
        crop = img_gray.crop((x0,y0,x1+1,y1+1))
        w,h = crop.size
        if w>h:
            new_w, new_h = 20, max(1, int(round(20*h/w)))
        else:
            new_h, new_w = 20, max(1, int(round(20*w/h)))
        small = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28,28), color=0)
        left, top = (28-new_w)//2, (28-new_h)//2
        canvas.paste(small, (left, top))
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.6))
        a = np.array(canvas, dtype=np.float32)/255.0
        yy,xx = np.mgrid[0:28,0:28]
        m = a.sum() + 1e-6
        cy = (yy*a).sum()/m; cx = (xx*a).sum()/m
        shift_y, shift_x = int(round(14-cy)), int(round(14-cx))
        a = np.roll(np.roll(a, shift_y, axis=0), shift_x, axis=1)
        a = (a - self.mean[0]) / self.std[0]
        tens = torch.from_numpy(a[None,None,:,:].astype(np.float32))
        return tens, canvas  # 返回张量以及 28×28 预览

    def predict(self):
        tens, prev = self._prep_mnist(self.img)
        if tens is None:
            self.result.set("结果：空白"); return
        tens = tens.to(self.device)
        with torch.inference_mode():
            prob = torch.softmax(self.model(tens), dim=1)[0]
            topv, topi = torch.topk(prob, k=3)
        msg = f"结果：{int(topi[0])}（置信度 {topv[0].item()*100:.1f}%） | Top3: " \
              f"{[int(x) for x in topi.tolist()]} {['%.1f%%'% (v*100) for v in topv.tolist()]}"
        self.result.set(msg)

    def save_sample(self):
        tens, prev = self._prep_mnist(self.img)
        if tens is None:
            messagebox.showwarning("提示","空白画布，无法保存。"); return
        label = simpledialog.askstring("标注","这是哪个数字？(0-9)")
        if label is None or not label.isdigit() or not (0<=int(label)<=9):
            messagebox.showwarning("提示","请输入 0—9 的数字。"); return
        # 保存 28×28 预处理结果
        out_dir = os.path.join("my_doodles", label); os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{int(time.time()*1000)}.png")
        prev.save(path)
        self.result.set(f"样本已保存：{path}")

def main():
    root = tk.Tk()
    App(root, MODEL_PATH)
    root.mainloop()

if __name__ == "__main__":
    main()
