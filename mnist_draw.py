# mnist_draw.py
import os
import io
import torch
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageOps, ImageGrab, ImageChops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageGrab, ImageChops, ImageFilter
from tkinter import simpledialog
import time

MODEL_PATH = r"runs_mnist\best_model.pt"

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
# 新增：与 train_fast.py 一致的 FastCNN
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

# 新增：BetterCNN（与 train_digits_aug.py 一致）
class BetterCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DrawApp:
    def __init__(self, root, model_path=MODEL_PATH):
        self.root = root
        self.root.title("MNIST 手写识别")
        self.last_x, self.last_y = None, None

        self.canvas_size = 300
        self.brush_width = 18
        self.fg = "black"
        self.bg = "white"

        # 画布
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg=self.bg, cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_pos)

        # 按钮
        tk.Button(root, text="清空", width=10, command=self.clear).grid(row=1, column=0, pady=6)
        tk.Button(root, text="识别", width=10, command=self.predict_canvas).grid(row=1, column=1, pady=6)
        tk.Button(root, text="退出", width=10, command=root.quit).grid(row=1, column=2, pady=6)
        tk.Button(root, text="保存样本", width=10, command=self.save_sample).grid(row=1, column=3, pady=6)
        # 并把右侧结果标签换到第4列：
        self.result_var = tk.StringVar(value="结果：-")
        tk.Label(root, textvariable=self.result_var, font=("Microsoft YaHei", 16)).grid(row=1, column=4, padx=8)

        self.result_var = tk.StringVar(value="结果：-")
        tk.Label(root, textvariable=self.result_var, font=("Microsoft YaHei", 16)).grid(row=1, column=3, padx=8)

        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.mean, self.std, self.classes = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, path):
        if not os.path.exists(path):
            messagebox.showerror("错误", f"未找到模型：{path}\n请先训练生成 best_model.pt。")
            raise FileNotFoundError(path)

        state = torch.load(path, map_location="cpu")
        sd = state["model"] if (isinstance(state, dict) and "model" in state) else state
        keys = list(sd.keys())

        # 依次判断权重键名，选择对应结构
        if any(k.startswith("features.") or k.startswith("classifier.") for k in keys):
            model = BetterCNN();
            arch = "BetterCNN"
        elif any(k.startswith("conv1.") for k in keys):
            model = SmallCNN();
            arch = "SmallCNN"
        elif any(k.startswith("conv.") or k.startswith("head.") for k in keys):
            model = FastCNN();
            arch = "FastCNN"
        else:
            # 兜底：依次尝试三种模型
            for cls, name in [(BetterCNN, "BetterCNN"), (SmallCNN, "SmallCNN"), (FastCNN, "FastCNN")]:
                m = cls()
                try:
                    m.load_state_dict(sd, strict=True)
                    model, arch = m, name
                    break
                except Exception:
                    continue
            else:
                messagebox.showerror("错误", "无法识别的模型权重结构。")
                raise RuntimeError("unknown checkpoint format")

        model.load_state_dict(sd, strict=True)

        mean = tuple(state.get("normalize_mean", (0.1307,)))
        std = tuple(state.get("normalize_std", (0.3081,)))
        classes = state.get("classes", list(range(10)))

        print(f"[Loader] 使用架构: {arch}  |  来自: {path}")
        return model, mean, std, classes

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill=self.fg, width=self.brush_width, capstyle=tk.ROUND, smooth=True)
        self.last_x, self.last_y = event.x, event.y

    def reset_pos(self, _):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.result_var.set("结果：-")

    def _get_canvas_image(self):
        # 抓取画布
        self.root.update()
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")  # 灰度

        # 反相：把“黑字白底”变成“白字黑底”，符合 MNIST
        img = ImageOps.invert(img)

        # 二值化（自适应阈值，抗屏幕灰度）
        arr = np.array(img, dtype=np.uint8)
        thr = max(10, int(arr.mean() * 0.9))  # 宽松点
        bin_ = (arr > thr).astype(np.uint8) * 255  # 前景=255

        # 找前景框
        ys, xs = np.where(bin_ > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # 裁剪到前景
        crop = Image.fromarray(arr[y_min:y_max + 1, x_min:x_max + 1])

        # 等比缩放：最长边=20（MNIST 的做法）
        w, h = crop.size
        if w > h:
            new_w, new_h = 20, max(1, int(round(20 * h / w)))
        else:
            new_h, new_w = 20, max(1, int(round(20 * w / h)))
        crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 放到 28×28 黑底中心，先粗略居中
        canvas = Image.new("L", (28, 28), 0)
        left = (28 - new_w) // 2
        top = (28 - new_h) // 2
        canvas.paste(crop, (left, top))

        # 轻微高斯模糊，贴近 MNIST 的笔画“发灰感”
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.6))

        # 质心微调：把前景质心挪到(14,14)
        a = np.array(canvas, dtype=np.float32)
        a /= 255.0
        yy, xx = np.mgrid[0:28, 0:28]
        m = a.sum() + 1e-6
        cy = (yy * a).sum() / m
        cx = (xx * a).sum() / m
        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))
        canvas = Image.fromarray(
            np.roll(np.roll((a * 255).astype(np.uint8), shift_y, axis=0), shift_x, axis=1)
        )

        # 转 tensor：标准化要与训练一致
        a = np.array(canvas, dtype=np.float32) / 255.0
        a = (a - self.mean[0]) / self.std[0]
        a = a[None, None, :, :]
        return torch.from_numpy(a)

    def predict_canvas(self):
        tensor = self._get_canvas_image()
        if tensor is None:
            self.result_var.set("结果：空白")
            return
        tensor = tensor.to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            prob = torch.softmax(logits, dim=1)[0]
            pred = int(prob.argmax().item())
            conf = float(prob.max().item())

        self.result_var.set(f"结果：{pred}（置信度 {conf*100:.1f}%）")

    def save_sample(self):
        tensor = self._get_canvas_image()
        if tensor is None:
            messagebox.showwarning("提示", "空白画布，无法保存。")
            return
        # 询问标签
        label = simpledialog.askstring("标注", "这是什么数字？(0-9)")
        if label is None or not label.isdigit() or not (0 <= int(label) <= 9):
            messagebox.showwarning("提示", "请输入 0-9 的数字。")
            return
        # 反标准化 -> 0~255
        a = tensor.squeeze().cpu().numpy()
        a = a * self.std[0] + self.mean[0]
        a = np.clip(a, 0, 1)
        arr = (a * 255).astype(np.uint8)
        img28 = Image.fromarray(arr, mode="L")
        # 保存
        outdir = os.path.join("my_doodles", str(label))
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{int(time.time()*1000)}.png")
        img28.save(path)
        self.result_var.set(f"样本已保存：{path}")




def main():
    root = tk.Tk()
    app = DrawApp(root, model_path=MODEL_PATH)
    root.mainloop()




if __name__ == "__main__":
    main()
