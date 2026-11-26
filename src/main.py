# -*- coding: utf-8 -*-
import os
import sys
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch.nn as nn

# مسیر مدل‌ها برای اجرای عادی و نسخه exe (PyInstaller)
if getattr(sys, "frozen", False):
    BASE = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    MODEL_DIR = os.path.join(BASE, "models")

    # سازگاری با نسخه‌های قبلی exe
    if not os.path.isdir(MODEL_DIR):
        MODEL_DIR = BASE
else:
    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

PERSIAN_MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.pt")
PERSIAN_V2_PATH = os.path.join(MODEL_DIR, "digit_model_v2.pt")
ENGLISH_V2_PATH = os.path.join(MODEL_DIR, "english_digit_model_v2.pt")
USE_V2 = os.path.isfile(PERSIAN_V2_PATH) and os.path.isfile(ENGLISH_V2_PATH)
LANG_V2_PATH = os.path.join(MODEL_DIR, "lang_model_v2.pt")
REVERSE_PERSIAN = False
ENGLISH_MODEL_PATH = os.path.join(MODEL_DIR, "english_digit_model.pt")
DEVICE = torch.device("cpu")
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
MIN_CONF = 0.35   # تشخیص‌های کم‌اطمینان‌تر نادیده گرفته می‌شوند
MODE_AUTO, MODE_FA, MODE_EN = "تشخیص خودکار", "فارسی", "English"


class PersianNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )


class EnglishNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class LangNet(nn.Sequential):
    """مدل ۲۰ کلاسه: ۰..۹ فارسی، ۱۰..۱۹ انگلیسی (فقط برای تشخیص زبان)."""
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 20),
        )


def load_model(path, cls):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"فایل مدل پیدا نشد:\n{path}")
    m = cls().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


# ---------------------------------------------------------------------
# پیش‌پردازش (اصلاح‌شده)
# ---------------------------------------------------------------------
def make_binary(gray):
    """مثل آموزش مدل: Otsu؛ رقم سفید روی زمینه سیاه. فقط اگر نتیجه خراب بود از آستانه محلی استفاده می‌شود."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    if min(gray.shape) >= 150:   # تصویر بزرگ: نویز زمینه را بیشتر صاف کن
        blur = cv2.medianBlur(cv2.GaussianBlur(gray, (5, 5), 0), 5)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 127:
        th = 255 - th
    if th.mean() > 0.40 * 255:   # بیش از ۴۰٪ سفید = Otsu شکست خورده
        h, w = gray.shape
        block = max(51, (min(h, w) // 4) | 1)
        d = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, block, 12)
        l = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block, -12)
        th = d if d.mean() <= l.mean() else l
    return th


def merge_boxes(boxes):
    """ادغام فقط اگر هم در x هم‌پوشان و در y نزدیک باشند (نقطه/بدنه یک رقم)."""
    boxes = [list(map(int, b)) for b in boxes]
    changed = True
    while changed:
        changed = False
        merged = []
        for b in boxes:
            for o in merged:
                ox = min(b[2], o[2]) - max(b[0], o[0])
                min_w = min(b[2] - b[0], o[2] - o[0])
                gap_y = max(b[1], o[1]) - min(b[3], o[3])
                min_h = min(b[3] - b[1], o[3] - o[1])
                if min_w > 0 and ox > 0.6 * min_w and gap_y < 0.3 * max(min_h, 1):
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    changed = True
                    break
            else:
                merged.append(b[:])
        boxes = merged
    return boxes


def group_rows(boxes):
    """گروه‌بندی جعبه‌ها در ردیف‌ها (بالا به پایین)."""
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    rows = []
    for b in boxes:
        cy = (b[1] + b[3]) / 2
        for row in rows:
            y0 = min(x[1] for x in row)
            y1 = max(x[3] for x in row)
            if y0 <= cy <= y1:
                row.append(b)
                break
        else:
            rows.append([b])
    rows.sort(key=lambda r: sum((x[1] + x[3]) / 2 for x in r) / len(r))
    return [sorted(r, key=lambda b: b[0]) for r in rows]


def segment_digits(image):
    if image is None:
        return None, []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    binary = make_binary(gray)
    H, W = binary.shape

    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = []
    for x, y, w, h, area in stats[1:]:
        if area < 12:
            continue
        fill = area / float(w * h)
        # کادر: تقریباً هم‌اندازه تصویر و تو‌خالی
        if w > 0.85 * W and h > 0.85 * H and fill < 0.15:
            continue
        # خط افقی/عمودی بلند و باریک (جدول)
        if (w > 0.5 * W and h < 0.04 * H) or (h > 0.5 * H and w < 0.04 * W and h > 15 * w):
            continue
        comps.append((int(x), int(y), int(x + w), int(y + h), int(area)))
    if not comps:
        return binary, []

    # ارتفاع مرجع رقم‌ها: میانه ارتفاع بزرگ‌ترین نیمه اجزا (از روی مساحت)
    comps.sort(key=lambda c: c[4], reverse=True)
    top = comps[:max(1, min(5, len(comps)))]
    hs = sorted(c[3] - c[1] for c in top)
    ref_h = hs[len(hs) // 2]
    max_area = comps[0][4]
    kept, small = [], []
    for c in comps:
        w, h = c[2] - c[0], c[3] - c[1]
        if 0.15 * ref_h <= h < 0.5 * ref_h and 0.6 <= w / float(h) <= 1.6 \
                and c[4] / float(w * h) > 0.6:
            small.append(c)     # کاندید «نقطه» (صفر فارسی)
        if h < 0.5 * ref_h or h > 2.0 * ref_h:      # لکه و نویز زمینه
            continue
        if c[4] < 0.04 * max_area:                  # مساحت خیلی کم نسبت به ارقام اصلی
            continue
        if w > 1.6 * h or h > 12 * w:               # شکل غیرعادی برای رقم
            continue
        kept.append(c)
    if not kept:
        return binary, []
    # نقطه فقط وقتی پذیرفته می‌شود که کنار یک رقم و حدود وسط ارتفاع آن باشد
    for c in small:
        cy = (c[1] + c[3]) / 2.0
        for k in kept:
            kh = k[3] - k[1]
            gap = max(k[0] - c[2], c[0] - k[2], 0)
            if gap < 1.2 * ref_h and k[1] + 0.3 * kh <= cy <= k[1] + 0.8 * kh:
                kept.append(c)
                break
    boxes = [list(c[:4]) for c in kept]
    boxes = merge_boxes(boxes)
    return binary, [tuple(b) for b in boxes]


def prepare_digit(binary, box):
    x0, y0, x1, y1 = box
    crop = binary[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    h, w = crop.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    small = cv2.resize(sq, (20, 20), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.uint8)
    out[4:24, 4:24] = small
    return out


def predict_one(img28, model, english=False):
    arr = img28.astype(np.float32) / 255.0
    if english and not USE_V2:   # مدل قدیمی انگلیسی با Normalize آموزش دیده بود
        arr = (arr - 0.1307) / 0.3081
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    conf, idx = probs.max(0)
    return int(idx.item()), float(conf.item())


def predict_joint(img28, model):
    t = torch.from_numpy(img28.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.softmax(model(t), dim=1)[0].numpy()


def decide_language(joint_probs):
    """جمع احتمال فارسی در برابر انگلیسی روی همه ارقام یک ردیف.
    ارقام هم‌شکل (مثل ۱ و 1) احتمال را نصف می‌کنند و تأثیری ندارند."""
    fa = sum(float(p[:10].sum()) for p in joint_probs)
    en = sum(float(p[10:].sum()) for p in joint_probs)
    return "فارسی" if fa >= en else "English"


def recognize(image, fa_model, en_model, mode=MODE_AUTO, lang_model=None):
    binary, boxes = segment_digits(image)
    output = image.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    if not boxes:
        return [], "رقمی پیدا نشد", output

    prepared = []
    for box in boxes:
        p = prepare_digit(binary, box)
        if p is not None:
            prepared.append((tuple(box), p))

    # حالت خودکار: زبان هر ردیف با مدل ۲۰ کلاسه تعیین می‌شود
    row_lang = {}
    if mode == MODE_AUTO and lang_model is not None and prepared:
        joint = {b: predict_joint(p, lang_model) for b, p in prepared}
        for row in group_rows([b for b, _ in prepared]):
            lang = decide_language([joint[b] for b in row])
            for b in row:
                row_lang[b] = lang

    results = []
    for box, p in prepared:
        if mode == MODE_FA:
            lang = "فارسی"
        elif mode == MODE_EN:
            lang = "English"
        else:
            lang = row_lang.get(box)

        if lang == "فارسی":
            d, c = predict_one(p, fa_model)
        elif lang == "English":
            d, c = predict_one(p, en_model, True)
        else:   # مدل زبان موجود نیست: روش قدیمی مقایسه اطمینان
            pd_, pc = predict_one(p, fa_model)
            ed, ec = predict_one(p, en_model, True)
            if pc >= ec:
                d, c, lang = pd_, pc, "فارسی"
            else:
                d, c, lang = ed, ec, "English"

        if c < MIN_CONF:       # اطمینان خیلی کم: با کادر قرمز نشان بده ولی در نتیجه نیاور
            x0, y0, x1, y1 = box
            cv2.rectangle(output, (x0, y0), (x1, y1), (60, 60, 230), 2)
            cv2.putText(output, "?", (x0, max(18, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 230), 2)
            continue
        shown = PERSIAN_DIGITS[d] if lang == "فارسی" else str(d)
        color = (40, 200, 80) if lang == "فارسی" else (255, 160, 30)
        x0, y0, x1, y1 = box
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
        cv2.putText(output, str(d), (x0, max(18, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        results.append({"digit": d, "text": shown, "confidence": c,
                        "language": lang, "box": box})

    # ردیف‌به‌ردیف، هر ردیف از چپ به راست
    by_box = {r["box"]: r for r in results}
    lines, ordered = [], []
    for row in group_rows([r["box"] for r in results]):
        row_res = [by_box[b] for b in row]
        if REVERSE_PERSIAN and all(r["language"] == "فارسی" for r in row_res):
            row_res = row_res[::-1]
        ordered.extend(row_res)
        lines.append("".join(r["text"] for r in row_res))
    return ordered, "\n".join(lines), output


# ---------------------------------------------------------------------
# رابط گرافیکی
# ---------------------------------------------------------------------
class DigitApp:
    def __init__(self, root):
        self.root = root
        root.title("تشخیص ارقام فارسی و انگلیسی")
        root.geometry("1100x800")
        root.configure(bg="#111827")
        self.original = None
        self.fa_model = None
        self.en_model = None
        self.lang_model = None
        self.orig_photo = None
        self.det_photo = None
        self.mode_var = tk.StringVar(value=MODE_AUTO)
        self.build()
        self.load_models()

    def build(self):
        r = self.root
        tk.Label(r, text="سیستم تشخیص ارقام فارسی و انگلیسی",
                 font=("Tahoma", 20, "bold"), bg="#111827", fg="white").pack(pady=12)
        bar = tk.Frame(r, bg="#111827")
        bar.pack(pady=8)
        for text, cmd, color in [
            ("انتخاب تصویر", self.open_image, "#2563EB"),
            ("تشخیص مجدد", self.process, "#059669"),
            ("نمایش تصویر اصلی", self.show_original, "#7C3AED"),
            ("پاک کردن", self.clear, "#DC2626"),
        ]:
            tk.Button(bar, text=text, command=cmd, bg=color, fg="white",
                      font=("Tahoma", 11, "bold"), padx=14, pady=8).pack(side="right", padx=5)

        mf = tk.Frame(r, bg="#111827")
        mf.pack(pady=4)
        tk.Label(mf, text="زبان ارقام:", bg="#111827", fg="white",
                 font=("Tahoma", 11)).pack(side="right", padx=6)
        ttk.Combobox(mf, textvariable=self.mode_var, state="readonly",
                     values=[MODE_AUTO, MODE_FA, MODE_EN], width=18).pack(side="right")

        imgs = tk.Frame(r, bg="#111827")
        imgs.pack(fill="both", expand=True, padx=18, pady=8)
        self.lbl_orig = self._panel(imgs, "تصویر اصلی")
        self.lbl_det = self._panel(imgs, "تصویر بررسی‌شده")

        rf = tk.LabelFrame(r, text="نتیجه", bg="#1F2937", fg="white",
                           font=("Tahoma", 12, "bold"), padx=10, pady=6)
        rf.pack(fill="x", padx=25, pady=6)
        self.result_var = tk.StringVar(value="—")
        tk.Label(rf, textvariable=self.result_var, bg="#1F2937", fg="#34D399",
                 font=("Tahoma", 20, "bold"), justify="center").pack(pady=4)

        self.details = tk.Text(r, height=7, bg="#1F2937", fg="white", font=("Consolas", 10))
        self.details.pack(fill="x", padx=25, pady=6)
        self.status = tk.Label(r, text="در حال بارگذاری...", bg="#111827",
                               fg="#9CA3AF", font=("Tahoma", 10))
        self.status.pack(pady=4)

    def _panel(self, parent, title):
        f = tk.LabelFrame(parent, text=title, bg="#1F2937", fg="white",
                          font=("Tahoma", 11, "bold"), padx=6, pady=6)
        f.pack(side="right", fill="both", expand=True, padx=6)
        lbl = tk.Label(f, bg="#1F2937", fg="white")
        lbl.pack(fill="both", expand=True)
        return lbl

    def load_models(self):
        try:
            if USE_V2:
                self.fa_model = load_model(PERSIAN_V2_PATH, PersianNet)
                self.en_model = load_model(ENGLISH_V2_PATH, PersianNet)
                self.status.config(text="مدل‌های جدید (v2) بارگذاری شدند")
                if os.path.isfile(LANG_V2_PATH):
                    self.lang_model = load_model(LANG_V2_PATH, LangNet)
                    self.status.config(text="مدل‌های جدید (v2) + مدل تشخیص زبان بارگذاری شدند")
            else:
                self.fa_model = load_model(PERSIAN_MODEL_PATH, PersianNet)
                self.en_model = load_model(ENGLISH_MODEL_PATH, EnglishNet)
                self.status.config(text="مدل‌های قدیمی بارگذاری شدند")
        except Exception as e:
            self.status.config(text="خطا در بارگذاری مدل")
            messagebox.showerror("خطای مدل", str(e))

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")])
        if not path:
            return
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("تصویر قابل خواندن نیست")
            self.original = img
            self.process()
        except Exception as e:
            messagebox.showerror("خطا", str(e))

    def _photo(self, img, size=(520, 390)):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 \
            else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(size)
        return ImageTk.PhotoImage(pil)

    def show(self, orig, det):
        self.orig_photo = self._photo(orig)
        self.det_photo = self._photo(det)
        self.lbl_orig.config(image=self.orig_photo)
        self.lbl_det.config(image=self.det_photo)

    def process(self):
        if self.original is None:
            messagebox.showwarning("توجه", "ابتدا یک تصویر انتخاب کن")
            return
        if self.fa_model is None or self.en_model is None:
            messagebox.showerror("خطا", "مدل‌ها بارگذاری نشده‌اند")
            return
        try:
            results, text, out = recognize(self.original, self.fa_model,
                                           self.en_model, self.mode_var.get(),
                                           self.lang_model)
            self.show(self.original, out)
            self.details.delete("1.0", tk.END)
            self.result_var.set(text)
            for i, r in enumerate(results, 1):
                self.details.insert(
                    tk.END, f"رقم {i}: {r['text']} | زبان: {r['language']} | "
                            f"اطمینان: {r['confidence'] * 100:.1f}%\n")
            if results:
                avg = sum(r["confidence"] for r in results) / len(results)
                langs = "، ".join(sorted({r["language"] for r in results}))
                self.status.config(text=f"{len(results)} رقم | زبان: {langs} | میانگین اطمینان: {avg * 100:.1f}%")
            else:
                self.status.config(text="هیچ رقمی شناسایی نشد")
        except Exception as e:
            messagebox.showerror("خطای تشخیص", str(e))

    def show_original(self):
        if self.original is not None:
            self.show(self.original, self.original)

    def clear(self):
        self.original = None
        self.result_var.set("—")
        self.details.delete("1.0", tk.END)
        self.lbl_orig.config(image="")
        self.lbl_det.config(image="")
        self.orig_photo = self.det_photo = None
        self.status.config(text="آماده")


if __name__ == "__main__":
    root = tk.Tk()
    DigitApp(root)
    root.mainloop()
