import os
import gc
import time
import torch
import threading
import psutil
from pathlib import Path

# ── Memory optimization trước mọi import nặng ─────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_TORCH_DTYPE"] = "float16"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"
os.environ["OMP_NUM_THREADS"]          = "1"

from paddleocr import PaddleOCRVL
import fitz  # pip install pymupdf

# ── RAM Monitor (background thread) ───────────────────────────────────────
class RAMMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.process  = psutil.Process(os.getpid())
        self.peak_ram  = 0.0   # MB
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                rss = self.process.memory_info().rss / 1024 / 1024  # MB
                if rss > self.peak_ram:
                    self.peak_ram = rss
            except psutil.NoSuchProcess:
                break
            time.sleep(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop_evt.set()
        self._thread.join()

    def report(self):
        # Cũng lấy VRAM nếu có GPU
        vram_info = ""
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024
            vram_reserv = torch.cuda.max_memory_reserved() / 1024 / 1024
            vram_info = (
                f"\n  🎮 VRAM allocated (peak): {vram_alloc:.1f} MB"
                f"\n  🎮 VRAM reserved  (peak): {vram_reserv:.1f} MB"
            )
        print(
            f"\n{'='*50}"
            f"\n📊 Memory Report:"
            f"\n  🧠 RAM  (peak RSS):       {self.peak_ram:.1f} MB"
            f"{vram_info}"
            f"\n{'='*50}"
        )

# ── Config ─────────────────────────────────────────────────────────────────
PDF_PATH   = "./picture.pdf"
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)
TMP_DIR    = Path("/tmp/pdf_pages")
TMP_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150  # Giảm xuống 120 nếu vẫn OOM; tăng 200 nếu cần chất lượng cao hơn

# ── Bắt đầu monitor ────────────────────────────────────────────────────────
monitor = RAMMonitor(interval=0.5).start()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()  # reset counter VRAM

# ── Load model ─────────────────────────────────────────────────────────────
print("⏳ Loading model...")
t0 = time.time()

pipeline = PaddleOCRVL(
    engine="transformers",
    device="gpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_chart_recognition=False,
)

print(f"✅ Model loaded ({time.time()-t0:.1f}s)")

# ── Export từng trang PDF → PNG ────────────────────────────────────────────
doc = fitz.open(PDF_PATH)
total_pages = len(doc)
print(f"📄 PDF có {total_pages} trang")

img_paths = []
for i, page in enumerate(doc):
    pix  = page.get_pixmap(dpi=DPI)
    path = TMP_DIR / f"page_{i:04d}.png"
    pix.save(str(path))
    img_paths.append(path)
    print(f"  → Exported page {i+1}/{total_pages}")
doc.close()

# ── Predict từng trang (giải phóng VRAM sau mỗi trang) ────────────────────
print("\n⏳ Processing pages...")
t1 = time.time()

pages_res = []
for i, img_path in enumerate(img_paths):
    try:
        for res in pipeline.predict(str(img_path)):
            pages_res.append(res)
        print(f"  → Page {i+1}/{total_pages} done")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"  ⚠️  OOM ở page {i+1}, thử giảm DPI hoặc dùng device='cpu' cho trang này")
            torch.cuda.empty_cache()
            gc.collect()
            for res in pipeline.predict(str(img_path)):
                pages_res.append(res)
            print(f"  → Page {i+1} done (CPU fallback)")
        else:
            raise

    torch.cuda.empty_cache()
    gc.collect()

print(f"✅ Predicted {len(pages_res)} pages ({time.time()-t1:.1f}s)")

# ── Restructure & Save ─────────────────────────────────────────────────────
print("\n⏳ Restructuring...")
result = pipeline.restructure_pages(
    pages_res,
    merge_tables=True,
    relevel_titles=True,
    concatenate_pages=True,
)

for res in result:
    res.save_to_markdown(save_path=output_dir)

# ── Cleanup tmp files ──────────────────────────────────────────────────────
for p in img_paths:
    p.unlink(missing_ok=True)

# ── Dừng monitor & in báo cáo ─────────────────────────────────────────────
monitor.stop()
monitor.report()

print(f"\n✅ Xong! ({time.time()-t0:.1f}s tổng)")
print(f"📄 Markdown: {output_dir}/")
print(f"🖼  Ảnh: {output_dir}/imgs/")