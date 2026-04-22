import os
import gc
import time
import torch
import tracemalloc
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_TORCH_DTYPE"] = "float16"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"
os.environ["OMP_NUM_THREADS"]          = "1"

from paddleocr import PaddleOCRVL
import fitz

# ── Bắt đầu theo dõi RAM ───────────────────────────────────────────────────
tracemalloc.start()
torch.cuda.reset_peak_memory_stats()  # Reset VRAM peak counter

def mem_snapshot(label=""):
    current, peak = tracemalloc.get_traced_memory()
    vram_peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{label}] RAM current: {current/1024**2:.1f} MB | "
          f"RAM peak: {peak/1024**2:.1f} MB | "
          f"VRAM peak: {vram_peak:.2f} GB")

# ── Config ─────────────────────────────────────────────────────────────────
PDF_PATH   = "./picture.pdf"
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)
TMP_DIR    = Path("/tmp/pdf_pages")
TMP_DIR.mkdir(parents=True, exist_ok=True)
DPI = 150

# ── Load model ─────────────────────────────────────────────────────────────
print("⏳ Loading model...")
t0 = time.time()
pipeline = PaddleOCRVL(
    engine="transformers", device="gpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_chart_recognition=False,
)
mem_snapshot("After model load")

# ── Export PDF → PNG ───────────────────────────────────────────────────────
doc = fitz.open(PDF_PATH)
total_pages = len(doc)
img_paths = []
for i, page in enumerate(doc):
    pix  = page.get_pixmap(dpi=DPI)
    path = TMP_DIR / f"page_{i:04d}.png"
    pix.save(str(path))
    img_paths.append(path)
doc.close()
mem_snapshot("After PDF export")

# ── Predict từng trang ─────────────────────────────────────────────────────
t1 = time.time()
pages_res = []
for i, img_path in enumerate(img_paths):
    try:
        for res in pipeline.predict(str(img_path)):
            pages_res.append(res)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache(); gc.collect()
            for res in pipeline.predict(str(img_path)):
                pages_res.append(res)
        else:
            raise
    torch.cuda.empty_cache(); gc.collect()
    mem_snapshot(f"Page {i+1}/{total_pages}")  # In sau mỗi trang

# ── Restructure & Save ─────────────────────────────────────────────────────
result = pipeline.restructure_pages(
    pages_res, merge_tables=True,
    relevel_titles=True, concatenate_pages=True,
)
for res in result:
    res.save_to_markdown(save_path=output_dir)

mem_snapshot("FINAL")
tracemalloc.stop()

for p in img_paths:
    p.unlink(missing_ok=True)

print(f"\n✅ Xong! ({time.time()-t0:.1f}s)")