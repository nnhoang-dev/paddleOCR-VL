# import os

# # Tối ưu memory trước khi import torch/transformers
# os.environ["TRANSFORMERS_TORCH_DTYPE"] = "float16"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"   # tránh warning + deadlock trên macOS
# os.environ["OMP_NUM_THREADS"] = "1"              # giới hạn CPU threads cho numpy/torch

# from pathlib import Path
# from paddleocr import PaddleOCRVL
# import time

# PDF_PATH = "./picture.pdf"
# output_dir = Path("./output")
# output_dir.mkdir(parents=True, exist_ok=True)

# print("⏳ Loading model...")
# t0 = time.time()

# pipeline = PaddleOCRVL(
#     engine="transformers",
#     device="cpu",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_chart_recognition=False,
# )

# print(f"✅ Model loaded ({time.time()-t0:.1f}s)")

# # Predict — dùng generator, không list() ngay để tránh load hết PDF vào RAM
# print("⏳ Processing PDF...")
# t1 = time.time()

# pages_res = []
# for i, page in enumerate(pipeline.predict(PDF_PATH)):
#     pages_res.append(page)
#     print(f"  → Page {i+1} done")

# print(f"✅ Predicted {len(pages_res)} pages ({time.time()-t1:.1f}s)")

# # Restructure
# result = pipeline.restructure_pages(
#     pages_res,
#     merge_tables=True,
#     relevel_titles=True,
#     concatenate_pages=True
# )

# # Save
# for res in result:
#     res.save_to_markdown(save_path=output_dir)

# print(f"\n✅ Xong! ({time.time()-t0:.1f}s tổng)")
# print(f"📄 Markdown: {output_dir}/")
# print(f"🖼  Ảnh: {output_dir}/imgs/")
import os
import gc
import time
import torch
from pathlib import Path

# ── Memory optimization trước mọi import nặng ─────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_TORCH_DTYPE"] = "float16"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"
os.environ["OMP_NUM_THREADS"]          = "1"

from paddleocr import PaddleOCRVL
import fitz  # pip install pymupdf

# ── Config ─────────────────────────────────────────────────────────────────
PDF_PATH   = "./picture.pdf"
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)
TMP_DIR    = Path("/tmp/pdf_pages")
TMP_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150  # Giảm xuống 120 nếu vẫn OOM; tăng 200 nếu cần chất lượng cao hơn

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
            # Fallback: predict trang này trên CPU
            for res in pipeline.predict(str(img_path)):
                pages_res.append(res)
            print(f"  → Page {i+1} done (CPU fallback)")
        else:
            raise

    # Giải phóng VRAM sau mỗi trang
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

print(f"\n✅ Xong! ({time.time()-t0:.1f}s tổng)")
print(f"📄 Markdown: {output_dir}/")
print(f"🖼  Ảnh: {output_dir}/imgs/")