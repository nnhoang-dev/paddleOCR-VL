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

import torch


# Tối ưu memory trước khi import torch/transformers
os.environ["TRANSFORMERS_TORCH_DTYPE"] = "float16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # tránh warning + deadlock trên macOS
os.environ["OMP_NUM_THREADS"] = "1"              # giới hạn CPU threads cho numpy/torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from paddleocr import PaddleOCRVL
import time


PDF_PATH = "./picture.pdf"
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)


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


# Predict — dùng generator, không list() ngay để tránh load hết PDF vào RAM
print("⏳ Processing PDF...")
t1 = time.time()


pages_res = []
for i, page in enumerate(pipeline.predict(PDF_PATH)):
    pages_res.append(page)
    print(f"  → Page {i+1} done")
    torch.cuda.empty_cache()

print(f"✅ Predicted {len(pages_res)} pages ({time.time()-t1:.1f}s)")


# Restructure
result = pipeline.restructure_pages(
    pages_res,
    merge_tables=True,
    relevel_titles=True,
    concatenate_pages=True
)


# Save
for res in result:
    res.save_to_markdown(save_path=output_dir)


print(f"\n✅ Xong! ({time.time()-t0:.1f}s tổng)")
print(f"📄 Markdown: {output_dir}/")
print(f"🖼  Ảnh: {output_dir}/imgs/")