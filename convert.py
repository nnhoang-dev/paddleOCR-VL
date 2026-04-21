import logging
from pathlib import Path
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

PDF_PATH = "image.png"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ⚠️ Phải bật 2 dòng này — mặc định Docling KHÔNG generate ảnh
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 2.0          # độ phân giải ảnh (2.0 = 144 DPI)
pipeline_options.generate_picture_images = True  # bật trích xuất figure/ảnh

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

conv_res = doc_converter.convert(PDF_PATH)
doc_filename = conv_res.input.file.stem

# Xuất markdown — ảnh lưu ra folder imgs/ bên cạnh file .md
md_file = OUTPUT_DIR / f"{doc_filename}.md"
conv_res.document.save_as_markdown(
    md_file,
    image_mode=ImageRefMode.REFERENCED  # ảnh lưu file riêng, chèn ![](imgs/...) vào md
)

print(f"✅ {md_file}")

# from pathlib import Path

# from paddleocr import PaddleOCRVL

# output_dir = Path("./output")
# output_dir.mkdir(parents=True, exist_ok=True)

# # NVIDIA GPU
# pipeline = PaddleOCRVL()
# # Kunlunxin XPU
# # pipeline = PaddleOCRVL(device="xpu")
# # Hygon DCU
# # pipeline = PaddleOCRVL(device="dcu")
# # MetaX GPU
# # pipeline = PaddleOCRVL(device="metax_gpu")
# # Apple Silicon
# # pipeline = PaddleOCRVL(device="cpu")
# # Huawei Ascend NPU 
# # Huawei Ascend NPU please refer to Chapter 3 for inference using PaddlePaddle + vLLM

# # pipeline = PaddleOCRVL(use_doc_orientation_classify=True) # Use use_doc_orientation_classify to enable/disable document orientation classification model
# # pipeline = PaddleOCRVL(use_doc_unwarping=True) # Use use_doc_unwarping to enable/disable document unwarping module
# # pipeline = PaddleOCRVL(use_layout_detection=False) # Use use_layout_detection to enable/disable layout detection module

# output = pipeline.predict("./picture.pdf")
# for res in output:
#     res.save_to_markdown(save_path=output_dir) ## Save the current image's result in Markdown format