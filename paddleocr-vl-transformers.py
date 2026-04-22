from pathlib import Path

from paddleocr import PaddleOCRVL

output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

pipeline = PaddleOCRVL(engine="transformers")
output = pipeline.predict("./picture.pdf")
for res in output:
    res.save_to_markdown(save_path=output_dir) ## Save the current image's result in Markdown format