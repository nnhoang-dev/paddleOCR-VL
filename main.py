import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from paddleocr import PaddleOCRVL

app = FastAPI(title="PaddleOCR-VL API")

# Load model once at startup (expensive operation)
pipeline = PaddleOCRVL(device="cpu")


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    # Save uploaded file to a temp path
    ext = os.path.splitext(file.filename)[-1]
    tmp_path = f"/tmp/{uuid.uuid4()}{ext}"

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        results = []
        for res in pipeline.predict(tmp_path):
            if hasattr(res, "save_to_markdown"):
                import tempfile
                import os as _os

                with tempfile.TemporaryDirectory() as tmpdir:
                    res.save_to_markdown(save_path=tmpdir)
                    md_files = [f for f in _os.listdir(tmpdir) if f.endswith(".md")]
                    if md_files:
                        with open(_os.path.join(tmpdir, md_files[0])) as f:
                            results.append({"markdown": f.read()})
                    else:
                        results.append({"markdown": ""})
            elif hasattr(res, "to_markdown"):
                md = res.to_markdown()
                if isinstance(md, dict):
                    md = md.get("markdown", "") or str(md)
                results.append({"markdown": md})
            else:
                if hasattr(res, "parsing_res_list"):
                    md = ""
                    for item in res.parsing_res_list:
                        if item.get("content"):
                            md += item["content"] + "\n"
                    results.append({"markdown": md})
                else:
                    results.append({"markdown": str(res)})
        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)  # Clean up temp file


@app.post("/ocr/file")
async def run_ocr_file(file: UploadFile = File(...)):
    # Save uploaded file to a temp path
    ext = os.path.splitext(file.filename)[-1]
    tmp_path = f"/tmp/{uuid.uuid4()}{ext}"

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        markdown_parts = []
        for res in pipeline.predict(tmp_path):
            if hasattr(res, "save_to_markdown"):
                import tempfile
                import os as _os

                with tempfile.TemporaryDirectory() as tmpdir:
                    res.save_to_markdown(save_path=tmpdir)
                    md_files = [f for f in _os.listdir(tmpdir) if f.endswith(".md")]
                    if md_files:
                        with open(_os.path.join(tmpdir, md_files[0])) as f:
                            markdown_parts.append(f.read())
            elif hasattr(res, "to_markdown"):
                md = res.to_markdown()
                if isinstance(md, dict):
                    md = md.get("markdown", "") or str(md)
                markdown_parts.append(md)
            else:
                if hasattr(res, "parsing_res_list"):
                    md = ""
                    for item in res.parsing_res_list:
                        if item.get("content"):
                            md += item["content"] + "\n"
                    markdown_parts.append(md)
                else:
                    markdown_parts.append(str(res))

        full_markdown = "\n\n---\n\n".join(markdown_parts)
        return Response(content=full_markdown, media_type="text/markdown")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)  # Clean up temp file


@app.get("/health")
def health():
    return {"status": "ok"}
