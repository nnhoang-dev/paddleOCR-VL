import os
import shutil
import uuid
import boto3
import re
import mimetypes
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCRVL

load_dotenv()

app = FastAPI(title="PaddleOCR-VL API")

pipeline = PaddleOCRVL(device="cpu")


@app.get("/health")
async def health_check():
    return {"status": "ok"}

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    endpoint_url=os.getenv("S3_ENDPOINT_URL"),
)

S3_BUCKET = os.getenv("S3_BUCKET", "pdf-markdown-converter")
S3_BASE_URL = os.getenv("S3_BASE_URL", f"https://{S3_BUCKET}.s3.amazonaws.com")
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL")
S3_ACL = os.getenv("S3_ACL", "public-read")


def build_public_object_url(key: str) -> str:
    normalized_key = key.lstrip("/")

    if S3_PUBLIC_BASE_URL:
        return f"{S3_PUBLIC_BASE_URL.rstrip('/')}/{normalized_key}"

    return f"{S3_BASE_URL.rstrip('/')}/{normalized_key}"


def upload_to_s3(file_path: str, key: str) -> str:
    try:
        with open(file_path, "rb") as f:
            file_content = f.read()

        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=file_content,
            ContentType=content_type,
            ACL=S3_ACL,
        )
        return build_public_object_url(key)
    except Exception as e:
        print(f"S3 upload error: {e}")
        raise


def build_uploaded_image_url_map(images_dirs: list[str]) -> dict[str, str]:
    image_url_map: dict[str, str] = {}

    for images_dir in images_dirs:
        if not images_dir or not os.path.exists(images_dir):
            print(f"Images dir does not exist: {images_dir}")
            continue

        image_files = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
        ]

        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            s3_key = f"uploads/{uuid.uuid4()}_{img_file}"
            s3_url = upload_to_s3(img_path, s3_key)
            image_url_map[img_file] = s3_url

    return image_url_map


def replace_images_with_s3_urls(markdown: str, image_url_map: dict[str, str]) -> str:
    if not image_url_map:
        return markdown

    def _resolve_local_src(src: str) -> str | None:
        if src.startswith(("http://", "https://", "data:")):
            return None
        normalized = src.strip().replace("\\", "/")
        basename = os.path.basename(normalized)
        return image_url_map.get(basename)

    # Replace markdown image syntax: ![alt](path)
    markdown = re.sub(
        r"(!\[[^\]]*\]\()([^\)]+)(\))",
        lambda m: f"{m.group(1)}{_resolve_local_src(m.group(2)) or m.group(2)}{m.group(3)}",
        markdown,
    )

    # Replace html image syntax: <img src="path" ...>
    markdown = re.sub(
        r"(<img\b[^>]*\bsrc=[\"'])([^\"']+)([\"'])",
        lambda m: f"{m.group(1)}{_resolve_local_src(m.group(2)) or m.group(2)}{m.group(3)}",
        markdown,
        flags=re.IGNORECASE,
    )

    return markdown


@app.post("/api/v1/ocr/convert-to-markdown")
async def convert_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    tmp_dir = f"/tmp/{uuid.uuid4()}"
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        pdf_path = os.path.join(tmp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        markdown_parts = []
        all_images_dirs = []

        for res in pipeline.predict(pdf_path):
            if hasattr(res, "save_to_markdown"):
                res.save_to_markdown(save_path=tmp_dir)

                for filename in os.listdir(tmp_dir):
                    if filename.endswith(".md"):
                        md_path = os.path.join(tmp_dir, filename)
                        with open(md_path, "r", encoding="utf-8") as f:
                            markdown_parts.append(f.read())

                        img_dir = os.path.join(tmp_dir, "imgs")
                        if os.path.exists(img_dir):
                            all_images_dirs.append(img_dir)
                        break
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

        image_url_map = build_uploaded_image_url_map(all_images_dirs)
        full_markdown = replace_images_with_s3_urls(full_markdown, image_url_map)

        return JSONResponse({"markdown": full_markdown})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
