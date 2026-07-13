from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from separator import (
    _load_as_rgb_array,
    _separate_colors,
    generate_separation_pdf,
)

JOBS_DIR = Path("/tmp/jobs")
MAX_FILE_SIZE = 24 * 1024 * 1024  # 24 MB
JOB_TTL = 3600  # 1 hour
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
VALID_PAGE_SIZES = {"auto", "a3", "a4", "a5", "letter", "legal", "tabloid"}
VALID_ORIENTATIONS = {"auto", "portrait", "landscape"}

executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    executor.shutdown(wait=False)


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(300)
        _purge_old_jobs()


def _purge_old_jobs() -> None:
    now = time.time()
    if not JOBS_DIR.exists():
        return
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        meta_path = job_dir / "meta.json"
        try:
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if now - meta.get("created_at", 0) > JOB_TTL:
                    shutil.rmtree(job_dir, ignore_errors=True)
            else:
                shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)


app = FastAPI(title="Position Print Separator", lifespan=lifespan)

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_job(job_id: str) -> Path:
    job_dir = JOBS_DIR / job_id
    if not job_dir.is_dir() or not (job_dir / "meta.json").exists():
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return job_dir


def _extract_colors(input_path: str, max_colors: int) -> List[dict]:
    """Run color detection only (no vectorization) to get color metadata."""
    with tempfile.TemporaryDirectory() as workdir:
        rgb = _load_as_rgb_array(input_path, workdir)
        _, masks, inks = _separate_colors(rgb, max_colors=max_colors)
    return [
        {
            "hex": "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2])),
            "pixels": int(m.sum()),
        }
        for m, c in zip(masks, inks)
    ]


def _run_job(
    input_path: str,
    job_dir: Path,
    original_filename: str,
    max_colors: int,
    page_size: str,
    orientation: str,
) -> dict:
    """Blocking: run full separation pipeline, save PDF + metadata. Returns response dict."""

    colors = _extract_colors(input_path, max_colors)
    color_count = len(colors)

    if color_count == 0:
        raise ValueError("No ink colors detected in this design")

    pdf_bytes = generate_separation_pdf(
        input_path=input_path,
        output_path=None,
        max_colors=max_colors,
        page_size=page_size,
        orientation=orientation,
    )

    (job_dir / "output.pdf").write_bytes(pdf_bytes)

    page_count = color_count + 1  # N ink layers + 1 composite

    meta = {
        "job_id": job_dir.name,
        "original_filename": original_filename,
        "max_colors": max_colors,
        "page_size": page_size,
        "orientation": orientation,
        "color_count": color_count,
        "colors": colors,
        "page_count": page_count,
        "created_at": time.time(),
    }
    (job_dir / "meta.json").write_text(json.dumps(meta))

    return {
        "job_id": job_dir.name,
        "color_count": color_count,
        "colors": colors,
        "page_count": page_count,
    }


def _render_thumbnails(job_dir: Path, page_count: int) -> None:
    """Blocking: rasterize PDF pages to PNG thumbnails at 100 DPI."""
    from pdf2image import convert_from_path

    thumb_dir = job_dir / "thumbs"
    thumb_dir.mkdir(exist_ok=True)

    if (thumb_dir / f"thumb_{page_count}.png").exists():
        return  # already rendered

    pages = convert_from_path(str(job_dir / "output.pdf"), dpi=100)
    for i, page in enumerate(pages, 1):
        page.save(str(thumb_dir / f"thumb_{i}.png"))


def _validate_params(page_size: str, orientation: str) -> None:
    if page_size.lower() not in VALID_PAGE_SIZES:
        raise HTTPException(status_code=400, detail=f"Invalid page_size: {page_size!r}")
    if orientation.lower() not in VALID_ORIENTATIONS:
        raise HTTPException(status_code=400, detail=f"Invalid orientation: {orientation!r}")


def _wrap_job_error(e: Exception) -> HTTPException:
    msg = str(e)
    if isinstance(e, ValueError):
        return HTTPException(status_code=422, detail=msg)
    return HTTPException(status_code=500, detail=f"Separation failed: {msg}")


@app.get("/")
async def root():
    return {"status": "ok", "service": "Position Print Separator"}


@app.post("/api/separate")
async def separate(
    file: UploadFile = File(...),
    max_colors: int = Form(4),
    page_size: str = Form("auto"),
    orientation: str = Form("auto"),
):
    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: PNG, JPG, PDF, SVG",
        )
    _validate_params(page_size, orientation)

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 24 MB limit")

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    original_filename = file.filename or f"upload{ext}"
    input_path = job_dir / f"original{ext}"
    input_path.write_bytes(content)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            partial(
                _run_job,
                str(input_path),
                job_dir,
                original_filename,
                max_colors,
                page_size.lower(),
                orientation.lower(),
            ),
        )
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise _wrap_job_error(e)

    return result


@app.get("/api/preview/{job_id}")
async def preview(job_id: str):
    job_dir = _require_job(job_id)
    meta = json.loads((job_dir / "meta.json").read_text())

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            executor,
            partial(_render_thumbnails, job_dir, meta["page_count"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thumbnail rendering failed: {e}")

    thumbnails = [f"/api/thumb/{job_id}/{i}" for i in range(1, meta["page_count"] + 1)]
    return {"thumbnails": thumbnails, "colors": meta["colors"]}


@app.get("/api/thumb/{job_id}/{page}")
async def thumb(job_id: str, page: int):
    job_dir = _require_job(job_id)
    thumb_path = job_dir / "thumbs" / f"thumb_{page}.png"
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(str(thumb_path), media_type="image/png")


class RegenerateRequest(BaseModel):
    max_colors: int = 4
    page_size: str = "auto"
    orientation: str = "auto"


@app.post("/api/regenerate/{job_id}")
async def regenerate(job_id: str, body: RegenerateRequest):
    _validate_params(body.page_size, body.orientation)

    old_dir = _require_job(job_id)
    old_meta = json.loads((old_dir / "meta.json").read_text())

    original_ext = Path(old_meta["original_filename"]).suffix.lower()
    original_file = old_dir / f"original{original_ext}"
    if not original_file.exists():
        raise HTTPException(status_code=404, detail="Original file not found in job")

    new_id = str(uuid.uuid4())
    new_dir = JOBS_DIR / new_id
    new_dir.mkdir(parents=True)

    new_input = new_dir / f"original{original_ext}"
    shutil.copy(str(original_file), str(new_input))

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            partial(
                _run_job,
                str(new_input),
                new_dir,
                old_meta["original_filename"],
                body.max_colors,
                body.page_size.lower(),
                body.orientation.lower(),
            ),
        )
    except Exception as e:
        shutil.rmtree(new_dir, ignore_errors=True)
        raise _wrap_job_error(e)

    return result


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job_dir = _require_job(job_id)
    pdf_path = job_dir / "output.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")

    meta = json.loads((job_dir / "meta.json").read_text())
    stem = Path(meta["original_filename"]).stem
    filename = f"{stem}_separation.pdf"

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
