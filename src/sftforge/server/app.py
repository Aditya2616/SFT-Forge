"""FastAPI app: serves the web UI and runs the pipeline on pasted or uploaded JSONL.

Safety of live runs: input is size- and row-capped, parsed only as JSON data (nothing is executed),
held in memory (never written to disk), and config overrides cannot change which models are loaded.
"""
from __future__ import annotations

import io
import os
import tempfile
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .. import __version__, export, loaders, pipeline, runjson
from ..models import Models
from ..schema import Config

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
MAX_BYTES = int(os.environ.get("SFTFORGE_MAX_UPLOAD_BYTES", 2_000_000))
MAX_ROWS = int(os.environ.get("SFTFORGE_MAX_ROWS", 500))
LOCKED_SECTIONS = {"models", "export"}   # live users may tune thresholds, not choose models or export formats


class HealthOut(BaseModel):
    status: str
    version: str
    max_rows: int
    max_bytes: int


class RunIn(BaseModel):
    text: str = Field(description="JSONL or a JSON array in Alpaca, Dolly, chat or ShareGPT format")
    config: dict[str, Any] | None = Field(default=None, description="threshold overrides, same shape as configs/default.yaml")


class JobOut(BaseModel):
    id: str
    status: str            # queued | running | done | error
    stage: str = ""
    progress: float = 0.0
    error: str | None = None
    result: dict[str, Any] | None = None


class _Job:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.status, self.stage, self.progress = "queued", "", 0.0
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.pipeline_result: pipeline.Result | None = None
        self.cfg: Config | None = None


app = FastAPI(title="SFT Forge", version=__version__)
_jobs: dict[str, _Job] = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)
_models: Models | None = None
_models_cfg = Config.load(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else Config()


def _get_models() -> Models:
    global _models
    if _models is None:
        _models = Models(_models_cfg.models)
    return _models


def make_config(overrides: dict[str, Any] | None) -> Config:
    base = _models_cfg.model_dump()
    for k, v in (overrides or {}).items():
        if k in LOCKED_SECTIONS:
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    try:
        return Config.model_validate(base)
    except ValidationError as exc:
        raise HTTPException(422, detail=f"invalid config: {exc.errors()[0]['msg']}") from exc


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _execute(job: _Job, text: str, cfg: Config) -> None:
    try:
        job.status = "running"
        loaded = loaders.load_text(text)
        if loaded.df.empty:
            raise ValueError("no usable rows found. " + (loaded.errors[0] if loaded.errors else ""))
        if len(loaded.df) > MAX_ROWS:
            raise ValueError(f"{len(loaded.df)} rows is over the live limit of {MAX_ROWS}")

        def progress(stage: str, frac: float) -> None:
            job.stage, job.progress = stage, frac

        res = pipeline.run(loaded.df, cfg, _get_models(), progress, load_errors=len(loaded.errors))
        meta = {"name": "Your data", "kind": "live", "label": "Live run", "rows": len(loaded.df),
                "load_errors": loaded.errors[:5], "formats": loaded.fmt}
        job.result = runjson.build(res, meta)
        job.pipeline_result, job.cfg = res, cfg
        job.status = "done"
    except Exception as exc:  # surfaced to the client, not swallowed
        job.status, job.error = "error", str(exc)


def _submit(text: str, config: dict[str, Any] | None) -> JobOut:
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise HTTPException(413, detail=f"input is over the {MAX_BYTES} byte limit")
    cfg = make_config(config)
    job = _Job()
    with _lock:
        _jobs[job.id] = job
        while len(_jobs) > 12:
            _jobs.pop(next(iter(_jobs)))
    _pool.submit(_execute, job, text, cfg)
    return _out(job, with_result=False)


def _out(job: _Job, with_result: bool) -> JobOut:
    return JobOut(id=job.id, status=job.status, stage=job.stage, progress=job.progress, error=job.error,
                  result=job.result if with_result else None)


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok", version=__version__, max_rows=MAX_ROWS, max_bytes=MAX_BYTES)


@app.post("/api/run", response_model=JobOut)
def run_text(body: RunIn) -> JobOut:
    return _submit(body.text, body.config)


@app.post("/api/run/upload", response_model=JobOut)
async def run_upload(file: UploadFile = File(...)) -> JobOut:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, detail=f"file is over the {MAX_BYTES} byte limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, detail="file is not UTF-8 text") from exc
    return _submit(text, None)


@app.get("/api/run/{job_id}", response_model=JobOut)
def get_run(job_id: str) -> JobOut:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, detail="unknown run")
    return _out(job, with_result=job.status == "done")


@app.get("/api/run/{job_id}/export.zip")
def export_zip(job_id: str) -> Response:
    job = _jobs.get(job_id)
    if job is None or job.pipeline_result is None or job.cfg is None:
        raise HTTPException(404, detail="run not finished")
    cfg = job.cfg.model_copy(update={"export": job.cfg.export.model_copy(update={"hf_dataset": False})})
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        export.export(job.pipeline_result, cfg, tmp, source_name="pasted data")
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(Path(tmp).iterdir()):
                if p.is_file():
                    zf.write(p, p.name)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=sft-forge-export.zip"})


@app.get("/api/config")
def default_config() -> dict[str, Any]:
    return _models_cfg.model_dump()


@app.get("/api/example")
def example() -> Response:
    """The messy sample used by the README quickstart and the 'Load example' button."""
    path = ROOT / "examples" / "messy_sample.jsonl"
    if not path.exists():
        raise HTTPException(404, detail="example file not found")
    return Response(path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")


@app.get("/")
def index() -> Response:
    # the static demo never calls the API; only the served page is told a backend exists
    html = (DOCS / "index.html").read_text(encoding="utf-8").replace("<head>", '<head><meta name="sftforge-server" content="1">', 1)
    return Response(html, media_type="text/html")


if DOCS.exists():
    app.mount("/", StaticFiles(directory=DOCS), name="site")
