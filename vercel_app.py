from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from detector import analyze, load_model


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
app = FastAPI(title="Verity News Screening", docs_url=None, redoc_url=None)


class ArticleRequest(BaseModel):
    text: str = Field(max_length=50_000)


@lru_cache(maxsize=1)
def get_model():
    return load_model()


@app.get("/api/health")
def health():
    model = get_model()
    return {"status": "ok", "mode": "trained model" if model is not None else "language-pattern screening"}


@app.post("/api/analyze")
def analyze_article(request: ArticleRequest):
    try:
        result = analyze(request.text, get_model())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return result.to_dict()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
