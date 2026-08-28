"""ASGI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import Settings, get_settings
from app.factory import build_pipeline
from app.ingest.loaders import SUPPORTED_SUFFIXES, load_path
from app.logging_config import configure_logging
from app.rag.pipeline import RagPipeline

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


async def _seed_if_empty(pipeline: RagPipeline, settings: Settings) -> None:
    """Index the seed corpus, but only into an empty store."""
    if not settings.seed_on_startup:
        return

    stats = await pipeline.store.stats()
    if stats["chunks"] > 0:
        return

    seed_dir = Path(settings.seed_path)
    if not seed_dir.is_absolute():
        seed_dir = Path(__file__).resolve().parent.parent / seed_dir
    if not seed_dir.is_dir():
        return

    for path in sorted(p for p in seed_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES):
        try:
            await pipeline.ingest(load_path(path))
        except ValueError:
            # A malformed seed file must never stop the service booting.
            continue


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    app.state.settings = settings
    app.state.pipeline = build_pipeline(settings)
    await _seed_if_empty(app.state.pipeline, settings)
    await app.state.pipeline.refresh_sparse_index()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="insight-rag",
        version="0.1.0",
        description=(
            "Retrieval-augmented generation with hybrid search, reranking, "
            "grounded citations and a reproducible eval harness."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")

    if UI_DIR.exists():
        app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(UI_DIR / "index.html")

    return app


app = create_app()
