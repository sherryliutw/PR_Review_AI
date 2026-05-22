"""
FastAPI application entry point.

Startup: loads fine-tuned model + initializes database.
Routes: /webhook, /health, /reviews
"""

import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request

from api.database import close_db, init_db
from api.model_service import model_service
from api.pipeline import get_reviews, review_pipeline
from api.webhook import parse_pr_event, verify_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model and initializing database")
    model_service.load()
    await init_db()
    yield
    logger.info("Shutting down")
    await close_db()


app = FastAPI(
    title="PR Review AI",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """
    GitHub webhook endpoint.
    Verifies signature, parses event, dispatches review pipeline in background.
    """
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    verify_signature(body, signature)

    payload = await request.json()
    pr_event = parse_pr_event(request, payload)

    if pr_event is None:
        return {"status": "ignored"}

    background_tasks.add_task(review_pipeline, **pr_event)
    return {"status": "accepted"}


@app.get("/health")
async def health():
    """Health check for K8s liveness/readiness probes."""
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded,
        "dry_run": model_service.dry_run,
    }


@app.get("/reviews/{repo_owner}/{repo_name}")
async def list_reviews(
    repo_owner: str,
    repo_name: str,
    pr_number: int | None = None,
):
    """Query review history. Optional filter by PR number."""
    repo_full_name = f"{repo_owner}/{repo_name}"
    return await get_reviews(repo_full_name, pr_number)
