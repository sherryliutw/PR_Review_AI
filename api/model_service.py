"""
Model service — loads the fine-tuned model once at startup and exposes
a review generation method for the API.

Wraps training.model.load_for_inference() and training.inference.generate_review().
"""

import logging

from api.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """Singleton-style service: call load() once at startup, then review() per request."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def dry_run(self) -> bool:
        return settings.skip_model_load

    def load(self) -> None:
        """Load base model + LoRA adapter. Called once during FastAPI lifespan startup."""
        if self.dry_run:
            logger.info("SKIP_MODEL_LOAD=true — running in dry-run mode (no GPU required)")
            return

        from training.model import load_for_inference

        logger.info("Loading model with adapter from %s", settings.adapter_path)
        self.model, self.tokenizer = load_for_inference(
            adapter_path=settings.adapter_path,
        )
        logger.info("Model loaded successfully")

    def review(self, before_code: str, after_code: str, file_path: str) -> str:
        """Generate a code review for a before/after code pair."""
        if self.dry_run:
            return f"[DRY RUN] Model inference skipped for {file_path}"

        if not self.is_loaded:
            raise RuntimeError("Model not loaded — call load() first")

        from training.inference import generate_review

        return generate_review(
            model=self.model,
            tokenizer=self.tokenizer,
            before_code=before_code,
            after_code=after_code,
            file_path=file_path,
        )


model_service = ModelService()
