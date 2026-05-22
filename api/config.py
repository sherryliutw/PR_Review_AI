"""
Centralized configuration for the Phase 2 API.

All settings are loaded from environment variables.
"""

from functools import cached_property
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # GitHub App credentials
    github_app_id: int
    github_private_key_path: str     # Path to PEM-encoded RSA private key
    github_webhook_secret: str

    # Database
    database_url: str                # e.g. postgresql+asyncpg://user:pass@host/db

    # Model
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    adapter_path: str = "./code-review-lora-adapter"
    skip_model_load: bool = False  # Set True for CPU dry-run (tests API without GPU)

    model_config = {"env_file": ".env", "extra": "ignore"}

    @cached_property
    def github_private_key(self) -> str:
        """Read the PEM private key from the file path."""
        return Path(self.github_private_key_path).read_text()


settings = Settings()
