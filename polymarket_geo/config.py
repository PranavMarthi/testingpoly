"""
Central configuration loaded from environment variables with sensible defaults.
All secrets come from env vars; no hardcoded credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = os.getenv("PG_HOST", "localhost")
    port: int = int(os.getenv("PG_PORT", "5432"))
    user: str = os.getenv("PG_USER", "polymarket")
    password: str = os.getenv("PG_PASSWORD", "polymarket")
    database: str = os.getenv("PG_DATABASE", "polymarket_geo")
    min_pool_size: int = int(os.getenv("PG_POOL_MIN", "2"))
    max_pool_size: int = int(os.getenv("PG_POOL_MAX", "10"))

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass(frozen=True)
class PolymarketConfig:
    base_url: str = os.getenv("POLYMARKET_API_URL", "https://gamma-api.polymarket.com")
    markets_endpoint: str = "/markets"
    page_size: int = int(os.getenv("POLYMARKET_PAGE_SIZE", "100"))
    max_pages: int = int(os.getenv("POLYMARKET_MAX_PAGES", "50"))  # safety cap
    request_timeout: int = int(os.getenv("POLYMARKET_TIMEOUT", "30"))
    # Rate limiting: requests per second
    rate_limit_rps: float = float(os.getenv("POLYMARKET_RATE_LIMIT", "5.0"))


@dataclass(frozen=True)
class GeocodingConfig:
    provider: str = os.getenv("GEOCODER_PROVIDER", "nominatim")  # nominatim | google
    nominatim_url: str = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org")
    nominatim_user_agent: str = os.getenv("NOMINATIM_USER_AGENT", "polymarket-geo-pipeline/1.0")
    google_api_key: str = os.getenv("GOOGLE_GEOCODING_KEY", "")
    # Rate limiting
    rate_limit_rps: float = float(os.getenv("GEOCODER_RATE_LIMIT", "1.0"))  # Nominatim wants <=1/s
    max_retries: int = int(os.getenv("GEOCODER_MAX_RETRIES", "3"))
    backoff_base: float = float(os.getenv("GEOCODER_BACKOFF_BASE", "2.0"))
    # Cache TTL in days
    cache_ttl_days: int = int(os.getenv("GEOCODER_CACHE_TTL_DAYS", "30"))


@dataclass(frozen=True)
class InferenceConfig:
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    # Minimum confidence to store a location candidate
    min_confidence: float = float(os.getenv("INFER_MIN_CONFIDENCE", "0.15"))
    # Maximum location candidates per market
    max_candidates: int = int(os.getenv("INFER_MAX_CANDIDATES", "5"))
    # LLM fallback (optional)
    llm_enabled: bool = os.getenv("LLM_ENABLED", "false").lower() == "true"
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # openai | anthropic
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    # Current inference version — bump when logic changes to re-process markets
    inference_version: int = int(os.getenv("INFERENCE_VERSION", "1"))


@dataclass(frozen=True)
class SchedulerConfig:
    enabled: bool = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
    interval_minutes: int = int(os.getenv("SCHEDULER_INTERVAL_MIN", "15"))
    # How many markets to process per pipeline run (0 = all unprocessed)
    batch_size: int = int(os.getenv("PIPELINE_BATCH_SIZE", "500"))


@dataclass(frozen=True)
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    # Default radius for /nearby queries in km
    default_radius_km: float = float(os.getenv("API_DEFAULT_RADIUS_KM", "50.0"))
    max_radius_km: float = float(os.getenv("API_MAX_RADIUS_KM", "500.0"))
    max_results: int = int(os.getenv("API_MAX_RESULTS", "100"))


@dataclass(frozen=True)
class Settings:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    geocoding: GeocodingConfig = field(default_factory=GeocodingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    env: str = os.getenv("APP_ENV", "development")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()
