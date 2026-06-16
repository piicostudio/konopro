from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_CORS_ALLOW_ORIGINS = (
    "http://127.0.0.1:5173,"
    "http://localhost:5173,"
    "http://127.0.0.1:5500,"
    "http://localhost:5500,"
    "http://127.0.0.1:8765,"
    "http://localhost:8765,"
    "http://127.0.0.1:8000,"
    "http://localhost:8000"
)


class BackendSettings(BaseSettings):
    """Runtime settings for the Konopro backend."""

    database_url: str = "sqlite:///./.local/konopro_backend.db"
    storage_root: Path = Field(default=Path("./.local/storage"))
    processing_root: Path = Field(default=Path("./.local/processing"))
    max_upload_mb: int = 500
    environment: str = "local"
    fingerprint_provider: str = "acrcloud"
    fingerprint_window_s: float = 10.0
    fingerprint_hop_s: float = 5.0
    fingerprint_max_windows: int = 120
    fingerprint_use_whole: bool = False
    fingerprint_timeout_s: float = 30.0
    audd_api_token: str | None = None
    acrcloud_host: str | None = None
    acrcloud_access_key: str | None = None
    acrcloud_access_secret: str | None = None
    shazamkit_helper_path: Path | None = None
    admin_api_key: str | None = None
    free_report_turnaround_hours: int = 72
    paid_report_turnaround_hours: int = 24
    manual_comp_report_turnaround_hours: int = 48
    cors_allow_origins: str = DEFAULT_CORS_ALLOW_ORIGINS
    cors_allow_all_in_local: bool = True
    reference_download_tool: str = "yt-dlp"
    reference_fetch_timeout_s: float = 180.0
    reference_scoring_use_demucs: bool = True
    reference_scoring_demucs_model: str = "htdemucs"
    reference_scoring_demucs_device: str | None = None
    reference_scoring_demucs_timeout_s: int = 1800
    reference_scoring_use_active_rms: bool = True
    reference_scoring_target_active_rms: float = 0.08
    reference_scoring_active_rms_percentile: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KONOPRO_",
        extra="ignore",
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def processing_path(self) -> Path:
        return self.processing_root.expanduser().resolve()

    @property
    def cors_origins(self) -> list[str]:
        if (
            self.environment == "local"
            and self.cors_allow_all_in_local
            and self.cors_allow_origins == DEFAULT_CORS_ALLOW_ORIGINS
        ):
            return ["*"]
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]
