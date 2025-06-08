import json
import logging
from typing import Literal, Optional

import cv2
import structlog
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_doc.common import ChunkType

_LOGGER = structlog.get_logger(__name__)
_MAX_PARALLEL_TASKS = 200
# Colors in BGR format (OpenCV uses BGR)
_COLOR_MAP = {
    ChunkType.marginalia: (128, 0, 255),  # Purple for marginalia
    ChunkType.table: (139, 69, 19),  # Brown for tables
    ChunkType.figure: (50, 205, 50),  # Lime green for figures
    ChunkType.text: (255, 0, 0),  # Blue for regular text
}


class LandingAISettings(BaseModel):
    api_key: str = Field(default="", description="API key for LandingAI")
    endpoint_host: str = Field(default="https://api.va.landing.ai", description="LandingAI endpoint host")


class HuggingFaceSettings(BaseModel):
    model_name: str = Field(default="", description="Name of the Hugging Face model to use")
    device: str = Field(default="cpu", description="Device to run the Hugging Face model on (e.g., 'cpu', 'cuda')")
    auth_token: Optional[str] = Field(default=None, description="Optional Hugging Face authentication token")


class OpenAISettings(BaseModel):
    api_key: str = Field(default="", description="API key for OpenAI")
    model_name: str = Field(default="gpt-4-vision-preview", description="OpenAI model name")


class GoogleAISettings(BaseModel):
    api_key: str = Field(default="", description="API key for Google AI")
    model_name: str = Field(default="gemini-pro-vision", description="Google AI model name")


class AnthropicSettings(BaseModel):
    api_key: str = Field(default="", description="API key for Anthropic")
    model_name: str = Field(default="claude-3-opus-20240229", description="Anthropic model name")


class Settings(BaseSettings):
    ai_provider_type: str = Field(default="landingai", description="The type of AI provider to use (e.g., 'landingai', 'huggingface', 'openai')")
    landingai: Optional[LandingAISettings] = None
    huggingface: Optional[HuggingFaceSettings] = None
    openai: Optional[OpenAISettings] = None
    googleai: Optional[GoogleAISettings] = None
    anthropic: Optional[AnthropicSettings] = None
    batch_size: int = Field(
        default=4,
        description="Number of documents to process in parallel",
        ge=1,
    )
    max_workers: int = Field(
        default=5,
        description="Maximum number of workers to use for parallel processing for each document",
        ge=1,
    )
    max_retries: int = Field(
        default=100,
        description="Maximum number of retries for a failed request",
        ge=0,
    )
    max_retry_wait_time: int = Field(
        default=60,
        description="Maximum wait time for a retry",
        ge=0,
    )
    retry_logging_style: Literal["none", "log_msg", "inline_block"] = Field(
        default="log_msg",
        description="Logging style for retries",
    )
    pdf_to_image_dpi: int = Field(
        default=96,
        description="DPI for converting PDF pages to images",
        ge=1,
    )
    split_size: int = Field(
        default=10,
        description="Pages per chunk for splitting the document",
        ge=1,
        le=50,
    )
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
        env_nested_delimiter='__',
    )

    def __str__(self) -> str:
        settings_dict = self.model_dump()
        for provider_key in ["landingai", "openai", "googleai", "anthropic"]:
            if provider_settings := settings_dict.get(provider_key):
                if "api_key" in provider_settings and provider_settings["api_key"]:
                    provider_settings["api_key"] = provider_settings["api_key"][:5] + "[REDACTED]"
        if huggingface_settings := settings_dict.get("huggingface"):
            if "auth_token" in huggingface_settings and huggingface_settings["auth_token"]:
                huggingface_settings["auth_token"] = huggingface_settings["auth_token"][:5] + "[REDACTED]"
        return f"{json.dumps(settings_dict, indent=2)}"


settings = Settings()

if settings.landingai is None:
    settings.landingai = LandingAISettings()
if settings.huggingface is None:
    settings.huggingface = HuggingFaceSettings()
if settings.openai is None:
    settings.openai = OpenAISettings()
if settings.googleai is None:
    settings.googleai = GoogleAISettings()
if settings.anthropic is None:
    settings.anthropic = AnthropicSettings()

_LOGGER.info(f"Settings loaded: {settings}")

if settings.batch_size * settings.max_workers > _MAX_PARALLEL_TASKS:
    raise ValueError(
        f"Batch size * max workers must be less than {_MAX_PARALLEL_TASKS}."
        " Please reduce the batch size or max workers."
        " Current settings: batch_size={settings.batch_size}, max_workers={settings.max_workers}"
    )

if settings.retry_logging_style == "inline_block":
    logging.getLogger("httpx").setLevel(logging.WARNING)


class VisualizationConfig(BaseSettings):
    thickness: int = Field(
        default=1,
        description="Thickness of the bounding box and text",
        ge=0,
    )
    text_bg_color: tuple[int, int, int] = Field(
        default=(211, 211, 211),  # Light gray
        description="Background color of the text, in BGR format",
    )
    text_bg_opacity: float = Field(
        default=0.7,
        description="Opacity of the text background",
        ge=0.0,
        le=1.0,
    )
    padding: int = Field(
        default=1,
        description="Padding of the text background box",
        ge=0,
    )
    font_scale: float = Field(
        default=0.5,
        description="Font scale of the text",
        ge=0.0,
    )
    font: int = Field(
        default=cv2.FONT_HERSHEY_SIMPLEX,
        description="Font of the text",
    )
    color_map: dict[ChunkType, tuple[int, int, int]] = Field(
        default=_COLOR_MAP,
        description="Color map for each chunk type",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )
