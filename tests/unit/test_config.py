import os
import pytest
from agentic_doc.config import (
    Settings,
    LandingAISettings,
    HuggingFaceSettings,
    OpenAISettings,
    GoogleAISettings,
    AnthropicSettings,
)

# Pytest fixtures can be defined here if needed, e.g. for monkeypatch
# For this set of tests, monkeypatch is used directly as a test function argument.

def test_settings_default_provider():
    """Test that the default AI provider type is correctly set."""
    settings = Settings()
    # Assuming 'landingai' is the default from config.py
    assert settings.ai_provider_type == "landingai"

def test_settings_override_provider_env_var(monkeypatch):
    """Test that AI_PROVIDER_TYPE environment variable overrides the default."""
    monkeypatch.setenv("AI_PROVIDER_TYPE", "huggingface")
    settings = Settings()
    assert settings.ai_provider_type == "huggingface"
    # Clean up env var for other tests if Settings is memoized or module-scoped
    monkeypatch.delenv("AI_PROVIDER_TYPE")


def test_settings_load_nested_landingai(monkeypatch):
    """Test loading nested LandingAI settings from environment variables."""
    monkeypatch.setenv("AI_PROVIDER_TYPE", "landingai")
    monkeypatch.setenv("LANDINGAI__API_KEY", "test_landing_key_from_env")
    monkeypatch.setenv("LANDINGAI__ENDPOINT_HOST", "http://testhost:8000")

    settings = Settings()

    assert settings.ai_provider_type == "landingai"
    assert settings.landingai is not None
    assert settings.landingai.api_key == "test_landing_key_from_env"
    assert settings.landingai.endpoint_host == "http://testhost:8000"

    monkeypatch.delenv("AI_PROVIDER_TYPE", raising=False)
    monkeypatch.delenv("LANDINGAI__API_KEY", raising=False)
    monkeypatch.delenv("LANDINGAI__ENDPOINT_HOST", raising=False)

def test_settings_load_nested_openai(monkeypatch):
    """Test loading nested OpenAI settings from environment variables."""
    monkeypatch.setenv("AI_PROVIDER_TYPE", "openai")
    monkeypatch.setenv("OPENAI__API_KEY", "sk-test_openai_key_from_env")
    monkeypatch.setenv("OPENAI__MODEL_NAME", "gpt-custom")

    settings = Settings()

    assert settings.ai_provider_type == "openai"
    assert settings.openai is not None
    assert settings.openai.api_key == "sk-test_openai_key_from_env"
    assert settings.openai.model_name == "gpt-custom"

    monkeypatch.delenv("AI_PROVIDER_TYPE", raising=False)
    monkeypatch.delenv("OPENAI__API_KEY", raising=False)
    monkeypatch.delenv("OPENAI__MODEL_NAME", raising=False)


def test_settings_str_redacts_api_keys(monkeypatch):
    """Test that the __str__ method redacts API keys."""
    raw_openai_key = "sk-thisisarealapikeyvalueforopenai"
    raw_landingai_key = "landingai_secret_key_value"

    monkeypatch.setenv("AI_PROVIDER_TYPE", "openai") # Set one to be active for initial dump
    monkeypatch.setenv("OPENAI__API_KEY", raw_openai_key)
    monkeypatch.setenv("LANDINGAI__API_KEY", raw_landingai_key)
    # For HuggingFace, it's auth_token
    raw_hf_token = "hf_thisisatesttoken"
    monkeypatch.setenv("HUGGINGFACE__AUTH_TOKEN", raw_hf_token)

    settings = Settings()
    settings_str = str(settings)

    assert raw_openai_key not in settings_str
    assert raw_landingai_key not in settings_str
    assert raw_hf_token not in settings_str

    # Check for redaction placeholder for OpenAI API Key
    # Need to handle both dict representations (repr vs json.dumps in __str__)
    assert f'"api_key": "{raw_openai_key[:5]}[REDACTED]"' in settings_str or \
           f"'api_key': '{raw_openai_key[:5]}[REDACTED]'" in settings_str or \
           f"api_key='{raw_openai_key[:5]}[REDACTED]'" in settings_str

    # Check for redaction placeholder for LandingAI API Key
    assert f'"api_key": "{raw_landingai_key[:5]}[REDACTED]"' in settings_str or \
           f"'api_key': '{raw_landingai_key[:5]}[REDACTED]'" in settings_str or \
           f"api_key='{raw_landingai_key[:5]}[REDACTED]'" in settings_str

    # Check for redaction placeholder for HuggingFace Auth Token
    assert f'"auth_token": "{raw_hf_token[:5]}[REDACTED]"' in settings_str or \
           f"'auth_token': '{raw_hf_token[:5]}[REDACTED]'" in settings_str or \
           f"auth_token='{raw_hf_token[:5]}[REDACTED]'" in settings_str

    monkeypatch.delenv("AI_PROVIDER_TYPE", raising=False)
    monkeypatch.delenv("OPENAI__API_KEY", raising=False)
    monkeypatch.delenv("LANDINGAI__API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE__AUTH_TOKEN", raising=False)


def test_settings_initializes_all_provider_configs():
    """
    Test that all provider-specific config objects are initialized by default,
    as per the logic added in config.py.
    """
    settings = Settings()
    assert settings.landingai is not None, "LandingAI settings should be initialized"
    assert isinstance(settings.landingai, LandingAISettings)

    assert settings.huggingface is not None, "HuggingFace settings should be initialized"
    assert isinstance(settings.huggingface, HuggingFaceSettings)

    assert settings.openai is not None, "OpenAI settings should be initialized"
    assert isinstance(settings.openai, OpenAISettings)

    assert settings.googleai is not None, "GoogleAI settings should be initialized"
    assert isinstance(settings.googleai, GoogleAISettings)

    assert settings.anthropic is not None, "Anthropic settings should be initialized"
    assert isinstance(settings.anthropic, AnthropicSettings)

def test_settings_override_huggingface_model_device(monkeypatch):
    """Test overriding HuggingFace model name and device."""
    monkeypatch.setenv("AI_PROVIDER_TYPE", "huggingface")
    monkeypatch.setenv("HUGGINGFACE__MODEL_NAME", "custom/model-test")
    monkeypatch.setenv("HUGGINGFACE__DEVICE", "cuda:1")

    settings = Settings()

    assert settings.huggingface is not None
    assert settings.huggingface.model_name == "custom/model-test"
    assert settings.huggingface.device == "cuda:1"

    monkeypatch.delenv("AI_PROVIDER_TYPE", raising=False)
    monkeypatch.delenv("HUGGINGFACE__MODEL_NAME", raising=False)
    monkeypatch.delenv("HUGGINGFACE__DEVICE", raising=False)
