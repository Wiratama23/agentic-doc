import os
import pytest
from agentic_doc.ai_providers import (
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import httpx # For LandingAI mock
import pdf2image # For HuggingFace mock

from agentic_doc.ai_providers import (
    get_ai_provider,
    BaseAIProvider,
    LandingAIProvider,
    HuggingFaceProvider,
    OpenAIProvider,
    GoogleAIProvider,
    AnthropicProvider,
)
from agentic_doc.config import (
    Settings,
    LandingAISettings,
    HuggingFaceSettings,
    OpenAISettings,
    GoogleAISettings,
    AnthropicSettings
)
from agentic_doc.common import ParsedDocument, Chunk, ChunkType, ChunkGrounding, ChunkGroundingBox, PageError, RetryableError

# Import client libraries for specific exception/error mocking if needed
import openai as openai_lib
import google.generativeai as genai_lib # For genai.configure error or specific model errors
import anthropic as anthropic_lib # For anthropic.APIError


# --- Fixtures ---
@pytest.fixture
def mock_landingai_settings():
    return LandingAISettings(api_key="test_landing_key", endpoint_host="http://fakehost.landing.ai")

@pytest.fixture
def mock_hf_settings():
    return HuggingFaceSettings(model_name="test/model", device="cpu")

@pytest.fixture
def mock_openai_settings():
    return OpenAISettings(api_key="test_openai_key", model_name="gpt-test")

@pytest.fixture
def mock_googleai_settings():
    return GoogleAISettings(api_key="test_google_key", model_name="gemini-test")

@pytest.fixture
def mock_anthropic_settings():
    return AnthropicSettings(api_key="test_anthropic_key", model_name="claude-test")

# --- Existing get_ai_provider tests ---
def test_get_ai_provider_landingai(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "landingai")
    monkeypatch.setenv("LANDINGAI__API_KEY", "fake_landing_key")
    # Ensure Settings is re-evaluated after monkeypatching
    settings_instance = Settings()
    provider = get_ai_provider()
    assert isinstance(provider, LandingAIProvider)
    monkeypatch.delenv("AI_PROVIDER_TYPE")
    monkeypatch.delenv("LANDINGAI__API_KEY")

def test_get_ai_provider_huggingface(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "huggingface")
    monkeypatch.setenv("HUGGINGFACE__MODEL_NAME", "some/test-model")
    settings_instance = Settings()
    provider = get_ai_provider()
    assert isinstance(provider, HuggingFaceProvider)
    monkeypatch.delenv("AI_PROVIDER_TYPE")
    monkeypatch.delenv("HUGGINGFACE__MODEL_NAME")

def test_get_ai_provider_openai(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "openai")
    monkeypatch.setenv("OPENAI__API_KEY", "sk-fake_openai_key")
    settings_instance = Settings()
    provider = get_ai_provider()
    assert isinstance(provider, OpenAIProvider)
    monkeypatch.delenv("AI_PROVIDER_TYPE")
    monkeypatch.delenv("OPENAI__API_KEY")

def test_get_ai_provider_googleai(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "googleai")
    monkeypatch.setenv("GOOGLEAI__API_KEY", "fake_google_key")
    settings_instance = Settings()
    provider = get_ai_provider()
    assert isinstance(provider, GoogleAIProvider)
    monkeypatch.delenv("AI_PROVIDER_TYPE")
    monkeypatch.delenv("GOOGLEAI__API_KEY")

def test_get_ai_provider_anthropic(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "anthropic")
    monkeypatch.setenv("ANTHROPIC__API_KEY", "fake_anthropic_key")
    settings_instance = Settings()
    provider = get_ai_provider()
    assert isinstance(provider, AnthropicProvider)
    monkeypatch.delenv("AI_PROVIDER_TYPE")
    monkeypatch.delenv("ANTHROPIC__API_KEY")

def test_get_ai_provider_unknown(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "unknown_provider")
    settings_instance = Settings()
    with pytest.raises(ValueError, match="Unsupported AI provider type: unknown_provider"):
        get_ai_provider()
    monkeypatch.delenv("AI_PROVIDER_TYPE")

# --- Tests for missing configuration ---

def test_get_ai_provider_missing_config_landingai(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "landingai")
    monkeypatch.delenv("LANDINGAI__API_KEY", raising=False)
    # Re-instantiate Settings to pick up the change from monkeypatch
    current_settings = Settings()
    # If the global settings.landingai was already populated by a previous test or default, clear its key
    if current_settings.landingai:
        current_settings.landingai.api_key = ""

    with pytest.raises(ValueError, match="LandingAI provider selected, but API key is missing in settings."):
        get_ai_provider() # get_ai_provider uses the global `settings` instance
    monkeypatch.delenv("AI_PROVIDER_TYPE")

def test_get_ai_provider_missing_config_huggingface(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "huggingface")
    monkeypatch.delenv("HUGGINGFACE__MODEL_NAME", raising=False)
    settings_instance = Settings()
    if settings_instance.huggingface:
        settings_instance.huggingface.model_name = ""

    with pytest.raises(ValueError, match="HuggingFace provider selected, but model_name is missing in settings."):
        get_ai_provider()
    monkeypatch.delenv("AI_PROVIDER_TYPE")

def test_get_ai_provider_missing_config_openai(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "openai")
    monkeypatch.delenv("OPENAI__API_KEY", raising=False)
    current_settings = Settings()
    if current_settings.openai:
         current_settings.openai.api_key = ""

    with pytest.raises(ValueError, match="OpenAI provider selected, but API key is missing in settings."):
        get_ai_provider()
    monkeypatch.delenv("AI_PROVIDER_TYPE")

def test_get_ai_provider_missing_config_googleai(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "googleai")
    monkeypatch.delenv("GOOGLEAI__API_KEY", raising=False)
    settings_instance = Settings()
    if settings_instance.googleai:
        settings_instance.googleai.api_key = ""

    with pytest.raises(ValueError, match="GoogleAI provider selected, but API key is missing in settings."):
        get_ai_provider()
    monkeypatch.delenv("AI_PROVIDER_TYPE")

def test_get_ai_provider_missing_config_anthropic(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_TYPE", "anthropic")
    monkeypatch.delenv("ANTHROPIC__API_KEY", raising=False)
    current_settings = Settings()
    if current_settings.anthropic:
        current_settings.anthropic.api_key = ""

    with pytest.raises(ValueError, match="Anthropic provider selected, but API key is missing in settings."):
        get_ai_provider()
    monkeypatch.delenv("AI_PROVIDER_TYPE")

# --- LandingAIProvider.analyze_document tests ---

@patch('agentic_doc.ai_providers.httpx.post')
def test_landingai_provider_analyze_image_success(mock_httpx_post, tmp_path, mock_landingai_settings):
    dummy_image_file = tmp_path / "test_image.png"
    dummy_image_file.write_bytes(b"dummy image data")

    provider = LandingAIProvider(mock_landingai_settings)

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    # This structure must match what ParsedDocument.model_validate expects after API call data extraction
    mock_response.json.return_value = {
        "data": {
            "markdown": "Test Markdown from LandingAI",
            "chunks": [
                {"text": "chunk1", "grounding": [{"page": 0, "box": {"l":0.1,"t":0.1,"r":0.2,"b":0.2}}], "chunk_type": "text", "chunk_id": "id1"}
            ]
        },
        # errors from API are part of the main dict, not under 'data' for LandingAI provider's current parsing
        "errors": []
    }
    mock_httpx_post.return_value = mock_response

    options = {"doc_type": "image", "start_page_idx": 0, "end_page_idx": 0, "include_marginalia": True, "include_metadata_in_markdown": True}
    result = provider.analyze_document(str(dummy_image_file), options)

    mock_httpx_post.assert_called_once()
    assert result.markdown == "Test Markdown from LandingAI"
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "chunk1"
    assert result.doc_type == "image"
    assert not result.errors

@patch('agentic_doc.ai_providers.httpx.post')
def test_landingai_provider_analyze_pdf_success(mock_httpx_post, tmp_path, mock_landingai_settings):
    dummy_pdf_file = tmp_path / "test_doc.pdf"
    dummy_pdf_file.write_bytes(b"dummy PDF data")

    provider = LandingAIProvider(mock_landingai_settings)

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "markdown": "PDF Markdown",
            "chunks": []
        },
        "errors": []
    }
    mock_httpx_post.return_value = mock_response

    options = {"doc_type": "pdf", "start_page_idx": 0, "end_page_idx": 0} # Assuming this is a single chunk PDF for test
    result = provider.analyze_document(str(dummy_pdf_file), options)

    mock_httpx_post.assert_called_once()
    assert result.markdown == "PDF Markdown"
    assert result.doc_type == "pdf" # This is taken from options
    assert not result.errors

@patch('agentic_doc.ai_providers.httpx.post')
def test_landingai_provider_api_error(mock_httpx_post, tmp_path, mock_landingai_settings):
    dummy_image_file = tmp_path / "test_image_api_error.png"
    dummy_image_file.write_bytes(b"dummy image data")

    provider = LandingAIProvider(mock_landingai_settings)

    # Simulate an HTTP error that is not automatically retried by tenacity for this test
    # or an error that makes it through retries.
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.request = MagicMock() # Needed for HTTPStatusError
    mock_httpx_post.side_effect = httpx.HTTPStatusError(
        message="Server error", request=mock_response.request, response=mock_response
    )

    options = {"doc_type": "image", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_image_file), options)

    mock_httpx_post.assert_called_once() # analyze_document itself is not retrying HTTPStatusError directly without tenacity setup for it
    assert result.markdown == ""
    assert len(result.errors) == 1
    assert result.errors[0].page_num == 0 # For single image, error on page 0
    assert "HTTP Error: 500 - Internal Server Error" in result.errors[0].error
    assert result.errors[0].error_code == 500


# --- HuggingFaceProvider.analyze_document tests (Placeholder Focused) ---

@patch('agentic_doc.ai_providers.transformers') # Mock the whole transformers module
def test_hf_provider_ensure_model_loaded_simulation(mock_transformers, mock_hf_settings, caplog):
    provider = HuggingFaceProvider(mock_hf_settings)

    # Simulate that no model is loaded initially
    provider.pipeline = None
    provider.model = None

    # For the placeholder, it just logs. If actual loading was tested,
    # mock_transformers.pipeline or .AutoModel would be set up.
    provider._ensure_model_loaded()

    # Check if the placeholder log message is present
    assert "MODEL LOADING PLACEHOLDER: Hugging Face model loading would occur here." in caplog.text
    assert f"MODEL LOADING PLACEHOLDER: Model Name: {mock_hf_settings.model_name}" in caplog.text

@patch('agentic_doc.ai_providers.PIL.Image.open')
@patch.object(HuggingFaceProvider, '_ensure_model_loaded') # Mock this to prevent actual model loading attempt
def test_hf_provider_analyze_image_placeholder(mock_ensure_loaded, mock_pil_open, tmp_path, mock_hf_settings):
    dummy_image_file = tmp_path / "hf_test_image.png"
    dummy_image_file.write_bytes(b"dummy image data for hf")

    mock_pil_open.return_value = MagicMock(size=(100,100), mode="RGB") # Simulate a PIL Image object

    provider = HuggingFaceProvider(mock_hf_settings)
    options = {"doc_type": "image", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_image_file), options)

    mock_ensure_loaded.assert_called_once()
    mock_pil_open.assert_called_once_with(dummy_image_file)

    assert "Placeholder text from HuggingFace model for page 1" in result.markdown
    assert len(result.chunks) == 1
    assert result.chunks[0].text.startswith("Placeholder text from HuggingFace model")
    assert result.doc_type == "image"
    assert not result.errors

@patch('agentic_doc.ai_providers.pdf2image.convert_from_path')
@patch.object(HuggingFaceProvider, '_ensure_model_loaded')
def test_hf_provider_analyze_pdf_placeholder(mock_ensure_loaded, mock_pdf2image, tmp_path, mock_hf_settings):
    dummy_pdf_file = tmp_path / "hf_test_doc.pdf"
    dummy_pdf_file.write_bytes(b"dummy pdf data for hf")

    # Simulate pdf2image returning two mock PIL Image objects
    mock_pil_image_page1 = MagicMock(spec=PIL.Image.Image, size=(100,100), mode="RGB")
    mock_pil_image_page2 = MagicMock(spec=PIL.Image.Image, size=(100,100), mode="RGB")
    mock_pdf2image.return_value = [mock_pil_image_page1, mock_pil_image_page2]

    provider = HuggingFaceProvider(mock_hf_settings)
    options = {"doc_type": "pdf", "start_page_idx": 0, "end_page_idx": 1} # For a 2-page PDF
    result = provider.analyze_document(str(dummy_pdf_file), options)

    mock_ensure_loaded.assert_called_once()
    mock_pdf2image.assert_called_once_with(dummy_pdf_file, output_folder=mock_pdf2image.call_args[1]['output_folder']) # Tmp dir is tricky to assert directly

    assert "Placeholder text from HuggingFace model for page 1" in result.markdown
    assert "Placeholder text from HuggingFace model for page 2" in result.markdown
    assert len(result.chunks) == 2
    assert result.chunks[0].text.startswith("Placeholder text from HuggingFace model for page 1")
    assert result.chunks[1].text.startswith("Placeholder text from HuggingFace model for page 2")
    assert result.doc_type == "pdf"
    assert not result.errors

@patch('agentic_doc.ai_providers.pdf2image.convert_from_path')
@patch.object(HuggingFaceProvider, '_ensure_model_loaded')
def test_hf_provider_pdf_conversion_error(mock_ensure_loaded, mock_pdf2image, tmp_path, mock_hf_settings):
    dummy_pdf_file = tmp_path / "hf_error_doc.pdf"
    dummy_pdf_file.write_bytes(b"dummy pdf data for hf error")

    mock_pdf2image.side_effect = pdf2image.exceptions.PDFInfoNotInstalledError("pdfinfo not found")

    provider = HuggingFaceProvider(mock_hf_settings)
    options = {"doc_type": "pdf", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_pdf_file), options)

    mock_ensure_loaded.assert_called_once() # Called before file processing starts
    assert result.markdown == ""
    assert len(result.errors) == 1
    assert result.errors[0].page_num == 0 # Error attributed to first expected page
    assert "PDF processing dependency (poppler) not installed" in result.errors[0].error
    assert result.errors[0].error_code == -1


# --- OpenAIProvider Tests (Placeholder Logic) ---

@patch('agentic_doc.ai_providers.openai.OpenAI')
def test_openai_provider_init(mock_openai_client, mock_openai_settings):
    # Test with API key
    provider = OpenAIProvider(mock_openai_settings)
    mock_openai_client.assert_called_once_with(api_key=mock_openai_settings.api_key)
    assert provider.client is not None

    # Test without API key
    mock_openai_client.reset_mock()
    settings_no_key = OpenAISettings(api_key=None)
    provider_no_key = OpenAIProvider(settings_no_key)
    mock_openai_client.assert_not_called() # Client should not be initialized
    assert provider_no_key.client is None

@patch('agentic_doc.ai_providers.PIL.Image.open')
@patch.object(OpenAIProvider, '_pil_to_base64', return_value="dummy_base64_string")
def test_openai_provider_analyze_image_placeholder(mock_pil_to_base64, mock_pil_open, tmp_path, mock_openai_settings):
    dummy_image_file = tmp_path / "oa_test_image.png"
    dummy_image_file.write_bytes(b"dummy image data for openai")

    mock_pil_open.return_value = MagicMock(size=(100,100), mode="RGB", format="PNG")

    provider = OpenAIProvider(mock_openai_settings)
    # Ensure client is mocked if analyze_document tries to use it, even for placeholders
    provider.client = MagicMock()

    options = {"doc_type": "image", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_image_file), options)

    mock_pil_open.assert_called_once_with(dummy_image_file)
    mock_pil_to_base64.assert_called_once() # Called with the mocked PIL image

    assert "Simulated OpenAI text for page 0" in result.markdown
    assert len(result.chunks) == 1
    assert result.chunks[0].text.startswith("Simulated OpenAI text")
    assert result.doc_type == "image"
    assert not result.errors

@patch('agentic_doc.ai_providers.pdf2image.convert_from_path')
@patch.object(OpenAIProvider, '_pil_to_base64', return_value="dummy_base64_string")
def test_openai_provider_analyze_pdf_placeholder(mock_pil_to_base64, mock_pdf2image, tmp_path, mock_openai_settings):
    dummy_pdf_file = tmp_path / "oa_test_doc.pdf"
    dummy_pdf_file.write_bytes(b"dummy pdf data for openai")

    mock_pil_image_page1 = MagicMock(spec=PIL.Image.Image, size=(100,100), mode="RGB", format="PNG")
    mock_pil_image_page2 = MagicMock(spec=PIL.Image.Image, size=(100,100), mode="RGB", format="PNG")
    mock_pdf2image.return_value = [mock_pil_image_page1, mock_pil_image_page2]

    provider = OpenAIProvider(mock_openai_settings)
    provider.client = MagicMock()

    options = {"doc_type": "pdf", "start_page_idx": 0, "end_page_idx": 1}
    result = provider.analyze_document(str(dummy_pdf_file), options)

    mock_pdf2image.assert_called_once()
    assert mock_pil_to_base64.call_count == 2

    assert "Simulated OpenAI text for page 0" in result.markdown # Based on parsed_doc_start_page_idx + page_idx_in_file
    assert "Simulated OpenAI text for page 1" in result.markdown
    assert len(result.chunks) == 2
    assert result.doc_type == "pdf"
    assert not result.errors

def test_openai_provider_missing_api_key_analyze(tmp_path):
    settings_no_key = OpenAISettings(api_key=None)
    provider = OpenAIProvider(settings_no_key)

    dummy_image_file = tmp_path / "oa_no_key.png"
    dummy_image_file.write_bytes(b"data")

    options = {"doc_type": "image", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_image_file), options)

    assert result.markdown == ""
    assert not result.chunks
    assert len(result.errors) == 1
    assert "OpenAI API key not configured" in result.errors[0].error
    assert result.errors[0].error_code == 400

@patch('agentic_doc.ai_providers.pdf2image.convert_from_path')
def test_openai_provider_pdf_conversion_error(mock_pdf2image, tmp_path, mock_openai_settings):
    dummy_pdf_file = tmp_path / "oa_pdf_error.pdf"
    dummy_pdf_file.write_bytes(b"dummy pdf data")

    mock_pdf2image.side_effect = pdf2image.exceptions.PDFInfoNotInstalledError("pdfinfo not found for test")

    provider = OpenAIProvider(mock_openai_settings)
    provider.client = MagicMock() # Mock client as it's not relevant for this error path

    options = {"doc_type": "pdf", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_pdf_file), options)

    assert result.markdown == ""
    assert not result.chunks
    assert len(result.errors) == 1
    assert "PDF processing dependency (poppler) not installed" in result.errors[0].error


# --- GoogleAIProvider Tests (Placeholder Logic) ---

@patch('agentic_doc.ai_providers.genai.GenerativeModel')
@patch('agentic_doc.ai_providers.genai.configure')
def test_googleai_provider_init(mock_genai_configure, mock_genai_model, mock_googleai_settings):
    # Test with API key
    provider = GoogleAIProvider(mock_googleai_settings)
    mock_genai_configure.assert_called_once_with(api_key=mock_googleai_settings.api_key)
    mock_genai_model.assert_called_once_with(mock_googleai_settings.model_name)
    assert provider.model is not None

    # Test without API key
    mock_genai_configure.reset_mock()
    mock_genai_model.reset_mock()
    settings_no_key = GoogleAISettings(api_key=None, model_name="gemini-test")
    provider_no_key = GoogleAIProvider(settings_no_key)
    mock_genai_configure.assert_not_called()
    mock_genai_model.assert_not_called()
    assert provider_no_key.model is None

    # Test genai.configure raising an error
    mock_genai_configure.reset_mock()
    mock_genai_model.reset_mock()
    mock_genai_configure.side_effect = Exception("GenAI Configure Error")
    provider_configure_error = GoogleAIProvider(mock_googleai_settings)
    assert provider_configure_error.model is None


@patch('agentic_doc.ai_providers.PIL.Image.open')
def test_googleai_provider_analyze_image_placeholder(mock_pil_open, tmp_path, mock_googleai_settings):
    dummy_image_file = tmp_path / "ga_test_image.jpg"
    dummy_image_file.write_bytes(b"dummy image data for google")
    mock_pil_open.return_value = MagicMock(size=(100,100), mode="RGB", format="JPEG")

    provider = GoogleAIProvider(mock_googleai_settings)
    provider.model = MagicMock() # Mock the model to prevent actual API calls

    options = {"doc_type": "image", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_image_file), options)

    assert "Simulated Google AI (Gemini) text for page 0" in result.markdown
    assert len(result.chunks) == 1
    assert result.chunks[0].text.startswith("Simulated Google AI (Gemini) text")
    assert not result.errors

@patch('agentic_doc.ai_providers.pdf2image.convert_from_path')
def test_googleai_provider_analyze_pdf_placeholder(mock_pdf2image, tmp_path, mock_googleai_settings):
    dummy_pdf_file = tmp_path / "ga_test_doc.pdf"
    dummy_pdf_file.write_bytes(b"dummy pdf data for google")
    mock_pil_image_page1 = MagicMock(spec=PIL.Image.Image, size=(100,100), mode="RGB", format="JPEG")
    mock_pdf2image.return_value = [mock_pil_image_page1]

    provider = GoogleAIProvider(mock_googleai_settings)
    provider.model = MagicMock()

    options = {"doc_type": "pdf", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_pdf_file), options)

    assert "Simulated Google AI (Gemini) text for page 0" in result.markdown
    assert len(result.chunks) == 1
    assert not result.errors

def test_googleai_provider_missing_api_key_analyze(tmp_path):
    settings_no_key = GoogleAISettings(api_key=None, model_name="gemini-test")
    provider = GoogleAIProvider(settings_no_key)
    dummy_image_file = tmp_path / "ga_no_key.jpg"
    dummy_image_file.write_bytes(b"data")

    options = {"doc_type": "image", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_image_file), options)

    assert len(result.errors) == 1
    assert "Google AI API key/model not configured" in result.errors[0].error


# --- AnthropicProvider Tests (Placeholder Logic) ---

@patch('agentic_doc.ai_providers.anthropic.Anthropic')
def test_anthropic_provider_init(mock_anthropic_client, mock_anthropic_settings):
    provider = AnthropicProvider(mock_anthropic_settings)
    mock_anthropic_client.assert_called_once_with(api_key=mock_anthropic_settings.api_key)
    assert provider.client is not None

    mock_anthropic_client.reset_mock()
    settings_no_key = AnthropicSettings(api_key=None)
    provider_no_key = AnthropicProvider(settings_no_key)
    mock_anthropic_client.assert_not_called()
    assert provider_no_key.client is None

@patch('agentic_doc.ai_providers.PIL.Image.open')
@patch.object(AnthropicProvider, '_pil_to_base64', return_value="dummy_base64_anthropic")
def test_anthropic_provider_analyze_image_placeholder(mock_pil_to_base64, mock_pil_open, tmp_path, mock_anthropic_settings):
    dummy_image_file = tmp_path / "ant_test_image.jpg"
    dummy_image_file.write_bytes(b"dummy image data for anthropic")
    mock_pil_open.return_value = MagicMock(size=(100,100), mode="RGB", format="JPEG")

    provider = AnthropicProvider(mock_anthropic_settings)
    provider.client = MagicMock()

    options = {"doc_type": "image", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_image_file), options)

    assert "Simulated Anthropic Claude text for page 0" in result.markdown
    assert len(result.chunks) == 1
    assert not result.errors

@patch('agentic_doc.ai_providers.pdf2image.convert_from_path')
@patch.object(AnthropicProvider, '_pil_to_base64', return_value="dummy_base64_anthropic")
def test_anthropic_provider_analyze_pdf_placeholder(mock_pil_to_base64, mock_pdf2image, tmp_path, mock_anthropic_settings):
    dummy_pdf_file = tmp_path / "ant_test_doc.pdf"
    dummy_pdf_file.write_bytes(b"dummy pdf data for anthropic")
    mock_pil_image_page1 = MagicMock(spec=PIL.Image.Image, size=(100,100), mode="RGB", format="JPEG")
    mock_pdf2image.return_value = [mock_pil_image_page1]

    provider = AnthropicProvider(mock_anthropic_settings)
    provider.client = MagicMock()

    options = {"doc_type": "pdf", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_pdf_file), options)

    assert "Simulated Anthropic Claude text for page 0" in result.markdown
    assert len(result.chunks) == 1
    assert not result.errors

def test_anthropic_provider_missing_api_key_analyze(tmp_path):
    settings_no_key = AnthropicSettings(api_key=None)
    provider = AnthropicProvider(settings_no_key)
    dummy_image_file = tmp_path / "ant_no_key.jpg"
    dummy_image_file.write_bytes(b"data")

    options = {"doc_type": "image", "start_page_idx": 0, "end_page_idx": 0}
    result = provider.analyze_document(str(dummy_image_file), options)

    assert len(result.errors) == 1
    assert "Anthropic client not configured" in result.errors[0].error
