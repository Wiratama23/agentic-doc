import abc
import httpx
import structlog
import tenacity
from pathlib import Path
from typing import Any, Optional, List
import uuid # For chunk IDs

# PDF and Image processing
import pdf2image
import PIL.Image
import tempfile

# Hugging Face
import torch
import transformers # Placeholder for specific imports like pipeline, AutoTokenizer, AutoModelForDocumentQuestionAnswering

from agentic_doc.common import (
    ParsedDocument, PageError, RetryableError, Timer, _LIB_VERSION,
    Chunk, ChunkType, ChunkGrounding, ChunkGroundingBox # PageError is already imported
)
from agentic_doc.config import (
    settings, LandingAISettings, HuggingFaceSettings, OpenAISettings, GoogleAISettings, AnthropicSettings
)

# Google AI specific imports
import google.generativeai as genai

# OpenAI specific imports
import openai # base64 and io are already imported for OpenAI, can be reused
import base64
import io

# Anthropic specific imports
import anthropic
# base64, io, PIL.Image, pdf2image etc. are already imported above.


_LOGGER = structlog.getLogger(__name__)


class BaseAIProvider(abc.ABC):
  """Base class for AI providers."""

  @abc.abstractmethod
  def analyze_document(self, file_path: str, options: dict) -> ParsedDocument:
    """Analyzes a document using an AI provider.

    Args:
      file_path: The path to the document to analyze.
      options: A dictionary of options for the AI provider.

    Returns:
      A ParsedDocument object.
    """
    pass


class LandingAIProvider(BaseAIProvider):
  """AI Provider for LandingAI's Agentic Document Analysis API."""

  def __init__(self, landingai_settings: Optional[LandingAISettings] = None):
    """Initializes the LandingAIProvider.

    Args:
      landingai_settings: Optional LandingAISettings. If not provided,
                          it uses the global settings.landingai.
    """
    self.landingai_settings = landingai_settings or settings.landingai
    if not self.landingai_settings:
        raise ValueError("LandingAI settings not found in global configuration and not provided.")
    self._endpoint_url = f"{self.landingai_settings.endpoint_host}/v1/tools/agentic-document-analysis"


  @tenacity.retry(
      wait=tenacity.wait_exponential_jitter(
          exp_base=1.5, initial=1, max=settings.max_retry_wait_time, jitter=10
      ),
      stop=tenacity.stop_after_attempt(settings.max_retries),
      retry=tenacity.retry_if_exception_type(RetryableError),
      after=tenacity.after_log(_LOGGER, tenacity.logging.INFO), # Adjusted for structlog
      before_sleep=tenacity.before_sleep_log(_LOGGER, tenacity.logging.WARNING) # Adjusted for structlog
  )
  def analyze_document(self, file_path: str, options: dict) -> ParsedDocument:
    """Analyzes a document using the LandingAI API.

    Args:
      file_path: The path to the document to analyze.
      options: A dictionary of options:
        include_marginalia (bool): Whether to include marginalia.
        include_metadata_in_markdown (bool): Whether to include metadata in markdown.
        doc_type (str): "pdf" or "image".
        start_page_idx (int): Start page index (for PDF chunks).
        end_page_idx (int): End page index (for PDF chunks).

    Returns:
      A ParsedDocument object.
    """
    include_marginalia = options.get("include_marginalia", True)
    include_metadata_in_markdown = options.get("include_metadata_in_markdown", True)
    doc_type = options.get("doc_type", "pdf") # Default to pdf, can be overwritten by caller
    # For single images, or unchunked PDFs, page indices might be 0.
    # For PDF chunks, these would be passed in options.
    start_page_idx = options.get("start_page_idx", 0)
    end_page_idx = options.get("end_page_idx", 0)

    with Timer() as timer:
      actual_doc_type = "pdf" if Path(file_path).suffix.lower() == ".pdf" else "image"
      # The 'doc_type' in options might be more about the original full document type,
      # while actual_doc_type is about the specific file being sent.
      files_key = actual_doc_type
      with open(file_path, "rb") as file:
        files = {files_key: file}
        data = {
            "include_marginalia": include_marginalia,
            "include_metadata_in_markdown": include_metadata_in_markdown,
        }
        headers = {
            "Authorization": f"Basic {self.landingai_settings.api_key}",
            "runtime_tag": f"agentic-doc-v{_LIB_VERSION}",
        }
        try:
            response = httpx.post(
                self._endpoint_url,
                files=files,
                data=data,
                headers=headers,
                timeout=None,
            )
            if response.status_code in [408, 429, 502, 503, 504]:
                raise RetryableError(response)
            response.raise_for_status()

            api_result = response.json()
            parsed_doc_data = {
                **api_result.get("data", {}), # Ensure 'data' key exists
                "errors": api_result.get("errors", []),
                "doc_type": doc_type, # Use the doc_type from options for consistency with how ParsedDocument was created before
                "start_page_idx": start_page_idx,
                "end_page_idx": end_page_idx,
            }
            # Ensure basic structure if 'data' was missing or empty
            if "markdown" not in parsed_doc_data: parsed_doc_data["markdown"] = ""
            if "chunks" not in parsed_doc_data: parsed_doc_data["chunks"] = []

            parsed_document = ParsedDocument.model_validate(parsed_doc_data)

        except httpx.HTTPStatusError as e:
            _LOGGER.error(f"HTTP error analyzing document '{file_path}': {e.response.status_code} - {e.response.text}")
            error_msg = f"HTTP Error: {e.response.status_code} - {e.response.text}"
            errors = [PageError(page_num=i, error=error_msg, error_code=e.response.status_code) for i in range(start_page_idx, end_page_idx +1)]
            # If it's an image, page range is just 0
            if doc_type == "image" and start_page_idx == 0 and end_page_idx == 0:
                 errors = [PageError(page_num=0, error=error_msg, error_code=e.response.status_code)]

            return ParsedDocument(
                markdown="",
                chunks=[],
                start_page_idx=start_page_idx,
                end_page_idx=end_page_idx,
                doc_type=doc_type,
                errors=errors,
            )
        except RetryableError as e: # Propagate retryable error
            raise e
        except Exception as e:
            _LOGGER.error(f"Error analyzing document '{file_path}': {e}")
            error_msg = str(e)
            errors = [PageError(page_num=i, error=error_msg, error_code=-1) for i in range(start_page_idx, end_page_idx +1)]
            # If it's an image, page range is just 0
            if doc_type == "image" and start_page_idx == 0 and end_page_idx == 0:
                 errors = [PageError(page_num=0, error=error_msg, error_code=-1)]
            return ParsedDocument(
                markdown="",
                chunks=[],
                start_page_idx=start_page_idx,
                end_page_idx=end_page_idx,
                doc_type=doc_type,
                errors=errors,
            )

    _LOGGER.info(
        f"Time taken to successfully parse a document chunk with LandingAI: {timer.elapsed:.2f} seconds",
        file_path=file_path,
        provider="LandingAI"
    )
    return parsed_document


class HuggingFaceProvider(BaseAIProvider):
    """AI Provider for Hugging Face models."""

    def __init__(self, hf_settings: Optional[HuggingFaceSettings] = None):
        """Initializes the HuggingFaceProvider.

        Args:
            hf_settings: Optional HuggingFaceSettings. If not provided,
                         it uses the global settings.huggingface.
        """
        self.hf_settings = hf_settings or settings.huggingface
        if not self.hf_settings:
            raise ValueError("HuggingFace settings not found in global configuration and not provided.")

        self.tokenizer = None
        self.model = None
        self.pipeline = None
        _LOGGER.info("HuggingFaceProvider initialized. Model will be loaded on first use or explicitly.",
                     model_name=self.hf_settings.model_name, device=self.hf_settings.device)

    def _ensure_model_loaded(self):
        """Ensures that the Hugging Face model and tokenizer (or pipeline) are loaded."""
        if self.pipeline is None and self.model is None: # Check if either is loaded
            _LOGGER.info(f"Loading Hugging Face model: {self.hf_settings.model_name} on device: {self.hf_settings.device}")
            try:
                # Option 1: Using a pipeline (preferred for simplicity if applicable)
                # _LOGGER.info("Attempting to load model using transformers.pipeline...")
                # self.pipeline = transformers.pipeline(
                #     task="document-question-answering", # Or other relevant task like 'object-detection'
                #     model=self.hf_settings.model_name,
                #     tokenizer=self.hf_settings.model_name, # Often tokenizer is same as model
                #     token=self.hf_settings.auth_token,
                #     device=self.hf_settings.device
                # )
                # _LOGGER.info("Hugging Face pipeline loaded successfully.")

                # Option 2: Loading model and tokenizer manually (more control)
                # _LOGGER.info("Attempting to load model and tokenizer manually...")
                # self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                #     self.hf_settings.model_name, token=self.hf_settings.auth_token
                # )
                # self.model = transformers.AutoModelForDocumentQuestionAnswering.from_pretrained( # Or other AutoModel class
                #     self.hf_settings.model_name, token=self.hf_settings.auth_token
                # )
                # self.model.to(self.hf_settings.device)
                # self.model.eval() # Set to evaluation mode
                # _LOGGER.info("Hugging Face model and tokenizer loaded successfully and moved to device.")

                # For this subtask: Placeholder message instead of actual loading
                _LOGGER.info("MODEL LOADING PLACEHOLDER: Hugging Face model loading would occur here.")
                _LOGGER.info(f"MODEL LOADING PLACEHOLDER: Model Name: {self.hf_settings.model_name}, Auth Token Used: {'Yes' if self.hf_settings.auth_token else 'No'}, Device: {self.hf_settings.device}")
                # Simulate successful loading for structure testing:
                # self.model = "mock_model_loaded" # Replace with actual model object after uncommenting
                # self.tokenizer = "mock_tokenizer_loaded" # Replace with actual tokenizer object

            except Exception as e:
                _LOGGER.error(f"Failed to load Hugging Face model '{self.hf_settings.model_name}': {e}", exc_info=True)
                # This exception will be caught by analyze_document and reported in ParsedDocument.errors
                raise RuntimeError(f"Failed to load Hugging Face model '{self.hf_settings.model_name}': {e}")

    def analyze_document(self, file_path: str, options: dict) -> ParsedDocument:
        """Analyzes a document using a Hugging Face model (placeholder implementation).

        Args:
            file_path: The path to the document (PDF or image).
            options: Dictionary of options, including:
                     'doc_type' (str): "pdf" or "image".
                     'start_page_idx' (int): Start page index for the context of this file.
                     'end_page_idx' (int): End page index for the context of this file.
        Returns:
            A ParsedDocument object.
        """
        _LOGGER.info("Starting HuggingFaceProvider.analyze_document", file_path=file_path)

        # doc_type from options refers to the original document type if it was part of a larger one (e.g. a PDF chunk)
        # For HuggingFace, we mostly care about the file_path's actual type for processing.
        # start_page_idx and end_page_idx from options are for context if this is a sub-document.
        # The actual page numbers for chunks will be relative to the start of this specific file_path.

        original_doc_type = options.get("doc_type", "image") # Default based on typical single file
        # For ParsedDocument, start/end_page_idx should reflect the indices within the *original* document if part of a split.
        # If file_path is a standalone doc, these would typically be 0 to num_pages-1.
        # The options dict should provide these values correctly.
        parsed_doc_start_page_idx = options.get("start_page_idx", 0)
        parsed_doc_end_page_idx = options.get("end_page_idx", 0)

        all_chunks: List[Chunk] = []
        all_markdown_parts: List[str] = []
        errors: List[PageError] = []
        images_to_process: List[PIL.Image.Image] = []

        try:
            self._ensure_model_loaded() # Load model if not already loaded

            input_path = Path(file_path)
            file_extension = input_path.suffix.lower()

            if file_extension == ".pdf":
                _LOGGER.info("Processing PDF for HuggingFaceProvider", file_path=file_path)
                # pdf2image handles multi-page PDFs.
                # For a chunked PDF (where file_path is a single-page or few-page PDF chunk),
                # this will convert those pages.
                # The page numbers for grounding should be relative to the start of *this* PDF file.
                # The final ParsedDocument's start/end_page_idx will use the values from `options`.
                with tempfile.TemporaryDirectory() as temp_pdf_image_dir:
                    images_to_process = pdf2image.convert_from_path(
                        input_path, output_folder=temp_pdf_image_dir
                    )
                if not images_to_process:
                    _LOGGER.warn("pdf2image converted PDF to 0 images.", file_path=file_path)
                    errors.append(PageError(page_num=parsed_doc_start_page_idx, error="PDF conversion resulted in no images.", error_code=-1))

            elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                _LOGGER.info("Processing Image for HuggingFaceProvider", file_path=file_path)
                img = PIL.Image.open(input_path).convert("RGB")
                images_to_process.append(img)
            else:
                _LOGGER.error(f"Unsupported file type for HuggingFaceProvider: {file_extension}", file_path=file_path)
                errors.append(PageError(page_num=parsed_doc_start_page_idx, error=f"Unsupported file type: {file_extension}", error_code=-1))
                # Return early if unsupported file type
                return ParsedDocument(
                    markdown="", chunks=[], doc_type=original_doc_type,
                    start_page_idx=parsed_doc_start_page_idx, end_page_idx=parsed_doc_end_page_idx, errors=errors
                )

            # Placeholder processing logic for each image (page)
            for page_idx_in_file, image in enumerate(images_to_process):
                current_original_page_num = parsed_doc_start_page_idx + page_idx_in_file # Page number in the context of the original document
                _LOGGER.info(f"HUGGINGFACE MODEL PLACEHOLDER: Simulating inference for page {page_idx_in_file} of file {file_path} (Original Page Num: {current_original_page_num}).",
                             model_name=self.hf_settings.model_name)

                # Simulate some text output
                placeholder_text = f"Placeholder text from HuggingFace model for page {page_idx_in_file + 1} of input file '{input_path.name}'."

                # Create a dummy chunk
                dummy_chunk = Chunk(
                    text=placeholder_text,
                    chunk_type=ChunkType.text,
                    chunk_id=str(uuid.uuid4()),
                    grounding=[
                        ChunkGrounding(
                            page=page_idx_in_file, # Page index *within this specific file*
                            box=ChunkGroundingBox(l=0.1, t=0.1, r=0.5, b=0.2)
                        )
                    ]
                )
                all_chunks.append(dummy_chunk)
                all_markdown_parts.append(f"## Page {page_idx_in_file + 1} (Original Page: {current_original_page_num})\n\n{placeholder_text}\n")

            # If this file was a multi-page PDF chunk, end_page_idx should be adjusted
            # based on number of pages processed.
            # However, the problem statement implies options.end_page_idx is authoritative for the ParsedDocument.
            # If images_to_process is empty and it was a PDF, it means conversion failed.
            # If it was an image, and images_to_process is empty, it means it was an unsupported type handled above.

            # Adjust end_page_idx for the ParsedDocument if the number of processed pages is less than expected
            # This is tricky because parsed_doc_end_page_idx comes from options.
            # For now, we trust options, but real processing might need to adjust this if a 3-page PDF chunk only yields 1 page.
            # For this placeholder, we assume options are correct for the scope of this file_path.

            final_markdown = "\n".join(all_markdown_parts)

            # If no images were processed and there were no errors yet, add a generic one.
            if not images_to_process and not errors and input_path.suffix.lower() == ".pdf":
                 _LOGGER.warn("No images were processed from the PDF, though no specific pdf2image error was caught.", file_path=file_path)
                 # This case should ideally be caught by pdf2image error handling if it returns empty list with no exception.
                 errors.append(PageError(page_num=parsed_doc_start_page_idx, error="No content processed from PDF.", error_code=-1))


        except RuntimeError as e: # Catch model loading errors from _ensure_model_loaded
            _LOGGER.error(f"Runtime error during HuggingFace analysis (likely model loading): {e}", file_path=file_path, exc_info=True)
            # Create page errors for all affected pages
            for i in range(parsed_doc_start_page_idx, parsed_doc_end_page_idx + 1):
                 errors.append(PageError(page_num=i, error=f"Model loading/runtime error: {e}", error_code=-1))
            final_markdown = "" # No markdown if model fails to load
        except pdf2image.exceptions.PDFInfoNotInstalledError:
            _LOGGER.error("pdfinfo/poppler not installed. pdf2image cannot function.", exc_info=True)
            errors.append(PageError(page_num=parsed_doc_start_page_idx, error="PDF processing dependency (poppler) not installed.", error_code=-1))
            final_markdown = ""
        except Exception as e:
            _LOGGER.error(f"Unexpected error in HuggingFaceProvider.analyze_document: {e}", file_path=file_path, exc_info=True)
            # Create page errors for all affected pages
            for i in range(parsed_doc_start_page_idx, parsed_doc_end_page_idx + 1):
                 errors.append(PageError(page_num=i, error=f"Unexpected error: {e}", error_code=-1))
            final_markdown = "" # No markdown on unexpected errors

        # Clean up temporary image files from pdf2image if they were not in a temp dir by pdf2image
        # pdf2image with output_folder to a temp dir handles its own cleanup when the temp dir context exits.
        # If images_to_process contains PIL Image objects, they are in memory.

        return ParsedDocument(
            markdown=final_markdown,
            chunks=all_chunks,
            doc_type=original_doc_type, # Use the original doc_type from options for consistency
            start_page_idx=parsed_doc_start_page_idx, # from options
            end_page_idx=parsed_doc_end_page_idx,   # from options
            errors=errors,
        )


class OpenAIProvider(BaseAIProvider):
    """AI Provider for OpenAI models like GPT-4 Vision."""

    def __init__(self, openai_settings: Optional[OpenAISettings] = None):
        """Initializes the OpenAIProvider.

        Args:
            openai_settings: Optional OpenAISettings. If not provided,
                             it uses the global settings.openai.
        """
        self.openai_settings = openai_settings or settings.openai
        if not self.openai_settings:
            raise ValueError("OpenAI settings not found in global configuration and not provided.")

        if not self.openai_settings.api_key:
            _LOGGER.warning("OpenAI API key is not set. Analyze_document will fail if called.")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=self.openai_settings.api_key)

        _LOGGER.info("OpenAIProvider initialized.", model_name=self.openai_settings.model_name)

    @staticmethod
    def _pil_to_base64(image: PIL.Image.Image, format="JPEG") -> str:
        """Converts a PIL Image object to a base64 encoded string."""
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_byte = buffered.getvalue()
        return base64.b64encode(img_byte).decode('utf-8')

    def analyze_document(self, file_path: str, options: dict) -> ParsedDocument:
        """Analyzes a document using the OpenAI API (placeholder for actual API call).

        Args:
            file_path: The path to the document (PDF or image).
            options: Dictionary of options, including:
                     'doc_type' (str): "pdf" or "image" (original type if part of split).
                     'start_page_idx' (int): Start page index for ParsedDocument context.
                     'end_page_idx' (int): End page index for ParsedDocument context.
        Returns:
            A ParsedDocument object.
        """
        _LOGGER.info("Starting OpenAIProvider.analyze_document", file_path=file_path)

        if not self.client:
            _LOGGER.error("OpenAI client not initialized because API key is missing.")
            # Create an error for each page expected to be processed
            doc_start_idx = options.get("start_page_idx", 0)
            doc_end_idx = options.get("end_page_idx", 0)
            page_errors = [
                PageError(page_num=i, error="OpenAI API key not configured.", error_code=400)
                for i in range(doc_start_idx, doc_end_idx + 1)
            ]
            return ParsedDocument(
                markdown="", chunks=[], doc_type=options.get("doc_type", "unknown"),
                start_page_idx=doc_start_idx, end_page_idx=doc_end_idx, errors=page_errors
            )

        original_doc_type = options.get("doc_type", "image")
        parsed_doc_start_page_idx = options.get("start_page_idx", 0)
        parsed_doc_end_page_idx = options.get("end_page_idx", 0)

        all_chunks: List[Chunk] = []
        all_markdown_parts: List[str] = []
        all_errors: List[PageError] = []
        images_to_process: List[PIL.Image.Image] = []

        try:
            input_path = Path(file_path)
            file_extension = input_path.suffix.lower()

            if file_extension == ".pdf":
                _LOGGER.info("Processing PDF for OpenAIProvider", file_path=file_path)
                with tempfile.TemporaryDirectory() as temp_dir:
                    images_to_process = pdf2image.convert_from_path(input_path, output_folder=temp_dir)
                if not images_to_process:
                     _LOGGER.warn("pdf2image converted PDF to 0 images.", file_path=file_path)
                     all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error="PDF conversion resulted in no images.", error_code=-1))


            elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                _LOGGER.info("Processing Image for OpenAIProvider", file_path=file_path)
                img = PIL.Image.open(input_path).convert("RGB") # Ensure RGB for JPEG conversion
                images_to_process.append(img)
            else:
                _LOGGER.error(f"Unsupported file type for OpenAIProvider: {file_extension}", file_path=file_path)
                all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error=f"Unsupported file type: {file_extension}", error_code=-1))
                return ParsedDocument(
                    markdown="", chunks=[], doc_type=original_doc_type,
                    start_page_idx=parsed_doc_start_page_idx, end_page_idx=parsed_doc_end_page_idx, errors=all_errors
                )

            MAX_TOKENS_PER_PAGE = options.get("max_tokens_per_page", 2000) # Allow override from options
            PROMPT = options.get("prompt", "Describe this document page in detail, extracting all text content and identifying its structure (paragraphs, headings, tables, figures if any). For each piece of text, if possible, provide its approximate bounding box on the page as [x1, y1, x2, y2] normalized coordinates (0.0 to 1.0).")


            for page_idx_in_file, pil_image in enumerate(images_to_process):
                # current_page_num_in_doc is the page number within the context of the potentially larger, original document
                current_page_num_in_doc = parsed_doc_start_page_idx + page_idx_in_file

                _LOGGER.info(f"Preparing page {page_idx_in_file} of file {file_path} for OpenAI (Original Page: {current_page_num_in_doc}).")
                base64_image_string = self._pil_to_base64(pil_image, format="JPEG") # OpenAI prefers JPEG

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image_string}"
                                }
                            }
                        ]
                    }
                ]

                _LOGGER.info(f"OPENAI API PLACEHOLDER: Simulating API call for page {current_page_num_in_doc}",
                             model_name=self.openai_settings.model_name)

                # ---- BEGIN SIMULATED RESPONSE ----
                page_chunks: List[Chunk] = []
                simulated_text_content = f"Simulated OpenAI text for page {current_page_num_in_doc} from file '{input_path.name}'."

                # Create one dummy chunk for the simulated text
                dummy_chunk = Chunk(
                    text=simulated_text_content,
                    chunk_type=ChunkType.text,
                    chunk_id=str(uuid.uuid4()),
                    grounding=[
                        ChunkGrounding(
                            page=page_idx_in_file, # Page index *within this specific file*
                            box=ChunkGroundingBox(l=0.05, t=0.05, r=0.95, b=0.95) # Full page box
                        )
                    ]
                )
                page_chunks.append(dummy_chunk)
                page_markdown = f"## Page {current_page_num_in_doc} (File Page: {page_idx_in_file + 1})\n\n{simulated_text_content}\n"
                # ---- END SIMULATED RESPONSE ----

                # ---- BEGIN ACTUAL API CALL (Commented out for this subtask) ----
                # try:
                #     api_response = self.client.chat.completions.create(
                #         model=self.openai_settings.model_name,
                #         messages=messages,
                #         max_tokens=MAX_TOKENS_PER_PAGE
                #     )
                #     # Actual response parsing would go here:
                #     # response_content = api_response.choices[0].message.content
                #     # page_markdown, page_chunks = self._parse_openai_response(response_content, page_idx_in_file)
                #     _LOGGER.info(f"OpenAI API call successful for page {current_page_num_in_doc}.")
                # except openai.APIError as e:
                #     _LOGGER.error(f"OpenAI API error on page {current_page_num_in_doc} of {file_path}: {e}", exc_info=True)
                #     all_errors.append(PageError(page_num=current_page_num_in_doc, error=f"OpenAI API Error: {e}", error_code=e.status_code if hasattr(e, 'status_code') else 500))
                #     continue # to next page
                # except Exception as e:
                #     _LOGGER.error(f"Generic error during OpenAI call for page {current_page_num_in_doc} of {file_path}: {e}", exc_info=True)
                #     all_errors.append(PageError(page_num=current_page_num_in_doc, error=f"Generic processing error: {e}", error_code=-1))
                #     continue # to next page
                # ---- END ACTUAL API CALL ----

                all_chunks.extend(page_chunks)
                all_markdown_parts.append(page_markdown)

            final_markdown = "\n".join(all_markdown_parts)
            if not images_to_process and not all_errors and input_path.suffix.lower() == ".pdf":
                 _LOGGER.warn("No images were processed from the PDF for OpenAI, though no specific pdf2image error was caught.", file_path=file_path)
                 all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error="No content processed from PDF for OpenAI.", error_code=-1))


        except pdf2image.exceptions.PDFInfoNotInstalledError:
            _LOGGER.error("pdfinfo/poppler not installed. pdf2image cannot function for OpenAIProvider.", exc_info=True)
            all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error="PDF processing dependency (poppler) not installed.", error_code=-1))
            final_markdown = ""
        except Exception as e:
            _LOGGER.error(f"Unexpected error in OpenAIProvider.analyze_document: {e}", file_path=file_path, exc_info=True)
            # Add error for all affected pages if it's a general error
            for i in range(parsed_doc_start_page_idx, parsed_doc_end_page_idx + 1):
                if not any(err.page_num == i for err in all_errors): # Avoid duplicate errors for same page
                    all_errors.append(PageError(page_num=i, error=f"Unexpected error: {e}", error_code=-1))
            final_markdown = ""


        return ParsedDocument(
            markdown=final_markdown,
            chunks=all_chunks,
            doc_type=original_doc_type,
            start_page_idx=parsed_doc_start_page_idx,
            end_page_idx=parsed_doc_end_page_idx,
            errors=all_errors,
        )

    # def _parse_openai_response(self, response_content: str, page_idx_in_file: int) -> (str, List[Chunk]):
    #     """
    #     Placeholder for parsing the complex string response from GPT-4V into markdown and structured chunks.
    #     This will be a significant piece of work.
    #     """
    #     _LOGGER.info("Placeholder: Parsing OpenAI response content.", page_idx_in_file=page_idx_in_file)
    #     # For now, just return basic stuff
    #     markdown = f"## Page {page_idx_in_file + 1}\n\n{response_content}\n"
    #     chunks = [
    #         Chunk(
    #             text=response_content,
    #             chunk_type=ChunkType.text,
    #             chunk_id=str(uuid.uuid4()),
    #             grounding=[ChunkGrounding(page=page_idx_in_file, box=ChunkGroundingBox(l=0,t=0,r=1,b=1))] # Whole page
    #         )
    #     ]
    #     return markdown, chunks


def get_ai_provider() -> BaseAIProvider:
    """
    Factory function to get the appropriate AI provider based on global settings.

    Returns:
        An instance of a class that inherits from BaseAIProvider.

    Raises:
        ValueError: If the provider type is unsupported or if essential settings
                    (like API key or model name) for the selected provider are missing.
    """
    provider_type = settings.ai_provider_type.lower()
    _LOGGER.info(f"Attempting to get AI provider for type: {provider_type}")

    if provider_type == "landingai":
        if not settings.landingai or not settings.landingai.api_key:
            err_msg = "LandingAI provider selected, but API key is missing in settings."
            _LOGGER.error(err_msg)
            raise ValueError(err_msg)
        _LOGGER.info("Returning LandingAIProvider.")
        return LandingAIProvider(settings.landingai)
    elif provider_type == "huggingface":
        if not settings.huggingface or not settings.huggingface.model_name:
            err_msg = "HuggingFace provider selected, but model_name is missing in settings."
            _LOGGER.error(err_msg)
            raise ValueError(err_msg)
        _LOGGER.info("Returning HuggingFaceProvider.")
        return HuggingFaceProvider(settings.huggingface)
    elif provider_type == "openai":
        if not settings.openai or not settings.openai.api_key:
            err_msg = "OpenAI provider selected, but API key is missing in settings."
            _LOGGER.error(err_msg)
            raise ValueError(err_msg)
        _LOGGER.info("Returning OpenAIProvider.")
        return OpenAIProvider(settings.openai)
    elif provider_type == "googleai":
        if not settings.googleai or not settings.googleai.api_key:
            err_msg = "GoogleAI provider selected, but API key is missing in settings."
            _LOGGER.error(err_msg)
            raise ValueError(err_msg)
        _LOGGER.info("Returning GoogleAIProvider.")
        return GoogleAIProvider(settings.googleai)
    elif provider_type == "anthropic":
        if not settings.anthropic or not settings.anthropic.api_key:
            err_msg = "Anthropic provider selected, but API key is missing in settings."
            _LOGGER.error(err_msg)
            raise ValueError(err_msg)
        _LOGGER.info("Returning AnthropicProvider.")
        return AnthropicProvider(settings.anthropic)
    else:
        err_msg = f"Unsupported AI provider type: {provider_type}"
        _LOGGER.error(err_msg)
        raise ValueError(err_msg)


class AnthropicProvider(BaseAIProvider):
    """AI Provider for Anthropic's Claude models."""

    def __init__(self, anthropic_settings: Optional[AnthropicSettings] = None):
        """Initializes the AnthropicProvider.

        Args:
            anthropic_settings: Optional AnthropicSettings. If not provided,
                                it uses the global settings.anthropic.
        """
        self.anthropic_settings = anthropic_settings or settings.anthropic
        if not self.anthropic_settings:
            raise ValueError("Anthropic settings not found in global configuration and not provided.")

        if not self.anthropic_settings.api_key:
            _LOGGER.warning("Anthropic API key is not set. Analyze_document will fail if called.")
            self.client = None
        else:
            try:
                self.client = anthropic.Anthropic(api_key=self.anthropic_settings.api_key)
                _LOGGER.info("AnthropicProvider initialized.", model_name=self.anthropic_settings.model_name)
            except Exception as e:
                _LOGGER.error(f"Failed to initialize Anthropic client: {e}", exc_info=True)
                self.client = None

    @staticmethod
    def _pil_to_base64(image: PIL.Image.Image, format="JPEG") -> str:
        """Converts a PIL Image object to a base64 encoded string."""
        # This helper can be moved to a common utility module if used by multiple providers
        buffered = io.BytesIO()
        # Ensure image is in RGB if saving as JPEG
        if format == "JPEG" and image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format=format)
        img_byte = buffered.getvalue()
        return base64.b64encode(img_byte).decode('utf-8')

    def analyze_document(self, file_path: str, options: dict) -> ParsedDocument:
        """Analyzes a document using the Anthropic API (placeholder for actual API call).

        Args:
            file_path: The path to the document (PDF or image).
            options: Dictionary of options.
        Returns:
            A ParsedDocument object.
        """
        _LOGGER.info("Starting AnthropicProvider.analyze_document", file_path=file_path)

        if not self.client:
            _LOGGER.error("Anthropic client not initialized (likely missing API key or init error).")
            doc_start_idx = options.get("start_page_idx", 0)
            doc_end_idx = options.get("end_page_idx", 0)
            page_errors = [
                PageError(page_num=i, error="Anthropic client not configured.", error_code=400)
                for i in range(doc_start_idx, doc_end_idx + 1)
            ]
            return ParsedDocument(
                markdown="", chunks=[], doc_type=options.get("doc_type", "unknown"),
                start_page_idx=doc_start_idx, end_page_idx=doc_end_idx, errors=page_errors
            )

        original_doc_type = options.get("doc_type", "image")
        parsed_doc_start_page_idx = options.get("start_page_idx", 0)
        parsed_doc_end_page_idx = options.get("end_page_idx", 0)

        all_chunks: List[Chunk] = []
        all_markdown_parts: List[str] = []
        all_errors: List[PageError] = []
        images_to_process: List[PIL.Image.Image] = []

        try:
            input_path = Path(file_path)
            file_extension = input_path.suffix.lower()

            if file_extension == ".pdf":
                _LOGGER.info("Processing PDF for AnthropicProvider", file_path=file_path)
                with tempfile.TemporaryDirectory() as temp_dir:
                    images_to_process = pdf2image.convert_from_path(input_path, output_folder=temp_dir)
                if not images_to_process:
                    _LOGGER.warn("pdf2image converted PDF to 0 images for Anthropic.", file_path=file_path)
                    all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error="PDF conversion resulted in no images.", error_code=-1))

            elif file_extension in [".png", ".jpg", ".jpeg", ".webp"]: # Claude 3 supports these
                _LOGGER.info("Processing Image for AnthropicProvider", file_path=file_path)
                img = PIL.Image.open(input_path)
                images_to_process.append(img)
            else:
                _LOGGER.error(f"Unsupported file type for AnthropicProvider: {file_extension}", file_path=file_path)
                all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error=f"Unsupported file type: {file_extension}", error_code=-1))
                return ParsedDocument(
                    markdown="", chunks=[], doc_type=original_doc_type,
                    start_page_idx=parsed_doc_start_page_idx, end_page_idx=parsed_doc_end_page_idx, errors=all_errors
                )

            PROMPT = options.get("prompt", "Describe this document page in detail, focusing on text content, layout, and any structural elements like tables or figures. If possible, provide bounding boxes for identified text segments as normalized [x1, y1, x2, y2] coordinates.")
            MAX_TOKENS_PER_PAGE = options.get("max_tokens_per_page", 2000) # Claude 3 Opus has 200k context, but output limit per call is lower

            for page_idx_in_file, pil_image in enumerate(images_to_process):
                current_page_num_in_doc = parsed_doc_start_page_idx + page_idx_in_file
                _LOGGER.info(f"Preparing page {page_idx_in_file} of file {file_path} for Anthropic (Original Page: {current_page_num_in_doc}).")

                # Determine media type and save format
                image_format = "JPEG"
                media_type = "image/jpeg"
                if pil_image.format == "PNG":
                    image_format = "PNG"
                    media_type = "image/png"
                elif pil_image.format == "WEBP": # Requires Pillow to be built with WebP support
                    image_format = "WEBP"
                    media_type = "image/webp"

                # Ensure image is in RGB for JPEG, can be RGBA for PNG/WEBP
                if image_format == "JPEG" and pil_image.mode != "RGB":
                    pil_image_to_encode = pil_image.convert("RGB")
                else:
                    pil_image_to_encode = pil_image

                base64_image_data = AnthropicProvider._pil_to_base64(pil_image_to_encode, format=image_format)

                messages_payload = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image_data,
                                },
                            },
                            {"type": "text", "text": PROMPT}
                        ],
                    }
                ]

                _LOGGER.info(f"ANTHROPIC API PLACEHOLDER: Simulating API call for page {current_page_num_in_doc}",
                             model_name=self.anthropic_settings.model_name)

                # ---- BEGIN SIMULATED RESPONSE ----
                page_chunks: List[Chunk] = []
                simulated_text_content = f"Simulated Anthropic Claude text for page {current_page_num_in_doc} from file '{input_path.name}'."

                dummy_chunk = Chunk(
                    text=simulated_text_content,
                    chunk_type=ChunkType.text,
                    chunk_id=str(uuid.uuid4()),
                    grounding=[
                        ChunkGrounding(
                            page=page_idx_in_file,
                            box=ChunkGroundingBox(l=0.05, t=0.05, r=0.95, b=0.95) # Full page
                        )
                    ]
                )
                page_chunks.append(dummy_chunk)
                page_markdown = f"## Page {current_page_num_in_doc} (File Page: {page_idx_in_file + 1})\n\n{simulated_text_content}\n"
                # ---- END SIMULATED RESPONSE ----

                # ---- BEGIN ACTUAL API CALL (Commented out for this subtask) ----
                # try:
                #     response = self.client.messages.create(
                #         model=self.anthropic_settings.model_name,
                #         max_tokens=MAX_TOKENS_PER_PAGE,
                #         messages=messages_payload
                #     )
                #     # Assuming response.content is a list of blocks, and first block is text
                #     # For Claude 3, the response structure is like: response.content = [ContentBlock(type="text", text="...")]
                #     response_text = ""
                #     if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                #          response_text = response.content[0].text
                #     else:
                #         _LOGGER.warn("Unexpected Anthropic response structure", response_raw=response)
                #     # page_markdown, page_chunks = self._parse_anthropic_response(response_text, page_idx_in_file)
                #     _LOGGER.info(f"Anthropic API call successful for page {current_page_num_in_doc}.")
                # except anthropic.APIError as e:
                #     _LOGGER.error(f"Anthropic API error on page {current_page_num_in_doc} of {file_path}: {e}", exc_info=True)
                #     all_errors.append(PageError(page_num=current_page_num_in_doc, error=f"Anthropic API Error: {e}", error_code=e.status_code if hasattr(e, 'status_code') else 500))
                #     continue
                # except Exception as e:
                #     _LOGGER.error(f"Generic error during Anthropic call for page {current_page_num_in_doc} of {file_path}: {e}", exc_info=True)
                #     all_errors.append(PageError(page_num=current_page_num_in_doc, error=f"Generic processing error: {e}", error_code=-1))
                #     continue
                # ---- END ACTUAL API CALL ----

                all_chunks.extend(page_chunks)
                all_markdown_parts.append(page_markdown)

            final_markdown = "\n".join(all_markdown_parts)
            if not images_to_process and not all_errors and input_path.suffix.lower() == ".pdf":
                 _LOGGER.warn("No images were processed from the PDF for Anthropic, though no specific pdf2image error was caught.", file_path=file_path)
                 all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error="No content processed from PDF for Anthropic.", error_code=-1))

        except pdf2image.exceptions.PDFInfoNotInstalledError:
            _LOGGER.error("pdfinfo/poppler not installed for AnthropicProvider.", exc_info=True)
            all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error="PDF processing dependency (poppler) not installed.", error_code=-1))
            final_markdown = ""
        except Exception as e:
            _LOGGER.error(f"Unexpected error in AnthropicProvider.analyze_document: {e}", file_path=file_path, exc_info=True)
            for i in range(parsed_doc_start_page_idx, parsed_doc_end_page_idx + 1):
                if not any(err.page_num == i for err in all_errors):
                    all_errors.append(PageError(page_num=i, error=f"Unexpected error: {e}", error_code=-1))
            final_markdown = ""

        return ParsedDocument(
            markdown=final_markdown,
            chunks=all_chunks,
            doc_type=original_doc_type,
            start_page_idx=parsed_doc_start_page_idx,
            end_page_idx=parsed_doc_end_page_idx,
            errors=all_errors,
        )

    # def _parse_anthropic_response(self, response_text: str, page_idx_in_file: int) -> (str, List[Chunk]):
    #     """Placeholder for parsing Anthropic Claude response."""
    #     _LOGGER.info("Placeholder: Parsing Anthropic response.", page_idx_in_file=page_idx_in_file)
    #     markdown = f"## Page {page_idx_in_file + 1}\n\n{response_text}\n"
    #     chunks = [Chunk(text=response_text, chunk_type=ChunkType.text, chunk_id=str(uuid.uuid4()),
    #                     grounding=[ChunkGrounding(page=page_idx_in_file, box=ChunkGroundingBox(l=0,t=0,r=1,b=1))])]
    #     return markdown, chunks


class GoogleAIProvider(BaseAIProvider):
    """AI Provider for Google AI models like Gemini."""

    def __init__(self, googleai_settings: Optional[GoogleAISettings] = None):
        """Initializes the GoogleAIProvider.

        Args:
            googleai_settings: Optional GoogleAISettings. If not provided,
                               it uses the global settings.googleai.
        """
        self.googleai_settings = googleai_settings or settings.googleai
        if not self.googleai_settings:
            raise ValueError("GoogleAI settings not found in global configuration and not provided.")

        if not self.googleai_settings.api_key:
            _LOGGER.warning("Google AI API key is not set. Analyze_document will fail if called.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.googleai_settings.api_key)
                self.model = genai.GenerativeModel(self.googleai_settings.model_name)
                _LOGGER.info("GoogleAIProvider initialized and model configured.", model_name=self.googleai_settings.model_name)
            except Exception as e:
                _LOGGER.error(f"Failed to configure Google AI or instantiate model: {e}", exc_info=True)
                self.model = None


    def analyze_document(self, file_path: str, options: dict) -> ParsedDocument:
        """Analyzes a document using the Google AI API (placeholder for actual API call).

        Args:
            file_path: The path to the document (PDF or image).
            options: Dictionary of options, including:
                     'doc_type' (str): "pdf" or "image" (original type if part of split).
                     'start_page_idx' (int): Start page index for ParsedDocument context.
                     'end_page_idx' (int): End page index for ParsedDocument context.
        Returns:
            A ParsedDocument object.
        """
        _LOGGER.info("Starting GoogleAIProvider.analyze_document", file_path=file_path)

        if not self.model:
            _LOGGER.error("Google AI model not initialized (likely missing API key or configuration error).")
            doc_start_idx = options.get("start_page_idx", 0)
            doc_end_idx = options.get("end_page_idx", 0)
            page_errors = [
                PageError(page_num=i, error="Google AI API key/model not configured.", error_code=400)
                for i in range(doc_start_idx, doc_end_idx + 1)
            ]
            return ParsedDocument(
                markdown="", chunks=[], doc_type=options.get("doc_type", "unknown"),
                start_page_idx=doc_start_idx, end_page_idx=doc_end_idx, errors=page_errors
            )

        original_doc_type = options.get("doc_type", "image")
        parsed_doc_start_page_idx = options.get("start_page_idx", 0)
        parsed_doc_end_page_idx = options.get("end_page_idx", 0)

        all_chunks: List[Chunk] = []
        all_markdown_parts: List[str] = []
        all_errors: List[PageError] = []
        images_to_process: List[PIL.Image.Image] = []

        try:
            input_path = Path(file_path)
            file_extension = input_path.suffix.lower()

            if file_extension == ".pdf":
                _LOGGER.info("Processing PDF for GoogleAIProvider", file_path=file_path)
                with tempfile.TemporaryDirectory() as temp_dir:
                    images_to_process = pdf2image.convert_from_path(input_path, output_folder=temp_dir)
                if not images_to_process:
                    _LOGGER.warn("pdf2image converted PDF to 0 images for GoogleAI.", file_path=file_path)
                    all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error="PDF conversion resulted in no images.", error_code=-1))

            elif file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]: # Gemini supports various image types
                _LOGGER.info("Processing Image for GoogleAIProvider", file_path=file_path)
                # Gemini can take bytes directly, ensure image is in a supported format like JPEG or PNG
                img = PIL.Image.open(input_path)
                images_to_process.append(img)
            else:
                _LOGGER.error(f"Unsupported file type for GoogleAIProvider: {file_extension}", file_path=file_path)
                all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error=f"Unsupported file type: {file_extension}", error_code=-1))
                return ParsedDocument(
                    markdown="", chunks=[], doc_type=original_doc_type,
                    start_page_idx=parsed_doc_start_page_idx, end_page_idx=parsed_doc_end_page_idx, errors=all_errors
                )

            PROMPT = options.get("prompt", "Describe this document page in detail, extracting all text content and identifying its structure. For text, provide bounding boxes if possible as normalized [x1,y1,x2,y2].")

            for page_idx_in_file, pil_image in enumerate(images_to_process):
                current_page_num_in_doc = parsed_doc_start_page_idx + page_idx_in_file
                _LOGGER.info(f"Preparing page {page_idx_in_file} of file {file_path} for Google AI (Original Page: {current_page_num_in_doc}).")

                # Convert PIL image to bytes for Google AI API
                # Gemini prefers JPEG or PNG. Let's use JPEG as a common default.
                # Ensure image is in RGB if saving as JPEG
                if pil_image.mode != "RGB":
                    pil_image_to_send = pil_image.convert("RGB")
                else:
                    pil_image_to_send = pil_image

                image_bytes_io = io.BytesIO()
                pil_image_to_send.save(image_bytes_io, format="JPEG") # Gemini also supports PNG, WEBP, HEIC, HEIF
                image_bytes = image_bytes_io.getvalue()

                # Prepare content parts for Gemini API
                # The new API prefers a list of parts directly.
                # Example: [prompt_text, {"mime_type": "image/jpeg", "data": image_bytes}, prompt_text_continued]
                # For simple cases: [prompt, image]

                # For Gemini 1.0 Pro Vision (gemini-pro-vision), the input is [string, vision_part, string, vision_part,...]
                # For Gemini 1.5 Pro, the input can be a list of Parts (text or inline_data)
                # Assuming we are targeting a model that can take [prompt, image] or [image, prompt]

                # Let's use the structure shown in Google AI SDK examples:
                # model.generate_content([prompt, image_part]) or model.generate_content([image_part, prompt])
                # depending on model specifics. gemini-pro-vision typically takes [prompt, image1, image2 ...]
                # but here we process one image at a time.

                # Using a structure that should be compatible with generate_content for single image + prompt
                # The genai.GenerativeModel().generate_content() method takes a list of content parts.
                # A part can be a string or a dictionary for inline data.
                # For gemini-pro-vision, it's often [prompt_string, image_blob1, image_blob2, ...]
                # Since we process page by page, it will be [prompt_string, image_blob_for_page]

                # The 'blob' part for an image for genai SDK:
                image_part_for_api = {
                    "mime_type": "image/jpeg", # Or image/png
                    "data": image_bytes
                }
                contents_for_api = [PROMPT, image_part_for_api]


                _LOGGER.info(f"GOOGLE AI API PLACEHOLDER: Simulating API call for page {current_page_num_in_doc}",
                             model_name=self.googleai_settings.model_name)

                # ---- BEGIN SIMULATED RESPONSE ----
                page_chunks: List[Chunk] = []
                simulated_text_content = f"Simulated Google AI (Gemini) text for page {current_page_num_in_doc} from file '{input_path.name}'."

                dummy_chunk = Chunk(
                    text=simulated_text_content,
                    chunk_type=ChunkType.text,
                    chunk_id=str(uuid.uuid4()),
                    grounding=[
                        ChunkGrounding(
                            page=page_idx_in_file, # Page index *within this specific file*
                            box=ChunkGroundingBox(l=0.05, t=0.05, r=0.95, b=0.95) # Full page box
                        )
                    ]
                )
                page_chunks.append(dummy_chunk)
                page_markdown = f"## Page {current_page_num_in_doc} (File Page: {page_idx_in_file + 1})\n\n{simulated_text_content}\n"
                # ---- END SIMULATED RESPONSE ----

                # ---- BEGIN ACTUAL API CALL (Commented out for this subtask) ----
                # try:
                #     # For streaming: response = self.model.generate_content(contents_for_api, stream=True)
                #     # For non-streaming:
                #     response = self.model.generate_content(contents_for_api)
                #     # response.resolve() # If any part of the response failed, this will raise an error.
                #     # Actual response parsing would go here:
                #     # response_text = response.text # This is the simple case.
                #     # For more complex outputs (e.g. if model is tuned for function calling or structured output):
                #     # response_parts = response.parts
                #     # page_markdown, page_chunks = self._parse_googleai_response(response_text, page_idx_in_file) # or response_parts
                #     _LOGGER.info(f"Google AI API call successful for page {current_page_num_in_doc}.")
                # except Exception as e: # Catching general Exception, specific Google AI errors can be caught too
                #     _LOGGER.error(f"Google AI API error on page {current_page_num_in_doc} of {file_path}: {e}", exc_info=True)
                #     # Check if e has specific attributes for error codes if using google.api_core.exceptions
                #     error_code_to_report = -1
                #     if hasattr(e, 'code'): # For gRPC errors
                #          error_code_to_report = e.code()
                #     elif hasattr(e, 'status_code'): # For HTTP errors (less common with this SDK path)
                #          error_code_to_report = e.status_code
                #     all_errors.append(PageError(page_num=current_page_num_in_doc, error=f"Google AI API Error: {e}", error_code=error_code_to_report))
                #     continue # to next page
                # ---- END ACTUAL API CALL ----

                all_chunks.extend(page_chunks)
                all_markdown_parts.append(page_markdown)

            final_markdown = "\n".join(all_markdown_parts)
            if not images_to_process and not all_errors and input_path.suffix.lower() == ".pdf":
                 _LOGGER.warn("No images were processed from the PDF for GoogleAI, though no specific pdf2image error was caught.", file_path=file_path)
                 all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error="No content processed from PDF for GoogleAI.", error_code=-1))


        except pdf2image.exceptions.PDFInfoNotInstalledError:
            _LOGGER.error("pdfinfo/poppler not installed. pdf2image cannot function for GoogleAIProvider.", exc_info=True)
            all_errors.append(PageError(page_num=parsed_doc_start_page_idx, error="PDF processing dependency (poppler) not installed.", error_code=-1))
            final_markdown = ""
        except Exception as e:
            _LOGGER.error(f"Unexpected error in GoogleAIProvider.analyze_document: {e}", file_path=file_path, exc_info=True)
            for i in range(parsed_doc_start_page_idx, parsed_doc_end_page_idx + 1):
                if not any(err.page_num == i for err in all_errors):
                    all_errors.append(PageError(page_num=i, error=f"Unexpected error: {e}", error_code=-1))
            final_markdown = ""

        return ParsedDocument(
            markdown=final_markdown,
            chunks=all_chunks,
            doc_type=original_doc_type,
            start_page_idx=parsed_doc_start_page_idx,
            end_page_idx=parsed_doc_end_page_idx,
            errors=all_errors,
        )

    # def _parse_googleai_response(self, response_text: str, page_idx_in_file: int) -> (str, List[Chunk]):
    #     """Placeholder for parsing Google AI response."""
    #     _LOGGER.info("Placeholder: Parsing Google AI response.", page_idx_in_file=page_idx_in_file)
    #     markdown = f"## Page {page_idx_in_file + 1}\n\n{response_text}\n"
    #     chunks = [Chunk(text=response_text, chunk_type=ChunkType.text, chunk_id=str(uuid.uuid4()),
    #                     grounding=[ChunkGrounding(page=page_idx_in_file, box=ChunkGroundingBox(l=0,t=0,r=1,b=1))])]
    #     return markdown, chunks
