import copy
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Union

import structlog
from pydantic_core import Url
from tqdm import tqdm

from agentic_doc.common import (
    Document,
    PageError,
    ParsedDocument,
)
from agentic_doc.config import settings
from agentic_doc.ai_providers import BaseAIProvider, get_ai_provider
from agentic_doc.connectors import (
    BaseConnector,
    ConnectorConfig,
    create_connector,
)
from agentic_doc.utils import (
    check_endpoint_and_api_key,
    download_file,
    get_file_type,
    is_valid_httpurl,
    log_retry_failure,
    save_groundings_as_images,
    split_pdf,
)

_LOGGER = structlog.getLogger(__name__)
# _ENDPOINT_URL, _LIB_VERSION and hardcoded _AI_PROVIDER are removed.


def parse(
    documents: Union[
        bytes,
        str,
        Path,
        Url,
        List[Union[str, Path, Url]],
        BaseConnector,
        ConnectorConfig,
    ],
    *,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    result_save_dir: Optional[Union[str, Path]] = None,
    grounding_save_dir: Optional[Union[str, Path]] = None,
    connector_path: Optional[str] = None,
    connector_pattern: Optional[str] = None,
) -> List[ParsedDocument]:
    """
    Universal parse function that can handle single documents, lists of documents,
    or documents from various connectors.

    Args:
        documents: Can be:
            - Single document path/URL (str, Path, Url)
            - List of document paths/URLs
            - Connector instance (BaseConnector)
            - Connector configuration (ConnectorConfig)
            - Raw bytes of a document (either PDF or Image bytes)
        include_marginalia: Whether to include marginalia in the analysis
        include_metadata_in_markdown: Whether to include metadata in markdown output
        result_save_dir: Directory to save results
        grounding_save_dir: Directory to save grounding images
        connector_path: Path for connector to search (when using connectors)
        connector_pattern: Pattern to filter files (when using connectors)

    Returns:
        List[ParsedDocument]
    """
    # Get the AI provider using the factory function
    ai_provider = get_ai_provider()

    # Convert input to list of document paths
    doc_paths = _get_document_paths(documents, connector_path, connector_pattern)

    if not doc_paths:
        _LOGGER.warning("No documents to parse")
        return []

    # Parse all documents
    parse_results = _parse_document_list(
        doc_paths,
        ai_provider=ai_provider, # Pass the provider
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        result_save_dir=result_save_dir,
        grounding_save_dir=grounding_save_dir,
    )

    # Convert results to ParsedDocument objects
    return _convert_to_parsed_documents(parse_results, result_save_dir)


def _get_document_paths(
    documents: Union[
        bytes,
        str,
        Path,
        Url,
        List[Union[str, Path, Url]],
        BaseConnector,
        ConnectorConfig,
    ],
    connector_path: Optional[str] = None,
    connector_pattern: Optional[str] = None,
) -> Sequence[Union[str, Path, Url]]:
    """Convert various input types to a list of document paths."""
    if isinstance(documents, (BaseConnector, ConnectorConfig)):
        return _get_paths_from_connector(documents, connector_path, connector_pattern)
    elif isinstance(documents, (str, Path, Url)):
        return [documents]
    elif isinstance(documents, list):
        return documents
    elif isinstance(documents, bytes):
        return _get_documents_from_bytes(documents)
    else:
        raise ValueError(f"Unsupported documents type: {type(documents)}")


def _get_paths_from_connector(
    connector_or_config: Union[BaseConnector, ConnectorConfig],
    connector_path: Optional[str],
    connector_pattern: Optional[str],
) -> List[Path]:
    """Download files from connector and return local paths."""
    connector = (
        connector_or_config
        if isinstance(connector_or_config, BaseConnector)
        else create_connector(connector_or_config)
    )

    file_list = connector.list_files(connector_path, connector_pattern)
    if not file_list:
        return []

    local_paths = []
    for file_id in file_list:
        try:
            local_path = connector.download_file(file_id)
            local_paths.append(local_path)
        except Exception as e:
            _LOGGER.error(f"Failed to download file {file_id}: {e}")

    return local_paths


def _get_documents_from_bytes(doc_bytes: bytes) -> List[Path]:
    """Save raw bytes to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(doc_bytes)
        temp_file_path = Path(temp_file.name)
    return [temp_file_path]


def _convert_to_parsed_documents(
    parse_results: Union[List[ParsedDocument], List[Path]],
    result_save_dir: Optional[Union[str, Path]],
) -> List[ParsedDocument]:
    """Convert parse results to ParsedDocument objects."""
    parsed_docs = []

    for result in parse_results:
        if isinstance(result, ParsedDocument):
            parsed_docs.append(result)
        elif isinstance(result, Path):
            with open(result) as f:
                data = json.load(f)
            parsed_doc = ParsedDocument.model_validate(data)
            if result_save_dir:
                parsed_doc.result_path = result
            parsed_docs.append(parsed_doc)
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

    return parsed_docs


def _parse_document_list(
    documents: Sequence[Union[str, Path, Url]],
    *,
    ai_provider: BaseAIProvider, # Added ai_provider
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    result_save_dir: Optional[Union[str, Path]] = None,
    grounding_save_dir: Optional[Union[str, Path]] = None,
) -> Union[List[ParsedDocument], List[Path]]:
    """Helper function to parse a list of documents."""
    documents_list = list(documents)
    if result_save_dir:
        return parse_and_save_documents(
            documents_list,
            ai_provider=ai_provider, # Pass provider
            result_save_dir=result_save_dir,
            grounding_save_dir=grounding_save_dir,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
        )
    else:
        return parse_documents(
            documents_list,
            ai_provider=ai_provider, # Pass provider
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
            grounding_save_dir=grounding_save_dir,
        )


def parse_documents(
    documents: list[Union[str, Path, Url]],
    *,
    ai_provider: BaseAIProvider, # Added ai_provider
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    grounding_save_dir: Union[str, Path, None] = None,
) -> list[ParsedDocument]:
    """
    Parse a list of documents using the specified AI provider.

    Args:
        documents (list[str | Path | Url]): The list of documents to parse.
        ai_provider (BaseAIProvider): The AI provider instance to use for parsing.
        grounding_save_dir (str | Path): The local directory to save the grounding images.
    Returns:
        list[ParsedDocument]: The list of parsed documents.
    """
    _LOGGER.info(f"Parsing {len(documents)} documents")
    _parse_func = partial(
        parse_and_save_document,
        ai_provider=ai_provider, # Pass provider
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        grounding_save_dir=grounding_save_dir,
    )
    with ThreadPoolExecutor(max_workers=settings.batch_size) as executor:
        return list(
            tqdm(
                executor.map(_parse_func, documents),  # type: ignore [arg-type]
                total=len(documents),
                desc="Parsing documents",
            )
        )


def parse_and_save_documents(
    documents: list[Union[str, Path, Url]],
    *,
    ai_provider: BaseAIProvider, # Added ai_provider
    result_save_dir: Union[str, Path],
    grounding_save_dir: Union[str, Path, None] = None,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> list[Path]:
    """
    Parse a list of documents and save the results to a local directory.

    Args:
        documents (list[str | Path | Url]): The list of documents to parse.
        ai_provider (BaseAIProvider): The AI provider instance to use for parsing.
        result_save_dir (str | Path): The local directory to save the results.
        grounding_save_dir (str | Path): The local directory to save the grounding images.
    Returns:
        list[Path]: A list of json file paths to the saved results.
    """
    _LOGGER.info(f"Parsing {len(documents)} documents")
    _parse_func = partial(
        parse_and_save_document,
        ai_provider=ai_provider, # Pass provider
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
        result_save_dir=result_save_dir,
        grounding_save_dir=grounding_save_dir,
    )
    with ThreadPoolExecutor(max_workers=settings.batch_size) as executor:
        return list(
            tqdm(
                executor.map(_parse_func, documents),  # type: ignore [arg-type]
                total=len(documents),
                desc="Parsing documents",
            )
        )


def parse_and_save_document(
    document: Union[str, Path, Url],
    *,
    ai_provider: BaseAIProvider, # Added ai_provider
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
    result_save_dir: Union[str, Path, None] = None,
    grounding_save_dir: Union[str, Path, None] = None,
) -> Union[Path, ParsedDocument]:
    """
    Parse a document and save the results to a local directory.

    Args:
        document (str | Path | Url): The document to parse.
        ai_provider (BaseAIProvider): The AI provider instance to use for parsing.
        result_save_dir (str | Path): The local directory to save the results. If None, the parsed document data is returned.

    Returns:
        Path | ParsedDocument: The file path to the saved result or the parsed document data.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        if isinstance(document, str) and is_valid_httpurl(document):
            document = Url(document)

        if isinstance(document, Url):
            output_file_path = Path(temp_dir) / Path(str(document)).name
            download_file(document, str(output_file_path))
            document = output_file_path
        else:
            document = Path(document)
            if isinstance(document, Path) and not document.exists():
                raise FileNotFoundError(f"File not found: {document}")

        file_type = get_file_type(document)

        if file_type == "image":
            result = _parse_image(
                document,
                ai_provider=ai_provider, # Pass provider
                include_marginalia=include_marginalia,
                include_metadata_in_markdown=include_metadata_in_markdown,
            )
        elif file_type == "pdf":
            result = _parse_pdf(
                document,
                ai_provider=ai_provider, # Pass provider
                include_marginalia=include_marginalia,
                include_metadata_in_markdown=include_metadata_in_markdown,
            )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_name = f"{Path(document).stem}_{ts}"
        if grounding_save_dir:
            grounding_save_dir = Path(grounding_save_dir) / result_name
            save_groundings_as_images(
                document, result.chunks, grounding_save_dir, inplace=True
            )
        if not result_save_dir:
            return result

        result_save_dir = Path(result_save_dir)
        result_save_dir.mkdir(parents=True, exist_ok=True)
        save_path = result_save_dir / f"{result_name}.json"
        save_path.write_text(result.model_dump_json())
        _LOGGER.info(f"Saved the parsed result to '{save_path}'")

        return save_path


def _parse_pdf(
    file_path: Union[str, Path],
    *,
    ai_provider: BaseAIProvider, # Added ai_provider
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> ParsedDocument:
    with tempfile.TemporaryDirectory() as temp_dir:
        parts = split_pdf(file_path, temp_dir, settings.split_size)
        file_path = Path(file_path)
        part_results = _parse_doc_in_parallel(
            parts,
            ai_provider=ai_provider, # Pass provider
            doc_name=file_path.name,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata_in_markdown,
        )
        return _merge_part_results(part_results)


def _parse_image(
    file_path: Union[str, Path],
    *,
    ai_provider: BaseAIProvider, # Added ai_provider
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> ParsedDocument:
    try:
        options = {
            "include_marginalia": include_marginalia,
            "include_metadata_in_markdown": include_metadata_in_markdown,
            "doc_type": "image", # This specific file is an image
            "start_page_idx": 0, # Images are single page in this context
            "end_page_idx": 0,
        }
        parsed_doc = ai_provider.analyze_document(str(file_path), options)
        if parsed_doc.errors:
             for err in parsed_doc.errors:
                _LOGGER.error(f"Provider error parsing image '{file_path}': {err.error_code} - {err.error}", page_num=err.page_num)
        return parsed_doc
    except Exception as e:
        # This catches errors from the analyze_document call itself (e.g., network issues not caught by tenacity, unexpected provider issues)
        # The provider's analyze_document is expected to return a ParsedDocument with errors for API-level errors.
        error_msg = str(e)
        _LOGGER.error(f"Exception during image parsing with AI provider '{file_path}': {error_msg}")
        return ParsedDocument(
            markdown="",
            chunks=[],
            start_page_idx=0,
            end_page_idx=0,
            doc_type="image",
            result_path=None,
            errors=[PageError(page_num=0, error=error_msg, error_code=-1)],
        )


def _merge_part_results(results: list[ParsedDocument]) -> ParsedDocument:
    if not results:
        _LOGGER.warning(
            f"No results to merge: {results}, returning empty ParsedDocument"
        )
        return ParsedDocument(
            markdown="",
            chunks=[],
            start_page_idx=0,
            end_page_idx=0,
            doc_type="pdf",
            result_path=None,
        )

    init_result = copy.deepcopy(results[0])
    for i in range(1, len(results)):
        _merge_next_part(init_result, results[i])

    return init_result


def _merge_next_part(curr: ParsedDocument, next: ParsedDocument) -> None:
    curr.markdown += "\n\n" + next.markdown
    next_chunks = next.chunks
    for chunk in next_chunks:
        for grounding in chunk.grounding:
            grounding.page += next.start_page_idx

    curr.chunks.extend(next_chunks)
    curr.end_page_idx = next.end_page_idx
    curr.errors.extend(next.errors)


def _parse_doc_in_parallel(
    doc_parts: list[Document],
    *,
    ai_provider: BaseAIProvider, # Added ai_provider
    doc_name: str,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> list[ParsedDocument]:
    _parse_func = partial(
        _parse_doc_parts,
        ai_provider=ai_provider, # Pass provider
        include_marginalia=include_marginalia,
        include_metadata_in_markdown=include_metadata_in_markdown,
    )
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        return list(
            tqdm(
                executor.map(_parse_func, doc_parts),
                total=len(doc_parts),
                desc=f"Parsing document parts from '{doc_name}'",
            )
        )


def _parse_doc_parts(
    doc: Document,
    *,
    ai_provider: BaseAIProvider, # Added ai_provider
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> ParsedDocument:
    try:
        _LOGGER.info(f"Start parsing document part with AI provider: '{doc}'")
        options = {
            "include_marginalia": include_marginalia,
            "include_metadata_in_markdown": include_metadata_in_markdown,
            "doc_type": "pdf",  # The original document was a PDF
            "start_page_idx": doc.start_page_idx, # Context for this chunk
            "end_page_idx": doc.end_page_idx,     # Context for this chunk
        }
        # The file at doc.file_path is the actual PDF chunk (potentially multiple pages if split_size > 1)
        # The provider's analyze_document will handle this file.
        parsed_doc = ai_provider.analyze_document(str(doc.file_path), options)
        if parsed_doc.errors:
            for err in parsed_doc.errors:
                _LOGGER.error(f"Provider error parsing doc part '{doc.file_path}': {err.error_code} - {err.error}", page_num=err.page_num)
        else:
            _LOGGER.info(f"Successfully parsed document part with AI provider: '{doc}'")
        return parsed_doc
    except Exception as e:
        # This catches errors from the analyze_document call itself
        error_msg = str(e)
        _LOGGER.error(f"Exception during PDF part parsing with AI provider '{doc}': {error_msg}")
        errors = [
            PageError(page_num=i, error=error_msg, error_code=-1)
            for i in range(doc.start_page_idx, doc.end_page_idx + 1)
        ]
        return ParsedDocument(
            markdown="",
            chunks=[],
            start_page_idx=doc.start_page_idx,
            end_page_idx=doc.end_page_idx,
            doc_type="pdf",
            result_path=Path(doc.file_path),
            errors=errors,
        )
# _send_parsing_request has been removed and its logic moved to LandingAIProvider.analyze_document
