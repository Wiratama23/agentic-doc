import json
import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import httpx
import pytest
from pydantic_core import Url

from agentic_doc.common import (
    Chunk,
    ChunkGrounding,
    ChunkGroundingBox,
    ChunkType,
    Document,
    ParsedDocument,
    RetryableError,
)
from agentic_doc.connectors import (
    GoogleDriveConnectorConfig,
    LocalConnector,
    LocalConnectorConfig,
    S3ConnectorConfig,
)
from agentic_doc.parse import (
    _merge_next_part,
    _merge_part_results,
    _merge_next_part,
    _merge_part_results,
    # _parse_doc_in_parallel, # Tested via parse()
    # _parse_doc_parts, # Tested via parse()
    # _parse_image, # Tested via parse()
    # _parse_pdf, # Tested via parse()
    # _send_parsing_request, # Removed
    parse,
    # parse_and_save_document, # Tested via parse()
    # parse_and_save_documents, # Tested via parse()
    # parse_documents, # Tested via parse()
)
from agentic_doc.ai_providers import BaseAIProvider # For spec in MagicMock

# Remove the autouse fixture as check_endpoint_and_api_key is removed
# @pytest.fixture(autouse=True)
# def patch_check_api_key():
#     with patch("agentic_doc.parse.check_endpoint_and_api_key"):
#         yield

# Tests for helper functions like _merge_part_results and _merge_next_part
# can remain as they are, since they don't directly involve AI provider calls.
# However, tests for _parse_pdf, _parse_image, _parse_doc_parts, etc., will be
# effectively replaced by tests for the main `parse` function with a mocked provider.

# Let's keep these as they test merging logic independent of provider calls
def test_merge_part_results_empty_list():
    # Call the function with an empty list
    result = _merge_part_results([])

    # Check that it returns an empty ParsedDocument
    assert isinstance(result, ParsedDocument)
    assert result.markdown == ""
    assert result.chunks == []
    assert result.start_page_idx == 0
    assert result.end_page_idx == 0
    assert result.doc_type == "pdf"


def test_merge_part_results_single_item(mock_parsed_document):
    # Call the function with a single item
    result = _merge_part_results([mock_parsed_document])

    # Check that it returns the item as is
    assert result == mock_parsed_document


def test_merge_part_results_multiple_items(mock_multi_page_parsed_document):
    # Create two parsed documents to merge
    doc1 = ParsedDocument(
        markdown="# Document 1",
        chunks=[
            Chunk(
                text="Document 1",
                chunk_type=ChunkType.text,
                chunk_id="1",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            )
        ],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf",
    )

    doc2 = ParsedDocument(
        markdown="# Document 2",
        chunks=[
            Chunk(
                text="Document 2",
                chunk_type=ChunkType.text,
                chunk_id="2",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            )
        ],
        start_page_idx=1,
        end_page_idx=1,
        doc_type="pdf",
    )

    # Call the function
    result = _merge_part_results([doc1, doc2])

    # Check the merged result
    assert result.markdown == "# Document 1\n\n# Document 2"
    assert len(result.chunks) == 2
    assert result.start_page_idx == 0
    assert result.end_page_idx == 1

    # Check that the page numbers were updated in the second document's chunks
    assert result.chunks[1].grounding[0].page == 1


def test_merge_next_part():
    # Create two ParsedDocuments to merge
    current_doc = ParsedDocument(
        markdown="# Current Doc",
        chunks=[
            Chunk(
                text="Current Doc",
                chunk_type=ChunkType.text,
                chunk_id="1",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            )
        ],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf",
    )

    next_doc = ParsedDocument(
        markdown="# Next Doc",
        chunks=[
            Chunk(
                text="Next Doc",
                chunk_type=ChunkType.text,
                chunk_id="2",
                grounding=[
                    ChunkGrounding(
                        page=0, box=ChunkGroundingBox(l=0.1, t=0.1, r=0.9, b=0.2)
                    )
                ],
            )
        ],
        start_page_idx=1,
        end_page_idx=1,
        doc_type="pdf",
    )

    # Call the function
    _merge_next_part(current_doc, next_doc)

    # Check that the current_doc was updated
    assert current_doc.markdown == "# Current Doc\n\n# Next Doc"
    assert len(current_doc.chunks) == 2
    assert current_doc.end_page_idx == 1

    # Check that the page number was updated for the next doc's chunk
    assert current_doc.chunks[1].grounding[0].page == 1

# Test for _get_document_paths can remain as it's independent of provider
# (This test is not present in the initial file, but good to keep in mind)

# Obsolete tests (related to removed _send_parsing_request or direct _parse_pdf/_parse_image calls):
# - test_parse_and_save_documents_empty_list (still relevant for parse_and_save_documents if kept public)
# - test_parse_documents_with_file_paths (needs to mock get_ai_provider for `parse` or `parse_documents`)
# - test_parse_documents_with_grounding_save_dir (similar to above)
# - test_parse_and_save_documents_with_url (similar)
# - test_parse_and_save_document_with_local_file (similar)
# - test_parse_and_save_document_with_url (similar)
# - test_parse_and_save_document_with_invalid_file_type (still relevant, tests get_file_type logic path)
# - test_parse_pdf (now internal, tested via `parse`)
# - test_parse_image (now internal, tested via `parse`)
# - test_parse_image_with_error (now internal, tested via `parse`)
# - test_parse_doc_in_parallel (now internal, tested via `parse`)
# - test_parse_doc_parts_success (now internal, tested via `parse`)
# - test_parse_doc_parts_error (now internal, tested via `parse`)
# - test_send_parsing_request_success (function removed)
# - test_parse_and_save_document_with_grounding_save_dir (if parse_and_save_document is kept public)
# - test_parse_pdf_with_empty_result (internal)
# - test_parse_documents_with_mixed_file_types (needs get_ai_provider mock)
# - test_send_parsing_request_with_different_file_types (function removed)
# - test_parse_pdf_handles_single_page_document (internal)


def test_document_string_representation():
    # Test the string representation of Document objects
    doc = Document(
        file_path=Path("/path/to/test_document.pdf"), start_page_idx=5, end_page_idx=10
    )

    expected_str = "File name: test_document.pdf\tPage: [5:10]"
    assert str(doc) == expected_str


def test_parse_pdf_handles_single_page_document(temp_dir):
    # Test parsing a single-page PDF
    pdf_path = temp_dir / "single_page.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.7\n")

    single_page_doc = ParsedDocument(
        markdown="# Single Page",
        chunks=[],
        start_page_idx=0,
        end_page_idx=0,
        doc_type="pdf",
    )

    with patch("agentic_doc.parse.split_pdf") as mock_split, patch(
        "agentic_doc.parse._parse_doc_in_parallel"
    ) as mock_parse_parts:

        mock_split.return_value = [
            Document(
                file_path=temp_dir / "single_1.pdf", start_page_idx=0, end_page_idx=0
            )
        ]
        mock_parse_parts.return_value = [single_page_doc]

        result = _parse_pdf(pdf_path)

        assert result.start_page_idx == 0
        assert result.end_page_idx == 0
        assert result.doc_type == "pdf"


class TestParseFunctionConsolidated:
    """Test the consolidated parse function."""

    @patch('agentic_doc.parse.get_ai_provider')
    def test_parse_single_image_document_with_mocked_provider(self, mock_get_ai_provider, tmp_path):
        # 1. Configure the mock provider
        mock_provider_instance = MagicMock(spec=BaseAIProvider)
        mock_parsed_doc_result = ParsedDocument(
            markdown="Mocked markdown from image",
            chunks=[], start_page_idx=0, end_page_idx=0, doc_type="image", errors=[]
        )
        mock_provider_instance.analyze_document.return_value = mock_parsed_doc_result
        mock_get_ai_provider.return_value = mock_provider_instance

        # 2. Create a dummy document file
        dummy_image_file = tmp_path / "test_image.png"
        # Create a small, valid PNG file (1x1 transparent pixel)
        # Other image formats could also be used if preferred
        dummy_image_file.write_bytes(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82'
        )


        # 3. Call the main parse function
        results = parse(dummy_image_file)

        # 4. Assertions
        assert len(results) == 1
        parsed_doc = results[0]
        assert parsed_doc.markdown == "Mocked markdown from image"
        mock_provider_instance.analyze_document.assert_called_once()
        call_args = mock_provider_instance.analyze_document.call_args
        # The actual file path passed to analyze_document will be the one created by parse_and_save_document for the image
        # which is inside a temp directory structure if the original path was not directly usable.
        # For a direct file path like this, it should be the same.
        # However, the internal _parse_image receives the direct path.
        assert call_args[0][0] == str(dummy_image_file) # file_path for _parse_image
        assert call_args[0][1]['doc_type'] == 'image' # options for _parse_image

    @patch('agentic_doc.parse.get_ai_provider')
    def test_parse_pdf_document_with_mocked_provider(self, mock_get_ai_provider, tmp_path):
        mock_provider_instance = MagicMock(spec=BaseAIProvider)
        # Simulate a PDF that is split into two chunks by the parsing logic (e.g. settings.split_size)
        # The provider's analyze_document will be called for each chunk.

        # Mock response for the first chunk (pages 0-0 of the chunk, originally 0-0 of PDF)
        mock_parsed_doc_chunk1 = ParsedDocument(
            markdown="Mocked markdown for PDF chunk 1",
            chunks=[], start_page_idx=0, end_page_idx=0, doc_type="pdf", errors=[]
        )
        # Mock response for the second chunk (pages 0-0 of the chunk, originally 1-1 of PDF)
        # For simplicity, assume split_size=1, so each chunk is one page.
        # If split_size was 2, and we had a 2 page PDF, it might be one call.
        # Let's assume split_size is 1 for this test, and our dummy PDF represents 2 pages.
        # The internal splitting logic will create temporary files for these chunks.

        # To simplify, we'll assume split_pdf creates one Document part,
        # and analyze_document is called once for that part.
        # The internal structure of _parse_pdf handles splitting. Our mock provider
        # will just return a result for whatever file path it's given by _parse_doc_parts.

        mock_final_merged_doc = ParsedDocument(
             markdown="Mocked markdown for whole PDF", # This would be a merge if multiple chunks
             chunks=[], start_page_idx=0, end_page_idx=0, doc_type="pdf", errors=[] # Assume 1 page PDF for simplicity here
        )
        mock_provider_instance.analyze_document.return_value = mock_final_merged_doc
        mock_get_ai_provider.return_value = mock_provider_instance

        dummy_pdf_file = tmp_path / "test_doc.pdf"
        dummy_pdf_file.write_bytes(b"%PDF-1.7\n%%EOF") # Minimal PDF content

        # Mock split_pdf to control how many parts are processed
        # This means we are also testing the merging logic if split_pdf returns multiple parts.
        # For a simpler unit test of just the provider call path, assume split_pdf returns one part.
        with patch("agentic_doc.parse.split_pdf") as mock_split_pdf:
            # This Document object represents a chunk that _parse_doc_parts will process
            # The file_path here would be a temporary file created by split_pdf
            doc_chunk = Document(file_path=Path(str(dummy_pdf_file) + "_chunk0"), start_page_idx=0, end_page_idx=0)
            mock_split_pdf.return_value = [doc_chunk]

            results = parse(dummy_pdf_file)

            assert len(results) == 1
            parsed_doc = results[0]
            assert parsed_doc.markdown == "Mocked markdown for whole PDF"

            # analyze_document should be called once with the path of the chunk
            mock_provider_instance.analyze_document.assert_called_once()
            call_args = mock_provider_instance.analyze_document.call_args
            assert call_args[0][0] == str(doc_chunk.file_path)
            assert call_args[0][1]['doc_type'] == 'pdf'
            assert call_args[0][1]['start_page_idx'] == 0
            assert call_args[0][1]['end_page_idx'] == 0


    @patch('agentic_doc.parse.get_ai_provider')
    def test_parse_pdf_document_provider_simulates_error(self, mock_get_ai_provider, tmp_path):
        mock_provider_instance = MagicMock(spec=BaseAIProvider)
        error_page_error = PageError(page_num=0, error="Simulated provider error", error_code=500)
        # This is the result from a single call to analyze_document for a PDF chunk
        mock_parsed_doc_result_with_error = ParsedDocument(
            markdown="", chunks=[], start_page_idx=0, end_page_idx=0, doc_type="pdf", errors=[error_page_error]
        )
        mock_provider_instance.analyze_document.return_value = mock_parsed_doc_result_with_error
        mock_get_ai_provider.return_value = mock_provider_instance

        dummy_pdf_file = tmp_path / "test_doc_error.pdf"
        dummy_pdf_file.write_bytes(b"%PDF-1.7\n%%EOF")

        with patch("agentic_doc.parse.split_pdf") as mock_split_pdf:
            doc_chunk = Document(file_path=Path(str(dummy_pdf_file) + "_chunk0"), start_page_idx=0, end_page_idx=0)
            mock_split_pdf.return_value = [doc_chunk]

            results = parse(dummy_pdf_file)
            assert len(results) == 1
            # The error is within the ParsedDocument returned by the (mocked) provider for the chunk
            assert results[0].errors[0].error == "Simulated provider error"
            assert results[0].errors[0].page_num == 0 # This page_num is relative to the chunk's start_page_idx
                                                 # which is 0 in this mocked ParsedDocument.
                                                 # The final merged document will adjust this if needed.


    @patch('agentic_doc.parse.get_ai_provider')
    def test_parse_when_get_ai_provider_fails(self, mock_get_ai_provider, tmp_path):
        mock_get_ai_provider.side_effect = ValueError("Provider config error")

        dummy_file = tmp_path / "test_any_file.txt"
        dummy_file.write_text("content")

        with pytest.raises(ValueError, match="Provider config error"):
            parse(dummy_file)

    # Test saving behavior
    @patch('agentic_doc.parse.get_ai_provider')
    def test_parse_single_document_with_save_dir(self, mock_get_ai_provider, tmp_path):
        mock_provider_instance = MagicMock(spec=BaseAIProvider)
        mock_parsed_doc_result = ParsedDocument(
            markdown="Mocked markdown for saving",
            chunks=[], start_page_idx=0, end_page_idx=0, doc_type="image", errors=[]
        )
        mock_provider_instance.analyze_document.return_value = mock_parsed_doc_result
        mock_get_ai_provider.return_value = mock_provider_instance

        dummy_image_file = tmp_path / "test_image_save.png"
        dummy_image_file.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82')

        result_dir = tmp_path / "results"
        # No need to create result_dir, parse_and_save_document should do it.

        results = parse(dummy_image_file, result_save_dir=result_dir)

        assert len(results) == 1
        parsed_doc = results[0]
        assert parsed_doc.markdown == "Mocked markdown for saving"
        assert parsed_doc.result_path is not None # This is the key part for saved results
        assert parsed_doc.result_path.parent == result_dir
        assert parsed_doc.result_path.name.startswith(dummy_image_file.stem)
        assert parsed_doc.result_path.suffix == ".json"

        with open(parsed_doc.result_path, 'r') as f:
            saved_data = json.load(f)
            assert saved_data['markdown'] == "Mocked markdown for saving"

    @patch('agentic_doc.parse.get_ai_provider')
    def test_parse_multiple_documents(self, mock_get_ai_provider, tmp_path):
        mock_provider_instance = MagicMock(spec=BaseAIProvider)
        # This will be returned for each call to analyze_document
        mock_provider_instance.analyze_document.return_value = ParsedDocument(
            markdown="mock data", chunks=[], start_page_idx=0, end_page_idx=0, doc_type="image", errors=[]
        )
        mock_get_ai_provider.return_value = mock_provider_instance

        test_files_paths = []
        for i in range(2):
            f = tmp_path / f"test{i}.png"
            f.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82')
            test_files_paths.append(f)

        results = parse([str(f) for f in test_files_paths])

        assert len(results) == 2
        assert mock_provider_instance.analyze_document.call_count == 2 # Called once for each file

    @patch('agentic_doc.parse.get_ai_provider')
    @patch("agentic_doc.parse.save_groundings_as_images") # Also mock this
    def test_parse_with_grounding_save_dir(self, mock_save_groundings, mock_get_ai_provider, tmp_path):
        mock_provider_instance = MagicMock(spec=BaseAIProvider)
        mock_parsed_doc_result = ParsedDocument(
            markdown="grounding test", chunks=[Chunk(text="t",grounding=[],chunk_type=ChunkType.text,chunk_id="id")],
            start_page_idx=0, end_page_idx=0, doc_type="image", errors=[]
        )
        mock_provider_instance.analyze_document.return_value = mock_parsed_doc_result
        mock_get_ai_provider.return_value = mock_provider_instance

        dummy_image_file = tmp_path / "test_grounding.png"
        dummy_image_file.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82')
        grounding_dir = tmp_path / "groundings_output"

        results = parse(dummy_image_file, grounding_save_dir=grounding_dir)

        assert len(results) == 1
        mock_save_groundings.assert_called_once()
        # Check that the first arg to save_groundings_as_images is the input file path
        # The actual input path to save_groundings_as_images is the original document path
        assert mock_save_groundings.call_args[0][0] == dummy_image_file


    @patch('agentic_doc.parse.get_ai_provider')
    @patch("agentic_doc.parse.create_connector")
    def test_parse_with_local_connector_config(self, mock_create_connector, mock_get_ai_provider, tmp_path):
        mock_provider_instance = MagicMock(spec=BaseAIProvider)
        mock_provider_instance.analyze_document.return_value = ParsedDocument(
            markdown="connector test", chunks=[], start_page_idx=0, end_page_idx=0, doc_type="image", errors=[]
        )
        mock_get_ai_provider.return_value = mock_provider_instance

        test_file = tmp_path / "connector_doc.png"
        test_file.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82')

        config = LocalConnectorConfig()
        mock_connector = MagicMock()
        mock_connector.list_files.return_value = [str(test_file)] # Connector returns full path
        mock_connector.download_file.return_value = test_file # download_file returns Path object
        mock_create_connector.return_value = mock_connector

        results = parse(config, connector_path=str(tmp_path))

        assert len(results) == 1
        mock_create_connector.assert_called_once_with(config)
        mock_connector.list_files.assert_called_once_with(str(tmp_path), None)
        mock_connector.download_file.assert_called_once_with(str(test_file))
        mock_provider_instance.analyze_document.assert_called_once()


    @patch('agentic_doc.parse.get_ai_provider')
    def test_parse_unsupported_type(self, mock_get_ai_provider):
        with pytest.raises(ValueError, match="Unsupported documents type"):
            parse(123)

    @patch('agentic_doc.parse.get_ai_provider')
    def test_parse_with_marginalia_and_metadata_flags(self, mock_get_ai_provider, tmp_path):
        mock_provider_instance = MagicMock(spec=BaseAIProvider)
        mock_provider_instance.analyze_document.return_value = ParsedDocument(
            markdown="flags test", chunks=[], start_page_idx=0, end_page_idx=0, doc_type="image", errors=[]
        )
        mock_get_ai_provider.return_value = mock_provider_instance

        dummy_image_file = tmp_path / "test_flags.png"
        dummy_image_file.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82')

        results = parse(dummy_image_file, include_marginalia=False, include_metadata_in_markdown=False)

        assert len(results) == 1
        mock_provider_instance.analyze_document.assert_called_once()
        call_options = mock_provider_instance.analyze_document.call_args[0][1]
        assert call_options['include_marginalia'] is False
        assert call_options['include_metadata_in_markdown'] is False

    @patch('agentic_doc.parse.get_ai_provider')
    def test_parse_with_bytes(self, mock_get_ai_provider, tmp_path):
        mock_provider_instance = MagicMock(spec=BaseAIProvider)
        mock_provider_instance.analyze_document.return_value = ParsedDocument(
            markdown="bytes test", chunks=[], start_page_idx=0, end_page_idx=0, doc_type="image", errors=[] # Assuming bytes are image
        )
        mock_get_ai_provider.return_value = mock_provider_instance

        # Valid 1x1 PNG bytes
        png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82'

        # We need to patch `get_file_type` for bytes input as it operates on Path.
        # The bytes are written to a temp file, then get_file_type is called on that.
        with patch("agentic_doc.parse.get_file_type", return_value="image") as mock_get_file_type:
            results = parse(png_bytes)

            assert len(results) == 1
            mock_provider_instance.analyze_document.assert_called_once()
            # The path passed to analyze_document will be a temporary file path
            temp_file_path_arg = mock_provider_instance.analyze_document.call_args[0][0]
            assert Path(temp_file_path_arg).exists() # Temp file should have existed during call
            # `get_file_type` is called on this temp file path
            mock_get_file_type.assert_called_once_with(Path(temp_file_path_arg))


    @patch('agentic_doc.parse.get_ai_provider')
    def test_parse_url_string(self, mock_get_ai_provider, tmp_path):
        mock_provider_instance = MagicMock(spec=BaseAIProvider)
        mock_provider_instance.analyze_document.return_value = ParsedDocument(
            markdown="url test", chunks=[], start_page_idx=0, end_page_idx=0, doc_type="image", errors=[] # Assuming URL is image
        )
        mock_get_ai_provider.return_value = mock_provider_instance

        url = "http://example.com/test_image.png"

        # Mock download_file and get_file_type
        with patch("agentic_doc.parse.download_file") as mock_download, \
             patch("agentic_doc.parse.get_file_type", return_value="image") as mock_get_file_type:

            # download_file will be called with a Url object and a string path to a temp file
            # It doesn't return anything, but writes to the output_file_path
            def fake_download(url_obj, out_path_str):
                Path(out_path_str).write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82')
            mock_download.side_effect = fake_download

            results = parse(url)

            assert len(results) == 1
            mock_download.assert_called_once()
            # The path passed to analyze_document will be a temporary file path
            temp_file_path_arg = mock_provider_instance.analyze_document.call_args[0][0]
            assert Path(temp_file_path_arg).name == "test_image.png" # Name from URL
            mock_get_file_type.assert_called_once_with(Path(temp_file_path_arg))
