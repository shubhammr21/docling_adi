"""
Tests for docling-adi (Azure Document Intelligence) OCR plugin.
Uses mocks to avoid real Azure API calls or credentials.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

if TYPE_CHECKING:
    from docling.datamodel.document import ConvertedDocument


# --------------------------------------------------------------------------- #
# Mock Azure SDK objects
# --------------------------------------------------------------------------- #

class MockWord:
    """Mock Azure DI word."""

    def __init__(
        self,
        content: str = "Hello",
        confidence: float = 0.98,
        polygon: list[float] | None = None,
    ):
        self.content = content
        self.confidence = confidence
        self.polygon = polygon or [0.5, 0.5, 2.0, 0.5, 2.0, 1.0, 0.5, 1.0]


class MockLine:
    """Mock Azure DI line."""

    def __init__(
        self,
        content: str = "Hello from Azure DI",
        polygon: list[float] | None = None,
    ):
        self.content = content
        self.polygon = polygon or [0.5, 0.5, 5.0, 0.5, 5.0, 1.0, 0.5, 1.0]


class MockAdiPage:
    """Mock Azure DI page."""

    def __init__(self):
        self.width = 8.5
        self.height = 11.0
        self.words = [
            MockWord("Hello", 0.98, [0.5, 0.5, 1.5, 0.5, 1.5, 0.8, 0.5, 0.8]),
            MockWord("from", 0.95, [1.7, 0.5, 2.5, 0.5, 2.5, 0.8, 1.7, 0.8]),
            MockWord("Azure", 0.99, [2.7, 0.5, 3.8, 0.5, 3.8, 0.8, 2.7, 0.8]),
            MockWord("DI", 0.97, [4.0, 0.5, 4.5, 0.5, 4.5, 0.8, 4.0, 0.8]),
        ]
        self.lines = [
            MockLine("Hello from Azure DI", [0.5, 0.5, 4.5, 0.5, 4.5, 0.8, 0.5, 0.8]),
        ]


class MockAnalyzeResult:
    """Mock Azure DI analyze result."""

    def __init__(self):
        self.pages = [MockAdiPage()]
        self.content = "Hello from Azure DI"


class MockPoller:
    """Mock Azure long-running operation poller."""

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def result(self) -> MockAnalyzeResult:
        return MockAnalyzeResult()


class MockDocumentIntelligenceClient:
    """Mock Azure DocumentIntelligenceClient."""

    def __init__(self, endpoint: str, credential: Any, **kwargs: Any):
        self.endpoint = endpoint
        self.credential = credential

    def begin_analyze_document(self, **kwargs: Any) -> MockPoller:
        return MockPoller()


class MockAzureKeyCredential:
    """Mock AzureKeyCredential."""

    def __init__(self, key: str):
        self.key = key


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def mock_azure_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock Azure SDK imports to avoid real API calls."""
    monkeypatch.setattr(
        "azure.ai.documentintelligence.DocumentIntelligenceClient",
        MockDocumentIntelligenceClient,
    )
    monkeypatch.setattr(
        "azure.core.credentials.AzureKeyCredential",
        MockAzureKeyCredential,
    )


@pytest.fixture(autouse=True)
def set_azure_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set mock Azure environment variables."""
    monkeypatch.setenv(
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "https://mock-instance.cognitiveservices.azure.com/",
    )
    monkeypatch.setenv(
        "AZURE_DOCUMENT_INTELLIGENCE_KEY",
        "mock-api-key-for-testing",
    )


@pytest.fixture
def sample_pdf(tmp_path: Path) -> str:
    """Create a minimal 1-page PDF with text."""
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(200, 200))
    c.drawString(50, 150, "Hello from Azure DI")
    c.save()
    return str(pdf_path)


# --------------------------------------------------------------------------- #
# Tests – Plugin discovery & factory
# --------------------------------------------------------------------------- #


def test_plugin_is_discoverable() -> None:
    """Test that the plugin is registered via entry points."""
    import importlib.metadata as im

    entry_points = im.entry_points(group="docling")
    names = [ep.name for ep in entry_points]
    assert "azure-di" in names, (
        f"Plugin 'azure-di' not found in entry points. Found: {names}"
    )


def test_ocr_engines_factory() -> None:
    """Test that the plugin factory returns the model class."""
    from docling_adi.plugin import AzureDocIntelOcrModel, ocr_engines

    engines = ocr_engines()
    assert "ocr_engines" in engines
    assert len(engines["ocr_engines"]) == 1
    assert engines["ocr_engines"][0] is AzureDocIntelOcrModel


# --------------------------------------------------------------------------- #
# Tests – Options
# --------------------------------------------------------------------------- #


def test_options_has_kind_as_class_attr() -> None:
    """Test that AzureDocIntelOcrOptions.kind is a ClassVar accessible on the class."""
    from docling_adi.plugin import AzureDocIntelOcrOptions

    assert hasattr(AzureDocIntelOcrOptions, "kind"), (
        "AzureDocIntelOcrOptions must have a 'kind' class attribute"
    )
    assert AzureDocIntelOcrOptions.kind == "azure_di", (
        f"Expected kind='azure_di', got '{AzureDocIntelOcrOptions.kind}'"
    )


def test_options_defaults() -> None:
    """Test that default option values are sensible."""
    from docling_adi.plugin import AzureDocIntelOcrOptions

    opts = AzureDocIntelOcrOptions()
    assert opts.model_id == "prebuilt-read"
    assert opts.locale is None
    assert opts.endpoint is None
    assert opts.api_key is None


def test_options_custom_values() -> None:
    """Test that custom option values are accepted."""
    from docling_adi.plugin import AzureDocIntelOcrOptions

    opts = AzureDocIntelOcrOptions(
        endpoint="https://my-instance.cognitiveservices.azure.com/",
        api_key="my-key",
        model_id="prebuilt-layout",
        locale="en-US",
    )
    assert opts.endpoint == "https://my-instance.cognitiveservices.azure.com/"
    assert opts.api_key == "my-key"
    assert opts.model_id == "prebuilt-layout"
    assert opts.locale == "en-US"


def test_options_forbids_extra_fields() -> None:
    """Test that extra fields are rejected (extra='forbid')."""
    from docling_adi.plugin import AzureDocIntelOcrOptions
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AzureDocIntelOcrOptions(unknown_field="oops")


def test_options_kind_on_instance() -> None:
    """Test that an instance also exposes 'kind'."""
    from docling_adi.plugin import AzureDocIntelOcrOptions

    opts = AzureDocIntelOcrOptions()
    assert opts.kind == "azure_di"


# --------------------------------------------------------------------------- #
# Tests – Model initialisation
# --------------------------------------------------------------------------- #


def test_model_initializes_with_api_key() -> None:
    """Test that AzureDocIntelOcrModel initialises with explicit key."""
    from docling.datamodel.accelerator_options import AcceleratorOptions

    from docling_adi.plugin import AzureDocIntelOcrModel, AzureDocIntelOcrOptions

    opts = AzureDocIntelOcrOptions(
        endpoint="https://mock.cognitiveservices.azure.com/",
        api_key="test-key",
    )
    accel = AcceleratorOptions()

    model = AzureDocIntelOcrModel(
        enabled=True,
        artifacts_path=None,
        options=opts,
        accelerator_options=accel,
    )
    assert model.enabled
    assert model.scale == 3
    assert hasattr(model, "client")
    assert model.model_id == "prebuilt-read"


def test_model_initializes_from_env_vars() -> None:
    """Test that the model reads endpoint/key from environment variables."""
    from docling.datamodel.accelerator_options import AcceleratorOptions

    from docling_adi.plugin import AzureDocIntelOcrModel, AzureDocIntelOcrOptions

    opts = AzureDocIntelOcrOptions()  # no explicit endpoint/key
    accel = AcceleratorOptions()

    model = AzureDocIntelOcrModel(
        enabled=True,
        artifacts_path=None,
        options=opts,
        accelerator_options=accel,
    )
    assert model.enabled
    assert hasattr(model, "client")


def test_model_raises_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a missing endpoint raises ValueError."""
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", raising=False)

    from docling.datamodel.accelerator_options import AcceleratorOptions

    from docling_adi.plugin import AzureDocIntelOcrModel, AzureDocIntelOcrOptions

    opts = AzureDocIntelOcrOptions()  # no endpoint anywhere
    accel = AcceleratorOptions()

    with pytest.raises(ValueError, match="endpoint is required"):
        AzureDocIntelOcrModel(
            enabled=True,
            artifacts_path=None,
            options=opts,
            accelerator_options=accel,
        )


def test_model_disabled_skips_init() -> None:
    """Test that disabled model skips Azure client creation."""
    from docling.datamodel.accelerator_options import AcceleratorOptions

    from docling_adi.plugin import AzureDocIntelOcrModel, AzureDocIntelOcrOptions

    opts = AzureDocIntelOcrOptions()
    accel = AcceleratorOptions()

    model = AzureDocIntelOcrModel(
        enabled=False,
        artifacts_path=None,
        options=opts,
        accelerator_options=accel,
    )
    assert not model.enabled
    assert not hasattr(model, "client")


def test_get_options_type() -> None:
    """Test that get_options_type returns the correct class."""
    from docling_adi.plugin import AzureDocIntelOcrModel, AzureDocIntelOcrOptions

    assert AzureDocIntelOcrModel.get_options_type() is AzureDocIntelOcrOptions


# --------------------------------------------------------------------------- #
# Tests – download_models (no-op for cloud service)
# --------------------------------------------------------------------------- #


def test_download_models_creates_directory(tmp_path: Path) -> None:
    """Test that download_models creates the cache directory."""
    from docling_adi.plugin import AzureDocIntelOcrModel

    out = AzureDocIntelOcrModel.download_models(local_dir=tmp_path / "azure_cache")
    assert out.exists()
    assert out.is_dir()


# --------------------------------------------------------------------------- #
# Tests – Coordinate mapping
# --------------------------------------------------------------------------- #


def test_polygon_to_page_bbox_basic() -> None:
    """Test polygon-to-bbox conversion with simple values."""
    from docling_core.types.doc import BoundingBox

    from docling_adi.plugin import AzureDocIntelOcrModel

    # Fake rect: entire page at 72 dpi → 612×792 points (8.5×11 inches)
    rect = BoundingBox.from_tuple(
        coord=(0.0, 0.0, 612.0, 792.0),
        origin="TOPLEFT",
    )

    # ADI page is 8.5×11 inches, polygon at (1, 1) to (2, 2) inches
    polygon = [1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0]

    bbox = AzureDocIntelOcrModel._polygon_to_page_bbox(
        polygon=polygon,
        adi_w=8.5,
        adi_h=11.0,
        rect=rect,
        rect_w=612.0,
        rect_h=792.0,
    )

    assert bbox is not None
    # 1/8.5 * 612 ≈ 72.0
    assert abs(bbox.l - 72.0) < 0.1
    # 1/11 * 792 = 72.0
    assert abs(bbox.t - 72.0) < 0.1
    # 2/8.5 * 612 ≈ 144.0
    assert abs(bbox.r - 144.0) < 0.1
    # 2/11 * 792 ≈ 144.0
    assert abs(bbox.b - 144.0) < 0.1


def test_polygon_to_page_bbox_with_offset() -> None:
    """Test polygon-to-bbox conversion with a non-zero rect offset."""
    from docling_core.types.doc import BoundingBox

    from docling_adi.plugin import AzureDocIntelOcrModel

    rect = BoundingBox.from_tuple(
        coord=(100.0, 200.0, 400.0, 500.0),
        origin="TOPLEFT",
    )

    # ADI coord in a 8.5×11 page, polygon at origin (0,0) → (8.5,11)
    polygon = [0.0, 0.0, 8.5, 0.0, 8.5, 11.0, 0.0, 11.0]

    bbox = AzureDocIntelOcrModel._polygon_to_page_bbox(
        polygon=polygon,
        adi_w=8.5,
        adi_h=11.0,
        rect=rect,
        rect_w=300.0,
        rect_h=300.0,
    )

    assert bbox is not None
    assert abs(bbox.l - 100.0) < 0.01
    assert abs(bbox.t - 200.0) < 0.01
    assert abs(bbox.r - 400.0) < 0.01
    assert abs(bbox.b - 500.0) < 0.01


def test_polygon_to_page_bbox_none_polygon() -> None:
    """Test that None polygon returns None."""
    from docling_core.types.doc import BoundingBox

    from docling_adi.plugin import AzureDocIntelOcrModel

    rect = BoundingBox.from_tuple(coord=(0, 0, 100, 100), origin="TOPLEFT")
    result = AzureDocIntelOcrModel._polygon_to_page_bbox(
        polygon=None, adi_w=8.5, adi_h=11.0, rect=rect, rect_w=100.0, rect_h=100.0
    )
    assert result is None


def test_polygon_to_page_bbox_short_polygon() -> None:
    """Test that polygon with fewer than 4 values returns None."""
    from docling_core.types.doc import BoundingBox

    from docling_adi.plugin import AzureDocIntelOcrModel

    rect = BoundingBox.from_tuple(coord=(0, 0, 100, 100), origin="TOPLEFT")
    result = AzureDocIntelOcrModel._polygon_to_page_bbox(
        polygon=[1.0, 2.0], adi_w=8.5, adi_h=11.0, rect=rect, rect_w=100.0, rect_h=100.0
    )
    assert result is None


# --------------------------------------------------------------------------- #
# Tests – End-to-end pipeline
# --------------------------------------------------------------------------- #


def test_pipeline_uses_azure_di_ocr(sample_pdf: str) -> None:
    """End-to-end test: convert PDF with Azure DI OCR."""
    from docling_adi.plugin import AzureDocIntelOcrOptions

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_model="azure_di",
        allow_external_plugins=True,
        ocr_options=AzureDocIntelOcrOptions(
            endpoint="https://mock.cognitiveservices.azure.com/",
            api_key="mock-key",
        ),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    result = converter.convert(sample_pdf)
    doc = result.document

    text = doc.export_to_markdown()
    # The mock returns "Hello", "from", "Azure", "DI" as separate words
    assert "Hello" in text or "Azure" in text, (
        f"Expected OCR text in output but got:\n{text}"
    )
