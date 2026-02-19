"""
Tests for docling-adi (Azure Document Intelligence) OCR plugin.
Uses mocks to avoid real Azure API calls or credentials.

Based on: tests/test_surya_ocr.py from docling_surya
  Repository : https://github.com/harrykhh/docling_surya
  Author     : Harry Ho (@harrykhh)
  License    : GPL-3.0

The test architecture — mock predictor classes, monkeypatch fixtures for the
OCR backend imports, entry-point discovery test, options/model initialisation
tests, reportlab-generated sample PDFs, and the end-to-end pipeline test
pattern — is directly adapted from that reference test suite.
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
# Tests - Plugin discovery & factory
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
# Tests - Options
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
    from pydantic import ValidationError

    from docling_adi.plugin import AzureDocIntelOcrOptions

    with pytest.raises(ValidationError):
        AzureDocIntelOcrOptions(unknown_field="oops")


def test_options_kind_on_instance() -> None:
    """Test that an instance also exposes 'kind'."""
    from docling_adi.plugin import AzureDocIntelOcrOptions

    opts = AzureDocIntelOcrOptions()
    assert opts.kind == "azure_di"


# --------------------------------------------------------------------------- #
# Tests - Model initialisation
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
# Tests - download_models (no-op for cloud service)
# --------------------------------------------------------------------------- #


def test_download_models_creates_directory(tmp_path: Path) -> None:
    """Test that download_models creates the cache directory."""
    from docling_adi.plugin import AzureDocIntelOcrModel

    out = AzureDocIntelOcrModel.download_models(local_dir=tmp_path / "azure_cache")
    assert out.exists()
    assert out.is_dir()


# --------------------------------------------------------------------------- #
# Tests - Coordinate mapping
# --------------------------------------------------------------------------- #


def test_polygon_to_page_bbox_basic() -> None:
    """Test polygon-to-bbox conversion with simple values."""
    from docling_core.types.doc import BoundingBox

    from docling_adi.plugin import AzureDocIntelOcrModel

    # Fake rect: entire page at 72 dpi -> 612x792 points (8.5x11 inches)
    rect = BoundingBox.from_tuple(
        coord=(0.0, 0.0, 612.0, 792.0),
        origin="TOPLEFT",
    )

    # ADI page is 8.5x11 inches, polygon at (1, 1) to (2, 2) inches
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

    # ADI coord in a 8.5x11 page, polygon at origin (0,0) -> (8.5,11)
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
# Tests - End-to-end pipeline
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


# --------------------------------------------------------------------------- #
# Helper: build an enabled model instance for unit tests
# --------------------------------------------------------------------------- #


def _make_model(
    endpoint: str = "https://mock.cognitiveservices.azure.com/",
    api_key: str = "mock-key",
    model_id: str = "prebuilt-read",
    locale: str | None = None,
    enabled: bool = True,
):
    from docling.datamodel.accelerator_options import AcceleratorOptions

    from docling_adi.plugin import AzureDocIntelOcrModel, AzureDocIntelOcrOptions

    opts = AzureDocIntelOcrOptions(
        endpoint=endpoint,
        api_key=api_key,
        model_id=model_id,
        locale=locale,
    )
    return AzureDocIntelOcrModel(
        enabled=enabled,
        artifacts_path=None,
        options=opts,
        accelerator_options=AcceleratorOptions(),
    )


# --------------------------------------------------------------------------- #
# Tests - _analyse_image (word-level path)
# --------------------------------------------------------------------------- #


def test_analyse_image_returns_word_cells() -> None:
    """Test _analyse_image returns one TextCell per word."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    cells = model._analyse_image(img, rect)
    # MockAdiPage has 4 words
    assert len(cells) == 4
    texts = [c.text for c in cells]
    assert texts == ["Hello", "from", "Azure", "DI"]
    for c in cells:
        assert c.from_ocr is True
        assert c.confidence > 0


def test_analyse_image_lines_fallback() -> None:
    """When words is empty, _analyse_image falls back to lines."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    # Patch the mock poller to return a page with no words, only lines
    class PageNoWords:
        width = 8.5
        height = 11.0
        words = []
        lines = [MockLine("Full line text")]

    class ResultNoWords:
        pages = [PageNoWords()]

    class PollerNoWords:
        def result(self):
            return ResultNoWords()

    model.client.begin_analyze_document = lambda **kw: PollerNoWords()

    cells = model._analyse_image(img, rect)
    assert len(cells) == 1
    assert cells[0].text == "Full line text"
    assert cells[0].confidence == 1.0


def test_analyse_image_no_pages() -> None:
    """When Azure DI returns no pages, _analyse_image returns []."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    class EmptyResult:
        pages = []

    class EmptyPoller:
        def result(self):
            return EmptyResult()

    model.client.begin_analyze_document = lambda **kw: EmptyPoller()

    cells = model._analyse_image(img, rect)
    assert cells == []


def test_analyse_image_zero_dimensions() -> None:
    """When ADI page has zero width/height, _analyse_image returns []."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    class ZeroPage:
        width = 0
        height = 0
        words = []
        lines = []

    class ZeroResult:
        pages = [ZeroPage()]

    class ZeroPoller:
        def result(self):
            return ZeroResult()

    model.client.begin_analyze_document = lambda **kw: ZeroPoller()

    cells = model._analyse_image(img, rect)
    assert cells == []


def test_analyse_image_api_exception() -> None:
    """When the Azure API raises, _analyse_image returns [] gracefully."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    def raise_error(**kw):
        raise RuntimeError("Service unavailable")

    model.client.begin_analyze_document = raise_error

    cells = model._analyse_image(img, rect)
    assert cells == []


def test_analyse_image_skips_blank_words() -> None:
    """Words with empty/whitespace content are skipped."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    class BlankWordPage:
        width = 8.5
        height = 11.0
        words = [
            MockWord("", 0.9),
            MockWord("   ", 0.9),
            MockWord("valid", 0.95),
        ]
        lines = []

    class BlankResult:
        pages = [BlankWordPage()]

    class BlankPoller:
        def result(self):
            return BlankResult()

    model.client.begin_analyze_document = lambda **kw: BlankPoller()

    cells = model._analyse_image(img, rect)
    assert len(cells) == 1
    assert cells[0].text == "valid"


def test_analyse_image_none_confidence_defaults_to_zero() -> None:
    """A word with confidence=None gets confidence 0.0."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    class NoneConfPage:
        width = 8.5
        height = 11.0
        words = [MockWord("word", None)]
        lines = []

    class NoneConfResult:
        pages = [NoneConfPage()]

    class NoneConfPoller:
        def result(self):
            return NoneConfResult()

    model.client.begin_analyze_document = lambda **kw: NoneConfPoller()

    cells = model._analyse_image(img, rect)
    assert len(cells) == 1
    assert cells[0].confidence == 0.0


def test_analyse_image_word_with_bad_polygon() -> None:
    """A word whose polygon is too short is skipped."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    class BadPolyPage:
        width = 8.5
        height = 11.0
        words = [MockWord("bad", 0.9, [1.0])]  # polygon too short
        lines = []

    class BadPolyResult:
        pages = [BadPolyPage()]

    class BadPolyPoller:
        def result(self):
            return BadPolyResult()

    model.client.begin_analyze_document = lambda **kw: BadPolyPoller()

    cells = model._analyse_image(img, rect)
    assert cells == []


def test_analyse_image_skips_blank_lines() -> None:
    """Lines with empty content are skipped in the fallback path."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    class BlankLinePage:
        width = 8.5
        height = 11.0
        words = []
        lines = [MockLine(""), MockLine("   "), MockLine("real line")]

    class BlankLineResult:
        pages = [BlankLinePage()]

    class BlankLinePoller:
        def result(self):
            return BlankLineResult()

    model.client.begin_analyze_document = lambda **kw: BlankLinePoller()

    cells = model._analyse_image(img, rect)
    assert len(cells) == 1
    assert cells[0].text == "real line"


def test_analyse_image_line_with_bad_polygon() -> None:
    """A line whose polygon is too short is skipped."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    class BadLinePolyPage:
        width = 8.5
        height = 11.0
        words = []
        lines = [MockLine("text")]

    # Override the polygon to be too short
    BadLinePolyPage.lines[0].polygon = [1.0]

    class BadLinePolyResult:
        pages = [BadLinePolyPage()]

    class BadLinePolyPoller:
        def result(self):
            return BadLinePolyResult()

    model.client.begin_analyze_document = lambda **kw: BadLinePolyPoller()

    cells = model._analyse_image(img, rect)
    assert cells == []


def test_analyse_image_with_locale() -> None:
    """When locale is set, it is passed through to the API call."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model(locale="en-US")
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    captured_kwargs = {}

    class CapturingPoller:
        def result(self):
            return MockAnalyzeResult()

    def capturing_analyze(**kw):
        captured_kwargs.update(kw)
        return CapturingPoller()

    model.client.begin_analyze_document = capturing_analyze

    cells = model._analyse_image(img, rect)
    assert len(cells) == 4
    assert captured_kwargs.get("locale") == "en-US"


def test_analyse_image_without_locale() -> None:
    """When locale is None, it is NOT passed to the API call."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model(locale=None)
    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    img = Image.new("RGB", (100, 100), color="white")

    captured_kwargs = {}

    class CapturingPoller:
        def result(self):
            return MockAnalyzeResult()

    def capturing_analyze(**kw):
        captured_kwargs.update(kw)
        return CapturingPoller()

    model.client.begin_analyze_document = capturing_analyze

    model._analyse_image(img, rect)
    assert "locale" not in captured_kwargs


# --------------------------------------------------------------------------- #
# Tests - __call__ method edge cases
# --------------------------------------------------------------------------- #


def test_call_disabled_yields_pages_unchanged() -> None:
    """A disabled model yields pages without processing."""
    model = _make_model(enabled=False)

    page1 = MagicMock()
    page2 = MagicMock()
    conv_res = MagicMock()

    result = list(model(conv_res, [page1, page2]))
    assert result == [page1, page2]


def test_call_invalid_backend_yields_page() -> None:
    """A page with no backend is yielded unchanged."""
    model = _make_model()

    page = MagicMock()
    page._backend = None
    conv_res = MagicMock()

    result = list(model(conv_res, [page]))
    assert result == [page]


def test_call_invalid_backend_is_valid_false() -> None:
    """A page whose backend.is_valid() is False is yielded unchanged."""
    model = _make_model()

    page = MagicMock()
    page._backend.is_valid.return_value = False
    conv_res = MagicMock()

    result = list(model(conv_res, [page]))
    assert result == [page]


def test_call_zero_area_rect_skipped() -> None:
    """A zero-area OCR rect is skipped (no API call)."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()

    page = MagicMock()
    page._backend.is_valid.return_value = True

    # get_ocr_rects returns a zero-area rect
    zero_rect = BoundingBox.from_tuple(coord=(10, 10, 10, 10), origin="TOPLEFT")
    model.get_ocr_rects = MagicMock(return_value=[zero_rect])
    model.post_process_cells = MagicMock()

    conv_res = MagicMock()

    result = list(model(conv_res, [page]))
    assert len(result) == 1
    # No call to get_page_image since rect area is 0
    page._backend.get_page_image.assert_not_called()


def test_call_normal_flow_processes_rect() -> None:
    """Normal flow: valid page, non-zero rect, calls _analyse_image."""
    from docling_core.types.doc import BoundingBox
    from PIL import Image

    model = _make_model()

    page = MagicMock()
    page._backend.is_valid.return_value = True
    page._backend.get_page_image.return_value = Image.new("RGB", (100, 100))

    rect = BoundingBox.from_tuple(coord=(0, 0, 612, 792), origin="TOPLEFT")
    model.get_ocr_rects = MagicMock(return_value=[rect])
    model.post_process_cells = MagicMock()

    conv_res = MagicMock()

    result = list(model(conv_res, [page]))
    assert len(result) == 1
    page._backend.get_page_image.assert_called_once()
    model.post_process_cells.assert_called_once()
    # Verify cells were passed (4 words from mock)
    cells_arg = model.post_process_cells.call_args[0][0]
    assert len(cells_arg) == 4


# --------------------------------------------------------------------------- #
# Tests - DefaultAzureCredential fallback
# --------------------------------------------------------------------------- #


def test_model_uses_default_credential_when_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no API key is available, DefaultAzureCredential is used."""
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", raising=False)

    from docling.datamodel.accelerator_options import AcceleratorOptions

    from docling_adi.plugin import AzureDocIntelOcrModel, AzureDocIntelOcrOptions

    opts = AzureDocIntelOcrOptions(
        endpoint="https://mock.cognitiveservices.azure.com/",
        # no api_key
    )
    accel = AcceleratorOptions()

    model = AzureDocIntelOcrModel(
        enabled=True,
        artifacts_path=None,
        options=opts,
        accelerator_options=accel,
    )
    assert model.enabled
    assert hasattr(model, "client")


# --------------------------------------------------------------------------- #
# Tests - download_models default path
# --------------------------------------------------------------------------- #


def test_download_models_default_path() -> None:
    """download_models with no args uses settings.cache_dir."""
    from docling_adi.plugin import AzureDocIntelOcrModel

    out = AzureDocIntelOcrModel.download_models()
    assert out.exists()
    assert out.is_dir()
    assert "AzureDocIntel" in str(out)
