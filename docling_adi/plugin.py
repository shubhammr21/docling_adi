"""Azure Document Intelligence OCR plugin for Docling.

This plugin integrates Azure Document Intelligence (formerly Azure Form
Recognizer) as an OCR backend for the Docling document-conversion framework.

Acknowledgements
----------------
This plugin is **directly based on** `docling_surya
<https://github.com/harrykhh/docling_surya>`_ by **Harry Ho (@harrykhh)**,
the first community Docling OCR plugin (GPL-3.0).

The following elements were adapted from that project:

* **Plugin architecture** — ``BaseOcrModel`` subclass, ``OcrOptions`` with a
  ``ClassVar[kind]`` discriminator, and the ``ocr_engines()`` factory function
  that Docling discovers via ``[project.entry-points.docling]``.
* **``__call__`` pipeline** — page iteration → ``get_ocr_rects`` → crop image
  at ``self.scale`` → run OCR engine → map coordinates back to page-point
  space → ``post_process_cells``.
* **Coordinate mapping** — proportional bounding-box conversion from the OCR
  engine's coordinate system onto the Docling crop rect.
* **Project scaffolding** — ``pyproject.toml`` entry-point layout,
  ``__init__.py`` exports, example script, and test suite structure with
  mocked OCR predictors and ``reportlab``-generated sample PDFs.

Reference repository : https://github.com/harrykhh/docling_surya
Reference PyPI       : https://pypi.org/project/docling-surya/
Reference author     : Harry Ho — https://github.com/harrykhh
"""

import io
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
class AzureDocIntelOcrOptions(OcrOptions):
    """Configuration for the Azure Document Intelligence OCR engine.

    Credentials are resolved in order:
      1. Explicit ``endpoint`` / ``api_key`` passed here.
      2. Environment variables ``AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT``
         and ``AZURE_DOCUMENT_INTELLIGENCE_KEY``.
      3. If no API key is available, ``DefaultAzureCredential`` from
         ``azure-identity`` is used (supports managed identity, CLI
         login, etc.).
    """

    kind: ClassVar[Literal["azure_di"]] = "azure_di"

    lang: list[str] = ["en"]
    endpoint: str | None = None
    api_key: str | None = None
    model_id: str = "prebuilt-read"
    locale: str | None = None  # e.g. "en-US", None = auto-detect

    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class AzureDocIntelOcrModel(BaseOcrModel):
    """Docling OCR model backed by Azure Document Intelligence."""

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: AzureDocIntelOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: AzureDocIntelOcrOptions
        self.scale = 3  # 72 dpi → 216 dpi

        if self.enabled:
            try:
                from azure.ai.documentintelligence import (
                    DocumentIntelligenceClient,
                )
                from azure.core.credentials import AzureKeyCredential
            except ImportError as exc:
                raise ImportError(
                    "azure-ai-documentintelligence is not installed. "
                    "Install via `uv add azure-ai-documentintelligence` or "
                    "`pip install azure-ai-documentintelligence`."
                ) from exc

            endpoint = options.endpoint or os.environ.get(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
            )
            if not endpoint:
                raise ValueError(
                    "Azure Document Intelligence endpoint is required. "
                    "Set it via AzureDocIntelOcrOptions(endpoint=...) or the "
                    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable."
                )

            api_key = options.api_key or os.environ.get(
                "AZURE_DOCUMENT_INTELLIGENCE_KEY"
            )

            if api_key:
                credential = AzureKeyCredential(api_key)
            else:
                try:
                    from azure.identity import DefaultAzureCredential

                    credential = DefaultAzureCredential()
                except ImportError as exc:
                    raise ImportError(
                        "No API key provided and azure-identity is not installed. "
                        "Either supply an api_key or install azure-identity for "
                        "DefaultAzureCredential support."
                    ) from exc

            self.client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=credential,
            )
            self.model_id = options.model_id

    # ------------------------------------------------------------------- #
    #  Static helper — no models to download for a cloud service
    # ------------------------------------------------------------------- #
    @staticmethod
    def download_models(local_dir: Path | None = None) -> Path:
        """No-op: Azure DI is a cloud service with no local models."""
        if local_dir is None:
            local_dir = settings.cache_dir / "models" / "AzureDocIntel"
        local_dir.mkdir(parents=True, exist_ok=True)
        return local_dir

    # ------------------------------------------------------------------- #
    #  Core inference
    # ------------------------------------------------------------------- #
    def __call__(
        self,
        conv_res: ConversionResult,
        page_batch: Iterable[Page],
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            if page._backend is None or not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_cells: list[TextCell] = []

                for rect in ocr_rects:
                    if rect.area() == 0:
                        continue

                    img = page._backend.get_page_image(scale=self.scale, cropbox=rect)

                    cells = self._analyse_image(img, rect)
                    del img

                    all_cells.extend(cells)

                self.post_process_cells(all_cells, page)

                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    # ------------------------------------------------------------------- #
    #  Send a single crop to Azure DI and parse the response
    # ------------------------------------------------------------------- #
    def _analyse_image(
        self,
        img: "Image.Image",  # noqa: F821  (PIL lazy import)
        rect: BoundingBox,
    ) -> list[TextCell]:
        """Send *img* to Azure Document Intelligence and return TextCells.

        Coordinates in the returned cells are mapped back into the
        original page coordinate system (points, 72 dpi, top-left origin).
        """
        # Encode PIL image → PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        buf.close()

        # Call Azure DI
        try:
            analyze_kwargs: dict = {
                "content_type": "application/octet-stream",
            }
            if self.options.locale:
                analyze_kwargs["locale"] = self.options.locale

            poller = self.client.begin_analyze_document(
                model_id=self.model_id,
                analyze_request=image_bytes,
                **analyze_kwargs,
            )
            result = poller.result()
        except Exception:
            _log.exception("Azure Document Intelligence call failed for rect %s", rect)
            return []

        if not result.pages:
            _log.warning("Azure DI returned no pages for rect %s", rect)
            return []

        # We sent a single image → expect exactly one page back
        adi_page = result.pages[0]
        adi_w = adi_page.width  # in page.unit (usually inches)
        adi_h = adi_page.height

        if not adi_w or not adi_h:
            _log.warning("Azure DI page dimensions are zero; skipping rect")
            return []

        rect_w = rect.r - rect.l
        rect_h = rect.b - rect.t

        cells: list[TextCell] = []
        cell_idx = 0

        # ---- Use *words* for granular bounding boxes + confidence ----
        if adi_page.words:
            for word in adi_page.words:
                text = word.content
                if not text or not text.strip():
                    continue

                confidence = word.confidence if word.confidence is not None else 0.0
                bbox = self._polygon_to_page_bbox(
                    word.polygon, adi_w, adi_h, rect, rect_w, rect_h
                )
                if bbox is None:
                    continue

                cells.append(
                    TextCell(
                        index=cell_idx,
                        text=text,
                        orig=text,
                        confidence=confidence,
                        from_ocr=True,
                        rect=BoundingRectangle.from_bounding_box(bbox),
                    )
                )
                cell_idx += 1

        # Fallback: if no words, try lines
        elif adi_page.lines:
            for line in adi_page.lines:
                text = line.content
                if not text or not text.strip():
                    continue

                bbox = self._polygon_to_page_bbox(
                    line.polygon, adi_w, adi_h, rect, rect_w, rect_h
                )
                if bbox is None:
                    continue

                cells.append(
                    TextCell(
                        index=cell_idx,
                        text=text,
                        orig=text,
                        confidence=1.0,  # lines have no confidence in the API
                        from_ocr=True,
                        rect=BoundingRectangle.from_bounding_box(bbox),
                    )
                )
                cell_idx += 1

        return cells

    # ------------------------------------------------------------------- #
    #  Coordinate helpers
    # ------------------------------------------------------------------- #
    @staticmethod
    def _polygon_to_page_bbox(
        polygon: list[float] | None,
        adi_w: float,
        adi_h: float,
        rect: BoundingBox,
        rect_w: float,
        rect_h: float,
    ) -> BoundingBox | None:
        """Convert an Azure DI polygon to a docling ``BoundingBox``.

        Azure DI polygons are flat lists ``[x1, y1, x2, y2, …]`` in the
        coordinate unit of the analysed page (typically inches).  We map
        them proportionally onto the crop *rect* in page-point space.
        """
        if not polygon or len(polygon) < 4:
            return None

        xs = polygon[0::2]
        ys = polygon[1::2]

        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)

        # Proportional mapping: ADI coords → page points
        page_l = (min_x / adi_w) * rect_w + rect.l
        page_t = (min_y / adi_h) * rect_h + rect.t
        page_r = (max_x / adi_w) * rect_w + rect.l
        page_b = (max_y / adi_h) * rect_h + rect.t

        return BoundingBox.from_tuple(
            coord=(page_l, page_t, page_r, page_b),
            origin=CoordOrigin.TOPLEFT,
        )

    # ------------------------------------------------------------------- #
    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return AzureDocIntelOcrOptions


# --------------------------------------------------------------------------- #
# Plugin factory (required by Docling)
# --------------------------------------------------------------------------- #
def ocr_engines():
    """Entry-point callable discovered by Docling's plugin system."""
    return {"ocr_engines": [AzureDocIntelOcrModel]}
