# Use Azure Document Intelligence with Docling to OCR a PDF and print Markdown.
#
# Based on: examples/docling_with_custom_models.py from docling_surya
#   Repository : https://github.com/harrykhh/docling_surya
#   Author     : Harry Ho (@harrykhh)
#   License    : GPL-3.0
#
# The overall example structure — configuring OCR options, building a
# PdfPipelineOptions with allow_external_plugins=True, creating a
# DocumentConverter with format_options, and converting a sample EPA PDF —
# is adapted from that reference script.
#
# What this example does
# - Configures `AzureDocIntelOcrOptions` for OCR processing via Azure DI.
# - Runs the PDF pipeline with Azure Document Intelligence and prints Markdown.
#
# Prerequisites
# - Install the plugin: `uv pip install docling-adi`
# - Set environment variables (or pass credentials directly):
#     AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
#     AZURE_DOCUMENT_INTELLIGENCE_KEY=<your-api-key>
#
# How to run
#   uv run python examples/docling_with_azure_di.py
#   uv run python examples/docling_with_azure_di.py --source path/to/document.pdf
#   uv run python examples/docling_with_azure_di.py --endpoint https://... --api-key ...
#
# Notes
# - The default `source` points to a sample EPA PDF URL; replace with a local path if desired.
# - If no --api-key is provided, the plugin falls back to the
#   AZURE_DOCUMENT_INTELLIGENCE_KEY env var, then to DefaultAzureCredential.

from __future__ import annotations

import argparse
import os
import sys

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_adi import AzureDocIntelOcrOptions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a document using Docling + Azure Document Intelligence OCR",
    )
    parser.add_argument(
        "--source",
        default="https://19january2021snapshot.epa.gov/sites/static/files/2016-02/documents/epa_sample_letter_sent_to_commissioners_dated_february_29_2015.pdf",
        help="Path or URL to the document to convert (default: sample EPA PDF)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Azure DI endpoint URL (overrides AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT env var)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Azure DI API key (overrides AZURE_DOCUMENT_INTELLIGENCE_KEY env var)",
    )
    parser.add_argument(
        "--model-id",
        default="prebuilt-read",
        help="Azure DI model ID (default: prebuilt-read)",
    )
    parser.add_argument(
        "--locale",
        default=None,
        help="Hint locale for OCR, e.g. 'en-US' (default: auto-detect)",
    )
    args = parser.parse_args()

    # Build OCR options
    ocr_options = AzureDocIntelOcrOptions(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model_id=args.model_id,
        locale=args.locale,
    )

    # Validate that we have at least an endpoint available
    endpoint = args.endpoint or os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not endpoint:
        print(
            "ERROR: No Azure Document Intelligence endpoint configured.\n"
            "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or pass --endpoint.",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_model="azure_di",
        allow_external_plugins=True,
        ocr_options=ocr_options,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    print(f"Converting: {args.source}")
    print(f"Azure DI model: {args.model_id}")
    print("-" * 60)

    result = converter.convert(args.source)
    markdown = result.document.export_to_markdown()

    print(markdown)


if __name__ == "__main__":
    main()
