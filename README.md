# Docling Azure Document Intelligence Plugin

**Docling plugin** that integrates **Azure Document Intelligence** (formerly Azure Form Recognizer) as an OCR engine in Docling.

Azure Document Intelligence is a cloud-based AI service that extracts text, structure, and data from documents with high accuracy — including scanned PDFs, images, handwritten text, and complex layouts.

## Features

- **Cloud-based OCR** — no local GPU or model downloads required.
- **Word-level bounding boxes** with per-word confidence scores.
- **Multiple model support** — `prebuilt-read`, `prebuilt-layout`, `prebuilt-document`, etc.
- **Flexible authentication** — API key or `DefaultAzureCredential` (managed identity, CLI login, etc.).
- **Locale hints** — optionally specify `locale` (e.g. `en-US`) for improved accuracy.

## Prerequisites

1. An **Azure subscription** with a [Document Intelligence resource](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/create-document-intelligence-resource).
2. The **endpoint URL** and **API key** (or configured Azure identity).

## Installation

```bash
uv pip install docling-adi
```

Or with pip:

```bash
pip install docling-adi
```

## Configuration

Set your Azure credentials via environment variables (recommended):

```bash
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com/"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="<your-api-key>"
```

Or copy the provided `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

> **Security:** Never hard-code API keys in source code. Use environment variables, `.env` files (git-ignored), or Azure managed identity.

## Python Usage

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_adi import AzureDocIntelOcrOptions

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    ocr_model="azure_di",              # Plugin engine name
    allow_external_plugins=True,        # Required for third-party plugins
    ocr_options=AzureDocIntelOcrOptions(
        # Credentials (optional here if set via env vars)
        # endpoint="https://<resource>.cognitiveservices.azure.com/",
        # api_key="<your-key>",
        model_id="prebuilt-read",       # Azure DI model (default)
        locale="en-US",                 # Optional locale hint
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
    }
)

result = converter.convert("path/to/document.pdf")
print(result.document.export_to_markdown())
```

## CLI Usage

```bash
# List available external plugins (should show "azure-di")
docling --show-external-plugins

# Run conversion with Azure Document Intelligence OCR
docling --allow-external-plugins --ocr-engine=azure_di path/to/document.pdf
```

## Example Script

See `examples/docling_with_azure_di.py`:

```bash
# Using environment variables for credentials
uv run python examples/docling_with_azure_di.py

# With explicit credentials
uv run python examples/docling_with_azure_di.py \
    --endpoint "https://my-resource.cognitiveservices.azure.com/" \
    --api-key "my-key" \
    --source path/to/document.pdf

# With a specific model and locale
uv run python examples/docling_with_azure_di.py \
    --model-id prebuilt-layout \
    --locale en-US \
    --source invoice.pdf
```

## Azure DI Models

| Model ID | Description |
|---|---|
| `prebuilt-read` | General OCR — text extraction from documents and images (default) |
| `prebuilt-layout` | Text + tables + figures + selection marks + structural layout |
| `prebuilt-document` | Key-value pairs + entities + layout |
| `prebuilt-invoice` | Specialised for invoices |
| `prebuilt-receipt` | Specialised for receipts |

See the [full list of models](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-model-overview) in the Azure documentation.

## Authentication

The plugin resolves credentials in this order:

1. **Explicit values** passed via `AzureDocIntelOcrOptions(endpoint=..., api_key=...)`.
2. **Environment variables** `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` and `AZURE_DOCUMENT_INTELLIGENCE_KEY`.
3. **`DefaultAzureCredential`** from the `azure-identity` package (supports managed identity, Azure CLI login, VS Code credentials, etc.) — used when no API key is available.

## Project Structure

```
docling_adi/
├── pyproject.toml                          # Project metadata, deps, entry-point
├── README.md
├── .env.example                            # Template for credentials
├── docling_adi/
│   ├── __init__.py                         # Public exports
│   └── plugin.py                           # AzureDocIntelOcrModel + factory
├── examples/
│   └── docling_with_azure_di.py            # End-to-end example script
└── tests/
    └── test_azure_di_ocr.py               # Full test suite with mocks
```

## Plugin Registration

The plugin registers itself via `pyproject.toml` entry-points:

```toml
[project.entry-points.docling]
"azure-di" = "docling_adi.plugin"
```

And exports the OCR engine via the factory function:

```python
def ocr_engines():
    return {"ocr_engines": [AzureDocIntelOcrModel]}
```

## Development

```bash
# Clone the repo
git clone https://github.com/<your-org>/docling_adi
cd docling_adi

# Create virtual environment + install deps
uv venv
uv sync --all-extras

# Run tests (uses mocks — no Azure credentials needed)
uv run pytest

# Run linter
uv run ruff check .

# Format code
uv run ruff format .

# Build wheel
uv build

# Install locally
uv pip install dist/docling_adi-*.whl
```

## License & Attribution

- **Azure Document Intelligence**: [Microsoft Azure Cognitive Services](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
- **Docling**: [DS4SD/docling](https://github.com/DS4SD/docling)