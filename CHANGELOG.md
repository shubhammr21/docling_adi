# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic changelog generation.

<!--next-version-placeholder-->

## v0.1.0 (2025-07-14)

### Features

- **plugin**: Initial Azure Document Intelligence OCR plugin for Docling
  - `AzureDocIntelOcrModel` with word-level bounding boxes and confidence scores
  - `AzureDocIntelOcrOptions` with flexible auth (API key / env vars / `DefaultAzureCredential`)
  - Plugin entry-point registered as `azure-di` (kind=`azure_di`)
  - Supports `prebuilt-read`, `prebuilt-layout`, and other Azure DI models
  - Optional locale hint for improved OCR accuracy
- **examples**: Add end-to-end example script with CLI arguments
- **tests**: Full test suite (37 tests) with mocked Azure SDK
- **coverage**: pytest-cov integration at 97.33% coverage
  - Terminal, HTML, and XML coverage reports
  - Minimum coverage threshold set to 80%
