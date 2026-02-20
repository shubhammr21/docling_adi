# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic changelog generation.

<!--next-version-placeholder-->

## v0.1.0 (2026-02-20)

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
- **ci**: Add pre-commit hooks (ruff lint/format, trailing whitespace, EOF, TOML, commitizen)
- **ci**: Add GitHub Actions CI workflow — lint + test matrix (Python 3.11/3.12/3.13) + coverage upload
- **ci**: Add GitHub Actions Release workflow — semantic-release version bump, changelog, GitHub Release, PyPI publish
- **ci**: Add semantic versioning with python-semantic-release and commitizen
- **docs**: Full credit and acknowledgements to [`docling_surya`](https://github.com/harrykhh/docling_surya) by Harry Ho

### Bug Fixes

- Replace `Optional[X]` with `X | None` to fix UP045 ruff errors on CI
- Remove stale `UP006`, `UP007`, `UP035` ignore rules from ruff config
- Fix `[project.urls]` TOML ordering that swallowed `dependencies` key
- Add missing `lang` default to `AzureDocIntelOcrOptions` for base class compatibility

### Acknowledgements

This plugin is directly based on [`docling_surya`](https://github.com/harrykhh/docling_surya)
by **Harry Ho ([@harrykhh](https://github.com/harrykhh))** (GPL-3.0). The plugin architecture,
`BaseOcrModel` subclass pattern, coordinate mapping, project scaffolding, and test strategy
are adapted from that work.
