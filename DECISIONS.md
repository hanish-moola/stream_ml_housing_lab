# Architectural & Technical Decisions

## 2025-10-31
- Adopted Poetry as the single source of dependency and script management. Runtime and dev dependencies defined in `pyproject.toml`; `poetry lock`/`poetry install` to be executed in connected environments once package indexes are accessible.
- Pruned legacy streaming/serving scaffolding to return the repo to a lean notebook-derived baseline so the new pipeline can be rebuilt cleanly.
- Introduced a configuration system (`config/config.yaml` + `src/config.py`) with environment-variable overrides to support reproducible experiments across environments.
- Established shared utility modules (`src/data.py`, `src/registry.py`, `src/logging_utils.py`) to centralise data access, artifact management, and logging, enabling later pipeline stages to compose these primitives.
- Seeded lightweight pytest coverage for config loading and artifact registry behaviour to guard future refactors.
