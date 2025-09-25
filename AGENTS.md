# Repository Guidelines

## Project Structure & Module Organization
The `mjcf_editor/` package implements a PyQt5 MVVM stack. `model/` stores geometry, XML parsing, and ray casting logic, while `viewmodel/` coordinates state changes feeding the widgets in `view/`. `main.py` wires the dock widgets and OpenGL viewport; update this file when introducing new panels. Shared loaders live in `utils/`, and runtime artifacts such as Gaussian splats reside under `save/gs_backups` and `save/history`—treat them as generated data and keep large binaries out of the repository.

## Build, Test, and Development Commands
Use Python 3.8+; create a virtual environment before installing. `pip install -e .` pulls the PyQt5/OpenGL stack defined in `pyproject.toml`, keeping the package editable. Launch the app with `python -m mjcf_editor.main` to verify UI changes. When you add optional Gaussian renderer features, install the extras with `pip install .[gsrenderer]` so `discoverse` is available.

## Coding Style & Naming Conventions
Follow the existing 4-space indentation and limit lines to roughly 100 characters so long docstrings stay readable. Classes use `CamelCase`, view-model properties and functions use `snake_case`, and enums stick to uppercase members. Prefer descriptive docstrings (Chinese is fine) and add type hints for new public APIs to match `geometry.py`. Keep Qt signal names in the `xChanged` form and mirror property names between view-models and panels.

## Testing Guidelines
There is no automated suite yet, so isolate business logic from PyQt widgets and cover it with `pytest` unit tests placed in a new `tests/` folder (`test_scene_viewmodel.py`, etc.). Include fixtures that mock `SceneViewModel` instead of real widgets to keep tests headless. Before opening a PR, run `pytest` locally and perform a smoke run of `python -m mjcf_editor.main` to confirm dock integration and OpenGL rendering still work.

## Commit & Pull Request Guidelines
Keep commit subjects short, imperative, and scoped (e.g. `refactor view/hierarchy_tree drag state`). Group related edits per commit and explain UI-impacting changes in the body. Pull requests should outline the motivation, list manual test steps, and attach screenshots or screen recordings for visual updates. Reference related issues or TODOs, call out follow-up work, and remind reviewers if any generated files in `save/` should be refreshed manually.
