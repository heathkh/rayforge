# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

Rayforge uses **Pixi** for dependency management and development environments. All development commands should be run through Pixi.

### Essential Commands

```bash
# Setup environment (run once)
pixi install

# Run the application
pixi run rayforge

# Run all tests
pixi run test

# Code quality checks
pixi run lint

# Build wheel package
pixi run wheel

# Translation management
pixi run update-translations
pixi run compile-translations

# Activate development shell (for interactive work)
pixi shell
```

### Single Test Execution

To run a specific test file or test function:
```bash
# Run specific test file
pytest tests/path/to/test_file.py

# Run specific test function
pytest tests/path/to/test_file.py::test_function_name

# Run with verbose output
pytest -v tests/path/to/test_file.py
```

## Code Architecture

Rayforge is a GTK4/Libadwaita-based desktop application for laser cutter control, built with a modular, pipeline-driven architecture:

### Core Architecture Components

- **`rayforge/core/`**: Document model and geometry handling
  - `doc.py`: Main document structure with layers and operations
  - `ops/`: Operation definitions (contour, raster, etc.)
  - `geo/`: Geometric primitives and transformations
  - `group.py`, `layer.py`: Document organization

- **`rayforge/pipeline/`**: Processing pipeline architecture
  - `producer/`: Converts input formats to geometry
  - `transformer/`: Modifies geometry (offsets, transforms, etc.)
  - `modifier/`: Advanced geometry modifications
  - `encoder/`: Converts to output formats (G-code, etc.)

- **`rayforge/machine/`**: Hardware interface layer
  - `driver/`: Device communication protocols
  - `transport/`: Low-level communication (serial, network, etc.)
  - `models/`: Machine configuration and profiles

- **`rayforge/doceditor/`**: Document editing interface
  - `editor.py`: Main document editor controller
  - `ui/`: Document-specific UI components

- **`rayforge/workbench/`**: Canvas and visualization
  - `surface.py`: 2D drawing surface
  - `canvas3d/`: 3D G-code preview system
  - `elements/`: Canvas drawable elements

- **`rayforge/image/`**: File format importers
  - Each subdirectory handles a specific format (svg, pdf, dxf, etc.)

### Key Design Patterns

1. **Pipeline Processing**: Jobs flow through producer → transformer → modifier → encoder stages
2. **Driver Architecture**: Composable transport + encoder for different machines
3. **Document Model**: Hierarchical structure with layers containing operations
4. **Async Task Management**: Background processing via `rayforge.shared.tasker`
5. **Configuration Management**: Centralized config in `rayforge.config`

### UI Architecture

- **GTK4/Libadwaita**: Modern Linux desktop UI framework
- **MainWindow**: Central application controller (`mainwindow.py`)
- **Modular UI**: Each major component has its own UI subdirectory
- **3D Visualization**: Optional OpenGL-based G-code preview

## Development Guidelines

### Code Style (from CONVENTIONS.md and .flake8)

- Follow PEP8 with 79-character line limit
- Use flake8 for linting (configured in `.flake8`)
- Keep cyclomatic complexity low and code testable
- Create/update unit tests for all new or changed functions

### File Import Structure

Files support multiple formats:
- **SVG**: Direct vector import or traced
- **DXF**: CAD file import
- **PDF**: Page-by-page vector extraction
- **PNG/BMP**: Raster images (traced to vectors)
- **Ruida (.rd)**: Proprietary laser format

### Testing

- Tests located in `tests/` directory (mirrors `rayforge/` structure)
- Uses pytest with asyncio support
- Mock external dependencies (hardware, file I/O)
- Test coverage tracked with pytest-cov

### Translation System

- Gettext-based internationalization
- Translations in `rayforge/locale/`
- Use `_("string")` for translatable strings
- Update .po files before building releases

### Machine Driver Development

See `docs/driver.md` for comprehensive driver development guide. Key concepts:
- Drivers compose Transport (communication) + OpsEncoder (command translation)
- Support for serial, network, and custom protocols
- Real-time status reporting and job execution

## Build System

- **pyproject.toml**: Python package configuration
- **pixi.toml**: Conda/pip dependency management and task definitions
- **setuptools**: Build backend with git versioning
- Cross-platform builds for Linux and Windows
- Packaging: Debian packages, Snap, Windows installer, PyPI

## Important Notes

- Application entry point: `rayforge.app:main`
- Requires system dependencies: GTK4, Libadwaita, various image libraries
- OpenGL support optional but recommended for 3D preview
- Async architecture: most I/O operations are async
- Configuration auto-saves on application exit


## Lint Style

When editing code keep lines 79 characters or less.

