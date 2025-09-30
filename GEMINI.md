# Rayforge Project Overview

This document provides a comprehensive overview of the Rayforge project, including its purpose, architecture, and development conventions.

## 1. Project Purpose and Technologies

Rayforge is a modern, cross-platform G-code sender and control software for GRBL-based laser cutters and engravers. It is built with Python and utilizes the Gtk4 and Libadwaita libraries for its graphical user interface, providing a clean and native look and feel on Linux and Windows.

**Key Technologies:**

*   **Programming Language:** Python
*   **UI Framework:** Gtk4 and Libadwaita
*   **Dependency Management:** Pixi
*   **Supported Platforms:** Linux and Windows

## 2. Project Architecture

The project is structured into several key directories and files:

*   `rayforge/`: Contains the main source code for the application.
    *   `app.py`: The main entry point of the application.
    *   `mainwindow.py`: Defines the main window of the application.
    *   `core/`: Contains the core logic of the application, such as the document model, G-code generation, and machine communication.
    *   `ui/`: Contains the UI components of the application.
*   `tests/`: Contains the unit tests for the project.
*   `pixi.toml`: Defines the project's dependencies and scripts.
*   `pyproject.toml`: Defines the project's metadata and build system.
*   `README.md`: Provides a general overview of the project.

## 3. Building and Running the Project

## System dependencies

You'll need to manually install some system dependencies:

```
# For ubuntu / debian linux
sudo apt update
sudo apt install python3-gi gir1.2-gtk-3.0 gir1.2-adw-1 gir1.2-gdkpixbuf-2.0 libgirepository-1.0-dev libgirepository-2.0-0 libvips42t64 libpotrace-dev libagg-dev libadwaita-1-0 libopencv-dev
```

The project uses Pixi to manage dependencies and development environments. The following commands can be used to build and run the project:

*   **Install pixi dependencies:** `pixi install`
*   **Run the application:** `pixi run rayforge`
*   **Run the tests:** `pixi run test`

## 4. Development Conventions

The project follows standard Python coding conventions and uses the following tools to enforce code quality:

*   **Linting:** `flake8`
*   **Type Checking:** `pyright` and `mypy`

All code should be formatted according to the project's `.flake8` configuration file.

## 5. Contribution Guidelines

Contributions to the project are welcome. Please follow these guidelines when submitting a pull request:

*   Ensure that all new code is covered by unit tests.
*   Ensure that the code passes all linting and type-checking tests.
*   Provide a clear and concise description of the changes in the pull request.
