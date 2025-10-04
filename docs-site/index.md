# Welcome to Rayforge Documentation

Rayforge is a modern, cross-platform G-code sender and control software for GRBL-based laser cutters and engravers. Built with Gtk4 and Libadwaita, it provides a clean, native interface for Linux and Windows, offering a full suite of tools for both hobbyists and professionals.

![Rayforge Main Interface](images/ss-main.png)

## What is Rayforge?

Rayforge transforms your laser cutter or engraver into a powerful creative tool. Whether you're a hobbyist working on personal projects or a professional running a business, Rayforge provides the features and reliability you need.

### Key Capabilities
## Key Features

| Feature                      | Description                                                                                        |
| :--------------------------- | :------------------------------------------------------------------------------------------------- |
| **Modern UI**                | Polished and modern UI built with Gtk4 and Libadwaita. Supports system, light, and dark themes.    |
| **Multi-Layer Operations**   | Assign different operations (e.g., engrave then cut) to layers in your design.                     |
| **Versatile Operations**     | Supports Contour, Raster Engraving (with cross-hatch fill), Shrink Wrap, and Depth Engraving.      |
| **Overscan & Kerf Comp.**    | Improve engraving quality with overscan and ensure dimensional accuracy with kerf compensation.    |
| **2.5D Cutting**             | Perform multi-pass cuts with a configurable step-down between each pass for thick materials.       |
| **3D G-code Preview**        | Visualize G-code toolpaths in 3D to verify the job before sending it to the machine.               |
| **Multi-Machine Profiles**   | Configure and instantly switch between multiple machine profiles.                                  |
| **GRBL Firmware Settings**   | Read and write firmware parameters (`$$`) directly from the UI.                                    |
| **Comprehensive 2D Canvas**  | Full suite of tools: alignment, transformation, measurement, zoom, pan, and more.                  |
| **Advanced Path Generation** | High-quality image tracing, travel time optimization, path smoothing, and spot size interpolation. |
| **Holding Tabs**             | Add tabs to contour cuts to hold pieces in place. Supports manual and automatic placement.         |
| **G-code Macros & Hooks**    | Run custom G-code snippets before/after jobs. Supports variable substitution.                      |
| **Broad File Support**       | Import from SVG, DXF, PDF, JPEG, PNG, BMP, and even Ruida files (`.rd`).                           |
| **Multi-Laser Operations**   | Choose different lasers for each operation in a job                                                |
| **Camera Integration**       | Use a USB camera for workpiece alignment, positioning, and background tracing.                     |
| **Cross-Platform**           | Native builds for Linux and Windows.                                                               |
| **Extensible**               | Open development model makes it easy to [add support for new devices](docs/driver.md).             |
| **Multi-Language**           | Available in English, Portuguese, Spanish, and German.                                             |
| **G-code Dialects**          | Supports GRBL, Smoothieware, and other GRBL-compatible firmwares.                 

## Quick Links

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __[Getting Started](getting-started/index.md)__

    ---

    New to Rayforge? Start here for installation instructions and your first project.

-   :material-view-dashboard:{ .lg .middle } __[User Interface](ui/index.md)__

    ---

    Learn about the main window, canvas tools, and interface elements.

-   :material-feature-search:{ .lg .middle } __[Features](features/index.md)__

    ---

    Explore all the operations and capabilities Rayforge offers.

-   :material-wrench:{ .lg .middle } __[Machine Setup](machine/index.md)__

    ---

    Configure your laser cutter or engraver to work with Rayforge.

</div>

## Platform Support

Rayforge runs natively on:

- **Linux** - Ubuntu 24.04+ (PPA), Snap package for all distributions, or install from source
- **Windows** - Native builds available

See the [Installation Guide](getting-started/installation.md) for detailed instructions.

## Device Compatibility

| Device Type      | Connection Method       | Notes                                                          |
| :--------------- | :---------------------- | :------------------------------------------------------------- |
| **GRBL**         | Serial Port             | Supported since version 0.13. The most common connection type. |
| **GRBL**         | Network (WiFi/Ethernet) | Connect to any GRBL device on your network.                    |
| **Smoothieware** | Telnet                  | Supported since version 0.15.                                  |

## Community & Support

- **Report Issues**: [GitHub Issues](https://github.com/barebaric/rayforge/issues)
- **Source Code**: [GitHub Repository](https://github.com/barebaric/rayforge)
- **Contribute**: See our [Contributing Guide](contributing/index.md)

## About This Documentation

This documentation is designed for end-users of Rayforge. If you're looking for developer documentation, please see the [Development Setup](contributing/development.md) guide.

---

Ready to get started? Head to the [Installation Guide](getting-started/installation.md) to install Rayforge on your system.
