# Drawing Module Design

This document outlines the design for a new drawing module in the Rayforge application.

## 1. Purpose and Scope

The purpose of the drawing module is to provide a set of basic drawing primitives that can be used by other parts of the application, such as internal tools or future extensions. This functionality will not be directly exposed in the main user interface but will be available for programmatic use.

The initial scope of this module will include the ability to draw:
- Rectangles
- Text

The implementation will use the Cairo graphics library.

## 2. Module Structure

The new drawing functionality will be located in a new `rayforge/drawing/` directory.

The proposed file structure is:

- `rayforge/drawing/__init__.py`: Initializes the Python package.
- `rayforge/drawing/drawing.py`: Contains the core drawing functions.
- `rayforge/drawing/font.py`: Will be used for font management.

## 3. API

The `rayforge/drawing/drawing.py` module will provide the following functions:

- `draw_rectangle(cr: cairo.Context, x: float, y: float, width: float, height: float)`: Draws a rectangle on the given Cairo context.
- `draw_text(cr: cairo.Context, text: str, x: float, y: float, font_size: float = 12, font_face: str = "Sans")`: Draws text on the given Cairo context.

This design ensures that the drawing functionality is well-encapsulated and can be easily extended in the future.
