import cairo

def draw_rectangle(cr: cairo.Context, x: float, y: float, width: float, height: float) -> None:
    """
    Draws a rectangle on the given Cairo context.

    Args:
        cr: The Cairo context to draw on.
        x: The x-coordinate of the top-left corner of the rectangle.
        y: The y-coordinate of the top-left corner of the rectangle.
        width: The width of the rectangle.
        height: The height of the rectangle.
    """
    cr.rectangle(x, y, width, height)


def draw_text(cr: cairo.Context, text: str, x: float, y: float, font_size: float = 12, font_face: str = "Sans") -> None:
    """
    Draws text on the given Cairo context.

    Args:
        cr: The Cairo context to draw on.
        text: The text to draw.
        x: The x-coordinate of the position of the text.
        y: The y-coordinate of the position of the text.
        font_size: The font size to use.
        font_face: The font face to use.
    """
    cr.select_font_face(font_face, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(font_size)
    cr.move_to(x, y)
    cr.show_text(text)
