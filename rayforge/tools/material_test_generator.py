from __future__ import annotations
import cairo
import math
from typing import List, Tuple, Optional

from rayforge.core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    MoveToCommand,
    LineToCommand,
    SetPowerCommand,
    SetCutSpeedCommand,
)
from rayforge.drawing.drawing import draw_text, draw_rectangle # Assuming these are available

def _text_to_ops(
    text: str,
    x: float,
    y: float,
    font_size: float,
    font_face: str,
    ops: Ops,
    power: float,
    speed: float,
) -> None:
    # Create a temporary surface to calculate text extents
    temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
    temp_cr = cairo.Context(temp_surface)
    temp_cr.select_font_face(font_face, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    temp_cr.set_font_size(font_size)
    extents = temp_cr.text_extents(text)

    # Create a new surface that is large enough for the text with a margin
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(extents.width) + 4, int(extents.height) + 4)
    cr = cairo.Context(surface)
    cr.select_font_face(font_face, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(font_size)
    cr.new_path()
    # Move the path to the start, considering the text's bearing
    cr.move_to(-extents.x_bearing + 2, -extents.y_bearing + 2)
    cr.text_path(text)
    path_data = cr.copy_path()

    ops.add(SetPowerCommand(power))
    ops.add(SetCutSpeedCommand(speed))

    for path_type, points in path_data:
        if path_type == cairo.PATH_MOVE_TO:
            px, py = points[0], points[1]
            ops.add(MoveToCommand((x + px, y - py, 0.0)))
        elif path_type == cairo.PATH_LINE_TO:
            px, py = points[0], points[1]
            ops.add(LineToCommand((x + px, y - py, 0.0)))
        elif path_type == cairo.PATH_CURVE_TO:
            p1x, p1y, p2x, p2y, p3x, p3y = points
            c1 = (x + p1x, y - p1y, 0.0)
            c2 = (x + p2x, y - p2y, 0.0)
            end = (x + p3x, y - p3y, 0.0)
            ops.bezier_to(c1, c2, end)
        elif path_type == cairo.PATH_CLOSE_PATH:
            pass

def generate_material_test_ops(
    test_type: str,  # "Engrave" or "Cut"
    laser_type: str, # "Diode" or "CO2"
    speed_range: Tuple[float, float], # (min_speed, max_speed)
    power_range: Tuple[float, float], # (min_power, max_power)
    grid_dimensions: Tuple[int, int], # (cols, rows)
    shape_size: float, # mm
    spacing: float, # mm
    line_interval: Optional[float] = None, # mm, for Engrave tests
    include_labels: bool = True,
) -> Ops:
    ops = Ops()
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "Material Test"))

    cols, rows = grid_dimensions
    min_speed, max_speed = speed_range
    min_power, max_power = power_range

    # Calculate step sizes
    speed_step = (max_speed - min_speed) / (cols - 1) if cols > 1 else 0
    power_step = (max_power - min_power) / (rows - 1) if rows > 1 else 0

    all_test_elements = [] # To store labels and boxes for sorting

    # --- Generate Box Data ---
    for r in range(rows):
        for c in range(cols):
            current_speed = min_speed + c * speed_step
            current_power = min_power + r * power_step

            x_pos = c * (shape_size + spacing)
            y_pos = r * (shape_size + spacing)

            # Add box data for later sorting
            all_test_elements.append({
                "type": "box",
                "x": x_pos,
                "y": y_pos,
                "width": shape_size,
                "height": shape_size,
                "speed": current_speed,
                "power": current_power,
            })

    # --- Generate Labels ---
    if include_labels:
        label_power = 10.0
        label_speed = 1000.0
        font_size = 2.5

        # Main X-axis label
        x_axis_label_text = "speed (mm/min)"
        x_pos = (cols * (shape_size + spacing)) / 2
        y_pos = -10
        _text_to_ops(x_axis_label_text, x_pos, y_pos, font_size, "Sans", ops, label_power, label_speed)

        # Main Y-axis label
        y_axis_label_text = "power"
        x_pos = -10
        y_pos = (rows * (shape_size + spacing)) / 2
        _text_to_ops(y_axis_label_text, x_pos, y_pos, font_size, "Sans", ops, label_power, label_speed)

        # X-axis numeric labels (Speed)
        for c in range(cols):
            current_speed = min_speed + c * speed_step
            label_text = f"{int(current_speed)}"
            x_pos = c * (shape_size + spacing) + shape_size / 2
            y_pos = -5 # Below the grid
            _text_to_ops(label_text, x_pos, y_pos, font_size, "Sans", ops, label_power, label_speed)

        # Y-axis numeric labels (Power)
        for r in range(rows):
            current_power = min_power + r * power_step
            label_text = f"{int(current_power)}"
            x_pos = -5 # To the left of the grid
            y_pos = r * (shape_size + spacing) + shape_size / 2
            _text_to_ops(label_text, x_pos, y_pos, font_size, "Sans", ops, label_power, label_speed)

    # --- Sort Box Elements by Risk ---
    # Sorting criteria: Speed Descending (highest speed is lowest risk), Power Ascending (lowest power is lowest risk)
    all_test_elements.sort(key=lambda e: (-e["speed"], e["power"]))

    # --- Generate Box Ops in Sorted Order ---
    for element in all_test_elements:
        if element["type"] == "box":
            ops.add(SetPowerCommand(element["power"]))
            ops.add(SetCutSpeedCommand(element["speed"]))
            
            # Draw rectangle
            ops.add(MoveToCommand((element["x"], element["y"], 0.0)))
            ops.add(LineToCommand((element["x"] + element["width"], element["y"], 0.0)))
            ops.add(LineToCommand((element["x"] + element["width"], element["y"] + element["height"], 0.0)))
            ops.add(LineToCommand((element["x"], element["y"] + element["height"], 0.0)))
            ops.add(LineToCommand((element["x"], element["y"], 0.0))) # Close the rectangle

    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    return ops