import math
from typing import List, Tuple, Any, Optional, TYPE_CHECKING
from itertools import groupby
from .linearize import linearize_arc
from .primitives import is_point_in_polygon

if TYPE_CHECKING:
    from .geometry import Geometry


def encloses(container: "Geometry", content: "Geometry") -> bool:
    """
    Checks if a container geometry fully encloses a content geometry.

    This function performs a series of checks to determine containment.
    The 'content' geometry must be fully inside the 'container' geometry's
    boundary, not intersecting it, and not located within any of the
    container's holes.

    Args:
        container: The Geometry object that might contain the other.
        content: The Geometry object to check for containment.

    Returns:
        True if the container encloses the content, False otherwise.
    """
    # 1. Sanity Checks
    if container.is_empty() or content.is_empty():
        return False

    # 2. Broad Phase: Bounding Box Check
    cont_min_x, cont_min_y, cont_max_x, cont_max_y = container.rect()
    cont_min_x, cont_min_y, cont_max_x, cont_max_y = container.rect()
    other_min_x, other_min_y, other_max_x, other_max_y = content.rect()
    if not (
        cont_min_x <= other_min_x
        and cont_min_y <= other_min_y
        and cont_max_x >= other_max_x
        and cont_max_y >= other_max_y
    ):
        return False

    # 3. Mid Phase: Intersection Check
    if container.intersects_with(content):
        return False

    # 4. Narrow Phase: Point-in-Polygon Check
    other_segments = content.segments()
    if not other_segments or not other_segments[0]:
        return False  # Other geometry is degenerate
    test_point = other_segments[0][0][:2]

    self_contours_geo = container.split_into_contours()
    self_contour_data = container._get_valid_contours_data(self_contours_geo)
    closed_contours = [c for c in self_contour_data if c["is_closed"]]
    if not closed_contours:
        return False  # Self has no closed contours to contain anything

    winding_number = 0
    for contour in closed_contours:
        # A single contour geometry always starts at command index 0
        area = get_subpath_area(contour["geo"].commands, 0)

        if is_point_in_polygon(test_point, contour["vertices"]):
            if area > 1e-9:  # CCW (outer boundary)
                winding_number += 1
            elif area < -1e-9:  # CW (hole)
                winding_number -= 1

    return winding_number > 0


def get_subpath_vertices(
    commands: List[Any], start_cmd_index: int
) -> List[Tuple[float, float]]:
    """
    Extracts all 2D vertices for a single continuous subpath starting at a
    given MoveToCommand index, linearizing any arcs.
    """
    from .geometry import MoveToCommand, LineToCommand, ArcToCommand

    vertices: List[Tuple[float, float]] = []
    if start_cmd_index >= len(commands):
        return []
    last_pos_3d = commands[start_cmd_index].end or (0.0, 0.0, 0.0)
    vertices.append(last_pos_3d[:2])

    for i in range(start_cmd_index + 1, len(commands)):
        cmd = commands[i]
        if isinstance(cmd, MoveToCommand):
            # End of the subpath
            break
        if not isinstance(cmd, (LineToCommand, ArcToCommand)) or not cmd.end:
            continue

        if isinstance(cmd, LineToCommand):
            vertices.append(cmd.end[:2])
        elif isinstance(cmd, ArcToCommand):
            segments = linearize_arc(cmd, last_pos_3d)
            for _, p2 in segments:
                vertices.append(p2[:2])
        last_pos_3d = cmd.end

    return vertices


def get_subpath_area(commands: List[Any], start_cmd_index: int) -> float:
    """
    Calculates the signed area of a subpath using the shoelace formula.
    Returns 0 for open or degenerate paths.
    """
    vertices = get_subpath_vertices(commands, start_cmd_index)
    if len(vertices) < 3:
        return 0.0

    # A subpath is closed if its first and last vertices are the same.
    p_start, p_end = vertices[0], vertices[-1]
    if not (
        math.isclose(p_start[0], p_end[0])
        and math.isclose(p_start[1], p_end[1])
    ):
        return 0.0

    # Shoelace formula to calculate signed area
    area = 0.0
    # The last point is a duplicate of the first, so we can ignore it.
    for i in range(len(vertices) - 1):
        p1 = vertices[i]
        p2 = vertices[i + 1]
        area += (p1[0] * p2[1]) - (p2[0] * p1[1])

    return area / 2.0


def get_path_winding_order(commands: List[Any], segment_index: int) -> str:
    """
    Determines winding order ('cw', 'ccw', 'unknown') for the subpath at a
    given index.
    """
    from .geometry import MoveToCommand

    # Find the start of the subpath for the given segment
    subpath_start_index = -1
    for i in range(segment_index, -1, -1):
        if isinstance(commands[i], MoveToCommand):
            subpath_start_index = i
            break
    if subpath_start_index == -1:
        return "unknown"

    area = get_subpath_area(commands, subpath_start_index)

    # Convention: positive area is CCW, negative is CW in a Y-up system
    if abs(area) < 1e-9:
        return "unknown"
    elif area > 0:
        return "ccw"
    else:
        return "cw"


def get_point_and_tangent_at(
    commands: List[Any], segment_index: int, t: float
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Calculates the 2D point and tangent vector at a parameter 't' along a
    segment.
    """
    from .geometry import LineToCommand, ArcToCommand, MovingCommand

    cmd = commands[segment_index]
    if not isinstance(cmd, MovingCommand) or cmd.end is None:
        return None

    # Find the start point of this segment
    start_pos_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    for i in range(segment_index - 1, -1, -1):
        prev_cmd = commands[i]
        if prev_cmd.end:
            start_pos_3d = prev_cmd.end
            break

    p0 = start_pos_3d[:2]
    p1 = cmd.end[:2]

    if isinstance(cmd, LineToCommand):
        point = (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
        tangent_vec = (p1[0] - p0[0], p1[1] - p0[1])
    elif isinstance(cmd, ArcToCommand):
        center = (
            p0[0] + cmd.center_offset[0],
            p0[1] + cmd.center_offset[1],
        )
        start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
        end_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
        angle_range = end_angle - start_angle

        if cmd.clockwise:
            if angle_range > 0:
                angle_range -= 2 * math.pi
        else:
            if angle_range < 0:
                angle_range += 2 * math.pi

        current_angle = start_angle + t * angle_range
        radius_start = math.dist(p0, center)
        radius_end = math.dist(p1, center)
        radius = radius_start + t * (radius_end - radius_start)

        point = (
            center[0] + radius * math.cos(current_angle),
            center[1] + radius * math.sin(current_angle),
        )

        radius_vec = (point[0] - center[0], point[1] - center[1])
        if cmd.clockwise:
            tangent_vec = (radius_vec[1], -radius_vec[0])
        else:
            tangent_vec = (-radius_vec[1], radius_vec[0])
    else:
        return None

    norm = math.hypot(tangent_vec[0], tangent_vec[1])
    if norm < 1e-9:
        return point, (1.0, 0.0)

    normalized_tangent = (tangent_vec[0] / norm, tangent_vec[1] / norm)
    return point, normalized_tangent


def get_outward_normal_at(
    commands: List[Any], segment_index: int, t: float
) -> Optional[Tuple[float, float]]:
    """
    Calculates the outward-pointing normal vector for a point on a closed
    path.
    """
    winding = get_path_winding_order(commands, segment_index)
    if winding == "unknown":
        return None

    result = get_point_and_tangent_at(commands, segment_index, t)
    if not result:
        return None

    _, tangent = result
    tx, ty = tangent

    if winding == "ccw":
        return (ty, -tx)
    else:  # winding == "cw"
        return (-ty, tx)


def get_angle_at_vertex(
    p0: Tuple[float, ...], p1: Tuple[float, ...], p2: Tuple[float, ...]
) -> float:
    """
    Calculates the internal angle of the corner at point p1 in the
    XY plane. Returns the angle in radians.
    """
    # Create vectors from p1 to p0 and p1 to p2.
    v1x, v1y = p0[0] - p1[0], p0[1] - p1[1]
    v2x, v2y = p2[0] - p1[0], p2[1] - p1[1]

    # Calculate magnitudes for normalization.
    mag_v1 = math.hypot(v1x, v1y)
    mag_v2 = math.hypot(v2x, v2y)
    mag_prod = mag_v1 * mag_v2
    if mag_prod < 1e-9:
        return math.pi  # Straight line if points are coincident.

    # Dot product and normalization to get cosine of the angle.
    dot = v1x * v2x + v1y * v2y
    cos_theta = min(1.0, max(-1.0, dot / mag_prod))

    # Return angle in radians.
    return math.acos(cos_theta)


def remove_duplicates(
    points: List[Tuple[float, ...]],
) -> List[Tuple[float, ...]]:
    """Removes consecutive duplicate points from a list."""
    return [k for k, v in groupby(points)]


def is_clockwise(points: List[Tuple[float, ...]]) -> bool:
    """
    Determines if the first three points in a list form a clockwise turn
    using the 2D cross product.
    """
    if len(points) < 3:
        return False  # Not enough points to determine direction

    p1, p2, p3 = points[0], points[1], points[2]
    cross_product = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (
        p3[0] - p2[0]
    )
    return cross_product < 0


def arc_direction_is_clockwise(
    points: List[Tuple[float, ...]], center: Tuple[float, float]
) -> bool:
    """
    Determines the winding direction of a sequence of points around a center
    by summing the cross products of vectors from the center to consecutive
    points. A negative sum indicates a net clockwise rotation.
    """
    xc, yc = center
    cross_product_sum = 0.0
    for i in range(len(points) - 1):
        x0, y0 = points[i][:2]
        x1, y1 = points[i + 1][:2]
        # Vectors from center to points
        v0x, v0y = x0 - xc, y0 - yc
        v1x, v1y = x1 - xc, y1 - yc
        # 2D Cross product
        cross_product_sum += v0x * v1y - v0y * v1x

    return cross_product_sum < 0
