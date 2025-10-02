from copy import deepcopy
import pytest
import math
import numpy as np
from typing import cast

from rayforge.core.geo import (
    Geometry,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
)
from rayforge.core.geo.query import get_total_distance


def _create_translate_matrix(x, y, z):
    """Creates a NumPy translation matrix."""
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _create_scale_matrix(sx, sy, sz):
    """Creates a NumPy scaling matrix."""
    return np.array(
        [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _create_z_rotate_matrix(angle_rad):
    """Creates a NumPy Z-axis rotation matrix."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


@pytest.fixture
def empty_geometry():
    return Geometry()


@pytest.fixture
def sample_geometry():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    geo.arc_to(20, 0, i=5, j=-10)
    return geo


def test_initialization(empty_geometry):
    assert len(empty_geometry.commands) == 0
    assert empty_geometry.last_move_to == (0.0, 0.0, 0.0)


def test_add_commands(empty_geometry):
    empty_geometry.move_to(5, 5)
    assert len(empty_geometry.commands) == 1
    assert isinstance(empty_geometry.commands[0], MoveToCommand)

    empty_geometry.line_to(10, 10)
    assert isinstance(empty_geometry.commands[1], LineToCommand)


def test_clear_commands(sample_geometry):
    sample_geometry.clear()
    assert len(sample_geometry.commands) == 0


def test_move_to(sample_geometry):
    sample_geometry.move_to(15, 15)
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, MoveToCommand)
    assert last_cmd.end == (15.0, 15.0, 0.0)


def test_line_to(sample_geometry):
    sample_geometry.line_to(20, 20)
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == (20.0, 20.0, 0.0)


def test_close_path(sample_geometry):
    sample_geometry.move_to(5, 5, -1.0)
    sample_geometry.close_path()
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == sample_geometry.last_move_to
    assert last_cmd.end == (5.0, 5.0, -1.0)


def test_arc_to(sample_geometry):
    sample_geometry.arc_to(5, 5, 2, 3, clockwise=False)
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, ArcToCommand)
    assert last_cmd.end == (5.0, 5.0, 0.0)
    assert last_cmd.clockwise is False


def test_serialization_deserialization(sample_geometry):
    geo_dict = sample_geometry.to_dict()
    new_geo = Geometry.from_dict(geo_dict)

    assert len(new_geo.commands) == len(sample_geometry.commands)
    assert new_geo.last_move_to == sample_geometry.last_move_to
    for cmd1, cmd2 in zip(new_geo.commands, sample_geometry.commands):
        assert type(cmd1) is type(cmd2)
        assert cmd1.end == cmd2.end


def test_from_dict_ignores_state_commands():
    geo_dict = {
        "commands": [
            {"type": "MoveToCommand", "end": [0, 0, 0]},
            {"type": "SetPowerCommand", "power": 500},
            {"type": "LineToCommand", "end": [10, 10, 0]},
        ],
        "last_move_to": [0, 0, 0],
    }
    geo = Geometry.from_dict(geo_dict)
    assert len(geo.commands) == 2
    assert isinstance(geo.commands[0], MoveToCommand)
    assert isinstance(geo.commands[1], LineToCommand)


def test_transform_identity():
    geo = Geometry()
    geo.move_to(10, 20, 30)
    geo.arc_to(50, 60, i=5, j=7, z=40)
    original_geo = deepcopy(geo)

    identity_matrix = np.identity(4, dtype=float)
    geo.transform(identity_matrix)

    arc_cmd = cast(ArcToCommand, geo.commands[1])
    orig_arc_cmd = cast(ArcToCommand, original_geo.commands[1])

    assert geo.commands[0].end == pytest.approx(original_geo.commands[0].end)
    assert arc_cmd.end == pytest.approx(orig_arc_cmd.end)
    assert arc_cmd.center_offset == pytest.approx(orig_arc_cmd.center_offset)
    assert geo.last_move_to == pytest.approx(original_geo.last_move_to)


def test_transform_translate():
    geo = Geometry()
    geo.move_to(10, 20, 30)
    geo.arc_to(50, 60, i=5, j=7, z=40)

    translate_matrix = _create_translate_matrix(10, -5, 15)
    geo.transform(translate_matrix)
    arc_cmd = cast(ArcToCommand, geo.commands[1])

    assert geo.commands[0].end == pytest.approx((20, 15, 45))
    assert arc_cmd.end == pytest.approx((60, 55, 55))
    assert arc_cmd.center_offset == pytest.approx((5, 7))
    assert geo.last_move_to == pytest.approx((20, 15, 45))


def test_transform_scale_non_uniform_linearizes_arc():
    geo = Geometry()
    geo.move_to(10, 20, 5)
    geo.arc_to(22, 22, i=5, j=7, z=-10)
    scale_matrix = _create_scale_matrix(2, 3, 4)
    geo.transform(scale_matrix)

    assert geo.commands[0].end == pytest.approx((20, 60, 20))
    # Arcs are linearized on non-uniform scale
    assert isinstance(geo.commands[1], LineToCommand)
    final_cmd = geo.commands[-1]
    assert final_cmd.end is not None
    final_point = final_cmd.end
    expected_final_point = (22 * 2, 22 * 3, -10 * 4)
    assert final_point == pytest.approx(expected_final_point)


def test_transform_rotate_preserves_z():
    geo = Geometry()
    geo.move_to(10, 10, -5)
    rotate_matrix = _create_z_rotate_matrix(math.radians(90))
    geo.transform(rotate_matrix)
    assert geo.commands[0].end is not None
    x, y, z = geo.commands[0].end
    assert z == -5
    assert x == pytest.approx(-10)
    assert y == pytest.approx(10)


def test_copy_method(sample_geometry):
    """Tests the deep copy functionality of the Geometry class."""
    original_geo = sample_geometry
    copied_geo = original_geo.copy()

    # Check for initial equality and deep copy semantics
    assert copied_geo is not original_geo
    assert copied_geo.commands is not original_geo.commands
    assert len(copied_geo.commands) == len(original_geo.commands)
    assert copied_geo.last_move_to == original_geo.last_move_to
    # Check a specific command's value to ensure it was copied
    original_line_to = cast(LineToCommand, original_geo.commands[1])
    copied_line_to = cast(LineToCommand, copied_geo.commands[1])
    assert copied_line_to.end == original_line_to.end

    # Modify the original and check that the copy is unaffected
    original_geo.line_to(100, 100)  # Adds a 4th command
    cast(MoveToCommand, original_geo.commands[0]).end = (99, 99, 99)

    # The copy should still have the original number of commands
    assert len(copied_geo.commands) == 3
    # The copy's first command should have its original value
    copied_move_to = cast(MoveToCommand, copied_geo.commands[0])
    assert copied_move_to.end == (0.0, 0.0, 0.0)


def test_distance(sample_geometry):
    """Tests the distance calculation for a Geometry object."""
    # move(0,0) -> line(10,10) -> arc(20,0)
    # Approximating arc as a line for distance calc
    dist1 = math.hypot(10 - 0, 10 - 0)
    dist2 = math.hypot(20 - 10, 0 - 10)
    expected_dist = dist1 + dist2

    assert sample_geometry.distance() == pytest.approx(expected_dist)
    # Also test the query function directly
    assert get_total_distance(sample_geometry.commands) == pytest.approx(
        expected_dist
    )


def test_geo_command_distance():
    last_point = (0.0, 0.0, 0.0)
    line_cmd = LineToCommand((3.0, 4.0, 0.0))
    assert line_cmd.distance(last_point) == pytest.approx(5.0)

    move_cmd = MoveToCommand((-3.0, -4.0, 0.0))
    assert move_cmd.distance(last_point) == pytest.approx(5.0)

    # Arc distance is approximated as a straight line
    arc_cmd = ArcToCommand(
        end=(3.0, 4.0, 0.0), center_offset=(0, 0), clockwise=False
    )
    assert arc_cmd.distance(last_point) == pytest.approx(5.0)


def test_area():
    # Test case 1: Empty and open paths
    assert Geometry().area() == 0.0
    assert Geometry.from_points([(0, 0), (10, 10)], close=False).area() == 0.0

    # Test case 2: Simple 10x10 CCW square
    square = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert square.area() == pytest.approx(100.0)

    # Test case 3: Simple 10x10 CW square (should have same positive area)
    square_cw = Geometry.from_points([(0, 0), (0, 10), (10, 10), (10, 0)])
    assert square_cw.area() == pytest.approx(100.0)

    # Test case 4: Shape with a hole
    # Outer CCW square (0,0) -> (10,10)
    geo_with_hole = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    # Inner CW square (hole) (2,2) -> (8,8)
    hole = Geometry.from_points([(2, 2), (2, 8), (8, 8), (8, 2)])
    geo_with_hole.commands.extend(hole.commands)
    # Expected area = 100 - (6*6) = 64
    assert geo_with_hole.area() == pytest.approx(64.0)

    # Test case 5: Two separate shapes
    geo_two_shapes = Geometry.from_points([(0, 0), (5, 0), (5, 5), (0, 5)])
    second_shape = Geometry.from_points(
        [(10, 10), (15, 10), (15, 15), (10, 15)]
    )
    geo_two_shapes.commands.extend(second_shape.commands)
    # Expected area = 25 + 25 = 50
    assert geo_two_shapes.area() == pytest.approx(50.0)


def test_split_into_components_empty_geometry():
    geo = Geometry()
    components = geo.split_into_components()
    assert len(components) == 0


def test_split_into_components_single_contour():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    components = geo.split_into_components()
    assert len(components) == 1
    assert len(components[0].commands) == 3


def test_split_into_components_two_separate_shapes():
    geo = Geometry()
    # Shape 1
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    geo.line_to(0, 10)
    geo.close_path()
    # Shape 2
    geo.move_to(20, 20)
    geo.line_to(30, 20)
    geo.line_to(30, 30)
    geo.line_to(20, 30)
    geo.close_path()

    components = geo.split_into_components()
    assert len(components) == 2
    assert len(components[0].commands) == 5
    assert len(components[1].commands) == 5


def test_split_into_components_containment_letter_o():
    geo = Geometry()
    # Outer circle (r=10, center=0,0)
    geo.move_to(10, 0)
    geo.arc_to(-10, 0, i=-10, j=0, clockwise=False)
    geo.arc_to(10, 0, i=10, j=0, clockwise=False)
    # Inner circle (r=5, center=0,0)
    geo.move_to(5, 0)
    geo.arc_to(-5, 0, i=-5, j=0, clockwise=False)
    geo.arc_to(5, 0, i=5, j=0, clockwise=False)

    components = geo.split_into_components()
    assert len(components) == 1
    assert len(components[0].commands) == 6  # 2 moves, 4 arcs


def test_split_into_contours_method(sample_geometry):
    """Tests the split_into_contours method on the Geometry class."""
    # sample_geometry has one MoveTo, so it's one contour.
    contours = sample_geometry.split_into_contours()
    assert len(contours) == 1
    assert len(contours[0].commands) == len(sample_geometry.commands)

    # Add another contour
    sample_geometry.move_to(100, 100)
    sample_geometry.line_to(110, 110)

    contours = sample_geometry.split_into_contours()
    assert len(contours) == 2
    assert len(contours[0].commands) == 3  # original M, L, A
    assert len(contours[1].commands) == 2  # new M, L

    # Check content of the split contours
    assert isinstance(contours[0].commands[0], MoveToCommand)
    assert contours[0].commands[0].end == (0, 0, 0)
    assert isinstance(contours[1].commands[0], MoveToCommand)
    assert contours[1].commands[0].end == (100, 100, 0)


def test_segments():
    """Tests the segments() method for extracting point lists."""
    # Test case 1: Empty geometry
    geo_empty = Geometry()
    assert geo_empty.segments() == []

    # Test case 2: Single open path
    geo_open = Geometry()
    geo_open.move_to(0, 0, 1)
    geo_open.line_to(10, 0, 2)
    geo_open.arc_to(10, 10, i=0, j=5, z=3)
    expected_open = [[(0, 0, 1), (10, 0, 2), (10, 10, 3)]]
    assert geo_open.segments() == expected_open

    # Test case 3: Single closed path
    geo_closed = Geometry.from_points([(0, 0), (10, 0), (0, 10)])
    expected_closed = [[(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 0)]]
    assert geo_closed.segments() == expected_closed

    # Test case 4: Multiple disjoint segments
    geo_multi = Geometry()
    # Segment 1
    geo_multi.move_to(0, 0)
    geo_multi.line_to(1, 1)
    # Segment 2
    geo_multi.move_to(10, 10)
    geo_multi.line_to(11, 11)
    geo_multi.line_to(12, 12)
    expected_multi = [
        [(0, 0, 0), (1, 1, 0)],
        [(10, 10, 0), (11, 11, 0), (12, 12, 0)],
    ]
    assert geo_multi.segments() == expected_multi

    # Test case 5: Path starting with a LineTo (implicit start at 0,0,0)
    geo_implicit_start = Geometry()
    geo_implicit_start.line_to(5, 5)
    geo_implicit_start.line_to(10, 0)
    expected_implicit = [[(0, 0, 0), (5, 5, 0), (10, 0, 0)]]
    assert geo_implicit_start.segments() == expected_implicit


def test_from_points():
    """Tests the Geometry.from_points classmethod."""
    # Test case 1: Empty list
    geo_empty = Geometry.from_points([])
    assert geo_empty.is_empty()

    # Test case 2: Single point
    geo_single = Geometry.from_points([(10, 20)])
    assert len(geo_single.commands) == 1
    assert isinstance(geo_single.commands[0], MoveToCommand)
    assert geo_single.commands[0].end == (10, 20, 0)
    assert geo_single.last_move_to == (10, 20, 0)
    # A single point doesn't get closed
    assert not any(
        isinstance(cmd, LineToCommand) for cmd in geo_single.commands
    )

    # Test case 3: Triangle (closed by default)
    points = [(0, 0), (10, 0), (5, 10)]
    geo_triangle = Geometry.from_points(points)
    assert len(geo_triangle.commands) == 4
    assert isinstance(geo_triangle.commands[0], MoveToCommand)
    assert geo_triangle.commands[0].end == (0, 0, 0)
    assert isinstance(geo_triangle.commands[3], LineToCommand)
    assert geo_triangle.commands[3].end == (0, 0, 0)  # from close_path

    # Test case 4: Triangle (open)
    geo_triangle_open = Geometry.from_points(points, close=False)
    assert len(geo_triangle_open.commands) == 3
    assert isinstance(geo_triangle_open.commands[0], MoveToCommand)
    assert geo_triangle_open.commands[0].end == (0, 0, 0)
    assert isinstance(geo_triangle_open.commands[2], LineToCommand)
    assert geo_triangle_open.commands[2].end == (5, 10, 0)  # last point
    # Final check: end point is not the start point
    assert (
        geo_triangle_open.commands[-1].end != geo_triangle_open.commands[0].end
    )

    # Test case 5: Points with Z coordinates (closed)
    points_3d = [(0, 0, 1), (10, 0, 2), (5, 10, 3)]
    geo_3d = Geometry.from_points(points_3d)
    assert len(geo_3d.commands) == 4
    assert geo_3d.commands[0].end == (0, 0, 1)
    assert geo_3d.commands[1].end == (10, 0, 2)
    assert geo_3d.commands[2].end == (5, 10, 3)
    assert geo_3d.commands[3].end == (0, 0, 1)  # from close_path
    assert geo_3d.last_move_to == (0, 0, 1)

    # Test case 6: Points with Z coordinates (open)
    geo_3d_open = Geometry.from_points(points_3d, close=False)
    assert len(geo_3d_open.commands) == 3
    assert geo_3d_open.commands[0].end == (0, 0, 1)
    assert geo_3d_open.commands[1].end == (10, 0, 2)
    assert geo_3d_open.commands[2].end == (5, 10, 3)
    assert geo_3d_open.last_move_to == (0, 0, 1)


def test_dump_and_load(sample_geometry):
    """
    Tests the dump() and load() methods for space-efficient serialization.
    """
    # Test with a non-empty geometry
    dumped_data = sample_geometry.dump()
    loaded_geo = Geometry.load(dumped_data)

    assert dumped_data["last_move_to"] == list(sample_geometry.last_move_to)
    assert len(dumped_data["commands"]) == 3
    # M 0 0 0
    assert dumped_data["commands"][0] == ["M", 0.0, 0.0, 0.0]
    # L 10 10 0
    assert dumped_data["commands"][1] == ["L", 10.0, 10.0, 0.0]
    # A 20 0 0 5 -10 1 (default clockwise is True)
    assert dumped_data["commands"][2] == ["A", 20.0, 0.0, 0.0, 5.0, -10.0, 1]

    assert loaded_geo.last_move_to == sample_geometry.last_move_to
    assert len(loaded_geo.commands) == len(sample_geometry.commands)
    for original_cmd, loaded_cmd in zip(
        sample_geometry.commands, loaded_geo.commands
    ):
        assert type(original_cmd) is type(loaded_cmd)
        # Easy way to check all attributes are the same
        assert original_cmd.to_dict() == loaded_cmd.to_dict()

    # Test with an empty geometry
    empty_geo = Geometry()
    dumped_empty = empty_geo.dump()
    loaded_empty = Geometry.load(dumped_empty)

    assert dumped_empty["last_move_to"] == [0.0, 0.0, 0.0]
    assert dumped_empty["commands"] == []
    assert loaded_empty.is_empty()
    assert loaded_empty.last_move_to == (0.0, 0.0, 0.0)


def test_close_gaps_on_empty_geometry():
    """Tests that close_gaps() doesn't fail on an empty Geometry."""
    geo = Geometry()
    geo.close_gaps()
    assert geo.is_empty()


def test_close_gaps_no_change_needed():
    """Tests that a clean geometry is not modified."""
    # Perfectly closed square
    geo_closed = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    original_cmds = geo_closed.copy().commands
    geo_closed.close_gaps()
    assert len(geo_closed.commands) == len(original_cmds)
    for cmd1, cmd2 in zip(geo_closed.commands, original_cmds):
        assert cmd1.to_dict() == cmd2.to_dict()

    # Open path with a large gap
    geo_open = Geometry()
    geo_open.move_to(0, 0)
    geo_open.line_to(10, 10)
    geo_open.move_to(50, 50)
    geo_open.line_to(60, 60)
    original_cmds_open = geo_open.copy().commands
    geo_open.close_gaps()
    assert len(geo_open.commands) == len(original_cmds_open)
    for cmd1, cmd2 in zip(geo_open.commands, original_cmds_open):
        assert cmd1.to_dict() == cmd2.to_dict()


def test_close_gaps_intra_contour():
    """Tests closing a small gap at the end of a single contour."""
    geo = Geometry()
    geo.move_to(0, 0, 5)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    geo.line_to(0, 10)
    geo.line_to(0.000001, -0.000002, 5.000001)  # Ends very close to (0,0,5)

    assert geo.commands[-1].end != geo.commands[0].end
    geo.close_gaps(tolerance=1e-5)
    # The final point should be snapped to the exact start point
    assert geo.commands[-1].end == geo.commands[0].end
    assert geo.commands[-1].end == (0, 0, 5)


def test_close_gaps_inter_contour():
    """Tests stitching two contours with a small jump between them."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10, 1)  # End of first contour
    geo.move_to(10.000001, 10.000002, 1.000003)  # Small jump
    geo.line_to(20, 20)

    assert isinstance(geo.commands[2], MoveToCommand)
    geo.close_gaps(tolerance=1e-5)
    # The MoveTo should be replaced by a LineTo
    assert isinstance(geo.commands[2], LineToCommand)
    # The new LineTo should connect to the exact previous end point
    assert geo.commands[2].end == (10, 10, 1)
    # The end point of the final LineTo should remain the same
    assert geo.commands[3].end == (20, 20, 0)
    # The total number of commands should not change
    assert len(geo.commands) == 4


def test_close_gaps_respects_tolerance():
    """Tests that the tolerance parameter is correctly used."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    geo.move_to(10.1, 10.1)  # A gap of sqrt(0.1^2 + 0.1^2) ~= 0.14
    geo.line_to(20, 20)

    # First, try with a tolerance that is too small
    geo_copy1 = geo.copy()
    geo_copy1.close_gaps(tolerance=0.1)
    # The MoveTo should NOT be replaced
    assert isinstance(geo_copy1.commands[2], MoveToCommand)
    assert geo_copy1.commands[2].end == (10.1, 10.1, 0)

    # Now, try with a tolerance that is large enough
    geo_copy2 = geo.copy()
    geo_copy2.close_gaps(tolerance=0.2)
    # The MoveTo SHOULD be replaced
    assert isinstance(geo_copy2.commands[2], LineToCommand)
    assert geo_copy2.commands[2].end == (10, 10, 0)
