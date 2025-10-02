import pytest
import math
import io
import numpy as np
from contextlib import redirect_stdout
from typing import cast

# Import real geo types to fix the test
from rayforge.core import geo
from rayforge.core.geo.geometry import Geometry

from rayforge.core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    SetPowerCommand,
    SetCutSpeedCommand,
    SetTravelSpeedCommand,
    EnableAirAssistCommand,
    DisableAirAssistCommand,
    State,
    MovingCommand,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    ScanLinePowerCommand,
    JobStartCommand,
    JobEndCommand,
    LayerStartCommand,
    LayerEndCommand,
    WorkpieceStartCommand,
    WorkpieceEndCommand,
    SetLaserCommand,
)


@pytest.fixture
def empty_ops():
    return Ops()


@pytest.fixture
def sample_ops():
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 10)
    ops.set_power(500)
    ops.enable_air_assist()
    return ops


def test_initialization(empty_ops):
    assert len(empty_ops.commands) == 0
    assert empty_ops.last_move_to == (0.0, 0.0, 0.0)


def test_add_commands(empty_ops):
    empty_ops.move_to(5, 5)
    assert len(empty_ops.commands) == 1
    assert isinstance(empty_ops.commands[0], MoveToCommand)

    empty_ops.line_to(10, 10)
    assert isinstance(empty_ops.commands[1], LineToCommand)


def test_clear_commands(sample_ops):
    sample_ops.clear()
    assert len(sample_ops.commands) == 0


def test_ops_addition(sample_ops):
    ops2 = Ops()
    ops2.move_to(20, 20)
    combined = sample_ops + ops2
    assert len(combined) == len(sample_ops) + len(ops2)


def test_ops_multiplication(sample_ops):
    multiplied = sample_ops * 3
    assert len(multiplied) == 3 * len(sample_ops)


def test_ops_extend(sample_ops):
    # Create another Ops object to extend with
    ops2 = Ops()
    ops2.move_to(20, 20)
    ops2.set_cut_speed(1000)

    original_len = len(sample_ops)
    len_to_add = len(ops2)

    # Perform the extend operation
    sample_ops.extend(ops2)

    # Verify the length has increased correctly
    assert len(sample_ops) == original_len + len_to_add

    # Verify the last two commands are the ones from ops2
    assert sample_ops.commands[-2] is ops2.commands[0]
    assert sample_ops.commands[-1] is ops2.commands[1]


def test_ops_extend_with_empty(sample_ops):
    empty_ops = Ops()
    original_len = len(sample_ops)
    sample_ops.extend(empty_ops)
    assert len(sample_ops) == original_len


def test_ops_extend_with_none(sample_ops):
    original_len = len(sample_ops)
    # This test is just to ensure it doesn't raise an exception.
    # The type hint is `Ops`, so this would be a type error, but
    # robust code should handle it.
    sample_ops.extend(None)  # type: ignore
    assert len(sample_ops) == original_len


def test_copy():
    ops_original = Ops()
    ops_original.move_to(10, 10)
    ops_original.line_to(20, 20)
    ops_original.last_move_to = (10, 10, 0)

    ops_copy = ops_original.copy()

    # Check for deepcopy: objects should not be the same instance
    assert ops_original is not ops_copy
    assert ops_original.commands is not ops_copy.commands
    assert ops_original.commands[0] is not ops_copy.commands[0]
    assert ops_original.last_move_to == ops_copy.last_move_to

    # Modify the copy and check that original is unchanged
    ops_copy.translate(5, 5)
    ops_copy.add(SetPowerCommand(100))

    assert len(ops_original.commands) == 2
    assert ops_original.commands[0].end == (10, 10, 0)
    assert ops_original.commands[1].end == (20, 20, 0)

    assert len(ops_copy.commands) == 3
    assert ops_copy.commands[0].end == (15, 15, 0)
    assert ops_copy.commands[1].end == (25, 25, 0)


def test_preload_state(sample_ops):
    sample_ops.preload_state()

    # Verify that non-state commands have their state attribute set
    for cmd in sample_ops.commands:
        if not cmd.is_state_command():
            assert cmd.state is not None
            assert isinstance(cmd.state, State)

    # Verify that state commands are still present in the commands list
    state_commands = [c for c in sample_ops.commands if c.is_state_command()]
    assert len(state_commands) > 0


def test_move_to(sample_ops):
    sample_ops.move_to(15, 15)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, MoveToCommand)
    assert last_cmd.end == (15.0, 15.0, 0.0)


def test_line_to(sample_ops):
    sample_ops.line_to(20, 20)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == (20.0, 20.0, 0.0)


def test_close_path(sample_ops):
    sample_ops.move_to(5, 5, -1.0)
    sample_ops.close_path()
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == sample_ops.last_move_to
    assert last_cmd.end == (5.0, 5.0, -1.0)


def test_arc_to(sample_ops):
    sample_ops.arc_to(5, 5, 2, 3, clockwise=False)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, ArcToCommand)
    assert last_cmd.end == (5.0, 5.0, 0.0)
    assert last_cmd.clockwise is False


def test_bezier_to():
    ops = Ops()
    ops.move_to(0.0, 0.0, 10.0)
    ops.bezier_to(
        c1=(1.0, 1.0, 10.0),
        c2=(2.0, 1.0, 20.0),
        end=(3.0, 0.0, 20.0),
        num_steps=2,
    )

    assert len(ops.commands) == 3  # move_to + 2 line_to
    assert isinstance(ops.commands[0], MoveToCommand)
    assert isinstance(ops.commands[1], LineToCommand)
    assert isinstance(ops.commands[2], LineToCommand)

    # Check interpolated points
    # t=0.5
    # B(0.5) = 0.125*p0 + 0.375*c1 + 0.375*c2 + 0.125*p1
    p0 = (0.0, 0.0, 10.0)
    c1 = (1.0, 1.0, 10.0)
    c2 = (2.0, 1.0, 20.0)
    p1 = (3.0, 0.0, 20.0)
    expected_x = 0.125 * p0[0] + 0.375 * c1[0] + 0.375 * c2[0] + 0.125 * p1[0]
    expected_y = 0.125 * p0[1] + 0.375 * c1[1] + 0.375 * c2[1] + 0.125 * p1[1]
    expected_z = 0.125 * p0[2] + 0.375 * c1[2] + 0.375 * c2[2] + 0.125 * p1[2]
    # This is the endpoint of the first segment (t=0.5).
    assert ops.commands[1].end == pytest.approx(
        (expected_x, expected_y, expected_z)
    )

    # Check final point (t=1.0)
    assert ops.commands[2].end == pytest.approx(p1)


def test_bezier_to_no_start_point():
    ops = Ops()
    # Should not raise an error, just do nothing and log a warning.
    ops.bezier_to((1, 1, 1), (2, 2, 2), (3, 3, 3))
    assert ops.is_empty()


def test_set_power(sample_ops):
    sample_ops.set_power(800)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, SetPowerCommand)
    assert last_cmd.power == 800.0


def test_set_cut_speed(sample_ops):
    sample_ops.set_cut_speed(300)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, SetCutSpeedCommand)
    assert last_cmd.speed == 300.0


def test_set_travel_speed(sample_ops):
    sample_ops.set_travel_speed(2000)
    last_cmd = sample_ops.commands[-1]
    assert isinstance(last_cmd, SetTravelSpeedCommand)
    assert last_cmd.speed == 2000.0


def test_set_laser():
    ops = Ops()
    ops.set_laser("laser-abc")
    last_cmd = ops.commands[-1]
    assert isinstance(last_cmd, SetLaserCommand)
    assert last_cmd.laser_uid == "laser-abc"


def test_enable_disable_air_assist(empty_ops):
    empty_ops.enable_air_assist()
    assert isinstance(empty_ops.commands[-1], EnableAirAssistCommand)

    empty_ops.disable_air_assist()
    assert isinstance(empty_ops.commands[-1], DisableAirAssistCommand)


def test_rect_default_ignores_travel():
    """Tests that Ops.rect() ignores travel moves by default."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 10)
    ops.move_to(100, 100)  # This move should be ignored
    min_x, min_y, max_x, max_y = ops.rect()
    assert (min_x, min_y, max_x, max_y) == (0.0, 0.0, 10.0, 10.0)


def test_rect_includes_travel():
    """Tests that Ops.rect(include_travel=True) includes travel moves."""
    ops = Ops()
    ops.move_to(-20, -20)
    ops.line_to(10, 10)
    ops.move_to(100, 100)  # This move should be included
    min_x, min_y, max_x, max_y = ops.rect(include_travel=True)
    # Points considered: (-20,-20), (10,10) from first segment,
    # and (10,10), (100,100) from second
    assert (min_x, min_y, max_x, max_y) == (-20.0, -20.0, 100.0, 100.0)


def test_get_frame(sample_ops):
    frame = sample_ops.get_frame(power=1000, speed=500)
    assert sum(1 for c in frame if c.is_travel_command()) == 1  # move_to
    assert sum(1 for c in frame if c.is_cutting_command()) == 4  # line_to

    min_x, min_y, max_x, max_y = sample_ops.rect()

    expected_points = [
        (min_x, min_y, 0.0),
        (min_x, max_y, 0.0),
        (max_x, max_y, 0.0),
        (max_x, min_y, 0.0),
        (min_x, min_y, 0.0),
    ]

    frame_points = [
        cmd.end for cmd in frame.commands if isinstance(cmd, MovingCommand)
    ]
    assert frame_points == expected_points


def test_get_frame_empty(empty_ops):
    frame = empty_ops.get_frame()
    assert len(frame.commands) == 0


def test_distance(sample_ops):
    sample_ops.move_to(20, 20, -5)  # Travel with Z change
    distance = sample_ops.distance()
    # Distance should be 2D
    expected = math.dist((0, 0), (10, 10)) + math.dist((10, 10), (20, 20))
    assert distance == pytest.approx(expected)


def test_cut_distance(sample_ops):
    # Add a travel move to ensure it's not counted
    sample_ops.move_to(100, 100)
    cut_dist = sample_ops.cut_distance()
    # Only the initial line_to(10, 10) from (0,0) should be counted
    expected = math.hypot(10, 10)
    assert cut_dist == pytest.approx(expected)


def test_segments(sample_ops):
    sample_ops.move_to(5, 5)  # Travel command
    segments = list(sample_ops.segments())
    assert len(segments) > 0
    # First segment should end before the travel command
    assert isinstance(segments[0][-1], LineToCommand)


def test_preload_state_application():
    ops = Ops()
    ops.set_power(300)
    ops.line_to(5, 5)
    ops.set_cut_speed(200)
    ops.preload_state()

    line_cmd = ops.commands[1]
    assert line_cmd.state is not None
    assert line_cmd.state.power == 300

    state_commands = [cmd for cmd in ops.commands if cmd.is_state_command()]
    assert len(state_commands) == 2

    for cmd in ops.commands:
        if not cmd.is_state_command():
            assert cmd.state is not None
            assert isinstance(cmd.state, State)


def test_translate_3d():
    ops = Ops()
    ops.move_to(10, 20, 30)
    ops.line_to(30, 40, 50)
    ops.translate(5, 10, -20)
    assert ops.commands[0].end is not None
    assert ops.commands[0].end == pytest.approx((15, 30, 10))
    assert ops.commands[1].end is not None
    assert ops.commands[1].end == pytest.approx((35, 50, 30))
    assert ops.last_move_to == pytest.approx((15, 30, 10))


def test_scale_3d():
    ops = Ops()
    ops.move_to(10, 20, 5)
    ops.arc_to(22, 22, 5, 7, z=-10)
    ops.scale(2, 3, 4)  # Non-uniform scale

    assert ops.commands[0].end is not None
    assert ops.commands[0].end == pytest.approx((20, 60, 20))
    assert isinstance(ops.commands[1], LineToCommand)
    final_cmd = ops.commands[-1]
    assert final_cmd.end is not None
    final_point = final_cmd.end
    expected_final_point = (22 * 2, 22 * 3, -10 * 4)
    assert final_point == pytest.approx(expected_final_point)
    assert ops.last_move_to == pytest.approx((20, 60, 20))


def test_rotate_preserves_z():
    ops = Ops()
    ops.move_to(10, 10, -5)
    ops.rotate(90, 0, 0)
    assert ops.commands[0].end is not None
    x, y, z = ops.commands[0].end
    assert z == -5
    assert x == pytest.approx(-10)
    assert y == pytest.approx(10)


def test_transform_uniform():
    """Tests applying a uniform transformation (rotation + translation)."""
    ops = Ops()
    ops.move_to(10, 0)
    ops.arc_to(0, 10, i=-10, j=0, clockwise=False)  # 90 deg arc

    # Rotate 90 degrees around origin and translate by (100, 0)
    angle_rad = math.radians(90)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    matrix = np.array(
        [
            [cos_a, -sin_a, 0, 100],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    ops.transform(matrix)

    move_cmd = ops.commands[0]
    arc_cmd = ops.commands[1]

    # Original (10,0) -> Rotated (0,10) -> Translated (100, 10)
    assert move_cmd.end == pytest.approx((100, 10, 0))
    # Original (0,10) -> Rotated (-10,0) -> Translated (90, 0)
    assert arc_cmd.end == pytest.approx((90, 0, 0))

    # Arc should NOT be linearized
    assert isinstance(arc_cmd, ArcToCommand)

    # Original offset (-10, 0) should be rotated to (0, -10)
    assert arc_cmd.center_offset[0] == pytest.approx(0)
    assert arc_cmd.center_offset[1] == pytest.approx(-10)


def test_transform_non_uniform():
    """Tests that a non-uniform scale linearizes arcs."""
    ops = Ops()
    ops.move_to(10, 0)
    ops.arc_to(0, 10, i=-10, j=0, clockwise=False)  # 90 deg arc

    # Scale X by 2, Y by 3
    matrix = np.diag([2.0, 3.0, 1.0, 1.0])
    ops.transform(matrix)

    # Original move_to (10,0) -> (20, 0)
    assert ops.commands[0].end == pytest.approx((20, 0, 0))

    # Arc must be linearized into LineToCommands
    assert all(isinstance(c, LineToCommand) for c in ops.commands[1:])
    assert len(ops.commands) > 2

    # Final point should be original arc end (0,10) scaled -> (0, 30)
    assert ops.commands[-1].end == pytest.approx((0, 30, 0))


def test_linearize_all():
    ops = Ops()
    ops.move_to(10, 0)
    ops.line_to(20, 0)
    ops.arc_to(10, 10, i=-10, j=0, clockwise=False)  # Semicircle
    ops.set_power(100)  # Should be preserved

    ops.linearize_all()

    assert len(ops.commands) > 4  # Move, Line, SetPower, plus linearized arc
    assert isinstance(ops.commands[0], MoveToCommand)
    assert isinstance(ops.commands[1], LineToCommand)
    # All geometric commands after the first two should be LineTo
    moving_cmds_after = [
        c for c in ops.commands[2:] if isinstance(c, MovingCommand)
    ]
    assert all(isinstance(c, LineToCommand) for c in moving_cmds_after)
    # Check that state command is still there
    assert any(isinstance(c, SetPowerCommand) for c in ops.commands)


@pytest.fixture
def clip_rect():
    return (0.0, 0.0, 100.0, 100.0)


def test_clip_fully_inside(clip_rect):
    ops = Ops()
    ops.move_to(10, 10, -1)
    ops.line_to(90, 90, -1)
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 2
    assert isinstance(clipped_ops.commands[0], MoveToCommand)
    assert isinstance(clipped_ops.commands[1], LineToCommand)
    assert clipped_ops.commands[1].end is not None
    assert clipped_ops.commands[1].end == pytest.approx((90.0, 90.0, -1.0))


def test_clip_fully_outside(clip_rect):
    ops = Ops()
    ops.move_to(110, 110, 0)
    ops.line_to(120, 120, 0)
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 0


def test_clip_with_arc():
    """Verify clip works on arcs via the new generic linearize interface."""
    ops = Ops()
    ops.move_to(0, 50)
    ops.arc_to(100, 50, i=50, j=0, clockwise=False)  # Semicircle
    clip_rect = (40.0, 0.0, 60.0, 100.0)  # A vertical slice through the middle
    clipped_ops = ops.clip(clip_rect)

    # Check that there are drawing commands left
    drawing_cmds = [c for c in clipped_ops if c.is_cutting_command()]
    assert len(drawing_cmds) > 0

    # Check that all remaining points are within the rect bounds
    for cmd in clipped_ops:
        if isinstance(cmd, MovingCommand) and cmd.end is not None:
            x, y, z = cmd.end
            assert clip_rect[0] <= x <= clip_rect[2]
            assert clip_rect[1] <= y <= clip_rect[3]


def test_clip_scanlinepowercommand_start_outside():
    """Tests clipping a scanline that starts outside and ends inside."""
    ops = Ops()
    ops.move_to(0, 50, 10)
    ops.add(
        ScanLinePowerCommand(
            end=(100, 50, 10),
            power_values=bytearray(range(100)),
        )
    )
    clip_rect = (50, 0, 150, 100)
    clipped_ops = ops.clip(clip_rect)

    assert len(clipped_ops.commands) == 2  # MoveTo, ScanLinePowerCommand
    assert isinstance(clipped_ops.commands[0], MoveToCommand)
    clipped_cmd = cast(ScanLinePowerCommand, clipped_ops.commands[1])

    # 1. Verify it's still a ScanLinePowerCommand (not linearized)
    assert isinstance(clipped_cmd, ScanLinePowerCommand)

    # 2. Verify new geometry (starts at the clip boundary)
    assert clipped_ops.commands[0].end == pytest.approx((50, 50, 10))
    assert clipped_cmd.end == pytest.approx((100, 50, 10))

    # 3. Verify power values are sliced correctly (original was 100 values)
    # The clip starts 50% of the way through the line.
    assert len(clipped_cmd.power_values) == 50
    assert clipped_cmd.power_values[0] == 50
    assert clipped_cmd.power_values[-1] == 99


def test_clip_scanlinepowercommand_crossing_with_z_interp():
    """Tests a scanline that crosses the clip rect with Z interpolation."""
    ops = Ops()
    # Line from (-50, 50, 0) to (150, 50, 200) -> total length 200
    ops.move_to(-50, 50, 0)
    ops.add(
        ScanLinePowerCommand(
            end=(150, 50, 200),
            power_values=bytearray(range(200)),
        )
    )
    clip_rect = (0, 0, 100, 100)
    clipped_ops = ops.clip(clip_rect)

    assert len(clipped_ops.commands) == 2
    clipped_cmd = cast(ScanLinePowerCommand, clipped_ops.commands[1])
    assert isinstance(clipped_cmd, ScanLinePowerCommand)

    # The line starts 50 units before x=0 and ends 50 units after x=100.
    # The clipped portion is from x=0 to x=100.
    # t_start = 50 / 200 = 0.25. t_end = 150 / 200 = 0.75
    expected_z_start = 0 + (0.25 * 200)  # 50
    expected_z_end = 0 + (0.75 * 200)  # 150

    assert clipped_ops.commands[0].end == pytest.approx(
        (0, 50, expected_z_start)
    )
    assert clipped_cmd.end == pytest.approx((100, 50, expected_z_end))

    # Power values should be sliced from index 50 to 150.
    expected_len = int(200 * 0.75) - int(200 * 0.25)
    assert len(clipped_cmd.power_values) == expected_len
    assert clipped_cmd.power_values[0] == 50
    assert clipped_cmd.power_values[-1] == 149


def test_clip_scanlinepowercommand_fully_outside():
    """Tests that a fully outside scanline is removed."""
    ops = Ops()
    ops.move_to(200, 50, 10)
    ops.add(
        ScanLinePowerCommand(
            end=(300, 50, 10),
            power_values=bytearray(range(100)),
        )
    )
    clip_rect = (0, 0, 100, 100)
    clipped_ops = ops.clip(clip_rect)
    assert len(clipped_ops.commands) == 0


def test_subtract_regions():
    ops = Ops()
    ops.move_to(0, 50, -5)
    ops.line_to(100, 50, 5)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]
    ops.subtract_regions([region])
    assert len(ops.commands) == 4
    assert ops.commands[1].end == pytest.approx((40.0, 50.0, -1.0))
    assert ops.commands[3].end == pytest.approx((100.0, 50.0, 5.0))


def test_subtract_regions_with_scanline():
    ops = Ops()
    ops.move_to(0, 50, 0)
    ops.add(
        ScanLinePowerCommand(
            end=(100, 50, 0), power_values=bytearray([100] * 100)
        )
    )
    # Region to cut out of the middle
    region = [(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)]
    ops.subtract_regions([region])

    # Expected: M(0,50), S(->40,50), M(60,50), S(->100,50)
    assert len(ops.commands) == 4
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == pytest.approx((0, 50, 0))

    scan1 = cast(ScanLinePowerCommand, ops.commands[1])
    assert isinstance(scan1, ScanLinePowerCommand)
    assert scan1.end == pytest.approx((40, 50, 0))
    assert len(scan1.power_values) == 40

    assert isinstance(ops.commands[2], MoveToCommand)
    assert ops.commands[2].end == pytest.approx((60, 50, 0))

    scan2 = cast(ScanLinePowerCommand, ops.commands[3])
    assert isinstance(scan2, ScanLinePowerCommand)
    assert scan2.end == pytest.approx((100, 50, 0))
    assert len(scan2.power_values) == 40


def test_from_geometry():
    # Use the actual Geometry class instead of mocks to ensure correct types
    geo_obj = Geometry()
    geo_obj.add(geo.MoveToCommand((10, 10, 0)))
    geo_obj.add(geo.LineToCommand((20, 20, 0)))
    geo_obj.add(geo.ArcToCommand((30, 10, 0), (-10, 0), False))
    # last_move_to is updated automatically by geo.add()

    ops = Ops.from_geometry(geo_obj)

    assert len(ops.commands) == 3
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == (10, 10, 0)
    assert isinstance(ops.commands[1], LineToCommand)
    assert ops.commands[1].end == (20, 20, 0)
    assert isinstance(ops.commands[2], ArcToCommand)
    assert ops.commands[2].end == (30, 10, 0)
    assert ops.commands[2].center_offset == (-10, 0)
    assert ops.commands[2].clockwise is False
    assert ops.last_move_to == geo_obj.last_move_to


def test_serialization_deserialization_all_types():
    """Tests that all command types can be serialized and deserialized."""
    ops = Ops()
    ops.add(JobStartCommand())
    ops.add(LayerStartCommand("layer-1"))
    ops.add(WorkpieceStartCommand("wp-1"))
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp-1"))
    ops.add(SetTravelSpeedCommand(5000))
    ops.add(SetCutSpeedCommand(1000))
    ops.add(SetPowerCommand(200))
    ops.add(EnableAirAssistCommand())
    ops.add(SetLaserCommand("laser-2"))
    ops.add(MoveToCommand((1, 1, 1)))
    ops.add(LineToCommand((2, 2, 2)))
    ops.add(ArcToCommand((3, 1, 2), (1, 1), False))
    ops.add(ScanLinePowerCommand((12, 2, 2), bytearray([50, 150])))
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
    ops.add(WorkpieceEndCommand("wp-1"))
    ops.add(LayerEndCommand("layer-1"))
    ops.add(JobEndCommand())
    ops.last_move_to = (1, 1, 1)

    data = ops.to_dict()
    new_ops = Ops.from_dict(data)

    assert len(ops.commands) == len(new_ops.commands)
    assert new_ops.last_move_to == (1, 1, 1)

    for old_cmd, new_cmd in zip(ops.commands, new_ops.commands):
        # Check type and dict representation to ensure all fields are preserved
        assert type(old_cmd) is type(new_cmd)
        assert old_cmd.to_dict() == new_cmd.to_dict()


def test_translate_with_scanline():
    """Tests that translate() correctly transforms ScanLinePowerCommand."""
    ops = Ops()
    ops.move_to(10, 20, 30)
    ops.add(
        ScanLinePowerCommand(
            end=(40, 50, 60),
            power_values=bytearray([1, 2, 3]),
        )
    )
    ops.translate(5, -10, 15)

    move_cmd = cast(MoveToCommand, ops.commands[0])
    translated_cmd = cast(ScanLinePowerCommand, ops.commands[1])
    assert isinstance(translated_cmd, ScanLinePowerCommand)

    # Check if both start_point (from move_to) and end are translated
    assert move_cmd.end == pytest.approx((15, 10, 45))
    assert translated_cmd.end == pytest.approx((45, 40, 75))


def test_clip_at_no_hit():
    """Tests that clip_at does nothing if no point is found."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 10)
    original_commands = ops.commands[:]
    # Point is far away from the path
    assert ops.clip_at(100, 100, 1.0) is False
    assert ops.commands == original_commands


def test_clip_at_on_line_segment():
    """Tests creating a gap in a simple line segment."""
    ops = Ops()
    ops.move_to(0, 50, 10)
    ops.line_to(100, 50, 20)  # Z should be interpolated

    # Clip near the midpoint
    assert ops.clip_at(50, 50, 10.0) is True

    # Expected:
    # Move(0,50,10), Line(45,50,14.5), Move(55,50,15.5), Line(100,50,20)
    assert len(ops.commands) == 4
    assert isinstance(ops.commands[0], MoveToCommand)
    assert isinstance(ops.commands[1], LineToCommand)
    assert isinstance(ops.commands[2], MoveToCommand)
    assert isinstance(ops.commands[3], LineToCommand)

    # Check the points
    assert ops.commands[1].end == pytest.approx((45.0, 50.0, 14.5))
    assert ops.commands[2].end == pytest.approx((55.0, 50.0, 15.5))
    assert ops.commands[3].end == pytest.approx((100.0, 50.0, 20.0))


def test_clip_at_on_arc_segment():
    """Tests creating a gap in an arc segment."""
    ops = Ops()
    ops.move_to(10, 0)
    # 90 deg CCW arc, radius 10, center (0,0)
    ops.arc_to(0, 10, i=-10, j=0, clockwise=False)

    # Clip near the 45-degree point on the arc
    point_on_arc_x = 10 * math.cos(math.radians(45))
    point_on_arc_y = 10 * math.sin(math.radians(45))
    assert ops.clip_at(point_on_arc_x, point_on_arc_y, 2.0) is True

    # The arc gets linearized by subtract_regions, so we expect a series
    # of LineTo commands with a gap in the middle.
    assert len(ops.commands) > 3
    # Verify there is a MoveTo command somewhere in the middle,
    # indicating a gap
    assert any(isinstance(cmd, MoveToCommand) for cmd in ops.commands[1:]), (
        "No MoveToCommand found, indicating no gap was created."
    )


def test_clip_at_start_of_subpath():
    """Tests clipping at the very beginning of a subpath."""
    ops = Ops()
    ops.move_to(0, 50)
    ops.line_to(100, 50)

    # Clip at x=1, width=2. Should clip from 0 to 2.
    assert ops.clip_at(1, 50, 2.0) is True

    # Expected: Move(0,50), Move(2,50), Line(100,50)
    assert len(ops.commands) == 3
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == pytest.approx((0, 50, 0))
    assert isinstance(ops.commands[1], MoveToCommand)
    assert ops.commands[1].end == pytest.approx((2.0, 50.0, 0.0))
    assert ops.commands[2].end == pytest.approx((100.0, 50.0, 0.0))


def test_clip_at_end_of_subpath():
    """Tests clipping at the very end of a subpath."""
    ops = Ops()
    ops.move_to(0, 50)
    ops.line_to(100, 50)

    # Clip at x=99, width=2. Should clip from 98 to 100.
    assert ops.clip_at(99, 50, 2.0) is True

    # Expected: Move(0,50), Line(98,50), Move(100,50)
    assert len(ops.commands) == 3
    assert isinstance(ops.commands[0], MoveToCommand)
    assert ops.commands[0].end == pytest.approx((0, 50, 0))
    assert isinstance(ops.commands[1], LineToCommand)
    assert ops.commands[1].end == pytest.approx((98.0, 50.0, 0.0))
    assert isinstance(ops.commands[2], MoveToCommand)
    assert ops.commands[2].end == pytest.approx((100.0, 50.0, 0.0))


def test_clip_at_spans_multiple_segments():
    """
    Tests that a clip correctly creates a gap across the boundary of two
    connected LineTo commands.
    """
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(50, 0)  # Segment 1 (index 1)
    ops.line_to(100, 50)  # Segment 2 (index 2)
    ops.line_to(100, 100)  # Segment 3 (index 3)

    # Clip at (50, 0) with a width of 20.
    # This should remove from x=40 on the first line to some point on the
    # second line.
    assert ops.clip_at(50, 0, 20.0) is True

    # Original: M, L, L, L -> 4 commands
    # Expected: M, L(shortened), M(to skip gap), L(shortened), L -> 5+ commands
    assert len(ops.commands) > 4
    assert isinstance(ops.commands[0], MoveToCommand)
    assert isinstance(ops.commands[1], LineToCommand)
    assert isinstance(ops.commands[2], MoveToCommand)

    # The first line segment should end before 50
    assert ops.commands[1].end[0] < 50
    # The new path should start after 50
    assert ops.commands[2].end[0] > 50

    # Ensure the entire original path after the clip is still present
    assert ops.commands[-1].end == pytest.approx((100, 100, 0))


def test_clip_at_ignores_state_commands():
    """
    Tests that clip_at correctly handles state commands, ensuring they are
    not part of the geometric subpath.
    """
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(50, 0)  # Subpath 1
    ops.set_power(100)
    ops.move_to(60, 0)
    ops.line_to(100, 0)  # Subpath 2

    # Clip the second line segment
    assert ops.clip_at(80, 0, 10.0) is True

    # Path 1 should be unchanged. Path 2 should be clipped.
    assert len(ops.commands) == 7
    # Path 1
    assert ops.commands[0].end == (0, 0, 0)
    assert ops.commands[1].end == (50, 0, 0)
    assert isinstance(ops.commands[2], SetPowerCommand)
    # Path 2 (clipped)
    assert ops.commands[3].end == (60, 0, 0)
    assert ops.commands[4].end == pytest.approx((75, 0, 0))
    assert ops.commands[5].end == pytest.approx((85, 0, 0))
    assert ops.commands[6].end == pytest.approx((100, 0, 0))


def test_dump(sample_ops):
    """Ensures dump() runs without error and produces output."""
    f = io.StringIO()
    with redirect_stdout(f):
        sample_ops.dump()
    output = f.getvalue()
    assert "MoveToCommand" in output
    assert "LineToCommand" in output
