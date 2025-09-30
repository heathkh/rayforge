import cairo
import pytest
from rayforge.tools.material_test_generator import generate_material_test_ops
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand, SetPowerCommand, SetCutSpeedCommand, OpsSectionStartCommand, OpsSectionEndCommand

def test_generate_material_test_ops_basic():
    ops = generate_material_test_ops(
        test_type="Engrave",
        laser_type="Diode",
        speed_range=(100, 1000),
        power_range=(10, 100),
        grid_dimensions=(3, 3),
        shape_size=10,
        spacing=1,
        include_labels=False,
    )

    assert isinstance(ops, Ops)
    # Expect 1 start, 1 end, and 9 boxes (each 5 commands: Move, Line, Line, Line, Line)
    # Plus 9 SetPower and 9 SetCutSpeed commands
    # Total: 1 + 1 + 9 * (5 + 2) = 65 commands
    assert len(ops.commands) == 65

    # Check a few commands to ensure they are of the correct type
    assert isinstance(ops.commands[0], OpsSectionStartCommand)
    assert isinstance(ops.commands[1], SetPowerCommand)
    assert isinstance(ops.commands[2], SetCutSpeedCommand)
    assert isinstance(ops.commands[3], MoveToCommand)
    assert isinstance(ops.commands[-1], OpsSectionEndCommand)


def test_generate_material_test_ops_with_labels():
    ops = generate_material_test_ops(
        test_type="Engrave",
        laser_type="Diode",
        speed_range=(100, 1000),
        power_range=(10, 100),
        grid_dimensions=(2, 2),
        shape_size=10,
        spacing=1,
        include_labels=True,
    )

    assert isinstance(ops, Ops)
    # The exact number of commands for text is variable, so we'll check for a reasonable minimum.
    # Let's just check that the total number of commands is greater when labels are included.
    ops_no_labels = generate_material_test_ops(
        test_type="Engrave",
        laser_type="Diode",
        speed_range=(100, 1000),
        power_range=(10, 100),
        grid_dimensions=(2, 2),
        shape_size=10,
        spacing=1,
        include_labels=False,
    )
    assert len(ops.commands) > len(ops_no_labels.commands)


def test_generate_material_test_ops_order():
    ops = generate_material_test_ops(
        test_type="Engrave",
        laser_type="Diode",
        speed_range=(100, 200),
        power_range=(10, 20),
        grid_dimensions=(2, 2),
        shape_size=10,
        spacing=1,
        include_labels=False,
    )

    # Expected order of (speed, power) for a 2x2 grid (min_s=100, max_s=200, min_p=10, max_p=20)
    # Cells: (100,10), (200,10), (100,20), (200,20)
    # Sorted by (-speed, power):
    # (200,10) -> (-200,10)
    # (200,20) -> (-200,20)
    # (100,10) -> (-100,10)
    # (100,20) -> (-100,20)
    expected_order = [
        (200.0, 10.0),
        (200.0, 20.0),
        (100.0, 10.0),
        (100.0, 20.0),
    ]

    actual_order = []
    for i, cmd in enumerate(ops.commands):
        if isinstance(cmd, SetCutSpeedCommand):
            speed = cmd.speed
            # The next command should be SetPowerCommand
            if i > 0 and isinstance(ops.commands[i-1], SetPowerCommand):
                power = ops.commands[i-1].power
                actual_order.append((speed, power))
    
    # Filter out any power/speed commands that might be from labels if labels were included
    # For this test, labels are False, so all should be from boxes.
    # We expect 4 pairs of (speed, power) for the 4 boxes.
    assert len(actual_order) == 4
    assert actual_order == expected_order


def test_generate_material_test_ops_coordinates():
    ops = generate_material_test_ops(
        test_type="Engrave",
        laser_type="Diode",
        speed_range=(100, 100),
        power_range=(10, 10),
        grid_dimensions=(1, 1),
        shape_size=10,
        spacing=0,
        include_labels=False,
    )

    # For a 1x1 grid, shape_size=10, spacing=0, the box should be at (0,0) to (10,10)
    # Commands: Start, SetPower, SetCutSpeed, MoveTo(0,0), LineTo(10,0), LineTo(10,10), LineTo(0,10), LineTo(0,0), End
    assert isinstance(ops.commands[3], MoveToCommand)
    assert ops.commands[3].end[0] == 0.0
    assert ops.commands[3].end[1] == 0.0

    assert isinstance(ops.commands[4], LineToCommand)
    assert ops.commands[4].end[0] == 10.0
    assert ops.commands[4].end[1] == 0.0

    assert isinstance(ops.commands[5], LineToCommand)
    assert ops.commands[5].end[0] == 10.0
    assert ops.commands[5].end[1] == 10.0

    assert isinstance(ops.commands[6], LineToCommand)
    assert ops.commands[6].end[0] == 0.0
    assert ops.commands[6].end[1] == 10.0

    assert isinstance(ops.commands[7], LineToCommand)
    assert ops.commands[7].end[0] == 0.0
    assert ops.commands[7].end[1] == 0.0

from rayforge.tools.material_test_generator import _text_to_ops

def test_text_to_ops_state():
    ops = Ops()

    # First call
    _text_to_ops("A", 10, 10, 12, "Sans", ops, 100, 1000)
    count1 = len(ops.commands)
    assert count1 > 2
    first_move = ops.commands[2] # Skip SetPower and SetCutSpeed
    assert isinstance(first_move, MoveToCommand)
    assert first_move.end[0] >= 10

    # Second call
    _text_to_ops("B", 20, 20, 12, "Sans", ops, 100, 1000)
    count2 = len(ops.commands)
    assert count2 > count1
    second_move = ops.commands[count1 + 2] # Skip SetPower and SetCutSpeed
    assert isinstance(second_move, MoveToCommand)
    assert second_move.end[0] >= 20
