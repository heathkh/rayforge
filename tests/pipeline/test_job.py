import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops, LineToCommand, MoveToCommand
from rayforge.core.import_source import ImportSource
from rayforge.machine.models.machine import Machine, Laser
from rayforge.pipeline.generator import OpsGenerator
from rayforge.shared.tasker.task import CancelledError
from rayforge.pipeline.job import generate_job_ops
from rayforge.pipeline.steps import create_contour_step
from rayforge.pipeline.transformer.multipass import MultiPassTransformer
from rayforge.image import SVG_RENDERER


@pytest.fixture
def machine():
    m = Machine()
    m.dimensions = (200, 150)
    m.y_axis_down = True
    m.add_head(Laser())
    return m


@pytest.fixture
def doc():
    d = Doc()
    # Start with a clean slate for tests by clearing the active (workpiece)
    # layer
    active_layer = d.active_layer
    assert active_layer.workflow is not None
    active_layer.workflow.set_steps([])
    return d


@pytest.fixture
def mock_ops_generator():
    """A mock OpsGenerator that we can configure for tests."""
    return MagicMock(spec=OpsGenerator)


@pytest.fixture
def real_workpiece():
    """Creates a realistic workpiece with a size and position."""
    # A workpiece must have a source to be rendered.
    # The data here informs the mock generator's scaling logic.
    wp = WorkPiece(name="wp1.svg")
    # Set properties in an order that is predictable and on-canvas.
    # 1. Set size first, which will adjust position to keep center at origin.
    wp.set_size(40, 30)
    # 2. Now explicitly set the position.
    wp.pos = (50, 60)
    # 3. Finally, set the angle.
    wp.angle = 90
    return wp


@pytest.mark.asyncio
async def test_generate_job_ops_assembles_correctly(
    doc, machine, mock_ops_generator, real_workpiece
):
    """
    Test that generate_job_ops correctly applies the workpiece's world
    transform matrix and then converts to machine coordinates.
    """
    # Arrange
    layer = doc.active_layer
    assert layer.workflow is not None
    with patch("rayforge.pipeline.steps.config", MagicMock()):
        # Create a contour step and configure it for "outline" behavior
        step = create_contour_step()
        step.opsproducer_dict["params"]["remove_inner_paths"] = True

    # The new way to specify passes is via a post-assembly transformer
    multi_pass_transformer = MultiPassTransformer(passes=2)
    step.post_step_transformers_dicts = [multi_pass_transformer.to_dict()]

    layer.workflow.add_step(step)

    # Link the workpiece to a source and add it to the document
    source = ImportSource(
        Path("wp1.svg"),
        b'<svg width="10" height="10" />',
        renderer=SVG_RENDERER,
    )
    doc.add_import_source(source)
    real_workpiece.import_source_uid = source.uid
    layer.add_workpiece(real_workpiece)

    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(10, 0)
    # The mock needs to return the *unscaled* ops. The generator's get_ops
    # method handles the scaling based on natural size from the source data.
    mock_ops_generator.get_ops.return_value = base_ops

    # Act
    final_ops = await generate_job_ops(doc, machine, mock_ops_generator)

    # Assert
    mock_ops_generator.get_ops.assert_called_once_with(step, real_workpiece)
    # The MultiPassTransformer with passes=2 should double the cutting
    # commands.
    assert len([c for c in final_ops.commands if c.is_cutting_command()]) == 2

    # Verify the coordinates of the transformed and y-flipped points.
    move_cmds = [c for c in final_ops.commands if isinstance(c, MoveToCommand)]
    line_cmds = [c for c in final_ops.commands if isinstance(c, LineToCommand)]

    # --- Trace the expected final coordinates ---
    # Point (0,0) from base_ops (in local mm, after scaling).
    # 1. World transf.: T(50,60) @ R(90,c=(20,15)) @ S(40,30) applied to (0,0)
    #    -> World coordinate (85, 55).
    # 2. Y-flip to machine coords: (85, 150-55) = (85, 95).
    assert move_cmds[0].end == pytest.approx((85.0, 95.0, 0.0))

    # Local point (10,0) with size (40,30) and pos (50,60), angle 90
    # Center is at (50+20, 60+15) = (70,75)
    # Point relative to center is (10-20, 0-15) = (-10, -15)
    # Rotated by 90 deg: (15, -10)
    # Translated back: (70+15, 75-10) = (85, 65)
    # 2. Y-flip to machine coords: (85, 150-65) = (85, 85).
    assert line_cmds[0].end == pytest.approx((85.0, 85.0, 0.0))

    # Verify the second pass is identical
    assert move_cmds[1].end == move_cmds[0].end
    assert line_cmds[1].end == line_cmds[0].end


@pytest.mark.asyncio
async def test_job_generation_cancellation(doc, machine, mock_ops_generator):
    """Test that job generation can be cancelled via the context."""
    # Arrange
    layer = doc.active_layer
    assert layer.workflow is not None
    with patch("rayforge.pipeline.steps.config", MagicMock()):
        step = create_contour_step()
        step.opsproducer_dict["params"]["remove_inner_paths"] = True
    layer.workflow.add_step(step)

    # Setup sources for the workpieces
    source1 = ImportSource(Path("wp1"), b"", renderer=SVG_RENDERER)
    source2 = ImportSource(Path("wp2"), b"", renderer=SVG_RENDERER)
    doc.add_import_source(source1)
    doc.add_import_source(source2)

    wp1 = WorkPiece("wp1")
    wp1.import_source_uid = source1.uid
    wp1.set_size(10, 10)
    wp1.pos = (0, 0)

    wp2 = WorkPiece("wp2")
    wp2.import_source_uid = source2.uid
    wp2.set_size(10, 10)
    wp2.pos = (20, 20)

    # There are two workpieces, so total_items = 2
    layer.add_workpiece(wp1)
    layer.add_workpiece(wp2)

    mock_context = MagicMock()
    # is_cancelled() is checked at the start of the loop for each item.
    # To cancel before item 2, the 2nd call must return True.
    mock_context.is_cancelled.side_effect = [False, True]

    # Act & Assert
    with pytest.raises(CancelledError):
        await generate_job_ops(
            doc, machine, mock_ops_generator, context=mock_context
        )

    # After the exception is caught, verify what happened before the
    # cancellation.
    mock_ops_generator.get_ops.assert_called_once_with(step, wp1)

    # Verify progress was set for the first item (processed_items = 1).
    # Progress is 1 / total_items (2) = 0.5
    mock_context.set_progress.assert_called_once_with(0.5)
