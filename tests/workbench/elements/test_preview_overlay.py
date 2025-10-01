"""Tests for the preview overlay canvas element."""

import pytest
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand, SetPowerCommand, SetCutSpeedCommand
from rayforge.workbench.elements.preview_overlay import (
    PreviewOverlay,
    OpsTimeline,
    speed_to_heatmap_color,
)


def test_speed_to_heatmap_color_basic():
    """Test heatmap color generation."""
    # Test minimum speed (should be blue)
    r, g, b = speed_to_heatmap_color(0.0, 0.0, 100.0)
    assert b == 1.0  # Blue component should be max

    # Test maximum speed (should be red)
    r, g, b = speed_to_heatmap_color(100.0, 0.0, 100.0)
    assert r == 1.0  # Red component should be max


def test_ops_timeline_creation():
    """Test timeline creation from operations."""
    ops = Ops()
    ops.add(SetPowerCommand(50))
    ops.add(SetCutSpeedCommand(1000))
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 0.0, 0.0)))

    timeline = OpsTimeline(ops)

    assert timeline.get_step_count() == 2
    min_speed, max_speed = timeline.speed_range
    assert min_speed == 1000.0
    assert max_speed == 1000.0


def test_ops_timeline_speed_range():
    """Test speed range calculation from multiple speeds."""
    ops = Ops()
    ops.add(SetCutSpeedCommand(100))
    ops.add(LineToCommand((5.0, 0.0, 0.0)))
    ops.add(SetCutSpeedCommand(500))
    ops.add(LineToCommand((10.0, 0.0, 0.0)))

    timeline = OpsTimeline(ops)

    min_speed, max_speed = timeline.speed_range
    assert min_speed == 100.0
    assert max_speed == 500.0


def test_preview_overlay_initialization():
    """Test preview overlay initialization."""
    overlay = PreviewOverlay((100.0, 100.0))

    assert overlay.width == 100.0
    assert overlay.height == 100.0
    assert overlay.get_step_count() == 0
    assert overlay.selectable is False


def test_preview_overlay_set_ops():
    """Test setting operations on overlay."""
    overlay = PreviewOverlay((100.0, 100.0))

    ops = Ops()
    ops.add(SetPowerCommand(50))
    ops.add(SetCutSpeedCommand(1000))
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 0.0, 0.0)))

    overlay.set_ops(ops)

    assert overlay.get_step_count() == 2


def test_preview_overlay_set_step():
    """Test setting current playback step."""
    overlay = PreviewOverlay((100.0, 100.0))

    ops = Ops()
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 10.0, 0.0)))

    overlay.set_ops(ops)
    overlay.set_step(1)

    assert overlay.current_step == 1


def test_preview_overlay_empty_ops():
    """Test overlay with no operations."""
    overlay = PreviewOverlay((100.0, 100.0))
    overlay.set_ops(None)

    assert overlay.get_step_count() == 0
    assert overlay.current_step == 0
