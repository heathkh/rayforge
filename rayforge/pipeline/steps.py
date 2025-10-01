from __future__ import annotations
from typing import Optional, List, Callable, Tuple
from .. import config
from ..core.step import Step
from .modifier import MakeTransparent, ToGrayscale
from .producer import (
    OutlineTracer,
    EdgeTracer,
    Rasterizer,
    DepthEngraver,
    ShrinkWrapProducer,
    FrameProducer,
    MaterialTestGridProducer,
    MaterialTestGridType,
)
from .transformer import (
    Optimize,
    Smooth,
    MultiPassTransformer,
    TabOpsTransformer,
)


def create_outline_step(name: Optional[str] = None) -> Step:
    """Factory to create and configure an Outline step."""
    assert config.config.machine
    step = Step(
        typelabel=_("External Outline"),
        name=name,
    )
    step.opsproducer_dict = OutlineTracer().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.opstransformers_dicts = [
        Smooth(enabled=False, amount=20).to_dict(),
        TabOpsTransformer().to_dict(),
        Optimize(enabled=True).to_dict(),
    ]
    step.post_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.laser_dict = config.config.machine.heads[0].to_dict()
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_contour_step(
    name: Optional[str] = None, optimize: bool = True
) -> Step:
    """Factory to create and configure a Contour step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Contour"),
        name=name,
    )
    step.opsproducer_dict = EdgeTracer().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.opstransformers_dicts = [
        Smooth(enabled=False, amount=20).to_dict(),
        TabOpsTransformer().to_dict(),
    ]
    if optimize:
        step.opstransformers_dicts.append(
            Optimize(enabled=True).to_dict(),
        )
    step.post_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.laser_dict = config.config.machine.heads[0].to_dict()
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_raster_step(name: Optional[str] = None) -> Step:
    """Factory to create and configure a Rasterize step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Raster Engrave"),
        name=name,
    )
    step.opsproducer_dict = Rasterizer().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.opstransformers_dicts = [
        Optimize(enabled=True).to_dict(),
    ]
    step.post_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.laser_dict = config.config.machine.heads[0].to_dict()
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_depth_engrave_step(name: Optional[str] = None) -> Step:
    """Factory to create and configure a Depth Engrave step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Depth Engrave"),
        name=name,
    )
    step.opsproducer_dict = DepthEngraver().to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.opstransformers_dicts = [Optimize(enabled=False).to_dict()]
    step.post_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict()
    ]
    step.laser_dict = config.config.machine.heads[0].to_dict()
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_shrinkwrap_step(name: Optional[str] = None) -> Step:
    """Factory to create and configure a Shrinkwrap (concave hull) step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Shrink Wrap"),
        name=name,
    )
    # Use the HullProducer with a default gravity to create the effect
    step.opsproducer_dict = ShrinkWrapProducer(gravity=0.0).to_dict()
    step.modifiers_dicts = [
        MakeTransparent().to_dict(),
        ToGrayscale().to_dict(),
    ]
    step.opstransformers_dicts = [
        Smooth(enabled=False, amount=20).to_dict(),
        TabOpsTransformer().to_dict(),
        Optimize(enabled=True).to_dict(),
    ]
    step.post_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.laser_dict = config.config.machine.heads[0].to_dict()
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_frame_step(name: Optional[str] = None) -> Step:
    """Factory to create and configure a Frame step."""
    assert config.config.machine
    step = Step(
        typelabel=_("Frame Outline"),
        name=name,
    )
    step.opsproducer_dict = FrameProducer(offset=1.0).to_dict()
    # FrameProducer does not use image data, so no modifiers are needed.
    step.modifiers_dicts = []
    step.opstransformers_dicts = [
        TabOpsTransformer().to_dict(),
        Optimize(enabled=True).to_dict(),
    ]
    step.post_step_transformers_dicts = [
        MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
    ]
    step.laser_dict = config.config.machine.heads[0].to_dict()
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed
    return step


def create_material_test_step(
    test_type: str = "Engrave",
    speed_range: Tuple[float, float] = (100.0, 500.0),
    power_range: Tuple[float, float] = (10.0, 100.0),
    grid_dimensions: Tuple[int, int] = (5, 5),
    shape_size: float = 10.0,
    spacing: float = 2.0,
    include_labels: bool = True,
    name: Optional[str] = None,
) -> Step:
    """
    Factory to create a Material Test step.

    Args:
        test_type: "Cut" or "Engrave"
        speed_range: (min_speed, max_speed) in mm/min
        power_range: (min_power, max_power) in percentage
        grid_dimensions: (columns, rows) for the test grid
        shape_size: Size of each test square in mm
        spacing: Gap between squares in mm
        include_labels: Whether to add axis labels
        name: Optional custom name for the step

    Returns:
        Configured Step object
    """
    assert config.config.machine, "Machine must be configured"

    step = Step(
        typelabel=_("Material Test Grid"),
        name=name or _("Material Test"),
    )

    # Convert string to enum for the producer
    test_type_enum = MaterialTestGridType(test_type)

    step.opsproducer_dict = MaterialTestGridProducer(
        test_type=test_type_enum,
        speed_range=speed_range,
        power_range=power_range,
        grid_dimensions=grid_dimensions,
        shape_size=shape_size,
        spacing=spacing,
        include_labels=include_labels,
    ).to_dict()

    # Material test doesn't use image modifiers
    step.modifiers_dicts = []

    # No transformers - ops are already optimally ordered
    step.opstransformers_dicts = []

    # No post-step transformers - we don't want path optimization
    step.post_step_transformers_dicts = []

    step.laser_dict = config.config.machine.heads[0].to_dict()
    step.max_cut_speed = config.config.machine.max_cut_speed
    step.max_travel_speed = config.config.machine.max_travel_speed

    # Note: Individual test cells override these with their own speeds/powers
    step.power = int((power_range[0] + power_range[1]) / 2)
    step.cut_speed = int((speed_range[0] + speed_range[1]) / 2)

    return step


STEP_FACTORIES: List[Callable[[Optional[str]], Step]] = [
    create_contour_step,
    create_outline_step,
    create_raster_step,
    create_depth_engrave_step,
    create_shrinkwrap_step,
    create_frame_step,
]
