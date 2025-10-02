# flake8: noqa:F401
import inspect
from .base import OpsProducer, PipelineArtifact, CutSide
from .depth import DepthEngraver
from .edge import EdgeTracer
from .frame import FrameProducer
from .material_test_grid import MaterialTestGridProducer, MaterialTestGridType
from .shrinkwrap import ShrinkWrapProducer
from .rasterize import Rasterizer

producer_by_name = dict(
    [
        (name, obj)
        for name, obj in locals().items()
        if inspect.isclass(obj)
        and issubclass(obj, OpsProducer)
        and not inspect.isabstract(obj)
    ]
)

__all__ = [
    "OpsProducer",
    "DepthEngraver",
    "EdgeTracer",
    "FrameProducer",
    "MaterialTestGridProducer",
    "MaterialTestGridType",
    "ShrinkWrapProducer",
    "Rasterizer",
    "producer_by_name",
]
