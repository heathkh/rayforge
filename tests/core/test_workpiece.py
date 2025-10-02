import cairo
import pytest
from pathlib import Path
from typing import Tuple, cast
from dataclasses import asdict
from blinker import Signal
from rayforge.core.doc import Doc
from rayforge.core.import_source import ImportSource
from rayforge.core.item import DocItem
from rayforge.core.matrix import Matrix
from rayforge.core.tab import Tab
from rayforge.core.geo import Geometry
from rayforge.core.workpiece import WorkPiece
from rayforge.image.svg.renderer import SvgRenderer
from rayforge.image import import_file


@pytest.fixture
def sample_svg_data() -> bytes:
    """Provides a simple SVG with defined dimensions in mm."""
    svg = """
    <svg width="100mm" height="50mm" viewBox="0 0 100 50"
         xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="100" height="50" fill="blue"/>
    </svg>
    """
    return svg.encode("utf-8")


@pytest.fixture
def doc_with_workpiece(
    sample_svg_data: bytes, tmp_path: Path
) -> Tuple[Doc, WorkPiece, ImportSource]:
    """
    Creates a Doc with a single WorkPiece linked to an ImportSource,
    which is the correct way to test a WorkPiece's data-dependent methods.
    """
    # Use the new convenience function to handle the entire import process.
    svg_file = tmp_path / "test_rect.svg"
    svg_file.write_bytes(sample_svg_data)
    payload = import_file(svg_file)

    assert payload is not None
    source = payload.source
    wp = cast(WorkPiece, payload.items[0])

    doc = Doc()
    doc.add_import_source(source)
    doc.active_layer.add_child(wp)
    return doc, wp, source


@pytest.fixture
def workpiece_instance(
    doc_with_workpiece: Tuple[Doc, WorkPiece, ImportSource],
):
    """Provides the WorkPiece instance from the doc_with_workpiece fixture."""
    return doc_with_workpiece[1]


class TestWorkPiece:
    def test_initialization(self, workpiece_instance, sample_svg_data):
        wp = workpiece_instance
        assert wp.name == "test_rect"
        assert wp.source_file is not None
        assert wp.source_file.name == "test_rect.svg"
        assert isinstance(wp.renderer, SvgRenderer)
        assert wp.data == sample_svg_data
        assert wp.source is not None
        assert wp.source.original_data == sample_svg_data
        assert wp.pos == pytest.approx((0.0, 0.0))
        assert wp.size == pytest.approx((100.0, 50.0))
        assert wp.angle == pytest.approx(0.0)
        # Importer sets size, so matrix is not identity
        assert wp.matrix != Matrix.identity()
        assert isinstance(wp.updated, Signal)
        assert isinstance(wp.transform_changed, Signal)
        assert wp.tabs == []
        assert wp.tabs_enabled is True
        assert wp.import_source_uid is not None

    def test_workpiece_is_docitem(self, workpiece_instance):
        assert isinstance(workpiece_instance, DocItem)
        assert hasattr(workpiece_instance, "get_world_transform")

    def test_serialization_deserialization(self, doc_with_workpiece):
        doc, wp, source = doc_with_workpiece
        wp.set_size(80.0, 40.0)
        wp.pos = (10.5, 20.2)
        wp.angle = 90
        wp.import_source_uid = "source-123"
        data_dict = wp.to_dict()

        # Key check: renderer and source_file are NOT part of the workpiece
        # dict
        assert "renderer_name" not in data_dict
        assert "source_file" not in data_dict
        assert "data" not in data_dict
        assert "size" not in data_dict
        assert isinstance(data_dict["matrix"], list)
        assert data_dict["import_source_uid"] == "source-123"

        new_wp = WorkPiece.from_dict(data_dict)

        # A free-floating workpiece cannot access its source properties
        assert new_wp.data is None
        assert new_wp.renderer is None
        assert new_wp.source_file is None

        # Add it to the doc to link it to its source
        source.uid = "source-123"  # Ensure UID matches for lookup
        doc.add_import_source(source)
        doc.active_layer.add_child(new_wp)

        assert new_wp.name == wp.name
        assert new_wp.source_file == source.source_file
        assert isinstance(new_wp.renderer, SvgRenderer)
        assert new_wp.pos == pytest.approx(wp.pos)
        assert new_wp.size == pytest.approx(wp.size)
        assert new_wp.angle == pytest.approx(wp.angle, abs=1e-9)
        assert new_wp.matrix == wp.matrix
        assert new_wp.get_natural_size() == (100.0, 50.0)
        assert new_wp.import_source_uid == "source-123"

    def test_serialization_with_tabs(self, workpiece_instance):
        """Tests that tabs are correctly serialized and deserialized."""
        wp = workpiece_instance
        wp.tabs = [
            Tab(width=3.0, segment_index=1, pos=0.5, uid="tab1"),
            Tab(width=3.0, segment_index=5, pos=0.25, uid="tab2"),
        ]
        wp.tabs_enabled = False

        data_dict = wp.to_dict()

        assert "tabs" in data_dict
        assert "tabs_enabled" in data_dict
        assert data_dict["tabs_enabled"] is False
        assert len(data_dict["tabs"]) == 2
        assert data_dict["tabs"][0] == asdict(wp.tabs[0])

        new_wp = WorkPiece.from_dict(data_dict)

        assert new_wp.tabs_enabled is False
        assert len(new_wp.tabs) == 2
        assert new_wp.tabs[0].uid == "tab1"
        assert new_wp.tabs[1].width == 3.0
        assert new_wp.tabs[1].pos == 0.25

    def test_setters_and_signals(self, workpiece_instance):
        wp = workpiece_instance
        updated_events, transform_events = [], []

        wp.updated.connect(
            lambda sender: updated_events.append(sender), weak=False
        )
        wp.transform_changed.connect(
            lambda sender: transform_events.append(sender), weak=False
        )

        # pos setter should fire transform_changed, NOT updated.
        wp.pos = (10, 20)
        assert wp.pos == pytest.approx((10.0, 20.0))
        assert len(updated_events) == 0
        assert len(transform_events) == 1

        # set_size should fire transform_changed, NOT updated.
        wp.set_size(150, 75)
        assert wp.size == pytest.approx((150.0, 75.0))
        assert len(updated_events) == 0
        assert len(transform_events) == 2

        # angle setter should fire transform_changed, NOT updated.
        wp.angle = 45
        assert wp.angle == pytest.approx(45.0)
        assert len(updated_events) == 0
        assert len(transform_events) == 3

    def test_sizing_and_aspect_ratio(self, workpiece_instance):
        wp = workpiece_instance
        assert wp.get_natural_aspect_ratio() == pytest.approx(2.0)
        # get_default_size should return the SVG's natural size.
        assert wp.get_default_size(bounds_width=1000, bounds_height=1000) == (
            100.0,
            50.0,
        )
        # The importer sets the size to the natural size.
        assert wp.size == pytest.approx((100.0, 50.0))

        wp.set_size(80, 20)
        assert wp.size == pytest.approx((80.0, 20.0))
        assert wp.get_current_aspect_ratio() == pytest.approx(4.0)

    def test_get_default_size_fallback(self):
        """
        Tests the fallback sizing logic when metadata is missing.
        """

        class MockNoSizeRenderer(SvgRenderer):
            # This override isn't strictly needed anymore since the logic
            # depends on metadata, but it clarifies the test's intent.
            def get_natural_size(self, workpiece: "WorkPiece"):
                return None

        # Setup doc and source with the mock renderer
        doc = Doc()
        source = ImportSource(
            source_file=Path("nosize.dat"),
            original_data=b"",
            renderer=MockNoSizeRenderer(),
        )
        doc.add_import_source(source)
        wp = WorkPiece("nosize.dat")
        wp.import_source_uid = source.uid
        doc.active_layer.add_child(wp)

        # The size should fall back to the provided bounds
        assert wp.get_default_size(
            bounds_width=400.0, bounds_height=300.0
        ) == (400.0, 300.0)

    def test_render_for_ops(self, workpiece_instance):
        """Tests that render_for_ops renders at the current size."""
        wp = workpiece_instance
        wp.set_size(100, 50)
        surface = wp.render_for_ops(pixels_per_mm_x=10, pixels_per_mm_y=10)

        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 1000
        assert surface.get_height() == 500

    def test_render_chunk(self, workpiece_instance):
        """
        Tests that render_chunk yields chunks that correctly tile the full
        image area.
        """
        wp = workpiece_instance
        wp.set_size(100, 50)
        chunks = list(
            wp.render_chunk(
                pixels_per_mm_x=1,
                pixels_per_mm_y=1,
                max_chunk_width=40,
                max_chunk_height=40,
            )
        )
        assert len(chunks) == 6  # 3x2 grid of chunks
        max_x = max((c[1][0] + c[0].get_width() for c in chunks), default=0)
        max_y = max((c[1][1] + c[0].get_height() for c in chunks), default=0)
        assert max_x == 100
        assert max_y == 50

    def test_dump(self, workpiece_instance, capsys):
        """
        Tests the console output of the dump method.
        """
        workpiece_instance.dump(indent=1)
        captured = capsys.readouterr()

        # Check for components to avoid issues with temporary paths in output
        assert workpiece_instance.source_file.name in captured.out
        assert "SvgRenderer" in captured.out

    def test_get_world_transform_simple_translation(self, workpiece_instance):
        wp = workpiece_instance
        wp.pos = (10, 20)
        matrix = wp.get_world_transform()

        p_in = (0, 0)
        p_out = matrix.transform_point(p_in)
        assert p_out == pytest.approx((10, 20))

    def test_get_world_transform_scale(self, workpiece_instance):
        wp = workpiece_instance
        # set_size preserves center. Original center is (50, 25).
        # New pos will be (50-10, 25-5) = (40, 20)
        wp.set_size(20, 10)
        matrix = wp.get_world_transform()
        p_in = (1, 1)  # Local corner in a 1x1 space
        p_out = matrix.transform_point(p_in)
        # After sizing, center is (50,25), pos is (40,20).
        # The new world transform is T(40,20) @ S(20,10).
        # It transforms local corner (1,1) to (40,20) + (20,10) = (60,30).
        assert p_out == pytest.approx((60.0, 30.0))

    def test_get_world_transform_rotation(self, workpiece_instance):
        wp = workpiece_instance
        # set_size preserves center. Initial center (50, 25) moves to a pos
        # that keeps it at the same world coords.
        wp.set_size(20, 10)
        # angle.setter rebuilds the matrix, rotating around the scaled center.
        wp.angle = 90
        matrix = wp.get_world_transform()
        # Calculation is complex, but we can check a known point.
        # Center of 100x50 is (0.5,0.5) in local coords.
        # After sizing to 20x10, world center is (50,25).
        # Rotating origin (0,0) around (50,25) by 90deg gives
        # (50 - (-25), 25 + 50) = (75, 75).
        # But that's in the unscaled space. The transform is more complex.
        # Let's check the transformed center instead.
        center_out = matrix.transform_point((0.5, 0.5))
        assert center_out == pytest.approx((50, 25))

    def test_get_world_transform_all(self, workpiece_instance):
        wp = workpiece_instance
        wp.set_size(20, 10)
        wp.pos = (100, 200)
        wp.angle = 90
        matrix = wp.get_world_transform()

        p_in = (0, 0)
        p_out = matrix.transform_point(p_in)
        # Center of 20x10 is (10,5). Pos is (100,200). So center is (110,205).
        # Rotate origin (0,0) 90 deg around (10,5) -> (15, -5).
        # Add original pos -> (115, 195).
        assert p_out == pytest.approx((115, 195))

    def test_decomposed_properties_consistency(self, workpiece_instance):
        wp = workpiece_instance
        target_pos = (12.3, 45.6)
        target_size = (78.9, 101.1)
        target_angle = 33.3

        # Set properties once and check. The order matters: operations that
        # preserve the object's center (set_size, angle) will change the
        # top-left position. Therefore, pos must be set last to achieve a
        # predictable final position.
        wp.set_size(*target_size)
        wp.angle = target_angle
        wp.pos = target_pos

        assert wp.pos == pytest.approx(target_pos, abs=1e-9)
        assert wp.size == pytest.approx(target_size, abs=1e-9)
        assert wp.angle == pytest.approx(target_angle, abs=1e-9)

        # Test again with a different order and values
        target_pos2 = (-5, -10)
        target_size2 = (20, 20)
        target_angle2 = 180

        # Set properties in a different order
        wp.angle = target_angle2
        wp.pos = target_pos2
        # The last operation is set_size, which preserves the center
        # based on the state right before it's called.
        wp.set_size(*target_size2)

        # Check the final size and angle, which should be correct
        assert wp.size == pytest.approx(target_size2, abs=1e-9)
        assert wp.angle == pytest.approx(target_angle2, abs=1e-9)
        # The final position will have been adjusted by set_size, so we don't
        # check it against target_pos2. Instead, we ensure it's consistent.
        final_pos = wp.pos
        wp.pos = final_pos
        assert wp.pos == pytest.approx(final_pos, abs=1e-9)

    def test_negative_angle_preservation(self, workpiece_instance):
        """
        Tests that a negative angle is correctly set and retrieved, which
        was the cause of the '1 -> 359' bug (actually '-1 -> 359').
        """
        wp = workpiece_instance
        wp.angle = -45.0
        assert wp.angle == pytest.approx(-45.0)

        wp.angle = -10.0
        assert wp.angle == pytest.approx(-10.0)

        # Also check a positive angle to ensure no regressions.
        wp.angle = 10.0
        assert wp.angle == pytest.approx(10.0)

    def test_get_tab_direction(self, workpiece_instance):
        wp = workpiece_instance
        # Create a simple CCW square geometry
        geo = Geometry()
        geo.move_to(0, 0)  # cmd 0
        geo.line_to(10, 0)  # cmd 1: bottom edge
        geo.line_to(10, 10)  # cmd 2: right edge
        geo.close_path()  # cmd 3
        wp.vectors = geo

        # Case 1: No vectors
        wp.vectors = None
        tab = Tab(width=1, segment_index=1, pos=0.5)
        assert wp.get_tab_direction(tab) is None
        wp.vectors = geo

        # Case 2: No transform
        wp.matrix = Matrix.identity()
        tab = Tab(width=1, segment_index=1, pos=0.5)  # Midpoint of bottom edge
        direction = wp.get_tab_direction(tab)
        assert direction is not None
        assert direction == pytest.approx((0, -1))

        # Case 3: 90 degree rotation
        wp.angle = 90
        direction = wp.get_tab_direction(tab)
        assert direction is not None
        # A (0, -1) vector rotated +90 deg becomes (1, 0)
        assert direction == pytest.approx((1, 0))

        # Case 4: Scale and rotation
        wp.set_size(20, 10)  # non-uniform scale
        wp.angle = -90
        direction = wp.get_tab_direction(tab)
        assert direction is not None
        # A (0, -1) vector rotated -90 deg becomes (-1, 0)
        assert direction == pytest.approx((-1, 0))

        # Case 5: Open path
        wp.vectors = Geometry()
        wp.vectors.move_to(0, 0)
        wp.vectors.line_to(10, 0)
        assert wp.get_tab_direction(tab) is None

    def test_get_tab_direction_non_uniform_scale_diagonal(
        self, workpiece_instance
    ):
        """
        Tests that the tab normal is correct for a diagonal path under
        non-uniform scaling, which is where the old method fails.
        """
        wp = workpiece_instance
        # A 45-degree rotated CCW square
        geo = Geometry()
        geo.move_to(10, 0)
        geo.line_to(
            20, 10
        )  # segment 1: diagonal, tangent proportional to (1,1)
        geo.line_to(10, 20)
        geo.line_to(0, 10)
        geo.close_path()
        wp.vectors = geo

        # The geometry's bounding box is 20x20.
        # Applying a 20x10 size results in a non-uniform scale of (1, 0.5).
        wp.set_size(20, 10)
        wp.angle = 0  # ensure no rotation

        tab = Tab(width=1, segment_index=1, pos=0.5)
        direction = wp.get_tab_direction(tab)
        assert direction is not None

        # The local tangent is (1, 1).
        # The world-space path segment is scaled by (1, 0.5).
        # The world tangent is therefore proportional to (1, 0.5).
        # The outward normal for a CCW path is a 90-deg CW rotation of the
        # tangent (ty, -tx).
        # So, the normal is proportional to (0.5, -1).
        expected_x, expected_y = (0.5, -1.0)
        norm = (expected_x**2 + expected_y**2) ** 0.5
        expected_direction = (expected_x / norm, expected_y / norm)

        assert direction == pytest.approx(expected_direction)
