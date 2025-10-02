from __future__ import annotations
import math
import logging
from copy import copy, deepcopy
from typing import (
    Iterator,
    List,
    Optional,
    Tuple,
    Generator,
    Dict,
    Any,
    TYPE_CHECKING,
)
import numpy as np
from ..geo import linearize, query, clipping
from .commands import (
    State,
    Command,
    MovingCommand,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    SetPowerCommand,
    SetCutSpeedCommand,
    SetTravelSpeedCommand,
    EnableAirAssistCommand,
    DisableAirAssistCommand,
    SetLaserCommand,
    JobStartCommand,
    JobEndCommand,
    LayerStartCommand,
    LayerEndCommand,
    WorkpieceStartCommand,
    WorkpieceEndCommand,
    SectionType,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    ScanLinePowerCommand,
)


if TYPE_CHECKING:
    from ..geo.geometry import Geometry

logger = logging.getLogger(__name__)


class Ops:
    """
    Represents a set of generated path segments and instructions that
    are used for making gcode, but also to generate vector graphics
    for display.
    """

    def __init__(self) -> None:
        self.commands: List[Command] = []
        self._commands_ref_for_pyreverse: Command
        self.last_move_to: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Ops object to a dictionary."""
        return {
            "commands": [cmd.to_dict() for cmd in self.commands],
            "last_move_to": self.last_move_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Ops:
        """Deserializes a dictionary into an Ops instance."""
        new_ops = cls()
        last_move = tuple(data.get("last_move_to", (0.0, 0.0, 0.0)))
        assert len(last_move) == 3, "last_move_to must be a 3-tuple"
        new_ops.last_move_to = last_move

        for cmd_data in data.get("commands", []):
            cmd_type = cmd_data.get("type")
            if cmd_type == "MoveToCommand":
                new_ops.add(MoveToCommand(end=tuple(cmd_data["end"])))
            elif cmd_type == "LineToCommand":
                new_ops.add(LineToCommand(end=tuple(cmd_data["end"])))
            elif cmd_type == "ArcToCommand":
                new_ops.add(
                    ArcToCommand(
                        end=tuple(cmd_data["end"]),
                        center_offset=tuple(cmd_data["center_offset"]),
                        clockwise=cmd_data["clockwise"],
                    )
                )
            elif cmd_type == "SetPowerCommand":
                new_ops.add(SetPowerCommand(power=cmd_data["power"]))
            elif cmd_type == "SetCutSpeedCommand":
                new_ops.add(SetCutSpeedCommand(speed=cmd_data["speed"]))
            elif cmd_type == "SetTravelSpeedCommand":
                new_ops.add(SetTravelSpeedCommand(speed=cmd_data["speed"]))
            elif cmd_type == "EnableAirAssistCommand":
                new_ops.add(EnableAirAssistCommand())
            elif cmd_type == "DisableAirAssistCommand":
                new_ops.add(DisableAirAssistCommand())
            elif cmd_type == "SetLaserCommand":
                new_ops.add(SetLaserCommand(laser_uid=cmd_data["laser_uid"]))
            elif cmd_type == "JobStartCommand":
                new_ops.add(JobStartCommand())
            elif cmd_type == "JobEndCommand":
                new_ops.add(JobEndCommand())
            elif cmd_type == "LayerStartCommand":
                new_ops.add(LayerStartCommand(layer_uid=cmd_data["layer_uid"]))
            elif cmd_type == "LayerEndCommand":
                new_ops.add(LayerEndCommand(layer_uid=cmd_data["layer_uid"]))
            elif cmd_type == "WorkpieceStartCommand":
                new_ops.add(
                    WorkpieceStartCommand(
                        workpiece_uid=cmd_data["workpiece_uid"]
                    )
                )
            elif cmd_type == "WorkpieceEndCommand":
                new_ops.add(
                    WorkpieceEndCommand(
                        workpiece_uid=cmd_data["workpiece_uid"]
                    )
                )
            elif cmd_type == "OpsSectionStartCommand":
                new_ops.add(
                    OpsSectionStartCommand(
                        section_type=SectionType[cmd_data["section_type"]],
                        workpiece_uid=cmd_data["workpiece_uid"],
                    )
                )
            elif cmd_type == "OpsSectionEndCommand":
                new_ops.add(
                    OpsSectionEndCommand(
                        section_type=SectionType[cmd_data["section_type"]]
                    )
                )
            elif cmd_type == "ScanLinePowerCommand":
                new_ops.add(
                    ScanLinePowerCommand(
                        end=tuple(cmd_data["end"]),
                        power_values=bytearray(cmd_data["power_values"]),
                    )
                )
            else:
                logger.warning(
                    "Skipping unknown command type during deserialization:"
                    f" {cmd_type}"
                )
        return new_ops

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> "Ops":
        """
        Creates an Ops object from a Geometry object, converting its path.
        """
        from .. import geo

        new_ops = cls()
        for cmd in geometry.commands:
            # Explicitly convert from geo.Command to ops.Command
            if isinstance(cmd, geo.MoveToCommand):
                new_ops.add(MoveToCommand(cmd.end))
            elif isinstance(cmd, geo.LineToCommand):
                new_ops.add(LineToCommand(cmd.end))
            elif isinstance(cmd, geo.ArcToCommand):
                new_ops.add(
                    ArcToCommand(cmd.end, cmd.center_offset, cmd.clockwise)
                )
        new_ops.last_move_to = geometry.last_move_to
        return new_ops

    def to_geometry(self) -> "Geometry":
        """
        Creates a Geometry path from this Ops object, including only the
        geometric commands.
        """
        from ..geo.geometry import Geometry

        new_geo = Geometry()
        for op in self.commands:
            if isinstance(op, MoveToCommand):
                if op.end:
                    new_geo.move_to(*op.end)
            elif isinstance(op, LineToCommand):
                if op.end:
                    new_geo.line_to(*op.end)
            elif isinstance(op, ArcToCommand):
                if op.end:
                    new_geo.arc_to(
                        op.end[0],
                        op.end[1],
                        op.center_offset[0],
                        op.center_offset[1],
                        op.clockwise,
                        op.end[2],
                    )
        return new_geo

    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    def __add__(self, ops: Ops) -> Ops:
        result = Ops()
        result.commands = self.commands + ops.commands
        return result

    def __mul__(self, count: int) -> Ops:
        result = Ops()
        result.commands = count * self.commands
        return result

    def __len__(self) -> int:
        return len(self.commands)

    def is_empty(self) -> bool:
        """Checks if the Ops object contains any commands."""
        return not self.commands

    def copy(self) -> Ops:
        """Creates a deep copy of the Ops object."""
        new_ops = Ops()
        new_ops.commands = deepcopy(self.commands)
        new_ops.last_move_to = self.last_move_to
        return new_ops

    def preload_state(self) -> None:
        """
        Walks through all commands, enriching each by the indended
        state of the machine. The state is useful for some post-processors
        that need to re-order commands without changing the intended
        state during each command.
        """
        state = State()
        for cmd in self.commands:
            if cmd.is_state_command():
                cmd.apply_to_state(state)
            elif not cmd.is_marker_command():
                cmd.state = copy(state)

    def clear(self) -> None:
        self.commands = []

    def add(self, command: Command) -> None:
        self.commands.append(command)

    def extend(self, other_ops: "Ops") -> None:
        """
        Appends all commands from another Ops object to this one.
        """
        if other_ops and other_ops.commands:
            self.commands.extend(other_ops.commands)

    def move_to(self, x: float, y: float, z: float = 0.0) -> None:
        self.last_move_to = (float(x), float(y), float(z))
        cmd = MoveToCommand(self.last_move_to)
        self.commands.append(cmd)

    def line_to(self, x: float, y: float, z: float = 0.0) -> None:
        cmd = LineToCommand((float(x), float(y), float(z)))
        self.commands.append(cmd)

    def close_path(self) -> None:
        """
        Convenience method that wraps line_to(). Makes a line to
        the last move_to point.
        """
        self.line_to(*self.last_move_to)

    def arc_to(
        self,
        x: float,
        y: float,
        i: float,
        j: float,
        clockwise: bool = True,
        z: float = 0.0,
    ) -> None:
        """
        Adds an arc command with specified endpoint, center offsets,
        and direction (cw/ccw).
        """
        self.commands.append(
            ArcToCommand(
                (float(x), float(y), float(z)),
                (float(i), float(j)),
                bool(clockwise),
            )
        )

    def bezier_to(
        self,
        c1: Tuple[float, float, float],
        c2: Tuple[float, float, float],
        end: Tuple[float, float, float],
        num_steps: int = 20,
    ) -> None:
        """
        Adds a cubic Bézier curve approximated by a series of LineToCommands.
        The curve starts from the current last point in the path. This method
        requires full 3D coordinates for all control and end points.
        """
        if not self.commands or self.commands[-1].end is None:
            logger.warning("bezier_to called without a starting point.")
            return

        start_point = self.commands[-1].end
        segments = linearize.linearize_bezier(
            start_point, c1, c2, end, num_steps
        )
        for _, end_point in segments:
            self.line_to(*end_point)

    def set_power(self, power: float) -> None:
        """
        Sets the intended laser power for subsequent cutting commands.
        This is a state declaration, not an immediate command to turn on
        the laser.
        """
        cmd = SetPowerCommand(int(power))
        self.commands.append(cmd)

    def set_cut_speed(self, speed: float) -> None:
        """
        Sets the intended feed rate for subsequent cutting commands.
        This is a state declaration.
        """
        cmd = SetCutSpeedCommand(int(speed))
        self.commands.append(cmd)

    def set_travel_speed(self, speed: float) -> None:
        """
        Sets the intended feed rate for subsequent travel commands.
        This is a state declaration.
        """
        cmd = SetTravelSpeedCommand(int(speed))
        self.commands.append(cmd)

    def enable_air_assist(self, enable: bool = True) -> None:
        """
        Sets the intended state of the air assist for subsequent commands.
        This is a state declaration.
        """
        if enable:
            self.commands.append(EnableAirAssistCommand())
        else:
            self.disable_air_assist()

    def disable_air_assist(self) -> None:
        """
        Sets the intended state of the air assist for subsequent commands.
        This is a state declaration.
        """
        self.commands.append(DisableAirAssistCommand())

    def set_laser(self, laser_uid: str) -> None:
        """
        Sets the intended active laser for subsequent commands.
        This is a state declaration.
        """
        cmd = SetLaserCommand(laser_uid)
        self.commands.append(cmd)

    def rect(
        self, include_travel: bool = False
    ) -> Tuple[float, float, float, float]:
        """
        Returns a rectangle (x1, y1, x2, y2) that encloses the
        occupied area in the XY plane.

        Args:
            include_travel: If True, travel moves are included in the bounds.
        """
        return query.get_bounding_rect(
            self.commands, include_travel=include_travel
        )

    def get_frame(
        self,
        power: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> Ops:
        """
        Returns a new Ops object containing four move_to operations forming
        a frame around the occupied area of the original Ops. The occupied
        area includes all points from line_to and close_path commands.
        """
        min_x, min_y, max_x, max_y = self.rect()
        if (min_x, min_y, max_x, max_y) == (0.0, 0.0, 0.0, 0.0):
            return Ops()

        frame_ops = Ops()
        if power is not None:
            frame_ops.set_power(power)
        if speed is not None:
            frame_ops.set_cut_speed(speed)
        frame_ops.move_to(min_x, min_y)
        frame_ops.line_to(min_x, max_y)
        frame_ops.line_to(max_x, max_y)
        frame_ops.line_to(max_x, min_y)
        frame_ops.line_to(min_x, min_y)
        return frame_ops

    def distance(self) -> float:
        """
        Calculates the total 2D path length for all moving commands.
        """
        return query.get_total_distance(self.commands)

    def cut_distance(self) -> float:
        """
        Like distance(), but only counts 2D cut distance.
        """
        total = 0.0
        last: Optional[Tuple[float, float, float]] = None
        for cmd in self.commands:
            if cmd.is_cutting_command():
                total += cmd.distance(last)

            if isinstance(cmd, MovingCommand):
                last = cmd.end
        return total

    def segments(self) -> Generator[List[Command], None, None]:
        segment: List[Command] = []
        for command in self.commands:
            if not segment:
                segment.append(command)
                continue

            if command.is_travel_command():
                yield segment
                segment = [command]

            elif command.is_cutting_command():
                segment.append(command)

            elif command.is_state_command() or command.is_marker_command():
                yield segment
                yield [command]
                segment = []

        if segment:
            yield segment

    def transform(self, matrix: "np.ndarray") -> "Ops":
        """
        Applies a transformation matrix to all geometric commands. If the
        transform is non-uniform (contains non-uniform scaling or shear),
        arcs will be linearized to preserve their shape.

        Args:
            matrix: A 4x4 NumPy transformation matrix.

        Returns:
            The Ops object itself for chaining.
        """
        # Check for non-uniform scaling or shear by comparing the length of
        # transformed basis vectors.
        v_x = matrix @ np.array([1, 0, 0, 0])
        v_y = matrix @ np.array([0, 1, 0, 0])
        len_x = np.linalg.norm(v_x[:2])
        len_y = np.linalg.norm(v_y[:2])
        is_non_uniform = not np.isclose(len_x, len_y)

        transformed_commands: List[Command] = []
        last_point_untransformed: Optional[Tuple[float, float, float]] = None

        for cmd in self.commands:
            original_cmd_end = (
                cmd.end if isinstance(cmd, MovingCommand) else None
            )

            if isinstance(cmd, ArcToCommand) and is_non_uniform:
                # Use the last known untransformed point as the start for
                # linearization
                start_point = last_point_untransformed or (0.0, 0.0, 0.0)
                segments = linearize.linearize_arc(cmd, start_point)
                for p1, p2 in segments:
                    point_vec = np.array([p2[0], p2[1], p2[2], 1.0])
                    transformed_vec = matrix @ point_vec
                    transformed_commands.append(
                        LineToCommand(tuple(transformed_vec[:3]))
                    )
            elif isinstance(cmd, MovingCommand):
                point_vec = np.array([*cmd.end, 1.0])
                transformed_vec = matrix @ point_vec
                cmd.end = tuple(transformed_vec[:3])

                if isinstance(cmd, ArcToCommand):
                    # For uniform transforms, we transform the center offset
                    # vector by the 3x3 rotation/scaling part of the matrix.
                    offset_vec_3d = np.array(
                        [cmd.center_offset[0], cmd.center_offset[1], 0]
                    )
                    rot_scale_matrix = matrix[:3, :3]
                    new_offset_vec_3d = rot_scale_matrix @ offset_vec_3d
                    cmd.center_offset = (
                        new_offset_vec_3d[0],
                        new_offset_vec_3d[1],
                    )
                transformed_commands.append(cmd)
            else:
                transformed_commands.append(cmd)

            # Crucially, update the last_point tracker with the endpoint
            # from BEFORE the transformation for the next iteration.
            if original_cmd_end is not None:
                last_point_untransformed = original_cmd_end

        self.commands = transformed_commands
        last_move_vec = np.array([*self.last_move_to, 1.0])
        transformed_last_move_vec = matrix @ last_move_vec
        self.last_move_to = tuple(transformed_last_move_vec[:3])
        return self

    def translate(self, dx: float, dy: float, dz: float = 0.0) -> Ops:
        """Translate geometric commands."""
        matrix = np.identity(4)
        matrix[0, 3] = dx
        matrix[1, 3] = dy
        matrix[2, 3] = dz
        return self.transform(matrix)

    def scale(self, sx: float, sy: float, sz: float = 1.0) -> Ops:
        """Scales all geometric commands."""
        matrix = np.diag([sx, sy, sz, 1.0])
        return self.transform(matrix)

    def rotate(self, angle_deg: float, cx: float, cy: float) -> Ops:
        """Rotates all points around a center (cx, cy) in the XY plane."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Create a 4x4 transformation matrix for rotation around a point
        # T(-cx, -cy) * R(angle) * T(cx, cy)
        translate_to_origin = np.identity(4)
        translate_to_origin[0, 3] = -cx
        translate_to_origin[1, 3] = -cy

        rotation_matrix = np.identity(4)
        rotation_matrix[0, 0] = cos_a
        rotation_matrix[0, 1] = -sin_a
        rotation_matrix[1, 0] = sin_a
        rotation_matrix[1, 1] = cos_a

        translate_back = np.identity(4)
        translate_back[0, 3] = cx
        translate_back[1, 3] = cy

        matrix = translate_back @ rotation_matrix @ translate_to_origin
        return self.transform(matrix)

    def linearize_all(self) -> None:
        """
        Replaces all complex commands (e.g., Arcs) with simple LineToCommands.
        """
        new_commands: List[Command] = []
        last_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Find initial position, in case path doesn't start with MoveTo
        for cmd in self.commands:
            if isinstance(cmd, MoveToCommand):
                last_point = cmd.end
                break

        for cmd in self.commands:
            if isinstance(cmd, MovingCommand):
                # The linearize method on the command itself will do the work.
                linearized_cmds = cmd.linearize(last_point)
                new_commands.extend(linearized_cmds)
                # Update last_point to the end of the last generated segment
                if linearized_cmds and linearized_cmds[-1].end is not None:
                    last_point = linearized_cmds[-1].end
                # If linearization is empty, original endpoint is the fallback
                elif cmd.end is not None:
                    last_point = cmd.end
            else:
                # Non-moving commands (state, markers) are passed through.
                new_commands.append(cmd)
        self.commands = new_commands

    def clip_at(self, x: float, y: float, width: float) -> bool:
        """
        Finds the closest point on a continuous path to (x, y) and creates a
        gap of the given width centered at that point, measured along the
        path's length.

        This method works by linearizing the target subpath into a polyline,
        calculating the clip region parametrically along the polyline's length,
        and then reconstructing the polyline with the gap. Arcs within the
        affected subpath will be converted to line segments.

        Args:
            x: The x-coordinate of the point to clip near.
            y: The y-coordinate of the point to clip near.
            width: The desired width of the gap in world units (e.g., mm).

        Returns:
            True if a clip was successfully performed, False otherwise.
        """
        if width <= 1e-6:
            return False

        # 1. Find the closest segment on the entire path
        closest = query.find_closest_point_on_path(self.commands, x, y)
        if not closest:
            return False

        segment_index, _, point_on_path = closest
        dist_sq = (x - point_on_path[0]) ** 2 + (y - point_on_path[1]) ** 2
        if dist_sq > (width * 2) ** 2:  # Generous tolerance for clicking
            return False

        # 2. Identify the continuous subpath containing the hit segment
        start_idx = 0
        for i in range(segment_index, -1, -1):
            if isinstance(self.commands[i], MoveToCommand):
                start_idx = i
                break

        end_idx = len(self.commands)
        for i in range(start_idx + 1, len(self.commands)):
            if isinstance(self.commands[i], MoveToCommand):
                end_idx = i
                break

        subpath_cmds = self.commands[start_idx:end_idx]
        if not subpath_cmds or not isinstance(subpath_cmds[0], MovingCommand):
            return False

        # 3. Create a temporary, linearized version of the subpath
        temp_ops = Ops()
        temp_ops.commands = deepcopy(subpath_cmds)
        temp_ops.preload_state()
        temp_ops.linearize_all()
        linear_cmds = temp_ops.commands

        if len(linear_cmds) < 2:
            return False

        # 4. Find closest point on the *linearized* path and calculate distance
        linear_closest = query.find_closest_point_on_path(linear_cmds, x, y)
        if not linear_closest:
            return False

        linear_segment_idx, linear_t, _ = linear_closest

        hit_dist = 0.0
        last_pos = linear_cmds[0].end
        assert last_pos is not None  # Should be MoveTo.end

        for i in range(1, linear_segment_idx):
            cmd = linear_cmds[i]
            if isinstance(cmd, LineToCommand):
                hit_dist += math.dist(last_pos[:2], cmd.end[:2])
                last_pos = cmd.end

        hit_segment_cmd = linear_cmds[linear_segment_idx]
        if isinstance(hit_segment_cmd, LineToCommand) and hit_segment_cmd.end:
            dist = math.dist(last_pos[:2], hit_segment_cmd.end[:2])
            hit_dist += linear_t * dist

        # 5. Define gap start and end distances
        gap_start_dist = max(0.0, hit_dist - width / 2.0)
        gap_end_dist = hit_dist + width / 2.0

        # 6. Rebuild the subpath
        def _clip_1d(d1, d2, g1, g2):
            kept = []
            if d1 < g1:
                kept.append((d1, min(d2, g1)))
            if d2 > g2:
                kept.append((max(d1, g2), d2))
            return kept

        new_subpath_cmds: List[Command] = [deepcopy(linear_cmds[0])]
        accum_dist = 0.0
        last_pos = linear_cmds[0].end
        assert last_pos is not None  # Should be MoveTo.end

        for cmd in linear_cmds[1:]:
            if isinstance(cmd, LineToCommand):
                p1, p2 = last_pos, cmd.end
                seg_len = math.dist(p1[:2], p2[:2])

                if seg_len < 1e-9:
                    last_pos = p2
                    continue

                seg_start_dist = accum_dist
                seg_end_dist = accum_dist + seg_len

                kept = _clip_1d(
                    seg_start_dist, seg_end_dist, gap_start_dist, gap_end_dist
                )

                p1_np, p2_np = np.array(p1), np.array(p2)
                vec = p2_np - p1_np

                for start_d, end_d in kept:
                    t_start = (start_d - seg_start_dist) / seg_len
                    t_end = (end_d - seg_start_dist) / seg_len
                    start_pt = tuple(p1_np + t_start * vec)
                    end_pt = tuple(p1_np + t_end * vec)

                    last_kept_pos: Optional[Tuple[float, ...]] = None
                    for rev_cmd in reversed(new_subpath_cmds):
                        if isinstance(rev_cmd, MovingCommand):
                            last_kept_pos = rev_cmd.end
                            break
                    assert last_kept_pos is not None

                    if math.dist(last_kept_pos, start_pt) > 1e-6:
                        new_subpath_cmds.append(MoveToCommand(start_pt))

                    new_line = LineToCommand(end_pt)
                    new_line.state = cmd.state
                    new_subpath_cmds.append(new_line)

                last_pos = p2
                accum_dist += seg_len
            else:  # Handle state/marker commands
                if not (gap_start_dist < accum_dist < gap_end_dist):
                    new_subpath_cmds.append(deepcopy(cmd))

        # Post-process to preserve original endpoint if it was clipped
        original_endpoint = subpath_cmds[-1].end
        new_endpoint = None
        if new_subpath_cmds and isinstance(
            new_subpath_cmds[-1], MovingCommand
        ):
            new_endpoint = new_subpath_cmds[-1].end

        if original_endpoint and (
            new_endpoint is None
            or math.dist(original_endpoint, new_endpoint) > 1e-6
        ):
            new_subpath_cmds.append(MoveToCommand(original_endpoint))

        # 7. Replace original subpath
        self.commands = (
            self.commands[:start_idx]
            + new_subpath_cmds
            + self.commands[end_idx:]
        )
        return True

    def _add_clipped_segment_to_ops(
        self,
        segment: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ],
        new_ops: Ops,
        current_pen_pos: Optional[Tuple[float, float, float]],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Processes a single clipped segment, adding MoveTo/LineTo commands
        to the new_ops object as needed.

        Returns the updated pen position.
        """
        if segment:
            p1_clipped, p2_clipped = segment

            # A new move is needed if the pen is up (None) or if there's a gap.
            # Using 3D distance for this check is correct.
            dist_to_start = (
                math.dist(current_pen_pos, p1_clipped)
                if current_pen_pos
                else float("inf")
            )

            # Use a small tolerance for floating point comparisons
            if dist_to_start > 1e-6:
                new_ops.move_to(*p1_clipped)

            new_ops.line_to(*p2_clipped)
            # The new pen position is the end of the clipped segment
            return p2_clipped
        else:
            # The segment was fully clipped, so the pen is now "up"
            return None

    def clip(self, rect: Tuple[float, float, float, float]) -> Ops:
        """
        Clips the Ops to the given rectangle.
        Returns a new, clipped Ops object.
        """
        new_ops = Ops()
        if not self.commands:
            return new_ops

        last_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Tracks the last known position of the pen *within the clipped area*.
        # None means the pen is "up" or outside the clip rect.
        clipped_pen_pos: Optional[Tuple[float, float, float]] = None

        for cmd in self.commands:
            if cmd.is_state_command() or cmd.is_marker_command():
                new_ops.add(deepcopy(cmd))
                continue

            if not isinstance(cmd, MovingCommand):
                continue

            # Special handling for ScanLinePowerCommand to prevent
            # linearization.
            if isinstance(cmd, ScanLinePowerCommand):
                clipped_segment = clipping.clip_line_segment(
                    last_point, cmd.end, rect
                )
                if clipped_segment:
                    new_start, new_end = clipped_segment

                    # Calculate the start and end `t` values (0-1) of the
                    # clipped segment relative to the original line.
                    p_start_orig = np.array(last_point)
                    p_end_orig = np.array(cmd.end)
                    vec_orig = p_end_orig - p_start_orig
                    len_sq = np.dot(vec_orig, vec_orig)

                    if len_sq > 1e-9:
                        t_start = (
                            np.dot(
                                np.array(new_start) - p_start_orig, vec_orig
                            )
                            / len_sq
                        )
                        t_end = (
                            np.dot(np.array(new_end) - p_start_orig, vec_orig)
                            / len_sq
                        )
                    else:
                        t_start, t_end = 0.0, 1.0

                    t_start = max(0.0, min(1.0, t_start))
                    t_end = max(0.0, min(1.0, t_end))

                    # Slice the power_values array based on the `t` values.
                    num_values = len(cmd.power_values)
                    idx_start = int(num_values * t_start)
                    idx_end = int(num_values * t_end)
                    new_power_values = cmd.power_values[idx_start:idx_end]

                    if new_power_values:
                        # Since scanlines are discrete, we always need a move
                        if (
                            clipped_pen_pos is None
                            or math.dist(clipped_pen_pos, new_start) > 1e-6
                        ):
                            new_ops.move_to(*new_start)
                        new_ops.add(
                            ScanLinePowerCommand(new_end, new_power_values)
                        )
                        clipped_pen_pos = new_end

                last_point = cmd.end
                continue  # Skip the generic linearization below

            if cmd.is_travel_command():
                if cmd.end is not None:
                    last_point = cmd.end
                clipped_pen_pos = None  # A travel move always lifts the pen
                continue

            # Linearize the command into a series of simpler commands
            linearized_commands = cmd.linearize(last_point)

            # Process each linearized segment
            p_current_segment_start = last_point
            for l_cmd in linearized_commands:
                if l_cmd.end is None:
                    continue
                p_current_segment_end = l_cmd.end

                clipped_segment = clipping.clip_line_segment(
                    p_current_segment_start, p_current_segment_end, rect
                )
                clipped_pen_pos = self._add_clipped_segment_to_ops(
                    clipped_segment, new_ops, clipped_pen_pos
                )
                p_current_segment_start = p_current_segment_end

            # The next command starts where the original unclipped command
            # ended
            if cmd.end is not None:
                last_point = cmd.end

        return new_ops

    def dump(self) -> None:
        for segment in self.segments():
            print(segment)

    def subtract_regions(
        self, regions: List[List[Tuple[float, float]]]
    ) -> "Ops":
        """
        Clips the Ops by subtracting a list of polygonal regions.
        This modifies the Ops object in place and returns it.
        """
        if not regions or not self.commands:
            return self

        new_ops = Ops()
        last_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Tracks the last known pen position of a *kept* segment
        pen_pos: Optional[Tuple[float, float, float]] = None

        # Add any leading state/marker commands before the first move
        first_move_idx = next(
            (
                i
                for i, cmd in enumerate(self.commands)
                if isinstance(cmd, MovingCommand)
            ),
            len(self.commands),
        )
        for i in range(first_move_idx):
            new_ops.add(deepcopy(self.commands[i]))

        for cmd in self.commands:
            if not isinstance(cmd, MovingCommand) or cmd.end is None:
                # State/marker commands are handled as they appear
                # between moves
                if not new_ops.commands or new_ops.commands[-1] is not cmd:
                    new_ops.add(deepcopy(cmd))
                continue

            if isinstance(cmd, MoveToCommand):
                last_point = cmd.end
                pen_pos = None  # Pen is up
                continue

            if isinstance(cmd, ScanLinePowerCommand):
                kept_segments = clipping.subtract_regions_from_line_segment(
                    last_point, cmd.end, regions
                )
                num_values = len(cmd.power_values)
                p_start_orig = np.array(last_point)
                p_end_orig = np.array(cmd.end)
                vec_orig = p_end_orig - p_start_orig
                len_sq = np.dot(vec_orig, vec_orig)

                for new_start, new_end in kept_segments:
                    if len_sq > 1e-9:
                        t_start = (
                            np.dot(
                                np.array(new_start) - p_start_orig, vec_orig
                            )
                            / len_sq
                        )
                        t_end = (
                            np.dot(np.array(new_end) - p_start_orig, vec_orig)
                            / len_sq
                        )
                    else:
                        t_start, t_end = 0.0, 1.0

                    idx_start = int(num_values * t_start)
                    idx_end = int(num_values * t_end)
                    new_power = cmd.power_values[idx_start:idx_end]

                    if new_power:
                        if (
                            pen_pos is None
                            or math.dist(pen_pos, new_start) > 1e-6
                        ):
                            new_ops.move_to(*new_start)
                        new_ops.add(ScanLinePowerCommand(new_end, new_power))
                        pen_pos = new_end
                last_point = cmd.end
                continue

            # Linearize cutting command into segments
            linearized_commands = cmd.linearize(last_point)

            p_current_segment_start = last_point
            for l_cmd in linearized_commands:
                if l_cmd.end is None:
                    continue
                p_current_segment_end = l_cmd.end

                kept_segments = clipping.subtract_regions_from_line_segment(
                    p_current_segment_start, p_current_segment_end, regions
                )
                for sub_p1, sub_p2 in kept_segments:
                    if pen_pos is None or math.dist(pen_pos, sub_p1) > 1e-6:
                        new_ops.move_to(*sub_p1)
                    new_ops.line_to(*sub_p2)
                    pen_pos = sub_p2
                p_current_segment_start = p_current_segment_end

            last_point = cmd.end

        self.commands = new_ops.commands
        # Update last_move_to to a valid point if ops is not empty
        if new_ops.commands:
            for cmd_rev in reversed(new_ops.commands):
                if isinstance(cmd_rev, MoveToCommand):
                    self.last_move_to = cmd_rev.end
                    break
        return self
