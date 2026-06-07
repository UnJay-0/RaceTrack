from __future__ import annotations

import math
import string
import sys
from collections import defaultdict
from functools import total_ordering

from numpy import sign

from src.utils import UnionFind, generate_square_edges, intersect

TRACKS_PATH = ""
GRASS = "G"
TRACK = "T"
OBSTACLE = "O"
START = "S"
FINISH = "F"
OUT_OF_BOUND = "OUT_OF_BOUND"
CORNER_GROUPING_THRESHOLD = 2
GATE_LENGTH = 5

# Symbols used to mark corners and apexes in the written track file.
# 26 uppercase letters for corner regions, digits 0-9 for apex cells.
CORNER_SYMBOLS = list(string.ascii_uppercase)
CORNER_SYMBOLS.remove("G")
CORNER_SYMBOLS.remove("T")
CORNER_SYMBOLS.remove("S")
CORNER_SYMBOLS.remove("F")
APEX_SYMBOLS = list(string.digits)


class Track:
    def __init__(self, filename: str) -> None:
        self.positions = []
        self.start_pos = Position(-1, -1, OUT_OF_BOUND)
        self.end_positions = []
        with open(filename, "r") as f:
            x = 0
            y = 0
            for line in f:
                line_content = []
                for content in line.strip():
                    position = Position(x, y, content)
                    line_content.append(position)
                    if content == START:
                        self.start_pos = position
                    if content == FINISH:
                        self.end_positions.append(position)
                    x += 1
                self.positions.append(line_content)
                x = 0
                y += 1

    @classmethod
    def from_positions(
        cls,
        positions: list[list[Position]],
        start_pos: Position | None = None,
        end_positions: list[Position] | None = None,
    ) -> Track:
        obj = cls.__new__(cls)
        obj.positions = positions
        obj.start_pos = (
            start_pos if start_pos is not None else Position(-1, -1, OUT_OF_BOUND)
        )
        obj.end_positions = end_positions if end_positions is not None else []
        return obj

    def deepcopy(self) -> Track:
        position_map: dict[tuple[int, int], Position] = {}
        new_positions: list[list[Position]] = []

        for row in self.positions:
            new_row = []
            for pos in row:
                copied = pos.deepcopy()
                position_map[(pos.x, pos.y)] = copied
                new_row.append(copied)
            new_positions.append(new_row)

        new_start = position_map.get(
            (self.start_pos.x, self.start_pos.y),
            self.start_pos.deepcopy(),
        )
        new_ends = [
            position_map.get((pos.x, pos.y), pos.deepcopy())
            for pos in self.end_positions
        ]

        return Track.from_positions(new_positions, new_start, new_ends)

    def get_boundaries(self) -> tuple[int, int]:
        return (len(self.positions), len(self.positions[0]))

    def get_position(self, coordinates: tuple[int, int]) -> Position:
        if self.boundaries_check(coordinates[0], coordinates[1]):
            return Position(-1, -1, OUT_OF_BOUND)
        return self.positions[coordinates[1]][coordinates[0]]

    def change_start(self, new_start: Position):
        # self.start_pos.content = TRACK
        self.start_pos = new_start
        # self.start_pos.content = START

    def change_finish_positions(self, new_finish_pos: list[Position]):
        for pos in self.end_positions:
            pos.content = TRACK
        self.end_positions: list[Position] = []
        for pos in new_finish_pos:
            # finish_pos = pos.deepcopy()
            pos.content = FINISH
            self.end_positions.append(pos)

    def is_valid_move(
        self, position: Position, next_position: Position, current, next
    ) -> bool:
        vx_prime, vy_prime = current.get_point()
        vx_second, vy_second = next.get_point()
        vx_second += vx_prime
        vy_second += vy_prime

        if position.content == OBSTACLE:
            return False
        if self.get_position((next_position.x, next_position.y)).content == OBSTACLE:
            return False

        if position.content == GRASS and next.magnitude > 1:
            if abs(vx_prime) >= 2 and not (vx_second - vx_prime) * sign(vx_prime) < 0:
                return False
            if abs(vy_prime) >= 2 and not (vy_second - vy_prime) * sign(vy_prime) < 0:
                return False
            if abs(vx_prime) == 1 and not (vx_second - vx_prime) * sign(vx_prime) <= 0:
                return False
            if abs(vy_prime) == 1 and not (vy_second - vy_prime) * sign(vy_prime) <= 0:
                return False

        return self.obstacles_check(
            position.x, position.y, next_position.x, next_position.y
        )

    def boundaries_check(self, x: int, y: int) -> bool:
        return x < 0 or y < 0 or x >= len(self.positions[0]) or y >= len(self.positions)

    def obstacles_check(self, x: int, y: int, nx: int, ny: int) -> bool:
        positions = self.supercover_line(x, y, nx, ny)
        for position in positions:
            if position.content == OBSTACLE:
                return False
            else:
                sign_x = 1 if (x - nx) > 0 else -1
                sign_y = 1 if (y - ny) > 0 else -1
                adjacent_positions = [
                    self.get_position((position.x + sign_x, position.y)),
                    self.get_position((position.x, position.y + sign_y)),
                ]
                for adjacent_pos in adjacent_positions:
                    if adjacent_pos.content == OBSTACLE:
                        edges = generate_square_edges((adjacent_pos.x, adjacent_pos.y))
                        for edge in edges:
                            if intersect((x, y), (nx, ny), edge[0], edge[1]):
                                return False
        return True

    def supercover_line(self, x0, y0, x1, y1):
        positions = []
        dx = x1 - x0
        dy = y1 - y0
        nx = abs(dx)
        ny = abs(dy)
        sign_x = 1 if dx > 0 else -1
        sign_y = 1 if dy > 0 else -1
        x = int(x0)
        y = int(y0)
        positions.append(self.get_position((x, y)))
        ix = 0
        iy = 0
        while ix < nx or iy < ny:
            if (1 + 2 * ix) * ny == (1 + 2 * iy) * nx:
                x += sign_x
                y += sign_y
                ix += 1
                iy += 1
            elif (1 + 2 * ix) * ny < (1 + 2 * iy) * nx:
                x += sign_x
                ix += 1
            else:
                y += sign_y
                iy += 1
            positions.append(self.get_position((x, y)))
        return positions

    @staticmethod
    def farthest_point_sample(
        positions: list[Position], n: int, on_track: bool = True
    ) -> list[Position]:
        """
        Returns n positions from `positions` that are as equally
        distributed by Euclidean distance as possible, using
        farthest-point (maximin) sampling. O(n * |positions|)
        """
        positions = [
            position for position in positions if not position.is_grass_or_obstacle()
        ]
        if n <= 0 or not positions:
            return []
        if n >= len(positions):
            return list(positions)

        # Start from the position closest to the centroid
        # for a stable, geometry-aware seed
        cx = sum(p.x for p in positions) / len(positions)
        cy = sum(p.y for p in positions) / len(positions)
        seed = min(positions, key=lambda p: (p.x - cx) ** 2 + (p.y - cy) ** 2)

        selected = [seed]
        # min_dist[i] = distance from positions[i] to its nearest selected point
        min_dist = [p.get_distance_to(seed) for p in positions]

        for _ in range(n - 1):
            # Pick the candidate farthest from all selected points
            farthest = max(range(len(positions)), key=lambda i: min_dist[i])
            new_point = positions[farthest]
            selected.append(new_point)

            # Update min distances with the new point
            for i, p in enumerate(positions):
                d = p.get_distance_to(new_point)
                if d < min_dist[i]:
                    min_dist[i] = d

        return selected

    def get_corners(self) -> list[Position]:
        corner_positions = []
        for line in self.positions:
            for pos in line:
                if pos.is_grass_or_obstacle() or pos.content == FINISH:
                    continue

                north = self.get_position((pos.x, pos.y + 1))
                south = self.get_position((pos.x, pos.y - 1))
                east = self.get_position((pos.x + 1, pos.y))
                west = self.get_position((pos.x - 1, pos.y))

                n_blocked = north.is_grass_or_obstacle()
                s_blocked = south.is_grass_or_obstacle()
                e_blocked = east.is_grass_or_obstacle()
                w_blocked = west.is_grass_or_obstacle()

                count = sum([n_blocked, s_blocked, e_blocked, w_blocked])

                if count == 2:
                    opposite = (n_blocked and s_blocked) or (e_blocked and w_blocked)
                    if not opposite:
                        corner_positions.append(pos)

                elif count == 1:
                    neighborhood = []
                    if n_blocked or s_blocked:
                        neighborhood = (
                            north if n_blocked else south
                        ).get_lateral_neighborhood(self, True)
                    elif w_blocked or e_blocked:
                        neighborhood = (
                            west if w_blocked else east
                        ).get_lateral_neighborhood(self, False)

                    blocked_count = sum(
                        int(p.is_grass_or_obstacle()) for p in neighborhood
                    )
                    if blocked_count <= 1:
                        corner_positions.append(pos)

        return corner_positions

    def write_track(self, filename: str = "track_test.t"):
        with open(filename, "w") as f:
            for line in self.positions:
                for pos in line:
                    f.write(pos.content)
                f.write("\n")

    def write_track_with_corners(
        self, corners: list[Corner], filename: str = "track_corners.t"
    ):
        symbol_map: dict[tuple[int, int], str] = {}

        print("=" * 50)
        print("Corner legend")
        print("=" * 50)

        for idx, corner in enumerate(corners):
            body_symbol = CORNER_SYMBOLS[idx % len(CORNER_SYMBOLS)]
            apex_symbol = APEX_SYMBOLS[idx % len(APEX_SYMBOLS)]
            apex = corner.get_apex()

            print(
                f"  Region {idx + 1:>2}  body='{body_symbol}'  apex='{apex_symbol}'  "
                f"apex_pos={apex}  size={len(corner.positions)}"
            )

            for pos in corner.positions:
                symbol_map[(pos.x, pos.y)] = body_symbol
            symbol_map[(apex.x, apex.y)] = apex_symbol

        print("=" * 50)
        print(f"Track written to: {filename}\n")

        with open(filename, "w") as f:
            for line in self.positions:
                for pos in line:
                    symbol = symbol_map.get((pos.x, pos.y))
                    f.write(symbol if symbol is not None else pos.content)
                f.write("\n")

    def __str__(self) -> str:
        output = ""
        for line in self.positions:
            for pos in line:
                output += pos.content
            output += "\n"
        return output

    def __repr__(self) -> str:
        return self.__str__()


class Corner:
    def __init__(self, positions: list[Position], gate_length: int = 5):
        self.positions = positions
        self.corner_gates = self.get_corner_gates(gate_length)

    def append(self, position: Position):
        self.positions.append(position)

    def get_apex(self) -> Position:
        cx = sum(p.x for p in self.positions) / len(self.positions)
        cy = sum(p.y for p in self.positions) / len(self.positions)
        return min(self.positions, key=lambda p: (p.x - cx) ** 2 + (p.y - cy) ** 2)

    def is_relevant(self, path: list[Position]) -> bool:
        return self.path_crossed_gate(path)

    # def get_corner_gates(self, track_width: int = 5):
    #     gate_pos_1 = self.positions[0]
    #     gate_pos_2 = self.positions[-1]

    #     ax = gate_pos_2.x - gate_pos_1.x
    #     ay = gate_pos_2.y - gate_pos_1.y
    #     length = math.sqrt(ax**2 + ay**2) or 1

    #     px, py = -ay / length, ax / length

    #     half = track_width / 2
    #     gate_entry = (
    #         (gate_pos_1.x - px * half, gate_pos_1.y - py * half),
    #         (gate_pos_1.x + px * half, gate_pos_1.y + py * half),
    #     )
    #     gate_exit = (
    #         (gate_pos_2.x - px * half, gate_pos_2.y - py * half),
    #         (gate_pos_2.x + px * half, gate_pos_2.y + py * half),
    #     )
    #     return gate_entry, gate_exit
    def get_corner_gates(self, track_width: int = 5):
        gate_pos_1 = self.positions[0]
        gate_pos_2 = self.positions[-1]
        apex = self.get_apex()

        def _make_gate(
            anchor: Position, toward: Position, half: float
        ) -> tuple[tuple[float, float], tuple[float, float]]:
            """
            Gate perpendicular to the vector anchor → toward,
            centred on anchor.
            """
            ax = toward.x - anchor.x
            ay = toward.y - anchor.y
            length = math.sqrt(ax**2 + ay**2) or 1
            # Perpendicular to the forward direction
            px, py = -ay / length, ax / length
            return (
                (anchor.x - px * half, anchor.y - py * half),
                (anchor.x + px * half, anchor.y + py * half),
            )

        half = track_width / 2
        gate_entry = _make_gate(gate_pos_1, apex, half)
        gate_exit = _make_gate(gate_pos_2, apex, half)
        return gate_entry, gate_exit

    def get_crossed_gate(
        self, pos_1: Position, pos_2: Position
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        for gate in self.corner_gates:
            if self.step_crossed_gate(pos_1, pos_2, gate):
                return gate
        return None

    def step_crossed_gate(
        self,
        pos_1: Position,
        pos_2: Position,
        gate: tuple[tuple[float, float], tuple[float, float]],
    ) -> bool:
        if intersect(
            pos_1.get_coordinates(),
            pos_2.get_coordinates(),
            gate[0],
            gate[1],
        ):
            return True
        return False

    def path_crossed_gate(self, path: list[Position]) -> bool:
        crossed_1 = False
        crossed_2 = False
        for i in range(len(path) - 1):
            if self.step_crossed_gate(
                path[i],
                path[i + 1],
                self.corner_gates[0],
            ):
                crossed_1 = True
            elif self.step_crossed_gate(
                path[i],
                path[i + 1],
                self.corner_gates[1],
            ):
                crossed_2 = True
        return crossed_1 and crossed_2

    @staticmethod
    def group_corner_positions(
        positions: list[Position],
        threshold: float = CORNER_GROUPING_THRESHOLD,
        gate_legth: int = GATE_LENGTH,
    ) -> list[Corner]:
        n = len(positions)
        uf = UnionFind(n)

        for i in range(n):
            for j in range(i + 1, n):
                if positions[i].get_distance_to(positions[j]) <= threshold:
                    uf.union(i, j)

        groups: dict[int, list[Position]] = defaultdict(list)
        for i, pos in enumerate(positions):
            groups[uf.find(i)].append(pos)

        return [Corner(group, gate_legth) for group in groups.values()]

    def __repr__(self) -> str:
        return self.positions.__str__()

    def __str__(self) -> str:
        return self.__repr__()


@total_ordering
class Position:
    def __init__(self, x: int, y: int, content: str = "") -> None:
        self.x = x
        self.y = y
        self.content = content

    def deepcopy(self) -> Position:
        return Position(self.x, self.y, self.content)

    def get_diagonal_neighborhood(
        self, track: Track, first_diagonal: bool
    ) -> list[Position]:
        if first_diagonal:
            return [
                track.get_position((self.x - 1, self.y - 1)),
                track.get_position((self.x + 1, self.y + 1)),
            ]
        else:
            return [
                track.get_position((self.x + 1, self.y - 1)),
                track.get_position((self.x - 1, self.y + 1)),
            ]

    def get_lateral_neighborhood(
        self, track: Track, horizontal: bool
    ) -> list[Position]:
        if horizontal:
            return [
                track.get_position((self.x - 1, self.y)),
                track.get_position((self.x + 1, self.y)),
            ]
        else:
            return [
                track.get_position((self.x, self.y - 1)),
                track.get_position((self.x, self.y + 1)),
            ]

    def get_neighborhood(self, track: Track) -> list[Position]:
        neighborhood = self.get_lateral_neighborhood(track, False)
        neighborhood.extend(self.get_lateral_neighborhood(track, True))
        neighborhood.extend(self.get_diagonal_neighborhood(track, False))
        neighborhood.extend(self.get_diagonal_neighborhood(track, True))
        return neighborhood

    def is_grass_or_obstacle(self) -> bool:
        return self.content == OBSTACLE or self.content == GRASS

    def is_finish(self) -> bool:
        return self.content == FINISH

    def get_coordinates(self) -> tuple[int, int]:
        return (self.x, self.y)

    def add_vector(self, vx: int, vy: int) -> tuple[int, int]:
        return (self.x + vx, self.y + vy)

    def get_distance_to(self, other: "Position") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.content))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if type(other) is not Position:
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y) and self.content == other.content

    def __repr__(self) -> str:
        return f"[{self.x} - {self.y} - {self.content}]"

    def __str__(self) -> str:
        return self.__repr__()


if __name__ == "__main__":
    track_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "track_corners.t"

    track = Track(track_file)
    corners = Corner.group_corner_positions(
        track.get_corners(), CORNER_GROUPING_THRESHOLD
    )

    corners = [corner for corner in corners if len(corner.positions) > 1]

    corners.sort(
        key=lambda c: (
            sum(p.y for p in c.positions) / len(c.positions),
            sum(p.x for p in c.positions) / len(c.positions),
        )
    )

    track.write_track_with_corners(corners, filename=output_file)