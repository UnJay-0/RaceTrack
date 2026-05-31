from __future__ import annotations

from functools import total_ordering

from numpy import sign

from src.vector import Vector

TRACKS_PATH = ""
GRASS = "G"
TRACK = "T"
OBSTACLE = "O"
START = "S"
FINISH = "F"


class Track:
    def __init__(self, filename: str) -> None:
        self.positions = []
        self.start_index = (0, 0)
        self.end_indexes = []
        with open(filename, "r") as f:
            x = 0
            y = 0
            for line in f:
                line_content = []
                for content in line.strip():
                    if content == START:
                        self.start_index = (x, y)
                    if content == FINISH:
                        self.end_indexes.append((x, y))
                    line_content.append(Position(x, y, content))
                    x += 1
                self.positions.append(line_content)
                x = 0
                y += 1
        # print(self.start_index)

    def get_boundaries(self) -> tuple[int, int]:
        return (len(self.positions), len(self.positions[0]))

    def get_position(self, coordinates: tuple[int, int]) -> Position:
        return self.positions[coordinates[1]][coordinates[0]]

    def is_valid_move(
        self, position: Position, next_position: Position, current: Vector, next: Vector
    ) -> bool:
        # print(f"Current position: {position}")
        # print(f"Next position: {next_position}")
        vx_prime, vy_prime = current.get_point()
        vx_second, vy_second = next.get_point()
        vx_second += vx_prime
        vy_second += vy_prime

        if position.content == OBSTACLE:
            return False

        # # Track boundaries check
        # if self.boundaries_check(ax, ay):
        #     print("boundaries")
        #     return False

        # arrival on an obstacle
        if self.get_position((next_position.x, next_position.y)).content == OBSTACLE:
            return False

        # Speed check
        if position.content == GRASS:
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
        x = x0
        y = y0
        positions.append(self.get_position((x, y)))

        ix = 0
        iy = 0

        while ix < nx or iy < ny:
            if (1 + 2 * ix) * ny == (1 + 2 * iy) * nx:
                # exact corner crossing
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


@total_ordering
class Position:
    def __init__(self, x: int, y: int, content: str = "") -> None:
        self.x = x
        self.y = y
        self.content = content

    def get_coordinates(self) -> tuple[int, int]:
        return (self.x, self.y)

    def add_vector(self, vx: int, vy: int) -> tuple[int, int]:
        return (self.x + vx, self.y + vy)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.content))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)

    def __eq__(self, other: object, /) -> bool:
        if other is self:
            return True

        if type(other) is not Position:
            return NotImplemented

        return (self.x, self.y) == (other.x, other.y) and self.content == other.content

    def __repr__(self) -> str:
        return f"[{self.x} - {self.y} - {self.content}]"

    def __str__(self) -> str:
        return self.__repr__()


def generate_square_edges(
    p: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [
        ((p[0] - 0.5, p[1] + 0.5), (p[0] + 0.5, p[1] + 0.5)),
        ((p[0] - 0.5, p[1] - 0.5), (p[0] + 0.5, p[1] - 0.5)),
        ((p[0] - 0.5, p[1] - 0.5), (p[0] - 0.5, p[1] + 0.5)),
        ((p[0] + 0.5, p[1] - 0.5), (p[0] + 0.5, p[1] + 0.5)),
    ]


def intersect(
    p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int], p4: tuple[int, int]
) -> bool:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:  # parallel
        return False
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:  # out of range
        return False
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:  # out of range
        return False
    # x = x1 + ua * (x2-x1)
    # y = y1 + ua * (y2-y1)
    return True
