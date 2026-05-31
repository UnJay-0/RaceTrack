from __future__ import annotations

import math

import numpy as np

EPS = 1e-6


class Vector:
    def __init__(self, magnitude: int, direction: float) -> None:
        self.magnitude = magnitude
        self.direction = direction

    @classmethod
    def from_point(cls, x, y) -> Vector:
        magnitude = math.hypot(x, y)  # sqrt(x² + y²)
        angle = math.atan2(y, x)  # angle in radians
        return cls(magnitude, angle)

    def get_point(self) -> tuple[int, int]:
        return (
            round(self.magnitude * math.cos(self.direction)),
            round(self.magnitude * math.sin(self.direction)),
        )

    def adjacent_vectors(self) -> np.array:
        v = np.array(
            [
                round(self.magnitude * math.cos(self.direction)),
                round(self.magnitude * math.sin(self.direction)),
            ]
        )
        d = np.array([-1, 0, 1])
        deltas = np.array(np.meshgrid(d, d)).T.reshape(-1, 2)  # shape (9, 2)
        return v + deltas  # shape (9, 2) — all neighbors in one step

    @staticmethod
    def get_vector(
        start_position: tuple[int, int], end_position: tuple[int, int]
    ) -> Vector:
        vx = end_position[0] - start_position[0]
        vy = end_position[1] - start_position[1]
        return Vector.from_point(vx, vy)

    def __eq__(self, other: object, /) -> bool:
        if other is self:
            return True
        if not isinstance(other, Vector):
            return NotImplemented
        return (
            self.magnitude == other.magnitude
            and self.direction - EPS >= other.direction
            and self.direction + EPS <= other.direction
        )

    def __hash__(self) -> int:
        return hash((self.magnitude, self.direction))

    def __repr__(self) -> str:
        vx, vy = self.get_point()
        return f"[{vx} - {vy}]"

    def __str__(self) -> str:
        return self.__repr__()
