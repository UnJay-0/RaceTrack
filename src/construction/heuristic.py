import heapq

from src.state import State, States
from src.track import GRASS, OBSTACLE, Position, Track
from src.vector import Vector

INF = 10**9


class Heuristic:
    def __init__(self, track: Track) -> None:
        self.track = track
        self.reverse_dist = Heuristic.reverse_dijkstra(self.track)

    @staticmethod
    def terrain_cost(position: Position) -> float:
        return 2.0 if position.content == GRASS else 1.0

    @staticmethod
    def reconstruct(state: State, parent_map: dict[State, State]) -> list[Position]:
        path = []
        while state is not None:
            path.append(state)
            state = parent_map[state]
        path.reverse()
        return path

    @staticmethod
    def reverse_dijkstra(track: Track) -> dict[Position, int]:
        dist = {}
        for line in track.positions:
            for pos in line:
                if pos.content != OBSTACLE:
                    dist[pos] = INF

        heap = []

        for pos in track.end_positions:
            dist[pos] = 0
            heapq.heappush(heap, (0, State(pos, Vector(0, 0))))

        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        while heap:
            d, state = heapq.heappop(heap)
            pos = state.position
            if d != dist[pos]:
                continue
            x = pos.x
            y = pos.y
            for dx, dy in directions:
                nx = x + dx
                ny = y + dy
                next_pos = track.get_position((nx, ny))
                next_vector = Vector.get_vector((x, y), (nx, ny))
                if track.boundaries_check(nx, ny):
                    continue
                if not track.is_valid_move(pos, next_pos, state.vector, next_vector):
                    continue
                move_cost = 1.4 if dx != 0 and dy != 0 else 1.0
                nd = d + move_cost
                if nd < dist[next_pos]:
                    dist[next_pos] = nd
                    heapq.heappush(heap, (nd, State(next_pos, next_vector)))
        return dist

    def apply(track: Track) -> States:
        pass
