import heapq

from src.construction.heuristic import INF, Heuristic
from src.state import State, States
from src.track import Track
from src.vector import Vector

MAX_SPEED = 8
WEIGHT = 2.0


class Rdasw(Heuristic):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def heuristic(state: State, reverse_dist) -> float:
        h = reverse_dist.get(state.position, INF)
        # small momentum reward
        h -= 0.15 * state.vector.magnitude
        return h

    @staticmethod
    def move_cost(track: Track, x0: int, y0: int, x1: int, y1: int):
        path = track.supercover_line(x0, y0, x1, y1)
        cost = 0.0
        for pos in path:
            cost += Heuristic.terrain_cost(pos)
        return cost

    @staticmethod
    def apply(track: Track) -> States:
        start = track.start_pos
        finishes = track.end_positions

        if start is None:
            raise Exception("No start found")

        if not finishes:
            raise Exception("No finish found")

        print("Building reverse Dijkstra heuristic...")

        reverse_dist = Heuristic.reverse_dijkstra(track)

        print("Heuristic map built")
        # print(reverse_dist)

        start_state = State(start, Vector(0, 0.0))

        open_set = []

        g_score = {start_state: 0}

        parent = {start_state: None}

        visited = set()

        h0 = Rdasw.heuristic(start_state, reverse_dist)

        heapq.heappush(open_set, (WEIGHT * h0, start_state))

        expanded = 0

        while open_set:
            _, state = heapq.heappop(open_set)

            if state in visited:
                continue

            visited.add(state)

            expanded += 1

            if expanded % 10000 == 0:
                print("Expanded:", expanded, "Open:", len(open_set))

            x = state.position.x
            y = state.position.y

            # print(f"current state: {state}")

            # GOAL
            if state.position in finishes:
                print("Finish reached")
                print("Expanded nodes:", expanded)

                return Heuristic.reconstruct(state, parent)

            for next_state in state.generate_moves(track):
                # print(f"evaluating: {next_state}")
                nx = next_state.position.x
                ny = next_state.position.y

                tentative_g = g_score[state] + Rdasw.move_cost(track, x, y, nx, ny)

                if next_state not in g_score or tentative_g < g_score[next_state]:
                    # print(f"confirming {next_state} for {state}")
                    g_score[next_state] = tentative_g

                    parent[next_state] = state

                    h = Rdasw.heuristic(next_state, reverse_dist)

                    fscore = tentative_g + WEIGHT * h

                    heapq.heappush(open_set, (fscore, next_state))

        print("No solution found")

        return States()
