from src.state import State, States
from src.track import GRASS, Position, Track


class Heuristic:
    def __init__(self) -> None:
        pass

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

    def apply(track: Track) -> States:
        pass
