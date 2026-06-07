from __future__ import annotations

from functools import total_ordering

from src.track import Position, Track
from src.vector import Vector


@total_ordering
class State:
    def __init__(self, position: Position, vector: Vector):
        self.position = position
        self.vector = vector

    def generate_moves(self, track: Track) -> States:
        next_states = States()
        next_vectors_points = self.vector.adjacent_vectors()
        for vector_point in next_vectors_points.tolist():
            new_coordinates = self.position.add_vector(vector_point[0], vector_point[1])
            if track.boundaries_check(new_coordinates[0], new_coordinates[1]):
                continue
            new_position = track.get_position(new_coordinates)
            new_vector = Vector.get_vector(
                self.position.get_coordinates(), new_position.get_coordinates()
            )
            if track.is_valid_move(
                self.position, new_position, self.vector, new_vector
            ):
                next_states.append(State(new_position, new_vector))
        return next_states

    def generate_unit_moves(self, track: Track) -> States:
        next_states = States()
        next_vectors_points = self.vector.adjacent_vectors()
        for vector_point in next_vectors_points.tolist():
            new_coordinates = self.position.add_vector(vector_point[0], vector_point[1])
            if track.boundaries_check(new_coordinates[0], new_coordinates[1]):
                continue
            new_position = track.get_position(new_coordinates)
            new_vector = Vector.get_vector(
                self.position.get_coordinates(), new_position.get_coordinates()
            )
            if int(new_vector.magnitude) != 1:
                continue

            vx, vy = self.vector.get_point()
            nvx, nvy = new_vector.get_point()
            if vx != 0 or vy != 0:
                dot = nvx * vx + nvy * vy
                if dot < 0:
                    continue

            if track.is_valid_move(
                self.position, new_position, self.vector, new_vector
            ):
                next_states.append(State(new_position, new_vector))
        return next_states

    def __hash__(self) -> int:
        return hash((self.position, self.vector))

    def __eq__(self, other: object, /) -> bool:
        if other is self:
            return True
        if not isinstance(other, State):
            return NotImplemented
        return self.position == other.position and self.vector == other.vector

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self.position < other.position

    def __repr__(self) -> str:
        return f"{self.position} - {self.vector}"

    def __str__(self) -> str:
        return self.__repr__()


class States:
    def __init__(self, states: list[State] = []):
        self.states = []
        self.states.extend(states)

    def append(self, state: State):
        self.states.append(state)

    def concat(self, other: States) -> States:
        return States(self.states + other.states)

    def get_positions(self) -> list[Position]:
        return [state.position for state in self]

    def __getitem__(self, arg):
        return self.states[arg]

    def __iter__(self) -> State:
        for state in self.states:
            yield state

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, States):
            return NotImplemented
        if len(self.states) != len(other.states):
            return False
        # return all(s == o for s, o in zip(self.states, other.states))
        # order-independent alternative (only if needed)
        return set(self.states) == set(other.states)

    def __hash__(self) -> int:
        return hash(tuple(self.states))  # requires State to be hashable, which it is

    def __len__(self):
        return len(self.states)

    def __str__(self) -> str:
        output = ""
        i = 0
        for state in self:
            output += str(i) + " - " + str(state) + "\n"
            i += 1
        return output
