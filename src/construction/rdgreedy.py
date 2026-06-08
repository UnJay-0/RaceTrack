import math

from src.construction.heuristic import Heuristic
from src.scripts.greedy_with_DijkstraReversed import INF
from src.state import State, States
from src.track import FINISH, GRASS, OBSTACLE, Position, Track
from src.vector import Vector

MAX_STEPS_FACTOR = 100

GRASS_PENALTY = 1000.0
REVISIT_WEIGHT = 10.0
TURN_WEIGHT = 0.15
LOOKAHEAD_STEPS = 2
LOOKAHEAD_WEIGHT = 1

# Penalties fade to zero within this many reverse-Dijkstra steps of the finish
FINISH_FADE_RADIUS = 15
# Speed limit applied within FINISH_FADE_RADIUS of the finish.
# Set to None (or math.inf) to disable entirely.
FINISH_SPEED_LIMIT: float | None = 2.0
FINISH_SPEED_PENALTY = 1e4  # large enough to always lose to a legal move


class RdGreedy(Heuristic):
    def __init__(self, track: Track) -> None:
        super().__init__(track)

    # ── helpers ────────────────────────────────────────────────────────

    def _finish_attenuation(self, next_h: float) -> float:
        """
        Returns a factor in [0, 1].
        1.0  = far from finish  → full penalties
        0.0  = at finish        → no penalties
        Linearly interpolates within FINISH_FADE_RADIUS steps.
        """
        if next_h >= INF:
            return 1.0
        return min(1.0, next_h / FINISH_FADE_RADIUS)

    def move_cost(self, position0: Position, position1: Position) -> float:
        path = self.track.supercover_line(
            position0.x, position0.y, position1.x, position1.y
        )
        cost = 0.0
        for pos in path:
            cost += Heuristic.terrain_cost(pos)
        return cost

    def obstacle_proximity_penalty(self, pos: Position, speed: float = 1.0) -> float:
        height, width = self.track.get_boundaries()
        penalty = 0.0
        radius = max(1, round(speed))  # larger radius at higher speed
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > radius:
                    continue
                nx = pos.x + dx
                ny = pos.y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    penalty += 1.5 / dist
                elif self.track.get_position((nx, ny)).is_grass_or_obstacle():
                    penalty += 1.0 / dist  # closer obstacles hurt more
        return penalty

    def velocity_lookahead_penalty(
        self, lookahead_steps: int, pos: Position, vx: int, vy: int
    ) -> float:
        penalty = 0.0
        for t in range(1, lookahead_steps + 1):
            nx, ny = pos.x + vx * t, pos.y + vy * t
            ahead = self.track.get_position((nx, ny))
            if ahead.content == FINISH:
                break
            if ahead.content == OBSTACLE:
                penalty += (lookahead_steps - t + 1) * 2.0
                break
            if ahead.content == GRASS:
                penalty += lookahead_steps - t + 1
                break
        return penalty

    def _speed_limit_penalty(
        self,
        next_h: float,
        speed: float,
        speed_limit: float | None,
    ) -> float:
        """
        Returns a huge penalty if the speed exceeds `speed_limit`
        while within FINISH_FADE_RADIUS steps of the finish.
        Returns 0.0 if the speed limit is disabled (None) or the
        car is far from the finish.
        """
        if speed_limit is None:
            return 0.0
        if next_h >= INF or next_h > FINISH_FADE_RADIUS:
            return 0.0
        if int(speed) >= int(speed_limit):
            return FINISH_SPEED_PENALTY * (speed - speed_limit)
        return 0.0

    # ── main scoring ───────────────────────────────────────────────────

    def greedy_score(
        self,
        current_state: State,
        next_state: State,
        reverse_dist: dict[Position, int],
        visit_count: dict[Position, int],
        lookahead_steps: int,
        speed_limit: float | None = FINISH_SPEED_LIMIT,
        obstacles: bool = True,
    ) -> float:
        vx, vy = current_state.vector.get_point()
        nvx, nvy = next_state.vector.get_point()

        current_h = reverse_dist.get(current_state.position, INF)
        next_h = reverse_dist.get(next_state.position, INF)

        if next_h >= INF:
            return INF

        # Attenuation: 1.0 far away, 0.0 at the finish
        attenuation = self._finish_attenuation(next_h)

        score = next_h

        # Wall-avoidance penalties — attenuated near finish
        if obstacles:
            score += attenuation * self.obstacle_proximity_penalty(
                next_state.position, next_state.vector.magnitude - 2
            )
        score += (
            attenuation
            * LOOKAHEAD_WEIGHT
            * self.velocity_lookahead_penalty(
                lookahead_steps, next_state.position, nvx, nvy
            )
        )

        score += REVISIT_WEIGHT * visit_count.get(next_state.position, 0)

        if next_h >= current_h:
            score += next_h - current_h + 1.0

        if next_state.position.content == GRASS:
            score += GRASS_PENALTY * attenuation

        # Turn penalty — attenuated near finish
        turn_amount = abs(nvx - vx) + abs(nvy - vy)
        score += (
            attenuation * TURN_WEIGHT * turn_amount * current_state.vector.magnitude
        )
        score += self._speed_limit_penalty(
            next_h, next_state.vector.magnitude, speed_limit
        )

        return score

    # ── constructor ────────────────────────────────────────────────────

    def greedy_path_constructor(
        self,
        force_unit: bool = True,
        starting_vector: Vector = Vector(0, 0.0),
        blocked_states_to_add: set = set(),
        speed_limit: float | None = FINISH_SPEED_LIMIT,
        lookahead_steps: int = LOOKAHEAD_STEPS,
        obstacles: bool = True,
    ) -> States | None:
        state = State(self.track.start_pos, starting_vector)
        path = [state]
        visit_count = {state.position: 1}

        height, width = self.track.get_boundaries()
        max_steps = MAX_STEPS_FACTOR * height * width
        previous_state_index = -1
        blocked_states = set()
        blocked_states.union(blocked_states_to_add)
        for step in range(1, max_steps + 1):
            pos = state.position

            if pos in self.track.end_positions and state not in blocked_states:
                print("Finish reached")
                print("Steps:", step - 1)
                return States(path)

            candidates = []

            next_states = (
                state.generate_unit_moves(self.track)
                if force_unit
                else state.generate_moves(self.track)
            )

            for next_state in next_states:
                if next_state in blocked_states:
                    continue

                score = self.greedy_score(
                    state,
                    next_state,
                    self.reverse_dist,
                    visit_count,
                    lookahead_steps,
                    speed_limit,
                    obstacles,
                )
                if score >= INF:
                    continue
                candidates.append((score, next_state))

            if not candidates:
                if len(path) == 0:
                    print("Greedy stuck: no valid candidate")
                    print("Last state:", state)
                    return None
                blocked_states.add(state)
                state = path[previous_state_index]
                previous_state_index -= 1
                path = path[:-1]
                continue

            candidates.sort(key=lambda item: item[0])
            best_score, best_state = candidates[0]

            previous_state_index += 1
            state = best_state
            path.append(state)
            visit_count[state.position] = visit_count.get(state.position, 0) + 1

        print("Greedy stopped: maximum steps reached")
        return None
