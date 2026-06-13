import math

from src.state import State, States
from src.track import GRASS, Position, Track
from src.utils import intersect

W1, W2, W3 = 0.4, 0.3, 0.3  # corner score weights
ALPHA = 0.5  # apex attraction
BETA = 2.0  # exit gate attraction
GAMMA = 0.2  # speed reward
BEAM_WIDTH = 40
MAX_STEPS = 50
GRASS_PENALTY = 1e4


def point_to_segment_dist(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Minimum distance from point (px,py) to segment (a,b)."""
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def evaluate_corner(
    path: list[State],
    apex: Position,
) -> float:
    """Corner quality score Q in [0,1]."""
    if len(path) < 2:
        return 0.0

    positions = [s.position for s in path]

    # 1. Apex distance error
    d_apex = min(p.get_distance_to(apex) for p in positions)

    # 2. Chord length vs straight-line minimum
    # Chord length
    L = sum(
        positions[i].get_distance_to(positions[i + 1])
        for i in range(len(positions) - 1)
    )
    # straight-line minimum
    L_min = positions[0].get_distance_to(positions[-1])

    # 3. Speed retention
    v_entry = path[0].vector.magnitude
    v_exit = path[-1].vector.magnitude
    v_ratio = (v_exit / v_entry) if v_entry > 0 else 0.0

    Q = W1 * (1.0 / (1.0 + d_apex)) + W2 * (L_min / L if L > 0 else 0.0) + W3 * v_ratio
    return Q


def step_heuristic(
    state: State,
    apex: Position,
    exit_gate: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Lower is better — guides beam toward apex then exit."""
    (gx0, gy0), (gx1, gy1) = exit_gate
    d_apex = state.position.get_distance_to(apex)
    d_gate = point_to_segment_dist(
        state.position.x, state.position.y, gx0, gy0, gx1, gy1
    )
    speed = state.vector.magnitude or 1e-9
    return ALPHA * d_apex + BETA * d_gate + GAMMA / speed


def crossed_gate(
    pos1: Position,
    pos2: Position,
    gate: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    return intersect(
        pos1.get_coordinates(),
        pos2.get_coordinates(),
        gate[0],
        gate[1],
    )


def optimize_corner_path(
    track: Track,
    entry_state: State,
    apex: Position,
    exit_gate: tuple[tuple[float, float], tuple[float, float]],
    beam_width: int = BEAM_WIDTH,
    max_steps: int = MAX_STEPS,
) -> States | None:
    """
    Beam search that finds the path from entry_state to exit_gate
    that maximises the corner score Q.

    Each beam entry is:  (heuristic_cost, path_so_far)
    """
    # beam: list of (cumulative_heuristic, path)
    beam = [(0.0, [entry_state])]
    completed = []  # paths that crossed the exit gate

    for _ in range(max_steps):
        candidates = []

        for cum_h, path in beam:
            current = path[-1]

            # Generate all valid next states
            for next_state in current.generate_moves(track):
                if track.boundaries_check(next_state.position.x, next_state.position.y):
                    continue
                if not track.is_valid_move(
                    current.position,
                    next_state.position,
                    current.vector,
                    next_state.vector,
                ):
                    continue

                # Check exit gate crossing
                if crossed_gate(current.position, next_state.position, exit_gate):
                    if crossed_gate(current.position, next_state.position, exit_gate):
                        # Only accept if the exit state has viable continuations
                        if (
                            len(next_state.generate_moves(track)) >= 6
                            and next_state.vector.magnitude <= 5
                        ):
                            # print(
                            #     f"\n{next_state} - {len(next_state.generate_moves(track))} - \n {next_state.generate_moves(track)}"
                            # )
                            completed.append(path + [next_state])
                        continue

                grass_cost = (
                    GRASS_PENALTY if next_state.position.content == GRASS else 0.0
                )
                dead_end_cost = 1e4 if len(next_state.generate_moves(track)) < 3 else 0
                h = (
                    step_heuristic(next_state, apex, exit_gate)
                    + grass_cost
                    + dead_end_cost
                )
                candidates.append((cum_h + h, path + [next_state]))

        if completed:
            # Among all completed paths pick the one with highest Q
            return States(max(completed, key=lambda p: evaluate_corner(p, apex)))

        if not candidates:
            break

        # Keep only the beam_width best partial paths
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    # No path reached the exit gate → return best non-dead-end partial path
    if beam:
        viable = [path for _, path in beam if len(path[-1].generate_moves(track)) >= 6]
        candidates_to_rank = viable if viable else [path for _, path in beam]
        return States(max(candidates_to_rank, key=lambda p: evaluate_corner(p, apex)))
    return None
