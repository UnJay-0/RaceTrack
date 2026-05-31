import sys
import math
import heapq


def distance_to_finish(x, y, finishes):
    return min(
        math.sqrt((fx - x) ** 2 + (fy - y) ** 2)
        for fx, fy in finishes
    )


def obstacle_ahead_penalty(track, x, y, vx, vy):

    speed = max(abs(vx), abs(vy))

    if speed <= 0:
        return 0

    r = max(4, speed * 2)

    height = len(track)
    width = len(track[0])

    penalty = 0

    vel_len = math.sqrt(vx * vx + vy * vy)

    if vel_len == 0:
        return 0

    # normalize velocity direction
    velx = vx / vel_len
    vely = vy / vel_len

    # number of rays
    NUM_RAYS = 64

    for i in range(NUM_RAYS):

        angle = (2 * math.pi * i) / NUM_RAYS

        dirx = math.cos(angle)
        diry = math.sin(angle)

        # ignore rays too far behind
        forward = dirx * velx + diry * vely

        if forward < -0.2:
            continue

        # stronger weight for forward rays
        directional_weight = max(0, forward)

        # march along the ray
        for step in range(1, r + 1):

            nx = round(x + dirx * step)
            ny = round(y + diry * step)

            if (
                nx < 0 or ny < 0 or
                nx >= width or ny >= height
            ):
                break

            cell = get_cell(track, nx, ny)

            weight = directional_weight / (step + 1)

            if cell == 'O':

                penalty += 120 * weight

                # STOP ray after obstacle
                break

            elif cell == 'T':

                penalty -= 0.4 * weight

            elif cell == 'G':

                penalty -= 0.15 * weight

    return penalty


def wall_proximity_penalty(track, x, y):

    penalty = 0
    radius = 2

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):

            if dx == 0 and dy == 0:
                continue

            nx = x + dx
            ny = y + dy

            if (
                nx < 0 or ny < 0 or
                nx >= len(track[0]) or
                ny >= len(track)
            ):
                penalty += 3
                continue

            if get_cell(track, nx, ny) == 'O':
                penalty += 3

    return penalty


def read_track(filename):
    with open(filename, "r") as f:
        return [list(line.strip()) for line in f]


def find_positions(track: list[list[str]]):

    start = None
    finishes = []
    height = len(track)

    for row in range(height):
        for x in range(len(track[0])):

            cell = track[row][x]
            y = height - 1 - row

            if cell == 'S':
                start = (x, y)
            elif cell == 'F':
                finishes.append((x, y))

    return start, finishes


def get_cell(track, x, y):
    height = len(track)
    row = height - 1 - y
    return track[row][x]


def line_cells(x0, y0, x1, y1):

    cells = []

    dx = x1 - x0
    dy = y1 - y0

    steps = max(abs(dx), abs(dy))

    if steps == 0:
        return [(x0, y0)]

    for i in range(steps + 1):

        t = i / steps

        x = round(x0 + dx * t)
        y = round(y0 + dy * t)

        if (x, y) not in cells:
            cells.append((x, y))

    return cells


def valid_move(track, x0, y0, x1, y1):

    height = len(track)
    width = len(track[0])

    cells = line_cells(x0, y0, x1, y1)

    for i, (x, y) in enumerate(cells):

        if x < 0 or y < 0 or x >= width or y >= height:
            return False

        if get_cell(track, x, y) == 'O':
            return False

        if i > 0:

            px, py = cells[i - 1]
            dx = x - px
            dy = y - py

            if abs(dx) == 1 and abs(dy) == 1:

                if get_cell(track, px, y) == 'O' and get_cell(track, x, py) == 'O':
                    return False

    return True

def clamp(v):
    return max(-8, min(8, v))

def generate_moves(state):

    x, y, vx, vy = state

    moves = []

    MAX_SPEED = 8

    for ax in [-1, 0, 1]:
        for ay in [-1, 0, 1]:

            nvx = clamp(vx + ax)
            nvy = clamp(vy + ay)

            # limit speed
            if abs(nvx) > MAX_SPEED:
                continue

            if abs(nvy) > MAX_SPEED:
                continue

            old_speed = math.sqrt(vx * vx + vy * vy)

            # avoid violent reversals
            if old_speed > 1:

                dot = vx * nvx + vy * nvy

                # cosine-like directional check
                directional_ratio = dot / (
                    old_speed *
                    (math.sqrt(nvx * nvx + nvy * nvy) + 1e-6)
                )

                # forbid near-180 turns
                if directional_ratio < -0.3:
                    continue

            nx = x + nvx
            ny = y + nvy

            moves.append((nx, ny, nvx, nvy))

    return moves


def beam_search_step(
    beam: list[tuple],
    track: list[list[str]],
    finishes: list[tuple],
    beam_width: int,
    visited: dict[tuple, int],
    position_visits: dict
):
    candidates = []

    for state in beam:

        x, y, vx, vy = state

        for nx, ny, nvx, nvy in generate_moves(state):

            if not valid_move(track, x, y, nx, ny):
                continue

            new_state = (nx, ny, nvx, nvy)

            visit_penalty = position_visits.get((nx, ny), 0) * 8

            dist = distance_to_finish(nx, ny, finishes)
            speed = math.sqrt(nvx * nvx + nvy * nvy)

            turn_penalty = 0

            old_speed = math.sqrt(vx * vx + vy * vy)

            if old_speed > 0 and speed > 0:

                dot = vx * nvx + vy * nvy

                cosine = dot / (old_speed * speed + 1e-6)

                turn_penalty = 8 * (1 - cosine)

            obstacle_penalty = obstacle_ahead_penalty(track, nx, ny, nvx, nvy)
            wall_penalty = wall_proximity_penalty(track, nx, ny)

            final_bonus = -50 / (dist + 1e-6)

            completion_force = 10 if dist < 5 else 0

            total_score = (
                dist +
                0.15 * speed +
                obstacle_penalty +
                0.4 * wall_penalty +
                turn_penalty +
                final_bonus -
                completion_force +
                visit_penalty
            )

            Q = 2 # quantization factor to reduce state space

            new_state = (
                nx,
                ny,
                (nvx // Q) * Q,
                (nvy // Q) * Q
            )

            if new_state in visited and visited[new_state] <= total_score:
                continue

            visited[new_state] = total_score

            position_visits[(nx, ny)] = (
                position_visits.get((nx, ny), 0) + 1
            )

            print("STATE DEBUG:")
            print("pos:", nx, ny)
            print("dist:", dist)
            print("speed:", speed)
            print("obstacle:", obstacle_penalty)
            print("wall:", wall_penalty)
            print("turn:", turn_penalty)
            print("visit:", visit_penalty)
            print("TOTAL:", total_score)
            print("---")

            candidates.append((total_score, new_state, state))

    if not candidates:
        return []

    best = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

    new_beam = [state for _, state, _ in best]

    parents = {state: parent for _, state, parent in best}

    return new_beam, parents


def reconstruct(state, parent_map):

    path = []

    while state is not None:
        x, y, vx, vy = state
        path.append((x, y))
        state = parent_map[state]

    path.reverse()
    return path


def solve(track: list[list[str]]):

    start, finishes = find_positions(track)

    start_state = (start[0], start[1], 0, 0)

    beam = [start_state]
    beam_width = 100
    max_steps = 1000

    parent_map = {start_state: None}

    visited = {}
    visited[start_state] = 0

    position_visits = {}

    debug_file = open("debug_states.txt", "w")   # <<< MOVE HERE

    for step in range(max_steps):

        result = beam_search_step(
            beam,
            track,
            finishes,
            beam_width,
            visited,
            position_visits
        )

        debug_file.write(
            f"STEP {step} | beam_size={len(beam)} | visited={len(visited)}\n"
        )

        for s in beam[:10]:
            debug_file.write(f"{s}\n")

        debug_file.write("\n")
        debug_file.flush()

        if not result:
            debug_file.close()
            return reconstruct(beam[0], parent_map)

        beam, parents = result
        parent_map.update(parents)

        for x, y, vx, vy in beam:
            if (x, y) in finishes:
                debug_file.close()
                return reconstruct((x, y, vx, vy), parent_map)

    debug_file.close()
    return reconstruct(beam[0], parent_map)


def write_csv(path, filename):

    with open(filename, "w") as f:
        for x, y in path:
            f.write(f"{x},{y}\n")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python beam_search.py track.t")
        sys.exit(1)

    track_file = sys.argv[1]

    track = read_track(track_file)
    path = solve(track)

    output_file = track_file.replace(".t", "_trip.csv")

    write_csv(path, output_file)

    print("Trip written to:", output_file)
