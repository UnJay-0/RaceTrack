import sys
import math
import heapq


INF = 10**9
MAX_STEPS_FACTOR = 100
REVISIT_WEIGHT = 10.0
TURN_WEIGHT = 0.15
DIAGONAL_WEIGHT = 0.05


def read_track(filename):

    with open(filename, "r") as f:
        return [list(line.strip()) for line in f]


def get_cell(track, x, y):

    height = len(track)

    row = height - 1 - y

    return track[row][x]


def find_positions(track):

    start = None
    finishes = []

    height = len(track)
    width = len(track[0])

    for row in range(height):

        for x in range(width):

            cell = track[row][x]

            y = height - 1 - row

            if cell == 'S':
                start = (x, y)

            elif cell == 'F':
                finishes.append((x, y))

    return start, finishes


def supercover_line(x0, y0, x1, y1):

    cells = []

    dx = x1 - x0
    dy = y1 - y0

    nx = abs(dx)
    ny = abs(dy)

    sign_x = 1 if dx > 0 else -1
    sign_y = 1 if dy > 0 else -1

    x = x0
    y = y0

    cells.append((x, y))

    ix = 0
    iy = 0

    while ix < nx or iy < ny:

        if (1 + ix) * ny == (1 + iy) * nx:

            # exact corner crossing
            x += sign_x
            y += sign_y

            ix += 1
            iy += 1

        elif (1 + ix) * ny < (1 + iy) * nx:

            x += sign_x
            ix += 1

        else:

            y += sign_y
            iy += 1

        cells.append((x, y))

    return cells


def terrain_cost(cell):
    if cell == 'T':
        return 1.0
    if cell == 'G':
        return 2.0
    if cell == 'S':
        return 1.0
    if cell == 'F':
        return 1.0
    return 1.0


def valid_move(track, x0, y0, x1, y1):

    height = len(track)
    width = len(track[0])

    path = supercover_line(x0, y0, x1, y1)

    for x, y in path:

        if x < 0 or y < 0:
            return False

        if x >= width or y >= height:
            return False

        if get_cell(track, x, y) == 'O':
            return False

    for i in range(len(path) - 1):

        xA, yA = path[i]
        xB, yB = path[i + 1]

        dx = xB - xA
        dy = yB - yA

        # diagonal transition
        if abs(dx) == 1 and abs(dy) == 1:

            try:

                side1 = get_cell(track, xA + dx, yA)
                side2 = get_cell(track, xA, yA + dy)

                # forbid squeezing through corners
                if side1 == 'O' or side2 == 'O':
                    return False

            except Exception:
                return False

    return True


def reverse_dijkstra(track, finishes):

    height = len(track)
    width = len(track[0])

    dist = {}

    for y in range(height):
        for x in range(width):

            if get_cell(track, x, y) != 'O':
                dist[(x, y)] = INF

    pq = []

    for fx, fy in finishes:

        dist[(fx, fy)] = 0

        heapq.heappush(
            pq,
            (0, fx, fy)
        )

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    while pq:

        d, x, y = heapq.heappop(pq)

        if d != dist[(x, y)]:
            continue

        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            if not valid_move(track, x, y, nx, ny):
                continue

            move_cost_value = 1.4 if dx != 0 and dy != 0 else 1.0

            nd = d + move_cost_value

            if nd < dist[(nx, ny)]:

                dist[(nx, ny)] = nd

                heapq.heappush(
                    pq,
                    (nd, nx, ny)
                )

    return dist


def move_cost(track, x0, y0, x1, y1):
    path = supercover_line(x0, y0, x1, y1)

    cost = 0.0
    for x, y in path:
        cost += terrain_cost(get_cell(track, x, y))

    return cost


def obstacle_proximity_penalty(track, x, y):

    height = len(track)
    width = len(track[0])

    penalty = 0.0

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue

            nx = x + dx
            ny = y + dy

            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                penalty += 1.5
            elif get_cell(track, nx, ny) == 'O':
                penalty += 1.0

    return penalty


def generate_unit_moves(track, state):

    x, y, vx, vy = state
    moves = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:

            if dx == 0 and dy == 0:
                continue

            if vx != 0 or vy != 0:
                dot = dx * vx + dy * vy
                if dot < 0:
                    continue

            nx = x + dx
            ny = y + dy

            next_state = (nx, ny, dx, dy)
            moves.append(next_state)

    return moves


def greedy_unit_score(track, current_state, next_state, reverse_dist, visit_count):
    x, y, vx, vy = current_state
    nx, ny, nvx, nvy = next_state

    current_h = reverse_dist.get((x, y), INF)
    next_h = reverse_dist.get((nx, ny), INF)

    if next_h >= INF:
        return INF

    score = next_h

    score += move_cost(track, x, y, nx, ny)

    score += obstacle_proximity_penalty(track, nx, ny)

    score += REVISIT_WEIGHT * visit_count.get((nx, ny), 0)

    if next_h >= current_h:
        score += next_h - current_h + 1.0

    turn_amount = abs(nvx - vx) + abs(nvy - vy)
    score += TURN_WEIGHT * turn_amount

    if nvx != 0 and nvy != 0:
        score += DIAGONAL_WEIGHT

    return score


def greedy_unit_step_constructor(track):

    start, finishes = find_positions(track)

    print("START:", start)
    print("FINISHES:", finishes)

    if start is None:
        raise Exception("No start found")

    if not finishes:
        raise Exception("No finish found")

    print("Building reverse Dijkstra heuristic...")

    reverse_dist = reverse_dijkstra(
        track,
        finishes
    )

    print("Heuristic map built")

    state = (
        start[0],
        start[1],
        0,
        0
    )

    path = [(state[0], state[1])]
    visit_count = {(state[0], state[1]): 1}

    height = len(track)
    width = len(track[0])
    max_steps = MAX_STEPS_FACTOR * height * width

    for step in range(1, max_steps + 1):

        x, y, vx, vy = state

        if (x, y) in finishes:
            print("Finish reached")
            print("Steps:", step - 1)
            return path

        candidates = []

        for next_state in generate_unit_moves(track, state):

            nx, ny, nvx, nvy = next_state

            if not valid_move(track, x, y, nx, ny):
                continue

            score = greedy_unit_score(
                track,
                state,
                next_state,
                reverse_dist,
                visit_count
            )

            if score >= INF:
                continue

            candidates.append((score, next_state))

        if not candidates:
            print("Greedy unit-step constructor got stuck: no valid candidate")
            print("Last state:", state)
            return []

        candidates.sort(key=lambda item: item[0])

        best_score, best_state = candidates[0]

        if step % 100 == 0:
            bx, by, bvx, bvy = best_state
            print(
                "Step:", step,
                "Position:", (bx, by),
                "Direction:", (bvx, bvy),
                "Score:", round(best_score, 3),
                "ReverseDist:", round(reverse_dist.get((bx, by), INF), 3)
            )

        state = best_state
        path.append((state[0], state[1]))
        visit_count[(state[0], state[1])] = visit_count.get((state[0], state[1]), 0) + 1

    print("Greedy unit-step constructor stopped: maximum number of steps reached")
    return []


def write_csv(path, filename):

    with open(filename, "w") as f:

        for x, y in path:
            f.write(f"{x},{y}\n")


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python greedy_unit_step_reverse_dijkstra.py track_05.t"
        )

        sys.exit(1)

    track_file = sys.argv[1]

    print("Reading track:", track_file)

    track = read_track(track_file)

    path = greedy_unit_step_constructor(track)

    if not path:

        print("No path found")
        sys.exit(1)

    output_file = track_file.replace(
        ".t",
        "_trip.csv"
    )

    write_csv(path, output_file)

    print("Trip written to:", output_file)
    print("Path length:", len(path))
