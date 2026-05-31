import heapq
import math
import sys

MAX_SPEED = 8
WEIGHT = 2.0
INF = 10**9


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

            if cell == "S":
                start = (x, y)

            elif cell == "F":
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

        cells.append((x, y))

    return cells


def terrain_cost(cell):
    if cell == "T":
        return 1.0
    if cell == "G":
        return 2.0
    if cell == "S":
        return 1.0
    if cell == "F":
        return 1.0
    return 1.0


def terrain_factor(cell):
    if cell == "G":
        return 0.5
    if cell == "T":
        return 2.0
    if cell == "S":
        return 1.0
    if cell == "F":
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

        if get_cell(track, x, y) == "O":
            return False

    # forbid touching obstacle corners
    dx_total = x1 - x0
    dy_total = y1 - y0

    steps = max(abs(dx_total), abs(dy_total))

    if steps > 0:
        for i in range(steps + 1):
            t = i / steps

            px = x0 + dx_total * t
            py = y0 + dy_total * t

            eps = 1e-9

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
                if side1 == "O" and side2 == "O":
                    return False

            except:
                return False

    return True


def reverse_dijkstra(track, finishes):

    height = len(track)
    width = len(track[0])

    dist = {}

    for y in range(height):
        for x in range(width):
            if get_cell(track, x, y) != "O":
                dist[(x, y)] = INF

    pq = []

    for fx, fy in finishes:
        dist[(fx, fy)] = 0

        heapq.heappush(pq, (0, fx, fy))

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while pq:
        d, x, y = heapq.heappop(pq)

        if d != dist[(x, y)]:
            continue

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            if nx < 0 or ny < 0:
                continue

            if nx >= width or ny >= height:
                continue

            if get_cell(track, nx, ny) == "O":
                continue

            if not valid_move(track, x, y, nx, ny):
                continue

            move_cost = 1.4 if dx != 0 and dy != 0 else 1.0

            nd = d + move_cost

            if nd < dist[(nx, ny)]:
                dist[(nx, ny)] = nd

                heapq.heappush(pq, (nd, nx, ny))

    return dist


def generate_moves(track, state):

    x, y, vx, vy = state
    cell = get_cell(track, x, y)

    factor = terrain_factor(cell)

    moves = []

    for ax in [-1, 0, 1]:
        for ay in [-1, 0, 1]:
            nvx = vx + ax * factor
            nvy = vy + ay * factor

            nvx = int(round(nvx))
            nvy = int(round(nvy))

            if abs(nvx) > MAX_SPEED or abs(nvy) > MAX_SPEED:
                continue

            nx = x + nvx
            ny = y + nvy

            moves.append((nx, ny, nvx, nvy))

    return moves


def heuristic(x, y, vx, vy, reverse_dist):

    h = reverse_dist.get((x, y), INF)

    speed = math.sqrt(vx * vx + vy * vy)

    # small momentum reward
    h -= 0.15 * speed

    return h


def reconstruct(state, parent_map):

    path = []

    while state is not None:
        x, y, vx, vy = state

        path.append((x, y))

        state = parent_map[state]

    path.reverse()

    return path


def move_cost(track, x0, y0, x1, y1):
    path = supercover_line(x0, y0, x1, y1)

    cost = 0.0
    for x, y in path:
        cost += terrain_cost(get_cell(track, x, y))

    return cost


def weighted_astar(track):

    start, finishes = find_positions(track)

    print("START:", start)
    print("FINISHES:", finishes)

    if start is None:
        raise Exception("No start found")

    if not finishes:
        raise Exception("No finish found")

    print("Building reverse Dijkstra heuristic...")

    reverse_dist = reverse_dijkstra(track, finishes)

    print("Heuristic map built")

    start_state = (start[0], start[1], 0, 0)

    open_set = []

    g_score = {start_state: 0}

    parent = {start_state: None}

    visited = set()

    h0 = heuristic(start[0], start[1], 0, 0, reverse_dist)

    heapq.heappush(open_set, (WEIGHT * h0, start_state))

    expanded = 0

    while open_set:
        f, state = heapq.heappop(open_set)

        if state in visited:
            continue

        visited.add(state)

        expanded += 1

        if expanded % 10000 == 0:
            print("Expanded:", expanded, "Open:", len(open_set))

        x, y, vx, vy = state

        # GOAL
        if (x, y) in finishes:
            print("Finish reached")
            print("Expanded nodes:", expanded)

            return reconstruct(state, parent)

        for next_state in generate_moves(track, state):
            nx, ny, nvx, nvy = next_state

            if not valid_move(track, x, y, nx, ny):
                continue

            tentative_g = g_score[state] + move_cost(track, x, y, nx, ny)

            if next_state not in g_score or tentative_g < g_score[next_state]:
                g_score[next_state] = tentative_g

                parent[next_state] = state

                h = heuristic(nx, ny, nvx, nvy, reverse_dist)

                fscore = tentative_g + WEIGHT * h

                heapq.heappush(open_set, (fscore, next_state))

    print("No solution found")

    return []


def write_csv(path, filename):

    with open(filename, "w") as f:
        for x, y in path:
            f.write(f"{x},{y}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solver.py track_05.t")

        sys.exit(1)

    track_file = sys.argv[1]

    print("Reading track:", track_file)

    track = read_track(track_file)

    path = weighted_astar(track)

    if not path:
        print("No path found")
        sys.exit(1)

    output_file = track_file.replace(".t", "_trip.csv")

    write_csv(path, output_file)

    print("Trip written to:", output_file)
    print("Path length:", len(path))
