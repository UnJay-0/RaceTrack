import sys
import math


def read_track(filename):

    with open(filename, "r") as f:
        track = [list(line.strip()) for line in f]

    return track

def find_positions(track):

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

def distance_to_finish(x, y, finishes):

    best = float("inf")

    for fx, fy in finishes:

        d = math.sqrt((fx - x) ** 2 + (fy - y) ** 2)

        if d < best:
            best = d

    return best


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

    for x, y in line_cells(x0, y0, x1, y1):

        #outside map
        if x < 0 or y < 0 or x >= width or y >= height:
            return False

        #obstacle
        if get_cell(track, x, y) == 'O':
            return False

    return True

def generate_moves(state, track):

    x, y, vx, vy = state

    current_cell = get_cell(track, x, y)

    moves = []

    for ax in [-1, 0, 1]:
        for ay in [-1, 0, 1]:

            nvx = vx + ax
            nvy = vy + ay

            if current_cell == 'G':

                if abs(nvx) > abs(vx):
                    continue

                if abs(nvy) > abs(vy):
                    continue

            nx = x + nvx
            ny = y + nvy

            moves.append((nx, ny, nvx, nvy))

    return moves

def choose_best_move(track, state, finishes, visited):

    x, y, vx, vy = state

    best_move = None
    best_distance = float("inf")

    for nx, ny, nvx, nvy in generate_moves(state, track):

        #collision
        if not valid_move(track, x, y, nx, ny):
            continue

        #avoid loops
        if (nx, ny, nvx, nvy) in visited:
            continue

        #closest to finish
        dist = distance_to_finish(nx, ny, finishes)

        if dist < best_distance:

            best_distance = dist

            best_move = (nx, ny, nvx, nvy)

    return best_move


def solve(track):

    start, finishes = find_positions(track)

    if start is None:
        raise Exception("No start position found")

    if len(finishes) == 0:
        raise Exception("No finish line found")

    #state = (x, y, vx, vy)
    state = (start[0], start[1], 0, 0)

    path = [(start[0], start[1])]

    visited = set()
    visited.add(state)

    max_steps = 10000

    for step in range(max_steps):

        x, y, vx, vy = state

        #finish reached
        if (x, y) in finishes:

            print("Finish reached in", step, "steps")

            return path

        next_state = choose_best_move(
            track,
            state,
            finishes,
            visited
        )

        #no valid move
        if next_state is None:

            print("No valid moves found")

            return path

        state = next_state

        visited.add(state)

        path.append((state[0], state[1]))

    print("Maximum number of steps reached")

    return path


def write_csv(path, filename):

    with open(filename, "w") as f:

        for x, y in path:
            f.write(f"{x},{y}\n")


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print("python heuristic.py track_02.t")

        sys.exit(1)

    track_file = sys.argv[1]

    print("Reading track:", track_file)

    track = read_track(track_file)

    path = solve(track)

    output_file = track_file.replace(".t", "_trip.csv")

    write_csv(path, output_file)

    print("Trip written to:", output_file)