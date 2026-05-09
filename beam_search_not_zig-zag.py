import sys
import math
import heapq


def distance_to_finish(x, y, finishes):
    return min(
        math.sqrt((fx - x) ** 2 + (fy - y) ** 2)
        for fx, fy in finishes
    )


def read_track(filename):
    with open(filename, "r") as f:
        return [list(line.strip()) for line in f]


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


def line_cells(x0, y0, x1, y1):
    cells = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    x, y = x0, y0

    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    cells.append((x1, y1))
    return cells


def valid_move(track, x0, y0, x1, y1):

    height = len(track)
    width = len(track[0])

    path = line_cells(x0, y0, x1, y1)

    for x, y in path:

        if x < 0 or y < 0 or x >= width or y >= height:
            return False

        if get_cell(track, x, y) == 'O':
            return False

    # check diagonal move through corners
    dx = x1 - x0
    dy = y1 - y0

    if abs(dx) == 1 and abs(dy) == 1:

        #check the two adjacent orthogonal cells
        if (get_cell(track, x0 + dx, y0) == 'O' and
            get_cell(track, x0, y0 + dy) == 'O'):
            return False

    return True


def generate_moves(state):

    x, y, vx, vy = state

    moves = []

    for ax in [-1, 0, 1]:
        for ay in [-1, 0, 1]:

            nvx = vx + ax
            nvy = vy + ay

            nx = x + nvx
            ny = y + nvy

            moves.append((nx, ny, nvx, nvy))

    return moves


def heuristic(nx, ny, nvx, nvy, finishes, depth):

    dist = distance_to_finish(nx, ny, finishes)

    speed = math.sqrt(nvx ** 2 + nvy ** 2)

    #f = g + h
    score = depth + dist

    #reward momentum
    score -= 0.8 * speed

    #penalize standing still
    if speed < 1:
        score += 2

    return score


def beam_search_step(
    beam,
    track,
    finishes,
    beam_width,
    best_score,
    depth
):

    candidates = []

    for state in beam:

        x, y, vx, vy = state

        for nx, ny, nvx, nvy in generate_moves(state):

            if not valid_move(track, x, y, nx, ny):
                continue

            new_state = (nx, ny, nvx, nvy)

            score = heuristic(
                nx,
                ny,
                nvx,
                nvy,
                finishes,
                depth
            )

            #keep better version of state
            if (
                new_state not in best_score
                or score < best_score[new_state]
            ):

                best_score[new_state] = score

                candidates.append(
                    (score, new_state, state)
                )

    if not candidates:
        return []

    #keep best beam_width states
    best = heapq.nsmallest(
        beam_width,
        candidates,
        key=lambda x: x[0]
    )

    new_beam = [state for _, state, _ in best]

    parents = {
        state: parent
        for _, state, parent in best
    }

    return new_beam, parents


def reconstruct(state, parent_map):

    path = []

    while state is not None:

        x, y, vx, vy = state

        path.append((x, y))

        state = parent_map[state]

    path.reverse()

    return path


def solve(track):

    start, finishes = find_positions(track)

    print("START:", start)
    print("FINISHES:", finishes)

    if start is None:
        raise Exception("No start position found")

    if not finishes:
        raise Exception("No finish line found")

    start_state = (start[0], start[1], 0, 0)

    beam_width = 100

    max_steps = 1000

    beam = [start_state]

    parent_map = {
        start_state: None
    }


    best_score = {
        start_state: 0
    }

    for step in range(max_steps):

        result = beam_search_step(
            beam,
            track,
            finishes,
            beam_width,
            best_score,
            step
        )

        if not result:

            print("Beam search failed")

            #return best partial path
            best_state = beam[0]

            return reconstruct(
                best_state,
                parent_map
            )

        beam, parents = result

        parent_map.update(parents)

        #check goal
        for state in beam:

            x, y, vx, vy = state

            if (x, y) in finishes:

                print("Finish reached in", step, "steps")

                return reconstruct(
                    state,
                    parent_map
                )

    print("Max steps reached")

    return []


def write_csv(path, filename):

    with open(filename, "w") as f:

        for x, y in path:
            f.write(f"{x},{y}\n")


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage: python heuristic.py track_02.t")

        sys.exit(1)

    track_file = sys.argv[1]

    print("Reading track:", track_file)

    track = read_track(track_file)

    path = solve(track)

    output_file = track_file.replace(
        ".t",
        "_trip.csv"
    )

    write_csv(path, output_file)

    print("Trip written to:", output_file)