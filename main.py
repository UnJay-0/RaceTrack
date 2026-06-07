import sys

from src.construction.rdasw import Rdasw
from src.construction.rdgreedy import RdGreedy
from src.improvement.evolutionary import EvolutionaryAlgo
from src.state import States
from src.track import Track


def write_csv(track: Track, path: States, filename: str, type: str = "a"):
    height, _ = track.get_boundaries()
    with open(f"output/track_{filename.split('_')[1]}/{type}/{filename}", "w") as f:
        for state in path:
            f.write(f"{state.position.x},{height - 1 - state.position.y}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solver.py track_05.t a|g")

        sys.exit(1)

    track_file = sys.argv[1]

    construction_type = sys.argv[2] if len(sys.argv) >= 3 else "a"

    print("Reading track:", track_file)

    track = Track(track_file)
    path = None
    if construction_type == "a":
        path = Rdasw.apply(track)
    elif construction_type == "g":
        greedy = RdGreedy(track)
        path_length = 10e9
        lookahead = 10
        while True:
            new_path = greedy.greedy_path_constructor(
                force_unit=True,
                speed_limit=None,
                lookahead_steps=lookahead,
                obstacles=False,
            )
            if path_length <= len(new_path):
                print(path_length, lookahead)
                break
            else:
                path_length = len(new_path)
                path = new_path
                lookahead -= 1

    if not path:
        print("No path found")
        sys.exit(1)

    output_file = track_file.replace(".t", "_trip.csv")

    write_csv(track, path, output_file.split("/")[1], construction_type)

    improver = EvolutionaryAlgo(track, corner_grouping_threshold=3, gate_length=7)
    path = improver.improve(path)

    if not path:
        print("No improved path found")
        sys.exit(1)

    output_file = track_file.replace(".t", "_trip.csv")

    write_csv(track, path, output_file.split("/")[1], "improved")

    print("Trip written to:", output_file)
    print("Path length:", len(path))
