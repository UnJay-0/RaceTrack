import sys

from src.construction.rdasw import Rdasw
from src.state import States
from src.track import Track


def write_csv(track: Track, path: States, filename: str):
    height, _ = track.get_boundaries()
    with open("output/track_" + filename.split("_")[1] + "/" + filename, "w") as f:
        for state in path:
            f.write(f"{state.position.x},{height - 1 - state.position.y}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solver.py track_05.t")

        sys.exit(1)

    track_file = sys.argv[1]

    print("Reading track:", track_file)

    track = Track(track_file)

    path = Rdasw.apply(track)

    if not path:
        print("No path found")
        sys.exit(1)

    output_file = track_file.replace(".t", "_trip.csv")

    write_csv(track, path, output_file.split("/")[1])

    print("Trip written to:", output_file)
    print("Path length:", len(path))
