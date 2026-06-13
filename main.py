import sys
from pathlib import Path

from src.construction.rdasw import Rdasw
from src.construction.rdgreedy import RdGreedy
from src.improvement.evolutionary import EvolutionaryAlgo
from src.state import States
from src.track import Track


def write_csv(track: Track, path: States, filename: str, type: str = "a"):
    height, _ = track.get_boundaries()

    out_dir = Path("output") / f"track_{filename.split('_')[1]}" / type
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / filename, "w") as f:
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
        for lookahead in range(11):
            new_path = greedy.greedy_path_constructor(
                force_unit=True,
                speed_limit=None,
                lookahead_steps=lookahead,
                obstacles=False,
            )
            if new_path is None:
                continue
            if path_length >= len(new_path):
                path_length = len(new_path)
                path = new_path

    if not path:
        print("No path found")
        sys.exit(1)


    output_filename = Path(track_file).name.replace(".t", "_trip.csv")
    write_csv(track, path, output_filename, construction_type)

    improvement_status = "not_run"
    improved_path = None

    try:
        improver = EvolutionaryAlgo(track, corner_grouping_threshold=4, gate_length=2)
        candidate = improver.improve(path)

        if candidate is None:
            improvement_status = "fallback_construction_none"
            improved_path = path
        elif len(candidate) <= len(path):
            improvement_status = "improved" if len(candidate) < len(path) else "unchanged"
            improved_path = candidate
        else:
            improvement_status = "fallback_construction_worse_candidate"
            improved_path = path

    except Exception as exc:
        improvement_status = f"fallback_construction_exception:{type(exc).__name__}"
        improved_path = path

    write_csv(track, improved_path, output_filename, "improved")

    print("Construction path length:", len(path))
    print("Construction moves:", len(path) - 1)
    print("Improved path length:", len(improved_path))
    print("Improved moves:", len(improved_path) - 1)
    print("Improvement status:", improvement_status)
