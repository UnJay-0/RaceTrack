from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(".").resolve()
TRACKS_DIR = PROJECT_ROOT / "tracks"
MAIN_PY = PROJECT_ROOT / "main.py"
OUTPUT_ROOT = PROJECT_ROOT / "output"
VISUALISE_PL = PROJECT_ROOT / "visualise.pl"

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_improvement"
SOLUTIONS_DIR = BENCHMARK_DIR / "solutions"
RESULTS_CSV = BENCHMARK_DIR / "benchmark_improvement_results.csv"
SUMMARY_CSV = BENCHMARK_DIR / "benchmark_improvement_summary.csv"
SYSTEM_SPECS_JSON = BENCHMARK_DIR / "system_specs.json"

CONSTRUCTION_TYPE = "g"
IMPROVED_TYPE = "improved"

RUNS_PER_TRACK = 3
SOLVER_TIMEOUT_SEC = 1800
VISUALISE_TIMEOUT_SEC = 90
PDFLATEX_TIMEOUT_SEC = 90
CLEAN_TRIPS_BEFORE_RUN = True
PYTHON_EXE = sys.executable

SIMPLIFY_PDF_ON_FAILURE = True


# =============================================================================
# Helpers
# =============================================================================


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def short_text(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    return text[-limit:]


def write_system_specs() -> None:
    specs: dict[str, Any] = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "cpu_count_logical": os.cpu_count(),
    }

    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            for line in meminfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        mem_total_kib = int(parts[1])
                        mem_total_bytes = mem_total_kib * 1024
                        specs["memory_total_bytes"] = mem_total_bytes
                        specs["memory_total_gib"] = round(mem_total_bytes / 1024**3, 3)
                    break
        except Exception as exc:
            specs["memory_read_error"] = repr(exc)

    write_json(SYSTEM_SPECS_JSON, specs)


def discover_tracks() -> list[Path]:
    if not TRACKS_DIR.exists():
        return []
    return sorted([p for p in TRACKS_DIR.glob("*.t") if p.is_file()], key=natural_key)


def expected_trip_path(track_file: Path, kind: str) -> Path:
    trip_name = track_file.name.replace(".t", "_trip.csv")
    parts = trip_name.split("_")

    if len(parts) >= 2:
        return OUTPUT_ROOT / f"track_{parts[1]}" / kind / trip_name

    return OUTPUT_ROOT / track_file.stem / kind / trip_name


def prefixed_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


# =============================================================================
# Trip parsing
# =============================================================================


def parse_trip_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "trip_exists": False,
            "trip_parse_error": "trip file missing",
            "coordinate_count": None,
            "move_count": None,
            "start_coordinate": None,
            "end_coordinate": None,
            "unique_position_count": None,
            "repeated_coordinate_count": None,
            "consecutive_repeat_count": None,
            "total_euclidean_length": None,
            "avg_step_length": None,
            "max_step_length": None,
        }

    coords: list[tuple[int, int]] = []

    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for line_no, row in enumerate(reader, start=1):
                if not row or all(not cell.strip() for cell in row):
                    continue

                if len(row) < 2:
                    raise ValueError(f"line {line_no}: fewer than two CSV fields: {row}")

                coords.append((int(row[0].strip()), int(row[1].strip())))

    except Exception as exc:
        return {
            "trip_exists": True,
            "trip_parse_error": repr(exc),
            "coordinate_count": None,
            "move_count": None,
            "start_coordinate": None,
            "end_coordinate": None,
            "unique_position_count": None,
            "repeated_coordinate_count": None,
            "consecutive_repeat_count": None,
            "total_euclidean_length": None,
            "avg_step_length": None,
            "max_step_length": None,
        }

    step_lengths = [
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(coords, coords[1:])
    ]
    total_length = sum(step_lengths)
    unique_count = len(set(coords))

    return {
        "trip_exists": True,
        "trip_parse_error": "",
        "coordinate_count": len(coords),
        "move_count": max(0, len(coords) - 1) if coords else None,
        "start_coordinate": coords[0] if coords else None,
        "end_coordinate": coords[-1] if coords else None,
        "unique_position_count": unique_count,
        "repeated_coordinate_count": len(coords) - unique_count,
        "consecutive_repeat_count": sum(1 for a, b in zip(coords, coords[1:]) if a == b),
        "total_euclidean_length": round(total_length, 6) if step_lengths else 0.0,
        "avg_step_length": round(total_length / len(step_lengths), 6) if step_lengths else 0.0,
        "max_step_length": round(max(step_lengths), 6) if step_lengths else 0.0,
    }


# =============================================================================
# Solver / improvement execution
# =============================================================================


def parse_improvement_status(stdout: str) -> str:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("Improvement status:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def run_solver_and_improver(track_file: Path, run_dir: Path, run_idx: int) -> dict[str, Any]:
    construction_trip = expected_trip_path(track_file, CONSTRUCTION_TYPE)
    improved_trip = expected_trip_path(track_file, IMPROVED_TYPE)

    for trip_path in [construction_trip, improved_trip]:
        trip_path.parent.mkdir(parents=True, exist_ok=True)
        if CLEAN_TRIPS_BEFORE_RUN and trip_path.exists():
            trip_path.unlink()

    cmd = [PYTHON_EXE, str(MAIN_PY), str(track_file), CONSTRUCTION_TYPE]
    start_time = datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.perf_counter()

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=SOLVER_TIMEOUT_SEC,
        )
        runtime = time.perf_counter() - t0
        status = "ok" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
        timeout = False
        stdout = completed.stdout
        stderr = completed.stderr

    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - t0
        status = "timeout"
        returncode = None
        timeout = True
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""

    construction_stats = parse_trip_csv(construction_trip)
    improved_stats = parse_trip_csv(improved_trip)

    row: dict[str, Any] = {
        "track": track_file.name,
        "track_stem": track_file.stem,
        "run": run_idx,
        "run_dir": str(run_dir),

        "solver_status": status,
        "solver_returncode": returncode,
        "solver_timeout": timeout,
        "solver_runtime_sec": round(runtime, 6),
        "solver_start_time": start_time,
        "solver_command": " ".join(cmd),

        "main_improvement_status": parse_improvement_status(stdout),
        "source_construction_trip_path": str(construction_trip),
        "source_improved_trip_path": str(improved_trip),
    }

    if status != "ok":
        row["solver_stdout_tail"] = short_text(stdout)
        row["solver_stderr_tail"] = short_text(stderr)

    row.update(prefixed_stats("construction", construction_stats))
    row.update(prefixed_stats("improved", improved_stats))

    construction_copy_path = ""
    improved_copy_path = ""

    if construction_trip.exists() and construction_stats.get("trip_parse_error") == "":
        construction_copy = run_dir / f"construction_{construction_trip.name}"
        shutil.copy2(construction_trip, construction_copy)
        construction_copy_path = str(construction_copy)

    if improved_trip.exists() and improved_stats.get("trip_parse_error") == "":
        improved_copy = run_dir / f"improved_{improved_trip.name}"
        shutil.copy2(improved_trip, improved_copy)
        improved_copy_path = str(improved_copy)

    row["construction_trip_path"] = construction_copy_path
    row["improved_trip_path"] = improved_copy_path

    compute_improvement_metrics(row)

    return row


def compute_improvement_metrics(row: dict[str, Any]) -> None:
    construction_moves = row.get("construction_move_count")
    improved_moves = row.get("improved_move_count")

    row["delta_moves"] = None
    row["relative_improvement_percent"] = None
    row["improvement_outcome"] = "unknown"

    if construction_moves is None:
        row["improvement_outcome"] = "missing_construction"
        return

    if improved_moves is None:
        row["improvement_outcome"] = "missing_improved"
        return

    try:
        construction_moves_i = int(construction_moves)
        improved_moves_i = int(improved_moves)

        delta = construction_moves_i - improved_moves_i
        row["delta_moves"] = delta

        if construction_moves_i > 0:
            row["relative_improvement_percent"] = round(
                100.0 * delta / construction_moves_i,
                6,
            )

        if delta > 0:
            row["improvement_outcome"] = "improved"
        elif delta == 0:
            row["improvement_outcome"] = "unchanged"
        else:
            row["improvement_outcome"] = "worse"

    except Exception as exc:
        row["improvement_outcome"] = f"metric_error:{type(exc).__name__}"


# =============================================================================
# visualise.pl and pdflatex
# =============================================================================


def run_visualise(track_file: Path, trip_file: Path, run_dir: Path) -> dict[str, Any]:
    if not VISUALISE_PL.exists():
        return {"visualise_status": "missing_visualise_pl"}

    if not trip_file.exists():
        return {"visualise_status": "missing_trip"}

    tex_path = run_dir / f"{track_file.stem}.tex"
    perl_exe = shutil.which("perl") or "perl"

    cmd = [
        perl_exe,
        str(VISUALISE_PL),
        str(track_file),
        str(trip_file),
        str(tex_path),
    ]

    t0 = time.perf_counter()

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=VISUALISE_TIMEOUT_SEC,
        )
        runtime = time.perf_counter() - t0
        status = "ok" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr

    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - t0
        status = "timeout"
        returncode = None
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""

    info: dict[str, Any] = {
        "visualise_status": status,
        "visualise_returncode": returncode,
        "visualise_runtime_sec": round(runtime, 6),
        "visualise_command": " ".join(cmd),
        "tex_path": str(tex_path) if tex_path.exists() else "",
    }

    if status != "ok":
        info["visualise_stdout_tail"] = short_text(stdout)
        info["visualise_stderr_tail"] = short_text(stderr)

    return info


def simplify_tex_remove_labels(tex_path: Path) -> Path:
    simplified = tex_path.with_name(tex_path.stem + "_no_labels" + tex_path.suffix)

    with tex_path.open("r", encoding="utf-8", errors="replace") as src, \
         simplified.open("w", encoding="utf-8") as dst:
        for line in src:
            if "\\node[anchor=south west]" in line:
                continue
            dst.write(line)

    return simplified


def run_pdflatex_once(tex_path: Path, suffix: str = "") -> dict[str, Any]:
    pdflatex_exe = shutil.which("pdflatex")

    if pdflatex_exe is None:
        return {
            f"pdflatex{suffix}_status": "missing_pdflatex",
            f"pdflatex{suffix}_pdf_path": "",
        }

    pdf_path = tex_path.with_suffix(".pdf")

    if pdf_path.exists():
        pdf_path.unlink()

    cmd = [
        pdflatex_exe,
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]

    t0 = time.perf_counter()

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(tex_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=PDFLATEX_TIMEOUT_SEC,
        )
        runtime = time.perf_counter() - t0
        status = "ok" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr

    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - t0
        status = "timeout"
        returncode = None
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""

    info: dict[str, Any] = {
        f"pdflatex{suffix}_status": status,
        f"pdflatex{suffix}_returncode": returncode,
        f"pdflatex{suffix}_runtime_sec": round(runtime, 6),
        f"pdflatex{suffix}_command": " ".join(cmd),
        f"pdflatex{suffix}_tex_path": str(tex_path),
        f"pdflatex{suffix}_pdf_path": str(pdf_path) if pdf_path.exists() else "",
    }

    if status != "ok":
        combined = f"{stdout}\n{stderr}"
        if "TeX capacity exceeded" in combined:
            info[f"pdflatex{suffix}_error"] = "TeX capacity exceeded"
        elif "Fatal error" in combined:
            info[f"pdflatex{suffix}_error"] = "Fatal error"
        else:
            info[f"pdflatex{suffix}_error"] = "pdflatex failed"

    return info


def run_pdflatex(tex_path: Path) -> dict[str, Any]:
    if not tex_path.exists():
        return {
            "pdflatex_status": "missing_tex_file",
            "pdf_path": "",
        }

    first = run_pdflatex_once(tex_path)

    if first.get("pdflatex_status") == "ok":
        return {
            **first,
            "pdflatex_status": "ok",
            "pdflatex_mode": "original",
            "pdf_path": first.get("pdflatex_pdf_path", ""),
        }

    if not SIMPLIFY_PDF_ON_FAILURE:
        return {
            **first,
            "pdflatex_status": first.get("pdflatex_status", "failed"),
            "pdflatex_mode": "original_failed_no_fallback",
            "pdf_path": first.get("pdflatex_pdf_path", ""),
        }

    simplified = simplify_tex_remove_labels(tex_path)
    second = run_pdflatex_once(simplified, suffix="_no_labels")

    if second.get("pdflatex_no_labels_status") == "ok":
        return {
            **first,
            **second,
            "pdflatex_status": "ok_simplified_no_labels",
            "pdflatex_mode": "simplified_no_labels",
            "simplified_tex_path": str(simplified),
            "pdf_path": second.get("pdflatex_no_labels_pdf_path", ""),
        }

    return {
        **first,
        **second,
        "pdflatex_status": "failed_all_attempts",
        "pdflatex_mode": "all_attempts_failed",
        "simplified_tex_path": str(simplified),
        "pdf_path": "",
    }


# =============================================================================
# Compact CSV rows
# =============================================================================


def compact_result_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "track": row.get("track"),
        "track_stem": row.get("track_stem"),
        "run": row.get("run"),

        "solver_status": row.get("solver_status"),
        "solver_returncode": row.get("solver_returncode"),
        "solver_timeout": row.get("solver_timeout"),
        "solver_runtime_sec": row.get("solver_runtime_sec"),

        "main_improvement_status": row.get("main_improvement_status"),
        "improvement_outcome": row.get("improvement_outcome"),

        "construction_trip_exists": row.get("construction_trip_exists"),
        "construction_trip_parse_error": row.get("construction_trip_parse_error"),
        "construction_coordinate_count": row.get("construction_coordinate_count"),
        "construction_move_count": row.get("construction_move_count"),
        "construction_unique_position_count": row.get("construction_unique_position_count"),
        "construction_repeated_coordinate_count": row.get("construction_repeated_coordinate_count"),
        "construction_total_euclidean_length": row.get("construction_total_euclidean_length"),

        "improved_trip_exists": row.get("improved_trip_exists"),
        "improved_trip_parse_error": row.get("improved_trip_parse_error"),
        "improved_coordinate_count": row.get("improved_coordinate_count"),
        "improved_move_count": row.get("improved_move_count"),
        "improved_unique_position_count": row.get("improved_unique_position_count"),
        "improved_repeated_coordinate_count": row.get("improved_repeated_coordinate_count"),
        "improved_total_euclidean_length": row.get("improved_total_euclidean_length"),

        "delta_moves": row.get("delta_moves"),
        "relative_improvement_percent": row.get("relative_improvement_percent"),

        "visualise_status": row.get("visualise_status"),
        "visualise_returncode": row.get("visualise_returncode"),
        "visualise_runtime_sec": row.get("visualise_runtime_sec"),

        "pdflatex_status": row.get("pdflatex_status"),
        "pdflatex_mode": row.get("pdflatex_mode"),
        "pdflatex_returncode": row.get("pdflatex_returncode"),
        "pdflatex_runtime_sec": row.get("pdflatex_runtime_sec"),
        "pdflatex_no_labels_status": row.get("pdflatex_no_labels_status"),
        "pdflatex_no_labels_returncode": row.get("pdflatex_no_labels_returncode"),
        "pdflatex_no_labels_runtime_sec": row.get("pdflatex_no_labels_runtime_sec"),
    }


# =============================================================================
# Aggregation
# =============================================================================


def is_success(row: dict[str, Any]) -> bool:
    return (
        row.get("solver_status") == "ok"
        and row.get("construction_trip_exists") is True
        and row.get("construction_trip_parse_error") == ""
        and row.get("construction_move_count") is not None
        and row.get("improved_trip_exists") is True
        and row.get("improved_trip_parse_error") == ""
        and row.get("improved_move_count") is not None
    )


def stdev_or_zero(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        grouped.setdefault(str(row["track"]), []).append(row)

    summary: list[dict[str, Any]] = []

    for track in sorted(grouped, key=lambda s: natural_key(Path(s))):
        group = grouped[track]
        successes = [r for r in group if is_success(r)]

        construction_moves = [int(r["construction_move_count"]) for r in successes]
        improved_moves = [int(r["improved_move_count"]) for r in successes]
        deltas = [int(r["delta_moves"]) for r in successes if r.get("delta_moves") is not None]
        relative_improvements = [
            float(r["relative_improvement_percent"])
            for r in successes
            if r.get("relative_improvement_percent") is not None
        ]

        runtimes_all = [float(r["solver_runtime_sec"]) for r in group]
        runtimes_success = [float(r["solver_runtime_sec"]) for r in successes]

        best = (
            min(
                successes,
                key=lambda r: (
                    int(r["improved_move_count"]),
                    float(r["solver_runtime_sec"]),
                ),
            )
            if successes
            else None
        )

        track_stem = str(group[0]["track_stem"]) if group else Path(track).stem
        best_dir = SOLUTIONS_DIR / track_stem / "best"

        summary.append({
            "track": track,
            "runs": len(group),
            "successes": len(successes),

            "construction_best_moves": min(construction_moves) if construction_moves else None,
            "construction_avg_moves": round(statistics.mean(construction_moves), 3) if construction_moves else None,
            "construction_stdev_moves": round(stdev_or_zero([float(x) for x in construction_moves]), 3) if construction_moves else None,

            "improved_best_moves": min(improved_moves) if improved_moves else None,
            "improved_best_run": best.get("run") if best else None,
            "improved_avg_moves": round(statistics.mean(improved_moves), 3) if improved_moves else None,
            "improved_stdev_moves": round(stdev_or_zero([float(x) for x in improved_moves]), 3) if improved_moves else None,

            "best_delta_moves": max(deltas) if deltas else None,
            "avg_delta_moves": round(statistics.mean(deltas), 3) if deltas else None,
            "stdev_delta_moves": round(stdev_or_zero([float(x) for x in deltas]), 3) if deltas else None,
            "avg_relative_improvement_percent": round(statistics.mean(relative_improvements), 6) if relative_improvements else None,

            "improved_runs": sum(1 for r in successes if r.get("improvement_outcome") == "improved"),
            "unchanged_runs": sum(1 for r in successes if r.get("improvement_outcome") == "unchanged"),
            "worse_runs": sum(1 for r in successes if r.get("improvement_outcome") == "worse"),

            "min_solver_runtime_sec_success": round(min(runtimes_success), 6) if runtimes_success else None,
            "avg_solver_runtime_sec_all": round(statistics.mean(runtimes_all), 6) if runtimes_all else None,
            "avg_solver_runtime_sec_success": round(statistics.mean(runtimes_success), 6) if runtimes_success else None,
            "stdev_solver_runtime_sec_success": round(stdev_or_zero(runtimes_success), 6) if runtimes_success else None,

            "timeouts": sum(1 for r in group if r.get("solver_timeout")),
            "failures": sum(1 for r in group if r.get("solver_status") != "ok"),
            "visualise_successes": sum(1 for r in group if r.get("visualise_status") == "ok"),
            "pdf_successes": sum(1 for r in group if str(r.get("pdflatex_status", "")).startswith("ok")),
            "simplified_pdf_successes": sum(1 for r in group if str(r.get("pdflatex_status", "")).startswith("ok_simplified")),

            "best_dir": str(best_dir) if best else "",
        })

    return summary


def copy_best_runs(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        grouped.setdefault(str(row["track_stem"]), []).append(row)

    for track_stem, group in grouped.items():
        successes = [row for row in group if is_success(row)]

        if not successes:
            continue

        best = min(
            successes,
            key=lambda row: (
                int(row["improved_move_count"]),
                float(row["solver_runtime_sec"]),
            ),
        )

        source_run_dir_raw = best.get("run_dir", "")
        if not source_run_dir_raw:
            continue

        source_run_dir = Path(str(source_run_dir_raw))
        if not source_run_dir.is_dir():
            continue

        best_dir = SOLUTIONS_DIR / track_stem / "best"

        if best_dir.exists():
            shutil.rmtree(best_dir)

        shutil.copytree(source_run_dir, best_dir)

        best_metadata = dict(best)
        best_metadata["best_copy"] = True
        best_metadata["best_source_run_dir"] = str(source_run_dir)
        best_metadata["best_dir"] = str(best_dir)

        for key in [
            "construction_trip_path",
            "improved_trip_path",
            "tex_path",
            "pdf_path",
            "simplified_tex_path",
        ]:
            value = best_metadata.get(key)
            if value:
                best_metadata[key] = str(best_dir / Path(str(value)).name)

        best_metadata["run_dir"] = str(best_dir)

        write_json(best_dir / "metadata.json", best_metadata)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    write_system_specs()

    tracks = discover_tracks()

    if not tracks:
        print(f"No .t track files found in {TRACKS_DIR}", file=sys.stderr)
        return 2

    print("=" * 80)
    print("Benchmark: construction + improvement heuristic")
    print("=" * 80)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Tracks directory: {TRACKS_DIR}")
    print(f"Output directory: {BENCHMARK_DIR}")
    print(f"Runs per track:   {RUNS_PER_TRACK}")
    print(f"pdflatex:         {shutil.which('pdflatex') or 'missing'}")
    print("Tracks:")

    tracks.reverse()  # Show in natural order with track_1 before track_10

    for track in tracks:
        print(f"  - {track}")

    print("=" * 80)

    rows: list[dict[str, Any]] = []

    for track_file in tracks:

        if track_file.name == "track_0circular.t":
            print(f"Skipping {track_file.name}")
            continue

        track_dir = SOLUTIONS_DIR / track_file.stem

        for run_idx in range(1, RUNS_PER_TRACK + 1):
            run_dir = track_dir / f"run{run_idx:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running {track_file.name}, run {run_idx:02d}/{RUNS_PER_TRACK:02d}...")

            row = run_solver_and_improver(track_file, run_dir, run_idx)

            improved_trip_path = (
                Path(str(row["improved_trip_path"]))
                if row.get("improved_trip_path")
                else None
            )

            if is_success(row) and improved_trip_path is not None and improved_trip_path.is_file():
                row.update(run_visualise(track_file, improved_trip_path, run_dir))

                tex_path = Path(str(row["tex_path"])) if row.get("tex_path") else None

                if tex_path is not None and tex_path.is_file() and row.get("visualise_status") == "ok":
                    row.update(run_pdflatex(tex_path))
                else:
                    row.update({
                        "pdflatex_status": "skipped_no_tex",
                        "pdf_path": "",
                    })
            else:
                row.update({
                    "visualise_status": "skipped_no_successful_improved_trip",
                    "pdflatex_status": "skipped_no_successful_improved_trip",
                    "pdf_path": "",
                })

            write_json(run_dir / "metadata.json", row)

            rows.append(row)
            write_csv(RESULTS_CSV, [compact_result_row(r) for r in rows])

            print(
                f"  solver={row.get('solver_status')} "
                f"time={row.get('solver_runtime_sec')}s "
                f"construction_moves={row.get('construction_move_count')} "
                f"improved_moves={row.get('improved_move_count')} "
                f"delta={row.get('delta_moves')} "
                f"outcome={row.get('improvement_outcome')} "
                f"visualise={row.get('visualise_status')} "
                f"pdflatex={row.get('pdflatex_status')}"
            )

    copy_best_runs(rows)

    summary = summarize(rows)
    write_csv(SUMMARY_CSV, summary)

    print("=" * 80)
    print("Benchmark finished")
    print("=" * 80)
    print(f"Detailed results: {RESULTS_CSV}")
    print(f"Summary CSV:      {SUMMARY_CSV}")
    print(f"System specs:     {SYSTEM_SPECS_JSON}")
    print(f"Solutions:        {SOLUTIONS_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())