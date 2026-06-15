from __future__ import annotations

import os
import platform
import csv
import json
import math
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

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_g_construction"
SOLUTIONS_DIR = BENCHMARK_DIR / "solutions"
RESULTS_CSV = BENCHMARK_DIR / "benchmark_g_construction_results.csv"
SUMMARY_CSV = BENCHMARK_DIR / "benchmark_g_construction_summary.csv"

CONSTRUCTION_TYPE = "g"
RUNS_PER_TRACK = 10
SOLVER_TIMEOUT_SEC = 180
VISUALISE_TIMEOUT_SEC = 60
PDFLATEX_TIMEOUT_SEC = 60
CLEAN_TRIP_BEFORE_RUN = True
PYTHON_EXE = sys.executable

SIMPLIFY_PDF_ON_FAILURE = True


# =============================================================================
# Helpers
# =============================================================================


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


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

def compact_result_row(row: dict[str, Any]) -> dict[str, Any]:
    pdflatex_original_runtime = row.get("pdflatex_runtime_sec")
    pdflatex_no_labels_runtime = row.get("pdflatex_no_labels_runtime_sec")

    pdf_total_runtime_sec = None
    try:
        parts = []
        if pdflatex_original_runtime is not None and pdflatex_original_runtime != "":
            parts.append(float(pdflatex_original_runtime))
        if pdflatex_no_labels_runtime is not None and pdflatex_no_labels_runtime != "":
            parts.append(float(pdflatex_no_labels_runtime))
        if parts:
            pdf_total_runtime_sec = round(sum(parts), 6)
    except Exception:
        pdf_total_runtime_sec = None

    return {
        "track": row.get("track"),
        "track_stem": row.get("track_stem"),
        "run": row.get("run"),

        "solver_status": row.get("solver_status"),
        "solver_returncode": row.get("solver_returncode"),
        "solver_timeout": row.get("solver_timeout"),
        "solver_runtime_sec": row.get("solver_runtime_sec"),

        "trip_exists": row.get("trip_exists"),
        "trip_parse_error": row.get("trip_parse_error"),
        "coordinate_count": row.get("coordinate_count"),
        "move_count": row.get("move_count"),
        "start_coordinate": row.get("start_coordinate"),
        "end_coordinate": row.get("end_coordinate"),
        "unique_position_count": row.get("unique_position_count"),
        "repeated_coordinate_count": row.get("repeated_coordinate_count"),
        "consecutive_repeat_count": row.get("consecutive_repeat_count"),
        "total_euclidean_length": row.get("total_euclidean_length"),
        "avg_step_length": row.get("avg_step_length"),
        "max_step_length": row.get("max_step_length"),

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


def short_text(text: str, limit: int = 4000) -> str:
    """Store only compact failure context, not full logs."""
    if not text:
        return ""
    return text[-limit:]


def discover_tracks() -> list[Path]:
    if not TRACKS_DIR.exists():
        return []
    return sorted([p for p in TRACKS_DIR.glob("*.t") if p.is_file()], key=natural_key)


def expected_trip_path(track_file: Path) -> Path:
    """Match main.py write_csv convention: output/track_<id>/g/<track>_trip.csv."""
    trip_name = track_file.name.replace(".t", "_trip.csv")
    parts = trip_name.split("_")
    if len(parts) >= 2:
        return OUTPUT_ROOT / f"track_{parts[1]}" / CONSTRUCTION_TYPE / trip_name
    return OUTPUT_ROOT / track_file.stem / CONSTRUCTION_TYPE / trip_name

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

    write_json(BENCHMARK_DIR / "system_specs.json", specs)

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

    step_lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(coords, coords[1:])]
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
# Solver, visualise.pl, pdflatex
# =============================================================================


def run_solver(track_file: Path, run_dir: Path, run_idx: int) -> dict[str, Any]:
    trip_path = expected_trip_path(track_file)
    trip_path.parent.mkdir(parents=True, exist_ok=True)
    if CLEAN_TRIP_BEFORE_RUN and trip_path.exists():
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

    row: dict[str, Any] = {
        "track": track_file.name,
        "track_stem": track_file.stem,
        "run": run_idx,
        "solver_status": status,
        "solver_returncode": returncode,
        "solver_timeout": timeout,
        "solver_runtime_sec": round(runtime, 6),
        "solver_start_time": start_time,
        "solver_command": " ".join(cmd),
        "source_trip_path": str(trip_path),
        "run_dir": str(run_dir),
    }

    if status != "ok":
        row["solver_stdout_tail"] = short_text(stdout)
        row["solver_stderr_tail"] = short_text(stderr)

    row.update(parse_trip_csv(trip_path))

    copied_trip_path = ""
    if trip_path.exists() and row.get("trip_parse_error") == "":
        copied_trip = run_dir / trip_path.name
        shutil.copy2(trip_path, copied_trip)
        copied_trip_path = str(copied_trip)

    row["trip_path"] = copied_trip_path
    return row


def run_visualise(track_file: Path, trip_file: Path, run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    if not VISUALISE_PL.exists():
        return {"visualise_status": "missing_visualise_pl"}
    if not trip_file.exists():
        return {"visualise_status": "missing_trip"}

    tex_path = run_dir / f"{track_file.stem}.tex"
    perl_exe = shutil.which("perl") or "perl"
    cmd = [perl_exe, str(VISUALISE_PL), str(track_file), str(trip_file), str(tex_path)]

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
    with tex_path.open("r", encoding="utf-8", errors="replace") as src, simplified.open("w", encoding="utf-8") as dst:
        for line in src:
            if "\\node[anchor=south west]" in line:
                continue
            dst.write(line)
    return simplified


def run_pdflatex_once(tex_path: Path, suffix: str = "") -> dict[str, Any]:
    pdflatex_exe = shutil.which("pdflatex")
    if pdflatex_exe is None:
        return {"pdflatex_status": "missing_pdflatex", "pdf_path": ""}

    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.exists():
        pdf_path.unlink()

    cmd = [pdflatex_exe, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
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
        return {"pdflatex_status": "missing_tex_file", "pdf_path": ""}

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
# Aggregation
# =============================================================================


def is_success(row: dict[str, Any]) -> bool:
    return (
        row.get("solver_status") == "ok"
        and row.get("trip_exists") is True
        and row.get("trip_parse_error") == ""
        and row.get("move_count") is not None
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
        moves = [int(r["move_count"]) for r in successes]
        runtimes_all = [float(r["solver_runtime_sec"]) for r in group]
        runtimes_success = [float(r["solver_runtime_sec"]) for r in successes]
        best = min(successes, key=lambda r: (int(r["move_count"]), float(r["solver_runtime_sec"]))) if successes else None

        summary.append({
            "track": track,
            "runs": len(group),
            "successes": len(successes),
            "best_moves": min(moves) if moves else None,
            "best_run": best.get("run") if best else None,
            "avg_moves": round(statistics.mean(moves), 3) if moves else None,
            "stdev_moves": round(stdev_or_zero([float(m) for m in moves]), 3) if moves else None,
            "min_solver_runtime_sec_success": round(min(runtimes_success), 6) if runtimes_success else None,
            "avg_solver_runtime_sec_all": round(statistics.mean(runtimes_all), 6) if runtimes_all else None,
            "avg_solver_runtime_sec_success": round(statistics.mean(runtimes_success), 6) if runtimes_success else None,
            "stdev_solver_runtime_sec_success": round(stdev_or_zero(runtimes_success), 6) if runtimes_success else None,
            "timeouts": sum(1 for r in group if r.get("solver_timeout")),
            "failures": sum(1 for r in group if r.get("solver_status") != "ok"),
            "visualise_successes": sum(1 for r in group if r.get("visualise_status") == "ok"),
            "pdf_successes": sum(1 for r in group if str(r.get("pdflatex_status", "")).startswith("ok")),
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
                int(row["move_count"]),
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
    print("Benchmark: g construction heuristic")
    print("=" * 80)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Tracks directory: {TRACKS_DIR}")
    print(f"Output directory: {BENCHMARK_DIR}")
    print(f"Runs per track:   {RUNS_PER_TRACK}")
    print(f"pdflatex:         {shutil.which('pdflatex') or 'missing'}")
    print("Tracks:")
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

            row = run_solver(track_file, run_dir, run_idx)

            trip_path = Path(row["trip_path"]) if row.get("trip_path") else None
            if is_success(row) and trip_path is not None and trip_path.is_file():
                row.update(run_visualise(track_file, trip_path, run_dir, row))
                tex_path = Path(row["tex_path"]) if row.get("tex_path") else None
                if tex_path is not None and tex_path.is_file() and row.get("visualise_status") == "ok":
                    row.update(run_pdflatex(tex_path))
                else:
                    row.update({"pdflatex_status": "skipped_no_tex", "pdf_path": ""})
            else:
                row.update({"visualise_status": "skipped_no_successful_trip"})
                row.update({"pdflatex_status": "skipped_no_successful_trip", "pdf_path": ""})

            write_json(run_dir / "metadata.json", row)
            rows.append(row)
            write_csv(RESULTS_CSV, [compact_result_row(r) for r in rows])

            print(
                f"  solver={row.get('solver_status')} "
                f"time={row.get('solver_runtime_sec')}s "
                f"moves={row.get('move_count')} "
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
    print(f"Solutions:        {SOLUTIONS_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())