from src.construction.rdgreedy import LOOKAHEAD_STEPS, RdGreedy
from src.improvement.corner_beam_search import optimize_corner_path
from src.state import States
from src.track import (
    CORNER_GROUPING_THRESHOLD,
    GATE_LENGTH,
    Corner,
    Position,
    Track,
)


class EvolutionaryAlgo:
    def __init__(
        self,
        track: Track,
        corner_grouping_threshold: float = CORNER_GROUPING_THRESHOLD,
        gate_length: int = GATE_LENGTH,
    ) -> None:
        self.track = track
        self.corners = Corner.group_corner_positions(
            track, self.track.get_corners(), corner_grouping_threshold, gate_length
        )

    def improve(self, path: States) -> States | None:
        # print(path)
        path_positions = path.get_positions()
        relevant_corners = [
            corner for corner in self.corners if corner.is_relevant(path_positions)
        ]
        # self.track.write_track_with_corners(relevant_corners)
        # for corner in relevant_corners:
        #     gates = corner.get_corner_gates(self.track)
        #     print(f"Corner apex={corner.get_apex()}: gates = {gates}")
        is_line_sector = True
        solution_paths: list[States] = [States([path[0]])]
        # print(len(path))
        crossed_gates = set()
        for i in range(len(path)-1):
            current_pos: Position = path[i].position
            if current_pos.is_finish():
                # print(len(solution_paths))
                solution_paths = self._mutate_to_finish(solution_paths)
                if len(solution_paths) == 0:
                    return None
                continue
            next_pos: Position = path[i + 1].position
            for corner in relevant_corners:
                gate: tuple[tuple[float, float], tuple[float, float]] = (
                    corner.get_crossed_gate(current_pos, next_pos)
                )
                # print(corner.get_corner_gates())
                # print(gate)
                if gate and gate not in crossed_gates:
                    crossed_gates.add(gate)
                    # print(
                    #     f"gate crossed at step {i}, state: {path[i]}, next: {path[i + 1]}"
                    # )
                    if is_line_sector:
                        # print("\nentry gate crossed")
                        # Generate N solution path with different corner entry
                        # to evaluate at the end of the corner
                        # print(f"evaluating {len(solution_paths)} paths: ")
                        # for sol_path in solution_paths:
                        #     print(sol_path)
                        solution_paths = self._mutate_line(solution_paths, gate)
                        if not solution_paths:
                            return None
                        is_line_sector = False
                    else:
                        # print("\nexit gate crossed")
                        # # Complete the N solution generated with corner performed
                        # print(f"evaluating {len(solution_paths)} paths: ")
                        # for sol_path in solution_paths:
                        #     print(sol_path)
                        solution_paths = self._mutate_corner(
                            solution_paths, corner, gate
                        )
                        is_line_sector = True
                        if len(solution_paths) == 0:
                            return None

        solution_paths = self._mutate_to_finish(solution_paths)
        if not solution_paths:
            return None

        return self._select_best(solution_paths)

    def _mutate_line(self, paths_to_mutate: list[States], gate, n=7):
        # print("#" * 50 + " MUTATE LINE " + "#" * 50)
        paths = set()
        negate_set = set()
        for path in paths_to_mutate:
            if path[-1] in negate_set:
                continue
            # print(f"mutating path: \n{path}\n")
            temp_track = self.track.deepcopy()
            temp_track.change_start(
                temp_track.get_position(path[-1].position.get_coordinates())
            )

            # Guard 1 — only offer real track cells as gate targets
            gate_candidates = [
                pos
                for pos in temp_track.supercover_line(
                    gate[0][0], gate[0][1], gate[1][0], gate[1][1]
                )
                if not pos.is_grass_or_obstacle()
            ]
            if not gate_candidates:
                continue

            # print(f"corner entries: {gate_candidates}")

            temp_track.change_finish_positions(gate_candidates)
            # print(f"On track: \n{temp_track}")
            speed_limit = 2
            lookahead = LOOKAHEAD_STEPS
            for _ in range(n):
                heuristic = RdGreedy(temp_track)
                # print(f"using speed limit: {speed_limit}")
                # print(f"using lookahead: {lookahead}")
                sol_path = heuristic.greedy_path_constructor(
                    False, path[-1].vector, negate_set, speed_limit, lookahead,
                )
                if not sol_path:
                    negate_set.add(path[-1])
                else:
                    result = path.concat(States(sol_path[1:]))
                    paths.add(result)
                speed_limit += 1
                if speed_limit >= 4:
                    lookahead += 1
        return list(paths)

    def _mutate_corner(self, paths_to_mutate, corner, exit_gate):
        print("#" * 50 + " MUTATE CORNER " + "#" * 50)
        output = []
        for path in paths_to_mutate:
            entry_state = path[-1]

            print(f"\nmutating path: \n{path}\n")
            corner_path = optimize_corner_path(
                self.track, entry_state, corner.get_apex(), exit_gate
            )
            if corner_path:
                paths = States(corner_path[1:])
                print(f"Found path: \n{paths}")
                output.append(path.concat(paths))
        return output

    def _mutate_to_finish(self, paths_to_mutate: list[States]):
        print("#" * 50 + " MUTATE TO FINISH " + "#" * 50)
        paths: list[States] = []
        temp_track = self.track.deepcopy()
        for path in paths_to_mutate:
            temp_track.change_start(
                temp_track.get_position(path[-1].position.get_coordinates())
            )
            heuristic = RdGreedy(temp_track)
            sol_path = heuristic.greedy_path_constructor(
                False, path[-1].vector, speed_limit=None
            )
            if sol_path:
                print(f"Found path: \n{sol_path}")
                paths.append(path.concat(States(sol_path[1:])))
        return paths

    def _select_best(self, solutions: list[States]) -> States:
        return sorted(solutions, key=lambda solution: len(solution))[0]
