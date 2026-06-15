from src.construction.rdgreedy import LOOKAHEAD_STEPS, RdGreedy
from src.improvement.corner_beam_search import optimize_corner_path
from src.state import State, States
from src.vector import Vector
from src.track import (
    CORNER_GROUPING_THRESHOLD,
    GATE_LENGTH,
    Corner,
    Position,
    Track,
)
import math

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

    # def improve(self, path: States) -> States | None:
    #     # print(path)
    #     path_positions = path.get_positions()
    #     relevant_corners = [
    #         corner for corner in self.corners if corner.is_relevant(path_positions)
    #     ]
    #     # self.track.write_track_with_corners(relevant_corners)
    #     # for corner in relevant_corners:
    #     #     gates = corner.get_corner_gates(self.track)
    #     #     print(f"Corner apex={corner.get_apex()}: gates = {gates}")
    #     is_line_sector = True
    #     solution_paths: list[States] = [States([path[0]])]
    #     # print(len(path))
    #     crossed_gates = set()
    #     for i in range(len(path)-1):
    #         current_pos: Position = path[i].position
    #         if current_pos.is_finish():
    #             # print(len(solution_paths))
    #             solution_paths = self._mutate_to_finish(solution_paths)
    #             if len(solution_paths) == 0:
    #                 return None
    #             continue
    #         next_pos: Position = path[i + 1].position
    #         for corner in relevant_corners:
    #             gate: tuple[tuple[float, float], tuple[float, float]] = (
    #                 corner.get_crossed_gate(current_pos, next_pos)
    #             )
    #             # print(corner.get_corner_gates())
    #             # print(gate)
    #             if gate and gate not in crossed_gates:
    #                 crossed_gates.add(gate)
    #                 # print(
    #                 #     f"gate crossed at step {i}, state: {path[i]}, next: {path[i + 1]}"
    #                 # )
    #                 if is_line_sector:
    #                     # print("\nentry gate crossed")
    #                     # Generate N solution path with different corner entry
    #                     # to evaluate at the end of the corner
    #                     # print(f"evaluating {len(solution_paths)} paths: ")
    #                     # for sol_path in solution_paths:
    #                     #     print(sol_path)
    #                     solution_paths = self._mutate_line(solution_paths, gate)
    #                     if not solution_paths:
    #                         return None
    #                     is_line_sector = False
    #                 else:
    #                     # print("\nexit gate crossed")
    #                     # # Complete the N solution generated with corner performed
    #                     # print(f"evaluating {len(solution_paths)} paths: ")
    #                     # for sol_path in solution_paths:
    #                     #     print(sol_path)
    #                     solution_paths = self._mutate_corner(
    #                         solution_paths, corner, gate
    #                     )
    #                     is_line_sector = True
    #                     if len(solution_paths) == 0:
    #                         return None

    #     construction_cleanup = self._skip_blobs_zigzags_loops(path)

    #     solution_paths = self._mutate_to_finish(solution_paths)
    #     if not solution_paths:
    #         return construction_cleanup

    #     best = self._select_best(solution_paths)
    #     best = self._skip_blobs_zigzags_loops(best)

    #     if len(best) <= len(construction_cleanup):
    #         return best

    #     return construction_cleanup


    #     # solution_paths = self._mutate_to_finish(solution_paths)
    #     # if not solution_paths:
    #     #     return None

    #     # return self._shortcut_path(self._select_best(solution_paths))

    def improve(self, path: States) -> States | None:
        path_positions = path.get_positions()
        relevant_corners = [
            corner for corner in self.corners if corner.is_relevant(path_positions)
        ]

        # Safe fallback: even if line/corner mutation fails, return a cleaned
        # construction path instead of None.
        construction_cleanup = self._skip_blobs_zigzags_loops(path)

        is_line_sector = True
        solution_paths: list[States] = [States([path[0]])]
        crossed_gates = set()

        for i in range(len(path) - 1):
            current_pos: Position = path[i].position
            next_pos: Position = path[i + 1].position

            for corner in relevant_corners:
                gate = corner.get_crossed_gate(current_pos, next_pos)

                if gate and gate not in crossed_gates:
                    crossed_gates.add(gate)

                    if is_line_sector:
                        solution_paths = self._mutate_line(solution_paths, gate)

                        if not solution_paths:
                            return construction_cleanup

                        is_line_sector = False

                    else:
                        solution_paths = self._mutate_corner(
                            solution_paths,
                            corner,
                            gate,
                        )

                        if not solution_paths:
                            return construction_cleanup

                        is_line_sector = True

        solution_paths = self._mutate_to_finish(solution_paths)

        if not solution_paths:
            return construction_cleanup

        best = self._select_best(solution_paths)
        best = self._skip_blobs_zigzags_loops(best)

        if len(best) <= len(construction_cleanup):
            return best

        return construction_cleanup

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
        
    def _shortcut_path(self, path: States) -> States:

        if len(path) <= 2:
            return path

        start_vx, start_vy = path[0].vector.get_point()
        start_key = (0, start_vx, start_vy)

        queue = [start_key]
        parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {
            start_key: None
        }
        parent_state: dict[tuple[int, int, int], State] = {
            start_key: path[0]
        }

        final_key = None
        head = 0

        while head < len(queue):
            i, vx, vy = queue[head]
            head += 1

            current_pos = path[i].position
            current_vector = Vector.from_point(vx, vy)

            if i == len(path) - 1:
                final_key = (i, vx, vy)
                break

            for j in range(len(path) - 1, i, -1):
                next_pos = path[j].position
                next_vector = Vector.get_vector(
                    current_pos.get_coordinates(),
                    next_pos.get_coordinates(),
                )
                nvx, nvy = next_vector.get_point()

                if abs(nvx - vx) > 1 or abs(nvy - vy) > 1:
                    continue

                if not self.track.is_valid_move(
                    current_pos,
                    next_pos,
                    current_vector,
                    next_vector,
                ):
                    continue

                key = (j, nvx, nvy)

                if key in parent:
                    continue

                parent[key] = (i, vx, vy)
                parent_state[key] = State(next_pos, next_vector)
                queue.append(key)

                if j == len(path) - 1:
                    final_key = key
                    queue = []
                    break

        if final_key is None:
            return path

        shortcut_states = []
        key = final_key

        while key is not None:
            shortcut_states.append(parent_state[key])
            key = parent[key]

        shortcut_states.reverse()
        shortcut = States(shortcut_states)

        if len(shortcut) < len(path):
            return shortcut

        return path

    def _skip_blobs_zigzags_loops(
        self,
        path: States,
        max_passes: int = 3,
        min_window: int = 5,
        max_window: int = 45,
        corridor_radius: float = 3.0,
        beam_width: int = 120,
        move_chord_ratio_threshold: float = 2.2,
        detour_ratio_threshold: float = 1.8,
    ) -> States:
        if len(path) <= 2:
            return path

        best = self._shortcut_path(path)

        for _ in range(max_passes):
            changed = False
            n = len(best)

            for i in range(0, n - min_window):
                last_j = min(n - 1, i + max_window)

                for j in range(last_j, i + min_window - 1, -1):
                    original_moves = j - i
                    max_bridge_moves = original_moves - 1

                    if max_bridge_moves <= 0:
                        continue

                    if not self._segment_looks_wasteful(
                        best,
                        i,
                        j,
                        move_chord_ratio_threshold,
                        detour_ratio_threshold,
                    ):
                        continue

                    bridge = self._find_local_bridge(
                        best,
                        i,
                        j,
                        max_moves=max_bridge_moves,
                        corridor_radius=corridor_radius,
                        beam_width=beam_width,
                    )

                    if bridge is None:
                        continue

                    if len(bridge) >= original_moves + 1:
                        continue

                    candidate = self._splice_bridge(best, i, j, bridge)
                    candidate = self._shortcut_path(candidate)

                    if len(candidate) < len(best):
                        best = candidate
                        changed = True
                        break

                if changed:
                    break

            if not changed:
                break

        if len(best) < len(path):
            return best

        return path

    def _segment_looks_wasteful(
        self,
        path: States,
        i: int,
        j: int,
        move_chord_ratio_threshold: float,
        detour_ratio_threshold: float,
    ) -> bool:
        if j <= i + 1:
            return False

        positions = [path[k].position for k in range(i, j + 1)]
        move_count = j - i

        start = positions[0]
        end = positions[-1]

        chord = max(start.get_distance_to(end), 1e-9)

        polyline_length = 0.0
        for k in range(len(positions) - 1):
            polyline_length += positions[k].get_distance_to(positions[k + 1])

        unique_positions = len(set(positions))
        repeated_positions = len(positions) - unique_positions

        min_x = min(p.x for p in positions)
        max_x = max(p.x for p in positions)
        min_y = min(p.y for p in positions)
        max_y = max(p.y for p in positions)

        bbox_area = max(1, (max_x - min_x + 1) * (max_y - min_y + 1))

        move_chord_ratio = move_count / chord
        detour_ratio = polyline_length / chord
        density_ratio = len(positions) / bbox_area

        if repeated_positions > 0:
            return True

        if move_chord_ratio >= move_chord_ratio_threshold:
            return True

        if detour_ratio >= detour_ratio_threshold:
            return True

        if move_count >= 8 and density_ratio >= 0.55:
            return True

        return False

    def _find_local_bridge(
        self,
        path: States,
        i: int,
        j: int,
        max_moves: int,
        corridor_radius: float,
        beam_width: int,
    ) -> States | None:
        entry_state = path[i]
        exit_pos = path[j].position

        ax = entry_state.position.x
        ay = entry_state.position.y
        bx = exit_pos.x
        by = exit_pos.y

        margin = math.ceil(corridor_radius) + 1
        min_x = min(ax, bx) - margin
        max_x = max(ax, bx) + margin
        min_y = min(ay, by) - margin
        max_y = max(ay, by) + margin

        def in_corridor(pos: Position) -> bool:
            if pos == exit_pos:
                return True

            if pos.x < min_x or pos.x > max_x:
                return False

            if pos.y < min_y or pos.y > max_y:
                return False

            distance = self._point_to_segment_distance(
                pos.x,
                pos.y,
                ax,
                ay,
                bx,
                by,
            )

            return distance <= corridor_radius

        start_vx, start_vy = entry_state.vector.get_point()
        start_key = (
            entry_state.position.x,
            entry_state.position.y,
            start_vx,
            start_vy,
        )

        parent: dict[
            tuple[int, int, int, int],
            tuple[int, int, int, int] | None,
        ] = {
            start_key: None
        }

        parent_state: dict[tuple[int, int, int, int], State] = {
            start_key: entry_state
        }

        current_layer = [start_key]
        final_key = None

        for _ in range(max_moves):
            candidate_by_key: dict[
                tuple[int, int, int, int],
                tuple[float, tuple[int, int, int, int], State],
            ] = {}

            for current_key in current_layer:
                current_state = parent_state[current_key]

                for next_state in current_state.generate_moves(self.track):
                    next_pos = next_state.position

                    if next_pos != exit_pos and not in_corridor(next_pos):
                        continue

                    nvx, nvy = next_state.vector.get_point()
                    next_key = (next_pos.x, next_pos.y, nvx, nvy)

                    if next_key in parent:
                        continue

                    if next_pos == exit_pos:
                        if not self._can_continue_after_bridge(
                            path,
                            j,
                            next_state,
                        ):
                            continue

                        parent[next_key] = current_key
                        parent_state[next_key] = next_state
                        final_key = next_key
                        break

                    distance_to_exit = next_pos.get_distance_to(exit_pos)
                    corridor_distance = self._point_to_segment_distance(
                        next_pos.x,
                        next_pos.y,
                        ax,
                        ay,
                        bx,
                        by,
                    )
                    speed = max(next_state.vector.magnitude, 1e-9)

                    score = (
                        distance_to_exit
                        + 0.25 * corridor_distance
                        + 0.05 / speed
                    )

                    old = candidate_by_key.get(next_key)

                    if old is None or score < old[0]:
                        candidate_by_key[next_key] = (
                            score,
                            current_key,
                            next_state,
                        )

                if final_key is not None:
                    break

            if final_key is not None:
                break

            if not candidate_by_key:
                return None

            ranked_candidates = sorted(
                candidate_by_key.items(),
                key=lambda item: item[1][0],
            )

            current_layer = []

            for next_key, (_, previous_key, next_state) in ranked_candidates[:beam_width]:
                parent[next_key] = previous_key
                parent_state[next_key] = next_state
                current_layer.append(next_key)

        if final_key is None:
            return None

        bridge_states = []
        key = final_key

        while key is not None:
            bridge_states.append(parent_state[key])
            key = parent[key]

        bridge_states.reverse()
        return States(bridge_states)

    def _can_continue_after_bridge(
        self,
        path: States,
        j: int,
        bridge_exit_state: State,
    ) -> bool:
        if j >= len(path) - 1:
            return True

        next_pos = path[j + 1].position
        next_vector = Vector.get_vector(
            bridge_exit_state.position.get_coordinates(),
            next_pos.get_coordinates(),
        )

        return self.track.is_valid_move(
            bridge_exit_state.position,
            next_pos,
            bridge_exit_state.vector,
            next_vector,
        )

    def _splice_bridge(
        self,
        path: States,
        i: int,
        j: int,
        bridge: States,
    ) -> States:
        return States(
            path.states[:i]
            + bridge.states
            + path.states[j + 1:]
        )

    @staticmethod
    def _point_to_segment_distance(
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> float:
        dx = bx - ax
        dy = by - ay

        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)

        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))

        closest_x = ax + t * dx
        closest_y = ay + t * dy

        return math.hypot(px - closest_x, py - closest_y)
        