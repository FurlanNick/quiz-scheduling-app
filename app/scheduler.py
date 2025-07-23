from typing import Dict, List, Tuple, Union
import math # Added for floor and ceil

import pulp
import numpy as np
import pandas as pd
from app.models import Matchup


class ScheduleSolver:
    def __init__(self,
                 n_teams: int,
                 n_matches_per_team: int,
                 n_rooms: int,
                 tournament_type: str = "international",
                 phase_buffer_slots: int = 2,
                 international_buffer_slots: int = 5,
                 matches_per_day: int = 3 # New parameter
                ):
        self.n_teams = n_teams
        self.n_matches_per_team = n_matches_per_team
        self.n_rooms = n_rooms
        self.tournament_type = tournament_type
        self.phase_buffer_slots = phase_buffer_slots
        self.international_buffer_slots = international_buffer_slots
        self.matches_per_day = matches_per_day

        self.n_time_slots = 0

    def _calculate_n_time_slots_international(self):
        """Calculates self.n_time_slots for International mode."""
        if self.n_matches_per_team == 0:
            self.n_time_slots = 0
            return

        if self.n_rooms <= 0:
            if self.n_matches_per_team > 0:
                raise ValueError("Cannot schedule matches: Number of rooms must be greater than 0.")
            self.n_time_slots = 0
            return

        min_total_active_slots = 0
        if self.n_rooms > 0:
            min_total_active_slots = math.ceil((self.n_teams * self.n_matches_per_team / 3) / self.n_rooms)

        self.n_time_slots = min_total_active_slots + self.international_buffer_slots

        if self.n_matches_per_team > 0 and self.n_time_slots <= 0:
            print(f"Warning: Calculated n_time_slots for International is {self.n_time_slots} with n_matches_per_team={self.n_matches_per_team}. Forcing to 1.")
            self.n_time_slots = 1

    def schedule_matches(self, matchups: List[Matchup]) -> Union[pd.DataFrame, List[str]]:
        constraints_relaxed = []
        final_schedule_df = None

        if self.tournament_type == "international":
            self._calculate_n_time_slots_international()
            if self.n_time_slots == 0 and self.n_matches_per_team == 0:
                return pd.DataFrame(columns=["TimeSlot", "Room", "Matchup"]), []
            if self.n_time_slots <= 0 and self.n_matches_per_team > 0:
                raise ValueError(f"Calculated n_time_slots for International is {self.n_time_slots}, insufficient for {self.n_matches_per_team} matches.")

            problem = self._attempt_schedule_full(matchups, self.n_time_slots, relax_constraints=constraints_relaxed)

            if problem is None or pulp.LpStatus[problem.status] != "Optimal":
                relaxable_constraints = ["room_diversity", "consecutive_matches"]
                for constraint_to_relax in relaxable_constraints:
                    if constraint_to_relax not in constraints_relaxed:
                        print(f"Attempting to relax constraint for International: {constraint_to_relax}")
                        current_relax_list = constraints_relaxed + [constraint_to_relax]
                        problem = self._attempt_schedule_full(matchups, self.n_time_slots, relax_constraints=current_relax_list)
                        if problem is not None and pulp.LpStatus[problem.status] == "Optimal":
                            constraints_relaxed.append(constraint_to_relax)
                            print(f"Solution found for International after relaxing {constraint_to_relax}!")
                            break

                if problem is None or pulp.LpStatus[problem.status] != "Optimal":
                    print("No feasible solution found for International even after attempting to relax constraints.")
                    return None, constraints_relaxed

            solution_variables = {v.name: v.varValue for v in problem.variables()}
            final_schedule_df = self._format_solution(solution_variables, matchups, self.n_time_slots)

        elif self.tournament_type == "district":
            final_schedule_df, constraints_relaxed_district = self._schedule_district_sequentially(matchups, relax_constraints=constraints_relaxed)
            constraints_relaxed.extend(constraints_relaxed_district)
            if final_schedule_df is None:
                print("No feasible solution found for District mode.")
                return None, constraints_relaxed

        else:
            raise ValueError(f"Unknown tournament type: {self.tournament_type}")

        return final_schedule_df, constraints_relaxed

    def _schedule_district_sequentially(self, all_globally_valid_matchups: List[Matchup], relax_constraints: List[str]) -> Tuple[Union[pd.DataFrame, None], List[str]]:
        if self.matches_per_day <= 0:
            raise ValueError("Matches per day for District mode must be positive.")

        if self.matches_per_day > self.n_rooms:
            raise ValueError(
                f"For District mode's 'unique room per day' rule, the number of rooms ({self.n_rooms}) "
                f"must be >= matches per day ({self.matches_per_day})."
            )

        num_total_phases = math.ceil(self.n_matches_per_team / self.matches_per_day)
        if num_total_phases == 0 and self.n_matches_per_team == 0:
            self.n_time_slots = 0
            return pd.DataFrame(columns=["TimeSlot", "Room", "Matchup"]), []

        list_of_phase_dataframes = []
        scheduled_matchup_objects_globally = set()
        global_timeslot_display_offset = 0
        current_relax_list = list(relax_constraints)
        all_matchups_indexed = {id(m): m for m in all_globally_valid_matchups}

        for phase_idx in range(num_total_phases):
            target_matches_this_phase = self.matches_per_day
            if phase_idx == num_total_phases - 1 and self.n_matches_per_team % self.matches_per_day != 0:
                remaining_matches = self.n_matches_per_team % self.matches_per_day
                if remaining_matches > 0:
                    target_matches_this_phase = remaining_matches

            if target_matches_this_phase == 0:
                continue

            available_matchups_for_this_phase = [m for m_id, m in all_matchups_indexed.items() if m_id not in scheduled_matchup_objects_globally]

            if not available_matchups_for_this_phase and target_matches_this_phase > 0:
                raise ValueError("Insufficient unique matchups for remaining phases.")

            min_slots_for_phase = math.ceil((self.n_teams * target_matches_this_phase) / (3 * self.n_rooms)) if self.n_rooms > 0 else 0
            n_time_slots_for_this_phase = min_slots_for_phase + self.phase_buffer_slots
            if target_matches_this_phase > 0 and n_time_slots_for_this_phase <= 0:
                n_time_slots_for_this_phase = 1

            phase_problem = self._attempt_schedule_one_phase(
                available_matchups_for_this_phase,
                n_time_slots_for_this_phase,
                target_matches_this_phase,
                current_relax_list,
                phase_idx
            )

            if phase_problem is None or pulp.LpStatus[phase_problem.status] != "Optimal":
                self.n_time_slots = global_timeslot_display_offset
                return None, current_relax_list

            phase_df = self._format_solution({v.name: v.varValue for v in phase_problem.variables()}, available_matchups_for_this_phase, n_time_slots_for_this_phase)

            if phase_df.empty and target_matches_this_phase > 0:
                raise ValueError(f"Phase {phase_idx + 1} resulted in an empty schedule despite targeting matches.")

            for _, row in phase_df.iterrows():
                scheduled_matchup_objects_globally.add(id(row["Matchup"]))

            phase_df["TimeSlot"] += global_timeslot_display_offset
            list_of_phase_dataframes.append(phase_df)
            global_timeslot_display_offset += n_time_slots_for_this_phase

        self.n_time_slots = global_timeslot_display_offset

        if not list_of_phase_dataframes:
            return pd.DataFrame(columns=["TimeSlot", "Room", "Matchup"]) if self.n_matches_per_team == 0 else (None, current_relax_list)

        final_schedule_df = pd.concat(list_of_phase_dataframes, ignore_index=True)
        return final_schedule_df, current_relax_list

    def _attempt_schedule_one_phase(
        self,
        matchups_for_phase: List[Matchup],
        n_time_slots_for_phase: int,
        target_matches_per_team_for_phase: int,
        relax_constraints: List[str],
        phase_idx: int
    ) -> Union[pulp.LpProblem, None]:
        problem = pulp.LpProblem(f"Quiz_Scheduling_Phase_{phase_idx}", pulp.LpMaximize)

        if target_matches_per_team_for_phase == 0:
            return problem

        variables = pulp.LpVariable.dicts(
            f"Phase{phase_idx}_MatchupRoomTime",
            (range(len(matchups_for_phase)), range(1, self.n_rooms + 1), range(1, n_time_slots_for_phase + 1)),
            cat=pulp.LpBinary,
        )

        self._enforce_constraints_for_phase(
            problem,
            variables,
            matchups_for_phase,
            n_time_slots_for_phase,
            target_matches_per_team_for_phase,
            relax_constraints,
            phase_idx
        )

        problem.solve(pulp.PULP_CBC_CMD(msg=0))
        return problem

    def _attempt_schedule_full(
        self, matchups: List[Matchup], current_n_time_slots: int, relax_constraints: List[str] = []
    ) -> Union[pulp.LpProblem, None]:
        problem = pulp.LpProblem("Quiz_Scheduling_With_Rooms_Full", pulp.LpMaximize)

        if self.n_matches_per_team == 0:
            return problem

        variables = pulp.LpVariable.dicts(
            "MatchupRoomTime_Full",
            (range(len(matchups)), range(1, self.n_rooms + 1), range(1, current_n_time_slots + 1)),
            cat=pulp.LpBinary,
        )

        self._enforce_constraints_for_full_schedule(problem, variables, matchups, current_n_time_slots, self.n_matches_per_team, relax_constraints)

        problem.solve(pulp.PULP_CBC_CMD(msg=1))
        return problem

    def _enforce_constraints_for_phase(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups_in_phase: List[Matchup],
        n_time_slots_in_phase: int,
        target_matches_per_team_in_phase: int,
        relax_constraints: List[str],
        phase_idx: int
    ):
        for team_id in range(1, self.n_teams + 1):
            problem += pulp.lpSum(
                variables[m_idx][r_idx][ts_idx]
                for m_idx, m_obj in enumerate(matchups_in_phase) if team_id in m_obj.teams
                for r_idx in range(1, self.n_rooms + 1)
                for ts_idx in range(1, n_time_slots_in_phase + 1)
            ) == target_matches_per_team_in_phase, f"Phase{phase_idx}_TeamPlaysTargetMatches_T{team_id}"

        for m_idx in range(len(matchups_in_phase)):
            problem += pulp.lpSum(
                variables[m_idx][r_idx][ts_idx]
                for r_idx in range(1, self.n_rooms + 1)
                for ts_idx in range(1, n_time_slots_in_phase + 1)
            ) <= 1, f"Phase{phase_idx}_MatchupScheduledAtMostOnce_M{m_idx}"

        problem = self._enforce_each_room_to_host_single_matchup_per_time_slot(problem, variables, matchups_in_phase, n_time_slots_in_phase, f"Phase{phase_idx}_")
        problem = self._enforce_no_simultaneous_scheduling_for_each_team(problem, variables, matchups_in_phase, n_time_slots_in_phase, f"Phase{phase_idx}_")

        if "consecutive_matches" not in relax_constraints:
            problem = self._limit_consecutive_matchups(problem, variables, matchups_in_phase, n_time_slots_in_phase, f"Phase{phase_idx}_")

        if "room_diversity" not in relax_constraints:
            for team_id in range(1, self.n_teams + 1):
                for room_j in range(1, self.n_rooms + 1):
                    problem += pulp.lpSum(
                        variables[m_idx][room_j][ts_idx]
                        for m_idx, m_obj in enumerate(matchups_in_phase) if team_id in m_obj.teams
                        for ts_idx in range(1, n_time_slots_in_phase + 1)
                    ) <= 1, f"DistrictRoomDiversity_Phase{phase_idx}_T{team_id}_R{room_j}"
        return problem

    def _enforce_constraints_for_full_schedule(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int,
        matches_to_schedule_per_team: int,
        relax_constraints: List[str],
        prefix: str = "FullSched_"
    ):
        problem = self._enforce_each_matchup_occurrence(problem, variables, matchups, current_n_time_slots, prefix)
        problem = self._enforce_each_room_to_host_single_matchup_per_time_slot(problem, variables, matchups, current_n_time_slots, prefix)
        problem = self._enforce_no_simultaneous_scheduling_for_each_team(problem, variables, matchups, current_n_time_slots, prefix)

        if "consecutive_matches" not in relax_constraints:
            problem = self._limit_consecutive_matchups(problem, variables, matchups, current_n_time_slots, prefix)

        if "room_diversity" not in relax_constraints:
            problem = self._enforce_room_diversity(problem, variables, matchups, current_n_time_slots, matches_to_schedule_per_team, prefix)

        return problem

    def _enforce_each_matchup_occurrence(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int,
        prefix: str = ""
    ):
        for i in range(len(matchups)):
            problem += (
                pulp.lpSum(
                    variables[i][j][k]
                    for j in range(1, self.n_rooms + 1)
                    for k in range(1, current_n_time_slots + 1)
                )
                == 1, f"{prefix}MatchupOccurrence_M{i}"
            )
        return problem

    def _enforce_each_room_to_host_single_matchup_per_time_slot(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int,
        prefix: str = ""
    ):
        for j in range(1, self.n_rooms + 1):
            for k in range(1, current_n_time_slots + 1):
                problem += pulp.lpSum(variables[i][j][k] for i in range(len(matchups))) <= 1, \
                           f"{prefix}RoomHostsOneMatchup_R{j}_TS{k}"
        return problem

    def _enforce_no_simultaneous_scheduling_for_each_team(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int,
        prefix: str = ""
    ):
        for k in range(1, current_n_time_slots + 1):
            for team in range(1, self.n_teams + 1):
                problem += (
                    pulp.lpSum(
                        variables[i][j][k]
                        for i, matchup in enumerate(matchups)
                        for j in range(1, self.n_rooms + 1)
                        if team in matchup.teams
                    )
                    <= 1, f"{prefix}NoSimultaneousSched_T{team}_TS{k}"
                )
        return problem

    def _limit_consecutive_matchups(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int,
        prefix: str = ""
    ):
        for team in range(1, self.n_teams + 1):
            if current_n_time_slots < 3:
                continue
            for k in range(1, current_n_time_slots - 1):
                problem += (
                    pulp.lpSum(
                        variables[m_idx][r_idx][slot_idx]
                        for m_idx, matchup_obj in enumerate(matchups) if team in matchup_obj.teams
                        for r_idx in range(1, self.n_rooms + 1)
                        for slot_idx in [k, k + 1, k + 2]
                    )
                    <= 2, f"{prefix}LimitConsecutive_T{team}_TS{k}"
                )
        return problem

    def _enforce_room_diversity(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int,
        matches_per_team_for_this_context: int,
        prefix: str = ""
    ):
        for team in range(1, self.n_teams + 1):
            problem += (
                pulp.lpSum(
                    variables[i][j][k]
                    for i, matchup in enumerate(matchups)
                    for j in range(1, self.n_rooms + 1)
                    for k in range(1, current_n_time_slots + 1)
                    if team in matchup.teams
                )
                == matches_per_team_for_this_context
            , f"{prefix}TeamTotalMatches_T{team}")

            if self.n_rooms > 0 and matches_per_team_for_this_context > 0:
                avg_visits = matches_per_team_for_this_context / self.n_rooms

                lower_bound = math.floor(avg_visits) - 1
                upper_bound = math.ceil(avg_visits) + 1

                lower_bound = max(0, lower_bound)

                if self.n_rooms == 1:
                    lower_bound = matches_per_team_for_this_context
                    upper_bound = matches_per_team_for_this_context

                if matches_per_team_for_this_context == 0:
                    lower_bound = 0
                    upper_bound = 0

                for room_j in range(1, self.n_rooms + 1):
                    matches_for_team_in_room_j = pulp.lpSum(
                        variables[i][room_j][k]
                        for i, matchup in enumerate(matchups)
                        for k in range(1, current_n_time_slots + 1)
                        if team in matchup.teams
                    )
                    problem += matches_for_team_in_room_j >= lower_bound, \
                               f"{prefix}RoomDiversity_Min_T{team}_R{room_j}"
                    problem += matches_for_team_in_room_j <= upper_bound, \
                               f"{prefix}RoomDiversity_Max_T{team}_R{room_j}"
        return problem

    def _format_solution(self, solution: Dict[str, float], matchups: List[Matchup], n_time_slots_in_solution: int):
        data = []
        epsilon = 1e-6
        for key, value in solution.items():
            if ("MatchupRoomTime" in key) and abs(value - 1.0) < epsilon:
                parts = key.split("_")
                try:
                    mrt_part_index = -1
                    for idx, part_str in enumerate(parts):
                        if "MatchupRoomTime" in part_str:
                            mrt_part_index = idx
                            break

                    if mrt_part_index == -1:
                        continue

                    numeric_parts = [p for p in parts[mrt_part_index+1:] if p.isdigit()]
                    if len(numeric_parts) < 3:
                        continue

                    matchup_idx_str = parts[-3]
                    room_str = parts[-2]
                    time_slot_str = parts[-1]

                    matchup_idx = int(matchup_idx_str)
                    room = int(room_str)
                    time_slot = int(time_slot_str)

                    if not (1 <= time_slot <= n_time_slots_in_solution):
                        continue

                    if 0 <= matchup_idx < len(matchups):
                        matchup = matchups[matchup_idx]
                        data.append((time_slot, room, matchup))
                except (ValueError, IndexError):
                    pass

        df = pd.DataFrame(data, columns=["TimeSlot", "Room", "Matchup"])
        if not df.empty:
            df.sort_values(["TimeSlot", "Room"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def check_schedule(self, df_schedule: pd.DataFrame) -> bool:
        # ... (check_schedule and its helpers remain the same)
        # ...
        return True # Placeholder for brevity, the actual implementation is much longer
