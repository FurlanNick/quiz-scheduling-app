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
                 international_buffer_slots: int = 5
                ):
        self.n_teams = n_teams
        self.n_matches_per_team = n_matches_per_team
        self.n_rooms = n_rooms
        self.tournament_type = tournament_type
        self.phase_buffer_slots = phase_buffer_slots # Buffer for each active phase in District mode
        self.international_buffer_slots = international_buffer_slots # Overall buffer for International mode

        self.n_time_slots = 0
        self.active_slots_per_phase_counts = [] # For District: stores calculated active slots for each phase
        self.break_slot_indices = []            # For District: stores timeslot indices that are breaks
        self.end_of_phase_active_slot_indices = [] # For District: stores timeslot indices marking end of an active phase

        self.CHUNK_SIZE = 3 # Matches per phase for District mode


    def _calculate_schedule_structure(self):
        """
        Calculates self.n_time_slots and, for District mode, the phase/break structure.
        Sets: self.n_time_slots, self.active_slots_per_phase_counts,
              self.break_slot_indices, self.end_of_phase_active_slot_indices.
        This method must be called before PuLP variables are dimensioned.
        """
        if self.n_matches_per_team == 0:
            self.n_time_slots = 0
            self.active_slots_per_phase_counts = []
            self.break_slot_indices = []
            self.end_of_phase_active_slot_indices = []
            return

        if self.n_rooms <= 0:
            if self.n_matches_per_team > 0:
                raise ValueError("Cannot schedule matches: Number of rooms must be greater than 0.")
            self.n_time_slots = 0
            return

        if self.tournament_type == "district" and self.n_matches_per_team > self.CHUNK_SIZE:
            num_phases = math.ceil(self.n_matches_per_team / self.CHUNK_SIZE)
            num_breaks = num_phases - 1

            self.active_slots_per_phase_counts = []
            calculated_total_active_slots = 0

            for p in range(num_phases):
                matches_this_phase = self.CHUNK_SIZE
                if p == num_phases - 1:
                    matches_this_phase = self.n_matches_per_team - (p * self.CHUNK_SIZE)
                    if matches_this_phase <= 0: matches_this_phase = self.CHUNK_SIZE

                if matches_this_phase == 0 :
                    self.active_slots_per_phase_counts.append(0)
                    continue

                min_slots_needed = math.ceil((self.n_teams * matches_this_phase) / (3 * self.n_rooms))
                buffer_for_this_phase = self.phase_buffer_slots
                if min_slots_needed == 0 and matches_this_phase > 0 :
                    buffer_for_this_phase = 0

                slots_allocated_for_phase = min_slots_needed + buffer_for_this_phase
                if slots_allocated_for_phase == 0 and matches_this_phase > 0:
                    slots_allocated_for_phase = 1

                self.active_slots_per_phase_counts.append(slots_allocated_for_phase)
                calculated_total_active_slots += slots_allocated_for_phase

            self.n_time_slots = calculated_total_active_slots + num_breaks

            self.break_slot_indices = []
            self.end_of_phase_active_slot_indices = []
            current_ts_cursor = 0
            for p in range(num_phases):
                if p >= len(self.active_slots_per_phase_counts):
                    break
                current_ts_cursor += self.active_slots_per_phase_counts[p]
                self.end_of_phase_active_slot_indices.append(current_ts_cursor)
                if p < num_breaks:
                    current_ts_cursor += 1
                    self.break_slot_indices.append(current_ts_cursor)

            if current_ts_cursor != self.n_time_slots:
                self.n_time_slots = current_ts_cursor
        else:
            min_total_active_slots = 0
            if self.n_rooms > 0:
                 min_total_active_slots = math.ceil((self.n_teams * self.n_matches_per_team / 3) / self.n_rooms)

            self.n_time_slots = min_total_active_slots + self.international_buffer_slots
            self.active_slots_per_phase_counts = []
            self.break_slot_indices = []
            self.end_of_phase_active_slot_indices = []

        if self.n_matches_per_team > 0 and self.n_time_slots <= 0:
            print(f"Warning: Calculated n_time_slots is {self.n_time_slots} with n_matches_per_team={self.n_matches_per_team}. Forcing to 1 to allow variable definition.")
            self.n_time_slots = 1


    def schedule_matches(self, matchups: List[Matchup]) -> Union[pd.DataFrame, List[str]]:
        constraints_relaxed = []
        self._calculate_schedule_structure()

        if self.n_time_slots == 0 and self.n_matches_per_team == 0:
            return pd.DataFrame(columns=["TimeSlot", "Room", "Matchup"]), []
        if self.n_time_slots <= 0 and self.n_matches_per_team > 0:
            raise ValueError(f"Calculated n_time_slots is {self.n_time_slots}, which is insufficient for {self.n_matches_per_team} matches per team. Check input parameters and buffer values.")

        problem = self.attempt_schedule(matchups=matchups)

        if problem is None or pulp.LpStatus[problem.status] != "Optimal": # Check if problem is None
            # Try relaxing constraints
            relaxable_constraints = ["room_diversity", "consecutive_matches"]
            original_constraints_relaxed_count = len(constraints_relaxed)

            for constraint_to_relax in relaxable_constraints:
                if constraint_to_relax not in constraints_relaxed:
                    print(f"Attempting to relax constraint: {constraint_to_relax}")
                    current_relax_list = constraints_relaxed + [constraint_to_relax]
                    # Create a new problem object for the new attempt
                    problem = self.attempt_schedule(matchups, relax_constraints=current_relax_list)
                    if problem is not None and pulp.LpStatus[problem.status] == "Optimal":
                        constraints_relaxed.append(constraint_to_relax)
                        print(f"Solution found after relaxing {constraint_to_relax}!")
                        break

            if problem is None or pulp.LpStatus[problem.status] != "Optimal":
                print("No feasible solution found even after attempting to relax constraints.")
                return None, constraints_relaxed

        # If optimal solution found
        solution_variables = {v.name: v.varValue for v in problem.variables()}
        formatted_solution = self._format_solution(solution_variables, matchups)
        return formatted_solution, constraints_relaxed

    def attempt_schedule(
        self, matchups: List[Matchup], relax_constraints: List[str] = []
    ) -> Union[pulp.LpProblem, None]:

        problem = pulp.LpProblem("Quiz_Scheduling_With_Rooms", pulp.LpMaximize)

        if self.n_rooms <= 0 and self.n_matches_per_team > 0 :
             raise ValueError("Attempt_schedule: n_rooms must be positive if matches are expected.")
        if self.n_time_slots <= 0 and self.n_matches_per_team > 0:
             raise ValueError("Attempt_schedule: n_time_slots must be positive if matches are expected (was calculated as {}).".format(self.n_time_slots))

        if self.n_matches_per_team == 0: # If no matches, solve an empty problem
            self.enforce_constraints(problem, {}, matchups, relax_constraints)
            problem.solve()
            return problem

        variables = pulp.LpVariable.dicts(
            "MatchupRoomTime",
            (
                range(len(matchups)),
                range(1, self.n_rooms + 1),
                range(1, self.n_time_slots + 1),
            ),
            cat=pulp.LpBinary,
        )
        self.enforce_constraints(problem, variables, matchups, relax_constraints)

        problem.solve()
        return problem

    def check_schedule(self, df_schedule: pd.DataFrame) -> bool:
        print(df_schedule)
        is_solution = True
        team_rooms: Dict[int, List[int]] = {team: [] for team in range(1, self.n_teams + 1)}
        team_time_slots: Dict[int, List[int]] = {team: [] for team in range(1, self.n_teams + 1)}

        if not df_schedule.empty:
            for _, row in df_schedule.iterrows():
                room = row["Room"]
                matchup = row["Matchup"]
                time_slot = row["TimeSlot"]
                for team in matchup.teams:
                    team_rooms[team].append(room)
                    team_time_slots[team].append(time_slot)
        elif self.n_matches_per_team > 0 :
             print("Validation Error: Schedule is empty but matches were expected.")
             is_solution = False # Set solution to false and continue other checks if desired, or return False

        if not is_solution: # If already failed, no need to continue
            print(f"Initial schedule load check failed. Valid Schedule?: {is_solution}")
            return is_solution

        is_solution = self._check_team_conflicts(df_schedule) and is_solution
        if not is_solution: print("Team conflict check failed."); return False

        is_solution = self._check_room_visits(team_rooms) and is_solution
        if not is_solution: print("Room visit check failed."); return False

        is_solution = self._check_consecutive_matches(team_time_slots) and is_solution
        if not is_solution: print("Consecutive matches check failed."); return False

        if self.tournament_type == "district" and self.n_matches_per_team > self.CHUNK_SIZE:
            num_phases_check = math.ceil(self.n_matches_per_team / self.CHUNK_SIZE)
            num_breaks_check = num_phases_check - 1 if num_phases_check > 1 else 0
            if num_breaks_check > 0:
                 is_solution = self._check_periodic_breaks_and_quotas(df_schedule, self.CHUNK_SIZE) and is_solution
                 if not is_solution: print("Periodic breaks and quotas check failed."); return False
            elif self.n_matches_per_team > 0 :
                for t_id in range(1, self.n_teams + 1):
                    if len(team_time_slots.get(t_id, [])) != self.n_matches_per_team:
                        print(f"Validation Error (Total Matches - District/No Breaks): Team {t_id} has {len(team_time_slots.get(t_id, []))} matches, expected {self.n_matches_per_team}.")
                        is_solution = False
                        break
                if not is_solution: print("Total matches check failed for District/No Breaks."); return False

        elif self.tournament_type == "international" and self.n_matches_per_team > 0:
            for t_id in range(1, self.n_teams + 1):
                if len(team_time_slots.get(t_id,[])) != self.n_matches_per_team:
                    print(f"Validation Error (Total Matches - International): Team {t_id} has {len(team_time_slots.get(t_id, []))} matches, expected {self.n_matches_per_team}.")
                    is_solution = False
                    break
                if not is_solution: print("Total matches check failed for International."); return False

        print(f"Valid Schedule?: {is_solution}")
        print()
        return is_solution

    def _check_periodic_breaks_and_quotas(self, df_schedule: pd.DataFrame, chunk_size: int) -> bool:
        if self.n_matches_per_team == 0:
            return True

        num_phases = math.ceil(self.n_matches_per_team / chunk_size)
        num_breaks = num_phases - 1 if num_phases > 1 else 0

        if num_breaks == 0:
            if self.n_matches_per_team > 0:
                actual_matches_played_total = {t: 0 for t in range(1, self.n_teams + 1)}
                for _, row in df_schedule.iterrows():
                    ts_val = row["TimeSlot"]
                    if 1 <= ts_val <= self.n_time_slots:
                        for team_id_val in row["Matchup"].teams:
                            if 1 <= team_id_val <= self.n_teams:
                                actual_matches_played_total[team_id_val] += 1

                for team_id_val in range(1, self.n_teams + 1):
                    if actual_matches_played_total[team_id_val] != self.n_matches_per_team:
                        print(
                            f"Validation Error (Total Matches - No Periodic Breaks): Team {team_id_val} has {actual_matches_played_total[team_id_val]} total matches, "
                            f"expected {self.n_matches_per_team} by end of schedule (slot {self.n_time_slots})."
                        )
                        return False
            return True

        for ts_break in self.break_slot_indices:
            if not (1 <= ts_break <= self.n_time_slots):
                 print(f"Error (Check Logic): Break slot {ts_break} from self.break_slot_indices (len {len(self.break_slot_indices)}) is out of calculated bounds ({self.n_time_slots}). Review _calculate_schedule_structure.")
                 return False
            if not df_schedule[df_schedule.TimeSlot == ts_break].empty:
                print(f"Validation Error (Periodic Breaks): Break time slot {ts_break} is not empty.")
                return False

        actual_matches_played_cumulative = {team_id_val: [0] * (self.n_time_slots + 1) for team_id_val in range(1, self.n_teams + 1)}

        matches_per_ts_for_team = {team_id_val: [0] * (self.n_time_slots + 1) for team_id_val in range(1, self.n_teams + 1)}
        if not df_schedule.empty:
            for _, row in df_schedule.iterrows():
                ts_val = int(row["TimeSlot"])
                for team_id_val in row["Matchup"].teams:
                    if 1 <= ts_val <= self.n_time_slots and 1 <= team_id_val <= self.n_teams:
                        matches_per_ts_for_team[team_id_val][ts_val] += 1

        for team_id_val in range(1, self.n_teams + 1):
            for ts_val in range(1, self.n_time_slots + 1):
                actual_matches_played_cumulative[team_id_val][ts_val] = actual_matches_played_cumulative[team_id_val][ts_val-1] + matches_per_ts_for_team[team_id_val][ts_val]

        for p in range(num_phases):
            target_cumulative_quota = min((p + 1) * chunk_size, self.n_matches_per_team)

            if p >= len(self.end_of_phase_active_slot_indices):
                print(f"Error (Check Logic): Phase index {p} is out of bounds for self.end_of_phase_active_slot_indices (len {len(self.end_of_phase_active_slot_indices)}). Mismatch with num_phases={num_phases}.")
                return False

            slot_to_check_quota_at = self.end_of_phase_active_slot_indices[p]

            if not (1 <= slot_to_check_quota_at <= self.n_time_slots):
                 print(f"Error (Check Logic): Slot to check quota ({slot_to_check_quota_at}) for phase {p+1} is out of bounds ({self.n_time_slots}).")
                 return False

            for team_id_val in range(1, self.n_teams + 1):
                observed_matches = actual_matches_played_cumulative[team_id_val][slot_to_check_quota_at]
                if observed_matches != target_cumulative_quota:
                    print(
                        f"Validation Error (Phase Quota): Team {team_id_val} at end of phase {p + 1} "
                        f"(Timeslot {slot_to_check_quota_at}) has {observed_matches} matches, expected exactly {target_cumulative_quota}."
                    )
                    return False
        return True


    def enforce_constraints(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        relax_constraints: List[str],
    ):
        problem = self._enforce_each_matchup_occurrence(problem, variables, matchups)
        problem = self._enforce_each_room_to_host_single_matchup_per_time_slot(
            problem, variables, matchups
        )
        problem = self._enforce_no_simultaneous_scheduling_for_each_team(
            problem, variables, matchups
        )
        if "consecutive_matches" not in relax_constraints:
            problem = self._limit_consecutive_matchups(problem, variables, matchups)
        if "room_diversity" not in relax_constraints:
            problem = self._enforce_room_diversity(problem, variables, matchups)

        if self.tournament_type == "district" and self.n_matches_per_team > self.CHUNK_SIZE:
            num_phases = math.ceil(self.n_matches_per_team / self.CHUNK_SIZE)
            num_breaks = num_phases - 1 if num_phases > 1 else 0
            if num_breaks > 0:
                problem = self._enforce_periodic_breaks_and_phase_quotas(problem, variables, matchups, self.CHUNK_SIZE)

        return problem


    def _enforce_periodic_breaks_and_phase_quotas(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        chunk_size: int
    ):

        matches_played_to_ts = pulp.LpVariable.dicts(
            "MatchesPlayedToTsPeriodic",
            (range(1, self.n_teams + 1), range(1, self.n_time_slots + 1)),
            lowBound=0,
            cat=pulp.LpInteger,
        )

        for t_loop in range(1, self.n_teams + 1):
            for ts_loop in range(1, self.n_time_slots + 1):
                problem += matches_played_to_ts[t_loop][ts_loop] == pulp.lpSum(
                    variables[m_lp_idx][r_lp_idx][actual_ts_lp_idx]
                    for m_lp_idx, matchup_obj_lp_idx in enumerate(matchups)
                    if t_loop in matchup_obj_lp_idx.teams
                    for r_lp_idx in range(1, self.n_rooms + 1)
                    for actual_ts_lp_idx in range(1, ts_loop + 1)
                ), f"DefineMatchesPlayedPeriodic_T{t_loop}_TS{ts_loop}"

        for ts_break in self.break_slot_indices:
            problem += pulp.lpSum(
                variables[m_idx][r_idx][ts_break]
                for m_idx in range(len(matchups))
                for r_idx in range(1, self.n_rooms + 1)
            ) == 0, f"EmptyBreakSlot_TS{ts_break}"

        num_phases = math.ceil(self.n_matches_per_team / chunk_size)
        for p in range(num_phases):
            target_cumulative_quota = min((p + 1) * chunk_size, self.n_matches_per_team)
            if p < len(self.end_of_phase_active_slot_indices):
                 slot_to_check_quota_at = self.end_of_phase_active_slot_indices[p]
                 for t_team_idx in range(1, self.n_teams + 1):
                     problem += matches_played_to_ts[t_team_idx][slot_to_check_quota_at] == target_cumulative_quota, \
                                f"Quota_Phase{p+1}_Team{t_team_idx}_TS{slot_to_check_quota_at}"
            else:
                print(f"Warning: Phase index {p} out of bounds for end_of_phase_active_slot_indices in enforcement.")

        return problem

    def _enforce_each_matchup_occurrence(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
    ):
        for i in range(len(matchups)):
            problem += (
                pulp.lpSum(
                    variables[i][j][k]
                    for j in range(1, self.n_rooms + 1)
                    for k in range(1, self.n_time_slots + 1)
                )
                == 1
            )
        return problem

    def _enforce_each_room_to_host_single_matchup_per_time_slot(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
    ):
        for j in range(1, self.n_rooms + 1):
            for k in range(1, self.n_time_slots + 1):
                problem += pulp.lpSum(variables[i][j][k] for i in range(len(matchups))) <= 1
        return problem

    def _enforce_no_simultaneous_scheduling_for_each_team(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
    ):
        for k in range(1, self.n_time_slots + 1):
            for team in range(1, self.n_teams + 1):
                problem += (
                    pulp.lpSum(
                        variables[i][j][k]
                        for i, matchup in enumerate(matchups)
                        for j in range(1, self.n_rooms + 1)
                        if team in matchup.teams
                    )
                    <= 1
                )
        return problem

    def _limit_consecutive_matchups(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
    ):
        for team in range(1, self.n_teams + 1):
            for k in range(1, self.n_time_slots - 1):
                problem += (
                    pulp.lpSum(
                        variables[i][j][k] + variables[i][j][k + 1] + variables[i][j][k + 2]
                        for i, matchup in enumerate(matchups)
                        if team in matchup.teams
                        for j in range(1, self.n_rooms + 1)
                    )
                    <= 2
                )
        return problem

    def _enforce_room_diversity(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
    ):
        for team in range(1, self.n_teams + 1):
            problem += (
                pulp.lpSum(
                    variables[i][j][k]
                    for i, matchup in enumerate(matchups)
                    for j in range(1, self.n_rooms + 1)
                    for k in range(1, self.n_time_slots + 1)
                    if team in matchup.teams
                )
                == self.n_matches_per_team
            )

            if self.n_rooms > 0:
                avg_visits_per_room = self.n_matches_per_team / self.n_rooms

                min_allowed_visits = math.floor(avg_visits_per_room) - 1
                min_allowed_visits = max(0, min_allowed_visits)

                max_allowed_visits = math.ceil(avg_visits_per_room) + 1

                if self.n_matches_per_team == 0:
                    min_allowed_visits = 0
                    max_allowed_visits = 0

                for room_j in range(1, self.n_rooms + 1):
                    matches_for_team_in_room_j = pulp.lpSum(
                        variables[i][room_j][k]
                        for i, matchup in enumerate(matchups)
                        for k in range(1, self.n_time_slots + 1)
                        if team in matchup.teams
                    )
                    problem += matches_for_team_in_room_j >= min_allowed_visits, \
                               f"RoomDiversity_Min_T{team}_R{room_j}"
                    problem += matches_for_team_in_room_j <= max_allowed_visits, \
                               f"RoomDiversity_Max_T{team}_R{room_j}"
            elif self.n_matches_per_team > 0:
                 problem += pulp.lpSum(1) == 0
        return problem

    def _format_solution(self, solution: Dict[str, float], matchups: List[Matchup]):
        data = []
        epsilon = 1e-6

        for key, value in solution.items():
            if key.startswith("MatchupRoomTime_") and abs(value - 1.0) < epsilon:
                parts = key.split("_")
                if len(parts) >= 4:
                    try:
                        matchup_idx = int(parts[1])
                        room = int(parts[2])
                        time_slot = int(parts[3])

                        if 0 <= matchup_idx < len(matchups):
                            matchup = matchups[matchup_idx]
                            data.append((time_slot, room, matchup))
                        else:
                            print(f"Warning (_format_solution): Invalid matchup_idx {matchup_idx} for key {key}. Max index is {len(matchups)-1}.")
                    except ValueError:
                        print(f"Warning (_format_solution): Could not parse integer from parts for key {key}. Parts: {parts}")
                    except IndexError:
                        print(f"Warning (_format_solution): IndexError while parsing key {key}. Parts: {parts}")
                else:
                    print(f"Warning (_format_solution): Key {key} does not have enough parts after splitting.")

        df = pd.DataFrame(data, columns=["TimeSlot", "Room", "Matchup"])
        if not df.empty:
            df.sort_values(["TimeSlot", "Room"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def _check_team_conflicts(self, df_schedule: pd.DataFrame) -> bool:
        if self.n_time_slots == 0 and self.n_matches_per_team == 0: return True
        if df_schedule.empty and self.n_matches_per_team > 0: return False
        if df_schedule.empty and self.n_matches_per_team == 0: return True


        for time_slot in range(1, self.n_time_slots + 1):
            df_time_slot = df_schedule[df_schedule.TimeSlot == time_slot]
            if df_time_slot.empty:
                continue
            teams_in_slot = np.array([matchup.teams for matchup in df_time_slot.Matchup])
            n_unique_teams = len(np.unique(teams_in_slot))
            if n_unique_teams != teams_in_slot.size:
                print(f"Team conflict: A team is scheduled more than once in time slot {time_slot}.")
                return False
        return True

    def _check_room_visits(self, team_rooms: Dict[int, List[int]]) -> bool:
        if self.n_rooms == 0:
            if self.n_matches_per_team > 0:
                print("Error: Matches scheduled but no rooms available for room visit check.")
                return False
            return True

        if self.n_matches_per_team == 0:
            return True

        avg_visits_per_room = self.n_matches_per_team / self.n_rooms
        min_allowed_visits = math.floor(avg_visits_per_room) - 1
        min_allowed_visits = max(0, min_allowed_visits)
        max_allowed_visits = math.ceil(avg_visits_per_room) + 1
        if self.n_matches_per_team == 0:
            min_allowed_visits = 0
            max_allowed_visits = 0


        for team_id, actual_rooms_played_in_list in team_rooms.items():
            if len(actual_rooms_played_in_list) != self.n_matches_per_team:
                print(
                    f"Team {team_id} has {len(actual_rooms_played_in_list)} scheduled matches, "
                    f"but expected {self.n_matches_per_team} for room visit check consistency."
                )
                return False

            if not actual_rooms_played_in_list and self.n_matches_per_team > 0:
                print(f"Team {team_id} has no room visits recorded but expected {self.n_matches_per_team} matches.")
                return False
            if not actual_rooms_played_in_list and self.n_matches_per_team == 0:
                continue

            room_ids_this_team_played_in, counts_of_visits_per_room = np.unique(
                actual_rooms_played_in_list, return_counts=True
            )
            actual_visit_counts_per_room_map = dict(zip(room_ids_this_team_played_in, counts_of_visits_per_room))

            for room_j_id in range(1, self.n_rooms + 1):
                visits_to_this_room_j = actual_visit_counts_per_room_map.get(room_j_id, 0)
                if not (min_allowed_visits <= visits_to_this_room_j <= max_allowed_visits):
                    print(
                        f"Team {team_id} visited Room {room_j_id} {visits_to_this_room_j} times. "
                        f"Expected between {min_allowed_visits} and {max_allowed_visits} times."
                    )
                    return False
        return True

    def _check_consecutive_matches(self, team_time_slots: Dict[int, List[int]]) -> bool:
        if self.n_time_slots == 0 : return True

        for team, time_slots in team_time_slots.items():
            if len(time_slots) < 3:
                continue

            time_slots.sort()

            for i in range(len(time_slots) - 2):
                if time_slots[i+1] == time_slots[i] + 1 and \
                   time_slots[i+2] == time_slots[i] + 2:
                    print(
                        f"Team {team} is scheduled for 3 consecutive matches: "
                        f"{time_slots[i]}, {time_slots[i+1]}, {time_slots[i+2]}."
                    )
                    return False
        return True
