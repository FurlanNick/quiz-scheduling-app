from typing import Dict, List, Tuple, Union
import math # Added for floor and ceil

import pulp
import numpy as np
import pandas as pd
from app.models import Matchup


class ScheduleSolver:
    def __init__(self, n_teams: int, n_matches_per_team: int, n_rooms: int, n_time_slots: int):
        self.n_teams = n_teams
        self.n_matches_per_team = n_matches_per_team
        self.n_rooms = n_rooms
        self.n_time_slots = n_time_slots

    def schedule_matches(self, matchups: List[Matchup]) -> Union[pd.DataFrame, List[str]]:
        constraints_relaxed = []
        problem = self.attempt_schedule(matchups=matchups)

        if pulp.LpStatus[problem.status] == "Optimal":
            print("Solution found!")
        else:
            for constraint in ["room_diversity", "consecutive_matches"]:
                constraints_relaxed.append(constraint)
                problem = self.attempt_schedule(matchups, relax_constraints=constraints_relaxed)
                if pulp.LpStatus[problem.status] == "Optimal":
                    break
            else:
                print("No feasible solution found even after relaxing constraints.")
                return None, constraints_relaxed

        solution_variables = {v.name: v.varValue for v in problem.variables() if v.varValue == 1}
        formatted_solution = self._format_solution(solution_variables, matchups)
        return formatted_solution, constraints_relaxed

    def attempt_schedule(
        self, matchups: List[Matchup], relax_constraints: List[str] = []
    ) -> Union[pd.DataFrame, None]:
        problem = pulp.LpProblem("Quiz_Scheduling_With_Rooms", pulp.LpMaximize)
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

        for _, row in df_schedule.iterrows():
            room = row["Room"]
            matchup = row["Matchup"]
            time_slot = row["TimeSlot"]
            for team in matchup.teams:
                team_rooms[team].append(room)
                team_time_slots[team].append(time_slot)

        # Check for conflicts
        is_solution = self._check_team_conflicts(df_schedule) and is_solution
        is_solution = self._check_room_visits(team_rooms) and is_solution
        is_solution = self._check_consecutive_matches(team_time_slots) and is_solution
        # Re-enabling phasing check
        if self.n_matches_per_team > 3: # Phasing check is relevant if more than 3 matches
            is_solution = self._check_phased_match_completion(df_schedule) and is_solution

        print(f"Valid Schedule?: {is_solution}")
        print()
        return is_solution

    def _check_phased_match_completion(self, df_schedule: pd.DataFrame) -> bool:
        """
        Checks if all teams complete 3 matches before any team starts their 4th match.
        """
        CHUNK_SIZE = 3
        # Create a structure to hold matches played per team up to each timeslot
        # matches_at_ts[ts][team_id] = count
        matches_up_to_ts = {ts: {t: 0 for t in range(1, self.n_teams + 1)} for ts in range(self.n_time_slots + 1)}

        # Populate matches_up_to_ts based on the schedule
        # df_schedule is sorted by TimeSlot, then Room
        for _, row in df_schedule.iterrows():
            current_ts = row["TimeSlot"]
            matchup_teams = row["Matchup"].teams
            for team_id in matchup_teams:
                # Increment count for this team for all timeslots from current_ts onwards
                for ts_iter in range(current_ts, self.n_time_slots + 1):
                    matches_up_to_ts[ts_iter][team_id] += 1

        # Check the phasing rule
        for ts_check in range(1, self.n_time_slots + 1): # For each timeslot where a 4th match could start
            for t1 in range(1, self.n_teams + 1):
                # Did team t1 start its (CHUNK_SIZE + 1)-th match in this timeslot (ts_check)?
                # Matches at end of previous timeslot (ts_check - 1)
                matches_t1_before_ts_check = matches_up_to_ts[ts_check - 1][t1] if ts_check > 0 else 0
                # Matches at end of current timeslot (ts_check)
                matches_t1_at_ts_check = matches_up_to_ts[ts_check][t1]

                if matches_t1_before_ts_check == CHUNK_SIZE and matches_t1_at_ts_check == CHUNK_SIZE + 1:
                    # Team t1 is starting its (CHUNK_SIZE + 1)-th match in timeslot ts_check.
                    # Now, check all other teams t2.
                    for t2 in range(1, self.n_teams + 1):
                        if t1 == t2:
                            continue

                        # Team t2 must have completed CHUNK_SIZE matches by the end of ts_check -1
                        # (i.e., before t1 started its CHUNK_SIZE + 1 -th match).
                        matches_t2_before_ts_check = matches_up_to_ts[ts_check - 1][t2] if ts_check > 0 else 0
                        if matches_t2_before_ts_check < CHUNK_SIZE:
                            print(
                                f"Phasing Violation: Team {t1} started its {CHUNK_SIZE + 1}-th match in timeslot {ts_check}, "
                                f"but team {t2} had only completed {matches_t2_before_ts_check} matches "
                                f"(expected at least {CHUNK_SIZE}) by the end of timeslot {ts_check -1 }."
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
        # Add call to new phasing constraint enforcement
        # Re-enabling phasing constraint
        if self.n_matches_per_team > 3: # Phasing relevant if more than 3 matches total
            problem = self._enforce_phased_match_completion(problem, variables, matchups)


    def _enforce_phased_match_completion(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
    ):
        CHUNK_SIZE = 3
        BIG_M = self.n_matches_per_team + 1 # A sufficiently large number

        # Auxiliary variables: matches_played[team][timeslot]
        # Stores cumulative matches played by team 't' up to and including timeslot 'ts'.
        matches_played_to_ts = pulp.LpVariable.dicts(
            "MatchesPlayedToTs",
            (range(1, self.n_teams + 1), range(1, self.n_time_slots + 1)),
            lowBound=0,
            cat=pulp.LpInteger, # Can be continuous too, sum of binaries will be int
        )

        for t in range(1, self.n_teams + 1):
            for ts_upper in range(1, self.n_time_slots + 1):
                problem += matches_played_to_ts[t][ts_upper] == pulp.lpSum(
                    variables[m_idx][r_idx][ts_actual]
                    for m_idx, matchup_obj in enumerate(matchups)
                    if t in matchup_obj.teams
                    for r_idx in range(1, self.n_rooms + 1)
                    for ts_actual in range(1, ts_upper + 1) # Summing up to and including ts_upper
                ), f"DefineMatchesPlayed_T{t}_TS{ts_upper}"

        # Binary indicator variables: Z_team_ge_CHUNKSIZE_plus_1_by_ts[team][timeslot]
        # Z = 1 if team 't' has played CHUNK_SIZE + 1 (e.g., 4) or more matches by timeslot 'ts'.
        # Z = 0 if team 't' has played CHUNK_SIZE (e.g., 3) or fewer matches by timeslot 'ts'.
        Z_team_ge_chunksize_plus_1_by_ts = pulp.LpVariable.dicts(
            "Z_TeamPlayedGeChunkPlus1ByTs",
            (range(1, self.n_teams + 1), range(1, self.n_time_slots + 1)),
            cat=pulp.LpBinary,
        )

        for t1 in range(1, self.n_teams + 1):
            for ts in range(1, self.n_time_slots + 1):
                # If matches_played[t1][ts] >= CHUNK_SIZE + 1, then Z_t1_ge_chunksize_plus_1_by_ts[t1][ts] must be 1.
                # matches_played[t1][ts] >= (CHUNK_SIZE + 1) - M * (1 - Z)
                # Z = 1 --> matches_played >= CHUNK_SIZE + 1 (correct)
                # Z = 0 --> matches_played >= CHUNK_SIZE + 1 - M (non-restrictive)
                problem += matches_played_to_ts[t1][ts] >= \
                           (CHUNK_SIZE + 1) - BIG_M * (1 - Z_team_ge_chunksize_plus_1_by_ts[t1][ts]), \
                           f"Z_LowerBound_T{t1}_TS{ts}"

                # If matches_played[t1][ts] <= CHUNK_SIZE, then Z_t1_ge_chunksize_plus_1_by_ts[t1][ts] must be 0.
                # matches_played[t1][ts] <= CHUNK_SIZE + M * Z
                # Z = 0 --> matches_played <= CHUNK_SIZE (correct)
                # Z = 1 --> matches_played <= CHUNK_SIZE + M (non-restrictive)
                problem += matches_played_to_ts[t1][ts] <= \
                           CHUNK_SIZE + BIG_M * Z_team_ge_chunksize_plus_1_by_ts[t1][ts], \
                           f"Z_UpperBound_T{t1}_TS{ts}"

        # Core Phasing Constraint:
        # If team 't1' has played CHUNK_SIZE + 1 or more matches by timeslot 'ts' (i.e., Z_...[t1][ts] == 1),
        # then every other team 't2' must have played at least CHUNK_SIZE matches by timeslot 'ts'.
        for t1 in range(1, self.n_teams + 1):
            for t2 in range(1, self.n_teams + 1):
                if t1 == t2:
                    continue
                for ts in range(1, self.n_time_slots + 1):
                    # If Z_...[t1][ts] = 1, then matches_played_to_ts[t2][ts] >= CHUNK_SIZE
                    # This can be written as: matches_played_to_ts[t2][ts] >= CHUNK_SIZE * Z_...[t1][ts]
                    # (If Z=1, t2 must have >= CHUNK_SIZE. If Z=0, t2 must have >= 0, which is always true)
                    problem += matches_played_to_ts[t2][ts] >= \
                               CHUNK_SIZE * Z_team_ge_chunksize_plus_1_by_ts[t1][ts], \
                               f"Phasing_T1_{t1}_T2_{t2}_TS{ts}"
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
            # Constraint 1: Each team plays exactly self.n_matches_per_team matches in total.
            # This constraint sums all instances where a team participates in a scheduled matchup.
            problem += (
                pulp.lpSum(
                    variables[i][j][k]  # Decision variable for matchup i in room j at time k
                    for i, matchup in enumerate(matchups) # Iterate over all possible pre-generated matchups
                    for j in range(1, self.n_rooms + 1)   # Iterate over all rooms
                    for k in range(1, self.n_time_slots + 1) # Iterate over all time slots
                    if team in matchup.teams # Condition: the current team is part of this specific matchup
                )
                == self.n_matches_per_team # The sum must equal the total number of matches for the team
            )

            # Constraint 2: Distribute matches per team across rooms with relaxed bounds.
            if self.n_rooms > 0:  # This logic only applies if there are rooms
                avg_visits_per_room = self.n_matches_per_team / self.n_rooms

                # Relaxed bounds: [max(0, floor(avg) - 1), ceil(avg) + 1]
                # K_lower = 1, K_upper = 1 for the +/-1 spread around avg's floor/ceil
                min_allowed_visits = math.floor(avg_visits_per_room) - 1
                min_allowed_visits = max(0, min_allowed_visits) # Ensure non-negative

                max_allowed_visits = math.ceil(avg_visits_per_room) + 1

                # Special case: if n_matches_per_team is 0, bounds should be 0.
                if self.n_matches_per_team == 0:
                    min_allowed_visits = 0
                    max_allowed_visits = 0

                for room_j in range(1, self.n_rooms + 1):  # For each room j
                    matches_for_team_in_room_j = pulp.lpSum(
                        variables[i][room_j][k] # Fixed room_j
                        for i, matchup in enumerate(matchups)
                        for k in range(1, self.n_time_slots + 1)
                        if team in matchup.teams
                    )
                    problem += matches_for_team_in_room_j >= min_allowed_visits, \
                               f"RoomDiversity_Min_T{team}_R{room_j}"
                    problem += matches_for_team_in_room_j <= max_allowed_visits, \
                               f"RoomDiversity_Max_T{team}_R{room_j}"
            elif self.n_matches_per_team > 0: # No rooms, but matches are expected
                 # This case should ideally be caught by validation earlier or will make the problem infeasible
                 # because variables are defined over an empty room range if n_rooms = 0.
                 # Adding a direct infeasible constraint if it reaches here with n_matches_per_team > 0 and n_rooms = 0
                 # to make it explicit, though PuLP would likely determine infeasibility anyway.
                 problem += pulp.lpSum(1) == 0 # Makes problem infeasible: 1 == 0
        return problem

    def _format_solution(self, solution: Dict[str, float], matchups: List[Matchup]):
        data = []
        for key, value in solution.items():
            if value == 1.0:
                parts = key.split("_")
                matchup_idx = int(parts[1])
                room = int(parts[2])
                time_slot = int(parts[3])
                matchup = matchups[matchup_idx]
                data.append((time_slot, room, matchup))
        df = pd.DataFrame(data, columns=["TimeSlot", "Room", "Matchup"])
        df.sort_values(["TimeSlot", "Room"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _check_team_conflicts(self, df_schedule: pd.DataFrame) -> bool:
        for time_slot in range(1, self.n_time_slots + 1):
            df_time_slot = df_schedule[df_schedule.TimeSlot == time_slot]
            teams_in_slot = np.array([matchup.teams for matchup in df_time_slot.Matchup])
            n_unique_teams = len(np.unique(teams_in_slot))
            if n_unique_teams != teams_in_slot.size:
                print("A team is scheduled more than once at the same time.")
                return False
        return True

    def _check_room_visits(self, team_rooms: Dict[int, List[int]]) -> bool:
        if self.n_rooms == 0:
            # If there are no rooms, check if any matches were expected.
            # self.n_matches_per_team could be derived also from sum(len(lst) for lst in team_rooms.values()) / self.n_teams
            # but direct use of self.n_matches_per_team is fine as it's an input parameter.
            if self.n_matches_per_team > 0:
                print("Error: Matches are scheduled/expected but no rooms are available.")
                return False
            return True # No matches expected and no rooms, so trivially valid.

        # Calculate relaxed bounds for room visits
        if self.n_matches_per_team == 0:
            min_allowed_visits = 0
            max_allowed_visits = 0
        else:
            avg_visits_per_room = self.n_matches_per_team / self.n_rooms
            min_allowed_visits = math.floor(avg_visits_per_room) - 1
            min_allowed_visits = max(0, min_allowed_visits)  # Ensure non-negative
            max_allowed_visits = math.ceil(avg_visits_per_room) + 1

        for team_id, actual_rooms_played_in_list in team_rooms.items():
            # Verify that the team played the correct total number of matches
            if len(actual_rooms_played_in_list) != self.n_matches_per_team:
                print(
                    f"Team {team_id} has {len(actual_rooms_played_in_list)} scheduled matches, "
                    f"but expected {self.n_matches_per_team}."
                )
                return False

            # Count how many times this team played in each room
            if not actual_rooms_played_in_list:
                 # If n_matches_per_team is 0, this is fine. All rooms should have 0 visits.
                 # If n_matches_per_team > 0, but list is empty, it's an error caught by above check.
                room_ids_this_team_played_in = np.array([])
                counts_of_visits_per_room = np.array([])
            else:
                room_ids_this_team_played_in, counts_of_visits_per_room = np.unique(
                    actual_rooms_played_in_list, return_counts=True
                )

            actual_visit_counts_per_room_map = dict(zip(room_ids_this_team_played_in, counts_of_visits_per_room))

            # Check each available room (from 1 to self.n_rooms)
            for room_j_id in range(1, self.n_rooms + 1):
                visits_to_this_room_j = actual_visit_counts_per_room_map.get(room_j_id, 0)

                if not (min_allowed_visits <= visits_to_this_room_j <= max_allowed_visits):
                    print(
                        f"Team {team_id} visited Room {room_j_id} {visits_to_this_room_j} times. "
                        f"Expected between {min_allowed_visits} and {max_allowed_visits} times."
                    )
                    return False

            # Ensure that even with relaxed per-room counts, the total matches for the team is correct.
            # This is already checked by `len(actual_rooms_played_in_list) != self.n_matches_per_team`.
            # We could also sum `actual_visit_counts_per_room_map.values()` as an additional sanity check.
            # total_counted_matches = sum(actual_visit_counts_per_room_map.values())
            # if total_counted_matches != self.n_matches_per_team:
            #     print(f"Team {team_id} total matches inconsistency: counted {total_counted_matches}, expected {self.n_matches_per_team}")
            #     return False

        return True

    def _check_consecutive_matches(self, team_time_slots: Dict[int, List[int]]) -> bool:
        for team, time_slots in team_time_slots.items():
            if len(time_slots) < 3:
                continue  # Cannot have 3 consecutive matches if playing less than 3 times

            time_slots.sort()

            # Iterate through the sorted time slots to check for any sequence of three
            # consecutive time slot numbers (e.g., 1, 2, 3 or 5, 6, 7)
            for i in range(len(time_slots) - 2):
                # Check if the current slot, the next, and the one after form a consecutive sequence
                if time_slots[i+1] == time_slots[i] + 1 and \
                   time_slots[i+2] == time_slots[i] + 2:
                    print(
                        f"Team {team} is scheduled for 3 consecutive matches: "
                        f"{time_slots[i]}, {time_slots[i+1]}, {time_slots[i+2]}."
                    )
                    return False
        return True
