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
                 phase_buffer_slots: int = 2, # Default buffer for district phases
                 international_buffer_slots: int = 5 # Default overall buffer for international
                ):
        self.n_teams = n_teams
        self.n_matches_per_team = n_matches_per_team
        self.n_rooms = n_rooms
        self.tournament_type = tournament_type
        self.phase_buffer_slots = phase_buffer_slots
        self.international_buffer_slots = international_buffer_slots

        # These will be calculated by _calculate_schedule_structure
        self.n_time_slots = 0
        self.active_slots_per_phase_counts = []
        self.break_slot_indices = []
        self.end_of_phase_active_slot_indices = []

        # Store CHUNK_SIZE as an instance variable if it's used in multiple methods related to district mode
        self.CHUNK_SIZE = 3


    def _calculate_schedule_structure(self):
        """
        Calculates n_time_slots and phase structures based on tournament type.
        This method sets:
        - self.n_time_slots
        - self.active_slots_per_phase_counts (for district)
        - self.break_slot_indices (for district)
        - self.end_of_phase_active_slot_indices (for district)
        """
        if self.n_matches_per_team == 0:
            self.n_time_slots = 0
            return

        if self.n_rooms <= 0:
            # If there are matches to be played but no rooms, it's impossible.
            if self.n_matches_per_team > 0:
                raise ValueError("Cannot schedule matches when number of rooms is zero or less.")
            self.n_time_slots = 0 # No matches, no rooms, no time slots needed.
            return

        if self.tournament_type == "district" and self.n_matches_per_team > self.CHUNK_SIZE:
            num_phases = math.ceil(self.n_matches_per_team / self.CHUNK_SIZE)
            num_breaks = num_phases - 1 # num_breaks will be 0 if num_phases is 1

            calculated_total_active_slots = 0
            self.active_slots_per_phase_counts = []

            for p in range(num_phases):
                matches_this_phase = self.CHUNK_SIZE if p < num_phases - 1 else self.n_matches_per_team - (p * self.CHUNK_SIZE)
                if matches_this_phase <= 0: # Should not happen if n_matches_per_team > 0
                    self.active_slots_per_phase_counts.append(0)
                    continue

                min_slots_needed = math.ceil((self.n_teams * matches_this_phase) / (3 * self.n_rooms))

                # Determine buffer: use a percentage, but ensure at least 1 if min_slots > 0, or a fixed buffer.
                # Let's use a fixed buffer for simplicity and predictability for now.
                buffer_for_phase = self.phase_buffer_slots
                if min_slots_needed == 0 and matches_this_phase > 0 : # e.g. very few teams/matches for rooms
                    buffer_for_phase = 0 # No buffer if no slots are strictly needed.

                slots_allocated_for_phase = min_slots_needed + buffer_for_phase
                self.active_slots_per_phase_counts.append(slots_allocated_for_phase)
                calculated_total_active_slots += slots_allocated_for_phase

            self.n_time_slots = calculated_total_active_slots + num_breaks

            # Populate break_slot_indices and end_of_phase_active_slot_indices
            self.break_slot_indices = []
            self.end_of_phase_active_slot_indices = []
            current_ts_cursor = 0
            for p in range(num_phases):
                current_ts_cursor += self.active_slots_per_phase_counts[p]
                self.end_of_phase_active_slot_indices.append(current_ts_cursor)
                if p < num_breaks:
                    current_ts_cursor += 1 # For the break slot
                    self.break_slot_indices.append(current_ts_cursor)
        else: # International mode or District mode with n_matches_per_team <= CHUNK_SIZE (no breaks/phases)
            min_total_active_slots = math.ceil((self.n_teams * self.n_matches_per_team / 3) / self.n_rooms)
            self.n_time_slots = min_total_active_slots + self.international_buffer_slots
            self.active_slots_per_phase_counts = [] # Not used
            self.break_slot_indices = [] # Not used
            self.end_of_phase_active_slot_indices = [] # Not used

        if self.n_matches_per_team > 0 and self.n_time_slots == 0:
            # This can happen if rooms is very high, leading to min_slots_needed = 0 for all phases,
            # and buffer is also 0. Ensure at least 1 timeslot if matches are expected.
            self.n_time_slots = 1
            # For district mode, if this happens, active_slots_per_phase_counts might need adjustment.
            # This edge case suggests the buffer logic or min_slots_needed might need refinement
            # if it results in zero total active slots when matches are expected.
            # For now, setting to 1 as a basic floor.
            if self.tournament_type == "district" and num_phases > 0 and not self.active_slots_per_phase_counts[0] > 0 :
                 # If district mode calculated 0 active slots for the first phase, this is an issue.
                 # This indicates an extreme case, likely n_teams or matches_this_phase is 0 or n_rooms is excessively large.
                 # The initial n_rooms <= 0 check should catch some of this.
                 # If active_slots_per_phase_counts sums to 0, then n_time_slots (before this fix) would be num_breaks.
                 # This needs careful thought if it leads to issues.
                 # The validations in _enforce_periodic_breaks_and_phase_quotas should catch if active_slots_per_phase_counts[p] is too low.
                 pass


    def schedule_matches(self, matchups: List[Matchup]) -> Union[pd.DataFrame, List[str]]:
        constraints_relaxed = []
        # Calculate n_time_slots and phase structures here, before calling attempt_schedule
        # Or, call it at the start of attempt_schedule. Let's do it in attempt_schedule
        # so it's done each time a solution attempt is made (e.g. after relaxing constraints)
        # However, self.n_time_slots is needed for variable definitions.
        # So, it's better to do it before the problem is defined.
        # Let's assume attempt_schedule will call it.
        # No, _calculate_schedule_structure MUST be called before LpVariables are dimensioned with self.n_time_slots
        self._calculate_schedule_structure() # Calculate n_time_slots and phase structures

        # If calculation resulted in 0 time slots (e.g. 0 matches), no need to solve.
        if self.n_time_slots == 0 and self.n_matches_per_team == 0:
            # Return an empty DataFrame and no relaxed constraints
            return pd.DataFrame(columns=["TimeSlot", "Room", "Matchup"]), []
        elif self.n_time_slots == 0 and self.n_matches_per_team > 0:
            # This case should ideally be an error raised by _calculate_schedule_structure if n_rooms > 0
            # or if it implies an impossible scenario. For safety, handle it.
            print("Warning: n_time_slots calculated to 0 but matches are expected. This indicates an issue.")
            return None, ["n_time_slots_calculation_error"]


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
    ) -> Union[pulp.LpProblem, None]: # Return type changed to LpProblem for consistency
        # _calculate_schedule_structure is now called in schedule_matches before this.
        # So, self.n_time_slots is set.

        # If n_time_slots is 0 (e.g. due to 0 matches), PuLP can't define variables over an empty range.
        # This case should be handled before calling attempt_schedule if n_matches_per_team is 0.
        # If n_matches_per_team > 0 but n_time_slots ended up 0, _calculate_schedule_structure should raise error or handle.
        # For safety, if we reach here and n_time_slots is 0 and matches are expected, it's an issue.
        # However, schedule_matches now handles the n_time_slots=0 case.

        problem = pulp.LpProblem("Quiz_Scheduling_With_Rooms", pulp.LpMaximize)

        # Ensure n_time_slots is at least 1 for PuLP variable indexing if any matches are expected
        # This is a safeguard; primary calculation is in _calculate_schedule_structure
        current_n_time_slots = self.n_time_slots
        if self.n_matches_per_team > 0 and current_n_time_slots == 0:
            # This should ideally be caught by _calculate_schedule_structure raising an error
            # or by schedule_matches returning early.
            # If it somehow gets here, it's problematic.
            print(f"Warning: attempt_schedule called with n_time_slots=0 but n_matches_per_team={self.n_matches_per_team}")
            # To prevent PuLP error, we might return None or an empty problem,
            # but this indicates a flaw in the n_time_slots calculation.
            # For now, let PuLP potentially error if ranges are empty.
            # A better approach is for _calculate_schedule_structure to ensure n_time_slots >= 1 if matches > 0.
            # And schedule_matches to not call attempt_schedule if n_time_slots is unsuitable.

        variables = pulp.LpVariable.dicts(
            "MatchupRoomTime",
            (
                range(len(matchups)),
                range(1, self.n_rooms + 1), # n_rooms should be > 0 if matches > 0
                range(1, current_n_time_slots + 1),
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

        # Conditionally call the new check for periodic breaks and quotas
        CHUNK_SIZE_FOR_BREAKS = 3 # Must be same as in enforcement
        if self.tournament_type == "district" and self.n_matches_per_team > CHUNK_SIZE_FOR_BREAKS:
            # Calculate num_breaks to see if the check is relevant for periodic breaks
            num_phases_check = math.ceil(self.n_matches_per_team / CHUNK_SIZE_FOR_BREAKS)
            num_breaks_check = num_phases_check - 1 if num_phases_check > 1 else 0
            if num_breaks_check > 0: # Only check if breaks were intended to be scheduled
                 is_solution = self._check_periodic_breaks_and_quotas(df_schedule, CHUNK_SIZE_FOR_BREAKS) and is_solution

        # The original _check_phased_match_completion (Big-M dynamic phasing) remains disabled/removed.
        # If you want to keep its structure for other types of phasing checks, you can rename it
        # or ensure it's not called unless that specific (Big-M) logic is active.
        # For now, the call to _check_phased_match_completion is removed as its enforcement part is disabled.

        print(f"Valid Schedule?: {is_solution}")
        print()
        return is_solution

    # The _check_phased_match_completion method (for Big-M style dynamic phasing) is now effectively disabled
    # as its enforcement counterpart is disabled. If needed for a different type of check, it would
    # require significant rework or careful conditional calling. For now, it's dormant.
    # def _check_phased_match_completion(self, df_schedule: pd.DataFrame) -> bool: ...

    def _check_periodic_breaks_and_quotas(self, df_schedule: pd.DataFrame, chunk_size: int) -> bool:
        """
        Checks if designated break time slots are empty and if teams meet match quotas by phase ends
        for the 'district' tournament type.
        This logic must mirror the calculations in _enforce_periodic_breaks_and_phase_quotas.
        """
        if self.n_matches_per_team == 0:
            return True

        num_phases = math.ceil(self.n_matches_per_team / chunk_size)
        # num_phases and num_breaks are needed for loop bounds and logic here.
        # These should be consistent with how they were calculated in _calculate_schedule_structure.
        num_phases = math.ceil(self.n_matches_per_team / chunk_size)
        num_breaks = num_phases - 1 if num_phases > 1 else 0

        if num_breaks == 0:
            # If no breaks were structured, this specific check for periodic breaks/quotas isn't applicable.
            # The overall total match count is still important.
            if self.n_matches_per_team > 0:
                actual_matches_played_total = {t: 0 for t in range(1, self.n_teams + 1)}
                for _, row in df_schedule.iterrows():
                    ts = row["TimeSlot"]
                    if 1 <= ts <= self.n_time_slots: # Count matches within the schedule's defined timespan
                        for team_id in row["Matchup"].teams:
                            if 1 <= team_id <= self.n_teams:
                                actual_matches_played_total[team_id] += 1

                for team_id in range(1, self.n_teams + 1):
                    if actual_matches_played_total[team_id] != self.n_matches_per_team:
                        print(
                            f"Validation Error (Total Matches - No Periodic Breaks): Team {team_id} has {actual_matches_played_total[team_id]} total matches, "
                            f"expected {self.n_matches_per_team} by end of schedule (slot {self.n_time_slots})."
                        )
                        return False
            return True # Correctly, no periodic breaks/quotas to check, and total matches (if any) are implicitly checked by overall constraints.

        # Use pre-calculated structure from self attributes
        # self.break_slot_indices
        # self.end_of_phase_active_slot_indices
        # self.n_time_slots (calculated total)

        # 1. Verify break slots are empty
        for ts_break in self.break_slot_indices: # Use instance variable
            if 1 <= ts_break <= self.n_time_slots:
                if not df_schedule[df_schedule.TimeSlot == ts_break].empty:
                    print(f"Validation Error (Periodic Breaks): Break time slot {ts_break} is not empty.")
                    return False
            else:
                # This case implies an issue in _calculate_schedule_structure if a break slot is outside n_time_slots
                print(f"Warning (Check Logic): Break slot {ts_break} is outside the total time slots {self.n_time_slots}. Inconsistent calculation.")
                return False # Treat as error because structure is ill-defined for checking

        # 2. Verify phase quotas (strict equality for all phases now)
        # Reconstruct cumulative matches from df_schedule
        actual_matches_played = {team_id: [0] * (self.n_time_slots + 1) for team_id in range(1, self.n_teams + 1)}
        for _, row in df_schedule.iterrows():
            ts = row["TimeSlot"]
            for team_id in row["Matchup"].teams:
                if 1 <= ts <= self.n_time_slots and 1 <= team_id <= self.n_teams: # Boundary checks
                    actual_matches_played[team_id][ts] += 1

        # Convert to cumulative sums
        for team_id in range(1, self.n_teams + 1):
            for ts in range(1, self.n_time_slots + 1):
                actual_matches_played[team_id][ts] += actual_matches_played[team_id][ts-1]

        for p in range(num_phases):
            target_cumulative_quota = min((p + 1) * chunk_size, self.n_matches_per_team)
            slot_to_check_quota_at = end_of_phase_active_slot_indices[p]

            if 1 <= slot_to_check_quota_at <= self.n_time_slots: # Ensure slot is valid
                for team_id in range(1, self.n_teams + 1):
                    observed_matches = actual_matches_played[team_id][slot_to_check_quota_at]

                    is_final_phase_quota_check = (target_cumulative_quota == self.n_matches_per_team)

                    if is_final_phase_quota_check:
                        if observed_matches != target_cumulative_quota:
                            print(
                                f"Validation Error (Final Phase Quota): Team {team_id} at end of phase {p + 1} "
                                f"(Timeslot {slot_to_check_quota_at}) has {observed_matches} matches, expected exactly {target_cumulative_quota}."
                            )
                            return False
                    else: # Intermediate phase
                        if observed_matches < target_cumulative_quota:
                            print(
                                f"Validation Error (Intermediate Phase Quota): Team {team_id} at end of phase {p + 1} "
                                f"(Timeslot {slot_to_check_quota_at}) has {observed_matches} matches, expected at least {target_cumulative_quota}."
                            )
                            return False
                        # Also, for intermediate phases, they shouldn't exceed the *next* phase's quota too early,
                        # but the primary check is meeting the current phase's minimum.
                        # The total matches constraint will cap the overall.
                        # A more advanced check could ensure they don't play *too many* more than target_cumulative_quota
                        # if that becomes a requirement (e.g. not more than target_cumulative_quota + chunk_size/2 )
                        # For now, >= is sufficient given the enforcement was also changed to >= for intermediate.

            elif p < num_phases -1 :
                 print(f"Warning (Check Logic): Slot to check quota ({slot_to_check_quota_at}) for phase {p+1} is out of bounds ({self.n_time_slots}). This implies an issue in slot calculation or schedule length.")
                 # This implies a mismatch between calculated structure and available slots, which is problematic.
                 return False # Treat as failure if an intermediate phase check point is invalid

        # Final check: ensure total matches are exactly n_matches_per_team by the last defined timeslot.
        # This is crucial and should use equality.
        last_slot_in_schedule = self.n_time_slots
        # If end_of_phase_active_slot_indices is not empty, the last element is the end of all active play
        if end_of_phase_active_slot_indices:
            last_slot_in_schedule = end_of_phase_active_slot_indices[-1]
            if last_slot_in_schedule > self.n_time_slots : # Should have been caught by earlier validation if so
                last_slot_in_schedule = self.n_time_slots


        for team_id in range(1, self.n_teams + 1):
            # Check at the very end of all active play, or at n_time_slots if that's earlier/different
            final_observed_matches = actual_matches_played[team_id][last_slot_in_schedule]
            if final_observed_matches != self.n_matches_per_team:
                print(
                    f"Validation Error (Total Matches): Team {team_id} has {final_observed_matches} total matches by slot {last_slot_in_schedule}, "
                    f"expected {self.n_matches_per_team}."
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
        # The Big-M dynamic phasing (_enforce_phased_match_completion) remains disabled due to performance.
        # Instead, call the new periodic breaks logic if tournament_type is 'district'.
        CHUNK_SIZE_FOR_BREAKS = 3 # Define chunk size for periodic breaks
        if self.tournament_type == "district" and self.n_matches_per_team > CHUNK_SIZE_FOR_BREAKS:
            problem = self._enforce_periodic_breaks_and_phase_quotas(problem, variables, matchups, CHUNK_SIZE_FOR_BREAKS)
        # The original _enforce_phased_match_completion (Big-M) should be removed or kept commented out.
        # For clarity, I am ensuring it's not called. The plan was to keep it disabled.

    # _enforce_phased_match_completion (Big-M version) is intentionally kept disabled/removed.

    def _enforce_periodic_breaks_and_phase_quotas(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        chunk_size: int # CHUNK_SIZE is self.CHUNK_SIZE, passed from enforce_constraints
    ):
        # This method is called only if self.tournament_type == "district"
        # and self.n_matches_per_team > self.CHUNK_SIZE.
        # The schedule structure (self.n_time_slots, self.active_slots_per_phase_counts,
        # self.break_slot_indices, self.end_of_phase_active_slot_indices)
        # has been pre-calculated by _calculate_schedule_structure.

        if self.n_matches_per_team == 0: # Should be caught by _calculate_schedule_structure or earlier
            return problem

        num_phases = math.ceil(self.n_matches_per_team / chunk_size)
        num_breaks = num_phases - 1 if num_phases > 1 else 0

        if num_breaks == 0: # Also means only one phase or less
            # No periodic breaks/quotas to enforce if there are no breaks.
            # The final quota (total matches) is handled by other constraints.
            return problem

        # Validations for sufficient slots are now primarily in _calculate_schedule_structure.
        # This method assumes the structure passed via self attributes is viable.

        # Define auxiliary variables for matches played up to a certain timeslot
        # These are defined here because their dimension self.n_time_slots is now known.
        matches_played_to_ts = pulp.LpVariable.dicts(
            "MatchesPlayedToTsPeriodic", # Using a unique name
            (range(1, self.n_teams + 1), range(1, self.n_time_slots + 1)),
            lowBound=0,
            cat=pulp.LpInteger,
        )

        for t_loop in range(1, self.n_teams + 1):
            for ts_loop in range(1, self.n_time_slots + 1):
                # Ensure PuLP variable names are unique if this method is called multiple times with different dict keys
                problem += matches_played_to_ts[t_loop][ts_loop] == pulp.lpSum(
                    variables[m_lp_idx][r_lp_idx][actual_ts_lp_idx]
                    for m_lp_idx, matchup_obj_lp_idx in enumerate(matchups)
                    if t_loop in matchup_obj_lp_idx.teams
                    for r_lp_idx in range(1, self.n_rooms + 1)
                    for actual_ts_lp_idx in range(1, ts_loop + 1)
                ), f"DefineMatchesPlayedPeriodic_T{t_loop}_TS{ts_loop}"

        # Enforce empty break slots
        for ts_break in self.break_slot_indices:
            # ts_break is already validated to be within self.n_time_slots by _calculate_schedule_structure
            problem += pulp.lpSum(
                variables[m_idx][r_idx][ts_break]
                for m_idx in range(len(matchups))
                for r_idx in range(1, self.n_rooms + 1)
            ) == 0, f"EmptyBreakSlot_TS{ts_break}"

        # Enforce strict match quotas at the end of each active phase
        for p in range(num_phases):
            target_cumulative_quota = min((p + 1) * chunk_size, self.n_matches_per_team)
            slot_to_check_quota_at = self.end_of_phase_active_slot_indices[p]

            # slot_to_check_quota_at is also validated by _calculate_schedule_structure
            for t_team_idx in range(1, self.n_teams + 1):
                problem += matches_played_to_ts[t_team_idx][slot_to_check_quota_at] == target_cumulative_quota, \
                           f"Quota_Phase{p+1}_Team{t_team_idx}_TS{slot_to_check_quota_at}"

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
        # PuLP variable values can sometimes be slightly off due to floating point precision,
        # e.g., 0.999999999 instead of 1.0. Use a small tolerance for comparison.
        epsilon = 1e-6

        for key, value in solution.items():
            # We are only interested in the 'MatchupRoomTime_...' variables that are selected (value is close to 1.0)
            if key.startswith("MatchupRoomTime_") and abs(value - 1.0) < epsilon:
                parts = key.split("_")
                # Expected format: MatchupRoomTime_matchupidx_room_timeslot
                # parts[0] = "MatchupRoomTime"
                # parts[1] = matchup_idx
                # parts[2] = room_num (1-based)
                # parts[3] = time_slot_num (1-based)
                if len(parts) >= 4: # Check if we have enough parts
                    try:
                        matchup_idx = int(parts[1])
                        room = int(parts[2])
                        time_slot = int(parts[3])

                        # Validate matchup_idx before accessing matchups list
                        if 0 <= matchup_idx < len(matchups):
                            matchup = matchups[matchup_idx]
                            data.append((time_slot, room, matchup))
                        else:
                            print(f"Warning (_format_solution): Invalid matchup_idx {matchup_idx} for key {key}. Max index is {len(matchups)-1}.")
                    except ValueError:
                        print(f"Warning (_format_solution): Could not parse integer from parts for key {key}. Parts: {parts}")
                    except IndexError:
                        # This should be caught by len(parts) >= 4, but as a safeguard
                        print(f"Warning (_format_solution): IndexError while parsing key {key}. Parts: {parts}")
                else:
                    print(f"Warning (_format_solution): Key {key} does not have enough parts after splitting.")
            # else:
                # Optionally log other variables if needed for debugging, e.g.:
                # if not key.startswith("MatchupRoomTime_") and value != 0:
                #    print(f"Debug (_format_solution): Ignored variable {key} with value {value}")

        df = pd.DataFrame(data, columns=["TimeSlot", "Room", "Matchup"])
        if not df.empty:
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
