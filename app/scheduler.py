from typing import Dict, List, Tuple, Union
import math # Added for floor and ceil

import pulp
import numpy as np
import pandas as pd
from app.models import Matchup


class ScheduleSolver:
    def __init__(self, n_teams: int, n_matches_per_team: int, n_rooms: int, n_time_slots: int, tournament_type: str = "international"):
        self.n_teams = n_teams
        self.n_matches_per_team = n_matches_per_team
        self.n_rooms = n_rooms
        self.n_time_slots = n_time_slots
        self.tournament_type = tournament_type # Store tournament type

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
        num_breaks = num_phases - 1 if num_phases > 1 else 0

        if num_breaks == 0: # No breaks means no specific quotas at break points to check by this method
            return True

        # Recalculate break schedule (must be identical to enforcement logic)
        # This calculation needs to be robust and identical to the one in _enforce_periodic_breaks_and_phase_quotas
        total_active_slots_for_play = self.n_time_slots - num_breaks

        # Basic validation (if these fail, problem setup was flawed, but check might still run)
        if total_active_slots_for_play < num_phases or \
           (total_active_slots_for_play <= 0 and self.n_matches_per_team > 0):
            print(f"Warning (Check Logic): Insufficient total active slots ({total_active_slots_for_play}) for {num_phases} phases. Cannot reliably check quotas.")
            return False # Or True, if we decide this implies setup error not a schedule error by this check

        active_slots_per_phase_counts = [(total_active_slots_for_play // num_phases) for _ in range(num_phases)]
        for i in range(total_active_slots_for_play % num_phases):
            active_slots_per_phase_counts[i] += 1

        break_slot_indices = []
        end_of_phase_active_slot_indices = []
        current_model_timeslot_cursor = 0

        for p in range(num_phases):
            matches_this_phase = chunk_size if p < num_phases - 1 else self.n_matches_per_team - (p * chunk_size)
            min_slots_theoretically_needed = 0
            if self.n_rooms > 0:
                 min_slots_theoretically_needed = math.ceil((self.n_teams * matches_this_phase) / (3 * self.n_rooms))

            if active_slots_per_phase_counts[p] < min_slots_theoretically_needed:
                print(f"Warning (Check Logic): Phase {p+1} allocated {active_slots_per_phase_counts[p]} slots, "
                      f"theoretically needs {min_slots_theoretically_needed}. Quota check might be problematic.")
                # This indicates a setup issue that should have been caught by validation in enforce.
                # If we proceed, it's to check what the solver did despite potentially impossible constraints.
                pass

            current_model_timeslot_cursor += active_slots_per_phase_counts[p]
            end_of_phase_active_slot_indices.append(current_model_timeslot_cursor)

            if p < num_breaks:
                current_model_timeslot_cursor += 1
                if current_model_timeslot_cursor <= self.n_time_slots : # Ensure break slot is within total timeslots
                    break_slot_indices.append(current_model_timeslot_cursor)

        if current_model_timeslot_cursor > self.n_time_slots:
            print(f"Warning (Check Logic): Calculated structure ({current_model_timeslot_cursor} slots) exceeds total timeslots ({self.n_time_slots}).")
            # This implies an issue with slot calculation consistency or that the schedule is fundamentally flawed.
            # For the check, we can only validate up to n_time_slots.
            # We might need to adjust how end_of_phase_active_slot_indices and break_slot_indices are used if this happens.
            # For now, assume this implies a problem that should have been caught earlier or means the schedule is invalid.
            # It's safer to return False if the calculated structure doesn't fit n_time_slots, as quota checks will be off.
            return False


        # 1. Verify break slots are empty
        for ts_break in break_slot_indices:
            # Ensure ts_break is a valid timeslot index for df_schedule
            if 1 <= ts_break <= self.n_time_slots:
                if not df_schedule[df_schedule.TimeSlot == ts_break].empty:
                    print(f"Validation Error (Periodic Breaks): Break time slot {ts_break} is not empty.")
                    return False
            else: # Should not happen if cursor logic is correct and validated
                print(f"Warning (Check Logic): Calculated break slot {ts_break} is out of bounds ({self.n_time_slots}).")


        # 2. Verify phase quotas
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
        chunk_size: int
    ):
        if self.n_matches_per_team == 0 :
            return problem

        num_phases = math.ceil(self.n_matches_per_team / chunk_size)
        num_breaks = num_phases - 1 if num_phases > 1 else 0

        if num_breaks == 0:
            # No breaks needed (e.g., <= chunk_size matches total, or only one phase)
            # We still need to ensure all teams play n_matches_per_team by the end
            # This is typically handled by _enforce_room_diversity's first constraint
            # but let's ensure matches_played_to_ts is defined for consistency if other parts rely on it.
            # However, if no breaks, this specific method's core logic isn't triggered.
            # For safety, define matches_played_to_ts if it might be used by a check function later.
            # Or, the check function should also know if num_breaks == 0.
            # For now, returning early if no breaks are part of this specific logic.
            return problem

        total_active_slots_for_play = self.n_time_slots - num_breaks
        if total_active_slots_for_play < num_phases:
            raise ValueError(
                f"Not enough total time slots for District mode ({self.n_time_slots} provided) to accommodate "
                f"{num_phases} active phases and {num_breaks} breaks. "
                f"Minimum active slots needed for phases: {num_phases}, available: {total_active_slots_for_play}."
            )
        if total_active_slots_for_play <= 0 and self.n_matches_per_team > 0 :
             raise ValueError(
                f"Not enough total time slots for District mode: 0 or fewer active slots available after accounting for {num_breaks} breaks."
            )


        active_slots_per_phase_counts = [(total_active_slots_for_play // num_phases) for _ in range(num_phases)]
        for i in range(total_active_slots_for_play % num_phases):
            active_slots_per_phase_counts[i] += 1

        break_slot_indices = []
        end_of_phase_active_slot_indices = []
        current_model_timeslot_cursor = 0

        for p in range(num_phases):
            matches_this_phase = chunk_size if p < num_phases - 1 else self.n_matches_per_team - (p * chunk_size)
            if self.n_rooms == 0 and matches_this_phase > 0: # Should be caught by other validation ideally
                raise ValueError("Cannot schedule matches in a phase if there are no rooms.")

            min_slots_theoretically_needed = 0
            if self.n_rooms > 0 : # Avoid division by zero if no rooms
                min_slots_theoretically_needed = math.ceil((self.n_teams * matches_this_phase) / (3 * self.n_rooms))

            if active_slots_per_phase_counts[p] < min_slots_theoretically_needed:
                raise ValueError(
                    f"Phase {p + 1} allocated {active_slots_per_phase_counts[p]} active slots, but theoretically "
                    f"requires at least {min_slots_theoretically_needed} slots for {self.n_teams} teams to play "
                    f"{matches_this_phase} matches in {self.n_rooms} rooms. Increase total time slots or adjust other parameters."
                )

            current_model_timeslot_cursor += active_slots_per_phase_counts[p]
            end_of_phase_active_slot_indices.append(current_model_timeslot_cursor)

            if p < num_breaks:
                current_model_timeslot_cursor += 1  # This is the break slot index
                break_slot_indices.append(current_model_timeslot_cursor)

        # Ensure the calculated total slots don't exceed available n_time_slots
        # This should be inherently true by how total_active_slots_for_play was calculated,
        # but as a safeguard for cursor logic:
        if current_model_timeslot_cursor > self.n_time_slots:
            raise ValueError(
                f"Calculated schedule structure ({current_model_timeslot_cursor} slots) exceeds "
                f"total available time slots ({self.n_time_slots})."
            )


        # Define auxiliary variables for matches played up to a certain timeslot
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
                    for actual_ts_lp_idx in range(1, ts_loop + 1) # Sum matches up to and including ts_loop
                ), f"DefineMatchesPlayedPeriodic_T{t_loop}_TS{ts_loop}"

        # Enforce empty break slots
        for ts_break in break_slot_indices:
            if 1 <= ts_break <= self.n_time_slots : # Ensure break slot is within bounds
                problem += pulp.lpSum(
                    variables[m_idx][r_idx][ts_break]
                    for m_idx in range(len(matchups))
                    for r_idx in range(1, self.n_rooms + 1)
                ) == 0, f"EmptyBreakSlot_TS{ts_break}"

        # Enforce match quotas at the end of each active phase
        for p in range(num_phases):
            target_cumulative_quota = min((p + 1) * chunk_size, self.n_matches_per_team)
            slot_to_check_quota_at = end_of_phase_active_slot_indices[p]

            if 1 <= slot_to_check_quota_at <= self.n_time_slots : # Ensure slot is within bounds
                for t_team_idx in range(1, self.n_teams + 1):
                    # If it's an intermediate phase, use >= (teams must complete at least the quota)
                    # If it's the final phase, use == (teams must complete exactly the total n_matches_per_team)
                    is_final_phase_quota = (target_cumulative_quota == self.n_matches_per_team)

                    if is_final_phase_quota:
                        problem += matches_played_to_ts[t_team_idx][slot_to_check_quota_at] == target_cumulative_quota, \
                                   f"Quota_FinalPhase_Team{t_team_idx}_TS{slot_to_check_quota_at}"
                    else:
                        problem += matches_played_to_ts[t_team_idx][slot_to_check_quota_at] >= target_cumulative_quota, \
                                   f"Quota_IntermediatePhase{p+1}_Team{t_team_idx}_TS{slot_to_check_quota_at}"

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
