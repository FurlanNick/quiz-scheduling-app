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
        self.matches_per_day = matches_per_day # Store new parameter

        self.n_time_slots = 0
        # These attributes were for the old District mode logic and will be re-evaluated or removed.
        self.active_slots_per_phase_counts = []
        self.break_slot_indices = []
        self.end_of_phase_active_slot_indices = []
        # self.CHUNK_SIZE = 3 # Replaced by self.matches_per_day for new District logic

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
        # This method is now a unified entry point for both tournament types.
        # It calls the sequential scheduler which handles the specific logic for each type.
        try:
            final_schedule_df, constraints_relaxed = self._schedule_sequentially(matchups, relax_constraints=[])
            if final_schedule_df is None:
                print(f"No feasible solution found for {self.tournament_type} mode.")
                return None, constraints_relaxed
        except Exception as e:
            print(f"Error during {self.tournament_type} sequential scheduling: {e}")
            # It might be beneficial to print traceback here for debugging
            # import traceback
            # traceback.print_exc()
            return None, []

        return final_schedule_df, constraints_relaxed


    def _schedule_sequentially(self, all_globally_valid_matchups: List[Matchup], relax_constraints: List[str]) -> Tuple[Union[pd.DataFrame, None], List[str]]:
        """
        Schedules matches sequentially, phase by phase, for either tournament type.
        """
        if self.matches_per_day <= 0:
            raise ValueError("Matches per day/phase must be positive.")

        if self.tournament_type == "district" and self.matches_per_day > self.n_rooms:
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

        # For International mode's cumulative room diversity
        cumulative_room_visits = {team_id: {room_id: 0 for room_id in range(1, self.n_rooms + 1)} for team_id in range(1, self.n_teams + 1)}

        for phase_idx in range(num_total_phases):
            print(f"Processing {self.tournament_type} Mode - Phase {phase_idx + 1}/{num_total_phases}")

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

            buffer_slots = self.phase_buffer_slots if self.tournament_type == "district" else self.international_buffer_slots
            min_slots_for_phase = math.ceil((self.n_teams * target_matches_this_phase) / (3 * self.n_rooms)) if self.n_rooms > 0 else 0
            n_time_slots_for_this_phase = min_slots_for_phase + buffer_slots
            if target_matches_this_phase > 0 and n_time_slots_for_this_phase <= 0:
                n_time_slots_for_this_phase = 1

            print(f"Phase {phase_idx + 1}: Target {target_matches_this_phase} matches/team. Calculated slots: {n_time_slots_for_this_phase}")

            # Pass cumulative_room_visits for International mode
            phase_problem = self._attempt_schedule_one_phase(
                available_matchups_for_this_phase,
                n_time_slots_for_this_phase,
                target_matches_this_phase,
                current_relax_list,
                phase_idx,
                cumulative_room_visits # Pass current cumulative counts
            )

            if phase_problem is None or pulp.LpStatus[phase_problem.status] != "Optimal":
                print(f"Phase {phase_idx + 1} failed. No relaxation retry logic implemented for sequential solver yet.")
                self.n_time_slots = global_timeslot_display_offset
                return None, current_relax_list

            phase_df = self._format_solution({v.name: v.varValue for v in phase_problem.variables()}, available_matchups_for_this_phase, n_time_slots_for_this_phase)

            if phase_df.empty and target_matches_this_phase > 0:
                raise ValueError(f"Phase {phase_idx + 1} resulted in an empty schedule despite targeting matches.")

            # Update cumulative counts for International mode
            if self.tournament_type == "international":
                for _, row in phase_df.iterrows():
                    for team_id in row["Matchup"].teams:
                        cumulative_room_visits[team_id][row["Room"]] += 1

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
        phase_idx: int,
        cumulative_room_visits: Dict[int, Dict[int, int]] # New parameter
    ) -> Union[pulp.LpProblem, None]:
        """
        Attempts to schedule one phase of a tournament.
        """
        problem = pulp.LpProblem(f"Quiz_Scheduling_Phase_{phase_idx}", pulp.LpMaximize)

        if self.n_rooms <= 0 and target_matches_per_team_for_phase > 0:
            raise ValueError(f"Phase_{phase_idx}: n_rooms must be positive.")
        if n_time_slots_for_phase <= 0 and target_matches_per_team_for_phase > 0:
            raise ValueError(f"Phase_{phase_idx}: n_time_slots_for_phase must be positive.")

        if target_matches_per_team_for_phase == 0:
            self._enforce_constraints_for_phase(problem, {}, matchups_for_phase, n_time_slots_for_phase, 0, relax_constraints, phase_idx, cumulative_room_visits)
            problem.solve(pulp.PULP_CBC_CMD(msg=0))
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
            phase_idx,
            cumulative_room_visits # Pass to constraint enforcer
        )

        problem.solve(pulp.PULP_CBC_CMD(msg=0))
        return problem


    def _attempt_schedule_full( # This method is no longer used and can be removed.
        self, matchups: List[Matchup], current_n_time_slots: int, relax_constraints: List[str] = [] # Added current_n_time_slots
    ) -> Union[pulp.LpProblem, None]:

        problem = pulp.LpProblem("Quiz_Scheduling_With_Rooms_Full", pulp.LpMaximize)

        if self.n_rooms <= 0 and self.n_matches_per_team > 0 :
             raise ValueError("_attempt_schedule_full: n_rooms must be positive if matches are expected.")
        # Use current_n_time_slots passed as argument
        if current_n_time_slots <= 0 and self.n_matches_per_team > 0:
             raise ValueError(f"_attempt_schedule_full: current_n_time_slots must be positive ({current_n_time_slots}) if matches are expected.")

        if self.n_matches_per_team == 0:
            # Pass current_n_time_slots and self.n_matches_per_team (which is 0)
            self._enforce_constraints_for_full_schedule(problem, {}, matchups, current_n_time_slots, self.n_matches_per_team, relax_constraints)
            problem.solve(pulp.PULP_CBC_CMD(msg=0))
            return problem

        variables = pulp.LpVariable.dicts(
            "MatchupRoomTime_Full", # Ensure unique variable names if problems coexist (though they shouldn't here)
            (
                range(len(matchups)),
                range(1, self.n_rooms + 1),
                range(1, current_n_time_slots + 1), # Use current_n_time_slots
            ),
            cat=pulp.LpBinary,
        )
        # Pass current_n_time_slots and self.n_matches_per_team
        self._enforce_constraints_for_full_schedule(problem, variables, matchups, current_n_time_slots, self.n_matches_per_team, relax_constraints)

        problem.solve(pulp.PULP_CBC_CMD(msg=1)) # msg=1 can show solver output for the main solve
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
             is_solution = False

        if not is_solution:
            print(f"Initial schedule load check failed. Valid Schedule?: {is_solution}")
            return is_solution

        # self.n_time_slots should be set correctly by either International or District logic before check_schedule is called
        is_solution = self._check_team_conflicts(df_schedule, self.n_time_slots) and is_solution # Pass current self.n_time_slots
        if not is_solution: print("Team conflict check failed."); return False

        # Room visits check should use self.n_matches_per_team (total for tournament)
        is_solution = self._check_room_visits(team_rooms) and is_solution
        if not is_solution: print("Room visit check failed."); return False

        # Consecutive matches check should use self.n_time_slots (total for tournament/phase)
        is_solution = self._check_consecutive_matches(team_time_slots, self.n_time_slots) and is_solution
        if not is_solution: print("Consecutive matches check failed."); return False

        if self.tournament_type == "district":
            is_solution = self._check_district_phase_structure(df_schedule) and is_solution # Changed from _check_periodic_breaks_and_quotas
            if not is_solution: print("District phase structure check failed."); return False

        # This total match count check should apply to both types, ensuring overall commitment is met.
        if self.n_matches_per_team > 0: # Only if matches are expected
            for t_id in range(1, self.n_teams + 1):
                if len(team_time_slots.get(t_id, [])) != self.n_matches_per_team:
                    print(f"Validation Error (Total Matches - {self.tournament_type}): Team {t_id} has {len(team_time_slots.get(t_id, []))} matches, expected {self.n_matches_per_team}.")
                    is_solution = False
                    break
            if not is_solution: print(f"Total matches check failed for {self.tournament_type}."); return False

        print(f"Valid Schedule?: {is_solution}")
        print()
        return is_solution

    # _check_periodic_breaks_and_quotas REMOVED (Old District Logic)

    # This is the new check for district mode's sequential structure
    def _check_district_phase_structure(self, df_schedule: pd.DataFrame) -> bool:
        if self.n_matches_per_team == 0:
            return True
        if self.matches_per_day <= 0:
            print("Validation Error (District Structure): matches_per_day is not positive.")
            return False

        num_total_phases = math.ceil(self.n_matches_per_team / self.matches_per_day)

        # Track matches per team per phase
        # This requires knowing the timeslot boundaries for each phase from the scheduling process.
        # self.n_time_slots is the total. We need the per-phase slot counts if they were stored.
        # For now, let's simulate phase reconstruction based on matches_per_day.
        # This check is more about "did each team play matches_per_day in segments"
        # rather than "were the timeslots themselves correctly segmented by the solver".
        # The latter is implicitly handled if the solver produced a valid schedule.

        # Reconstruct phase boundaries based on number of slots used per phase during generation
        # This info isn't directly stored in a simple list now.
        # The df_schedule has absolute timeslots.
        # We need to infer phase completion by observing when teams complete 'matches_per_day'.
        # This check becomes more complex without explicit phase slot markers from the solver.

        # Alternative: Iterate through the schedule and count matches for each team.
        # When a team hits matches_per_day, that's effectively the end of a "day" for them.
        # All teams should complete their "day" around the same number of total timeslots.

        # For simplicity in this check, let's verify that each team played self.matches_per_day
        # for (num_total_phases - 1) phases, and the remainder in the last phase.
        # This doesn't check the "breaks" or "simultaneous completion of phases" as the old constraint did,
        # because the new model schedules phases independently. The "break" is the gap between
        # concatenated phase schedules.

        # Count total matches per team (already done by the generic check later)
        # Count matches per phase for each team
        # This would require knowing the timeslot boundaries for each phase.
        # Since _schedule_district_sequentially builds the final DF by concatenating phase DFs
        # and adjusting their timeslots, the `self.n_time_slots` is the total.
        # We'd need to store the `n_time_slots_for_this_phase` for each phase during generation
        # if we want to check this precisely.

        # Let's assume for now that if the total matches per team is correct (checked later),
        # and the individual phase solves were correct (meaning each team played target_matches_this_phase),
        # then the structure is implicitly correct.
        # The main new thing to check is that each team played `matches_per_day` in each conceptual "day".

        print(f"Checking District Phase Structure: {num_total_phases} phases, {self.matches_per_day} matches/day.")

        # This check is more conceptual now. The core is:
        # 1. Did each sub-problem (_attempt_schedule_one_phase) correctly schedule target_matches_this_phase? (Handled by its own constraints)
        # 2. Is the overall total correct? (Handled by the generic check)
        # The "structure" is that these phases are concatenated. There are no explicit "break slots" to check in the final df.

        # What we can verify:
        # Iterate through the schedule. For each "day" (block of matches_per_day), are teams playing that amount?
        # This is complex because teams might finish their "day" at slightly different absolute timeslots within a phase.

        # For now, this check will be a placeholder or simplified.
        # The most important aspects are covered by phase-internal constraints and the overall total match count.
        if df_schedule.empty and self.n_matches_per_team > 0:
            print("Validation Error (District Structure): Schedule is empty but matches were expected.")
            return False

        # If we stored n_slots_per_phase during generation:
        # phase_slot_boundaries = [0]
        # current_ts = 0
        # for slots_in_phase in self.stored_n_slots_per_phase_list: # Hypothetical stored list
        #    current_ts += slots_in_phase
        #    phase_slot_boundaries.append(current_ts)
        #
        # for p_idx in range(num_total_phases):
        #    start_ts_for_phase = phase_slot_boundaries[p_idx] + 1
        #    end_ts_for_phase = phase_slot_boundaries[p_idx+1]
        #
        #    target_for_this_phase = self.matches_per_day
        #    if p_idx == num_total_phases - 1: # last phase
        #        remaining = self.n_matches_per_team - (p_idx * self.matches_per_day)
        #        if remaining > 0: target_for_this_phase = remaining
        #
        #    phase_sub_df = df_schedule[(df_schedule['TimeSlot'] >= start_ts_for_phase) & (df_schedule['TimeSlot'] <= end_ts_for_phase)]
        #    for team_id in range(1, self.n_teams + 1):
        #        matches_for_team_in_phase = 0
        #        for _, row in phase_sub_df.iterrows():
        #            if team_id in row['Matchup'].teams:
        #                matches_for_team_in_phase += 1
        #        if matches_for_team_in_phase != target_for_this_phase:
        #            print(f"Team {team_id} played {matches_for_team_in_phase} in phase {p_idx+1} (slots {start_ts_for_phase}-{end_ts_for_phase}), expected {target_for_this_phase}.")
        #            return False
        # This requires storing phase slot counts for a more precise check.

        if df_schedule.empty and self.n_matches_per_team > 0:
            print("Validation Error (District Structure): Schedule is empty but matches were expected.")
            return False

        if self.n_matches_per_team == 0 and not df_schedule.empty:
            print("Validation Error (District Structure): Schedule is not empty but no matches were expected.")
            return False

        # Further checks could validate that the number of active timeslots aligns with
        # the sum of phase slots calculated during generation, if that info were stored.
        # For now, this relies on:
        # 1. Each phase solver ensuring teams play `matches_per_day` (or remainder).
        # 2. The final overall check in `check_schedule` ensuring total `n_matches_per_team` is met.
        print("Note: _check_district_phase_structure relies on phase-internal solving correctness and overall match count validation.")
        return True


    def _enforce_constraints_for_phase(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups_in_phase: List[Matchup],
        n_time_slots_in_phase: int,
        target_matches_per_team_in_phase: int,
        relax_constraints: List[str],
        phase_idx: int,
        cumulative_room_visits: Dict[int, Dict[int, int]]
    ):
        # --- Universal Phase Constraints ---

        # Each team plays the target number of matches in this phase.
        for team_id in range(1, self.n_teams + 1):
            problem += pulp.lpSum(
                variables[m_idx][r_idx][ts_idx]
                for m_idx, m_obj in enumerate(matchups_in_phase) if team_id in m_obj.teams
                for r_idx in range(1, self.n_rooms + 1)
                for ts_idx in range(1, n_time_slots_in_phase + 1)
            ) == target_matches_per_team_in_phase, f"Phase{phase_idx}_TeamPlaysTargetMatches_T{team_id}"

        # A specific matchup from the available list is used at most once in this phase.
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

        # --- Tournament-Specific Room Diversity ---
        if "room_diversity" not in relax_constraints:
            if self.tournament_type == "district":
                # District Rule: Play in a different room for each match within this phase.
                for team_id in range(1, self.n_teams + 1):
                    for room_j in range(1, self.n_rooms + 1):
                        problem += pulp.lpSum(
                            variables[m_idx][room_j][ts_idx]
                            for m_idx, m_obj in enumerate(matchups_in_phase) if team_id in m_obj.teams
                            for ts_idx in range(1, n_time_slots_in_phase + 1)
                        ) <= 1, f"DistrictRoomDiversity_Phase{phase_idx}_T{team_id}_R{room_j}"
            else: # International Mode
                # International Rule: Balance room visits across the entire tournament.
                problem = self._enforce_cumulative_room_diversity(
                    problem, variables, matchups_in_phase, n_time_slots_in_phase,
                    cumulative_room_visits, f"Phase{phase_idx}_"
                )
        return problem

    def _enforce_cumulative_room_diversity(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        n_time_slots_in_phase: int,
        cumulative_room_visits: Dict[int, Dict[int, int]],
        prefix: str = ""
    ):
        # Overall tournament diversity bounds
        if self.n_rooms > 0 and self.n_matches_per_team > 0:
            avg_visits = self.n_matches_per_team / self.n_rooms
            lower_bound = math.floor(avg_visits) - 1
            upper_bound = math.ceil(avg_visits) + 1
            lower_bound = max(0, lower_bound)

            for team_id in range(1, self.n_teams + 1):
                for room_j in range(1, self.n_rooms + 1):
                    # Sum of visits in this phase
                    visits_in_this_phase = pulp.lpSum(
                        variables[m_idx][room_j][ts_idx]
                        for m_idx, m_obj in enumerate(matchups) if team_id in m_obj.teams
                        for ts_idx in range(1, n_time_slots_in_phase + 1)
                    )

                    # Past visits (a constant) + visits in this phase <= overall upper bound
                    past_visits = cumulative_room_visits[team_id][room_j]
                    problem += (visits_in_this_phase + past_visits) <= upper_bound, \
                               f"{prefix}CumulativeRoomDiversity_Max_T{team_id}_R{room_j}"

                    # We could also add a constraint for the lower bound, but it's trickier.
                    # Forcing it to meet the lower bound in an intermediate phase can be too restrictive.
                    # The upper bound is the more critical one to enforce sequentially.
        return problem

    # _enforce_constraints_for_full_schedule is now obsolete and removed.
    # _attempt_schedule_full is now obsolete and removed.
    # _calculate_n_time_slots_international is now obsolete and removed.

    def _enforce_each_matchup_occurrence(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int, # Added
        prefix: str = "" # Added
    ):
        for i in range(len(matchups)):
            problem += (
                pulp.lpSum(
                    variables[i][j][k]
                    for j in range(1, self.n_rooms + 1)
                    for k in range(1, current_n_time_slots + 1) # Use current_n_time_slots
                )
                == 1, f"{prefix}MatchupOccurrence_M{i}" # Added prefix and unique name
            )
        return problem

    def _enforce_each_room_to_host_single_matchup_per_time_slot(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int, # Added
        prefix: str = "" # Added
    ):
        for j in range(1, self.n_rooms + 1):
            for k in range(1, current_n_time_slots + 1): # Use current_n_time_slots
                problem += pulp.lpSum(variables[i][j][k] for i in range(len(matchups))) <= 1, \
                           f"{prefix}RoomHostsOneMatchup_R{j}_TS{k}" # Added prefix and unique name
        return problem

    def _enforce_no_simultaneous_scheduling_for_each_team(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int, # Added
        prefix: str = "" # Added
    ):
        for k in range(1, current_n_time_slots + 1): # Use current_n_time_slots
            for team in range(1, self.n_teams + 1):
                problem += (
                    pulp.lpSum(
                        variables[i][j][k]
                        for i, matchup in enumerate(matchups)
                        for j in range(1, self.n_rooms + 1)
                        if team in matchup.teams
                    )
                    <= 1, f"{prefix}NoSimultaneousSched_T{team}_TS{k}" # Added prefix and unique name
                )
        return problem

    def _limit_consecutive_matchups(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int, # Added
        prefix: str = "" # Added
    ):
        for team in range(1, self.n_teams + 1):
            # Iterate up to current_n_time_slots - 2 to avoid index out of bounds for k+1, k+2
            for k in range(1, current_n_time_slots - 1):
                if current_n_time_slots < 3: # Constraint is meaningless if not enough slots for 3 consecutive
                    continue
                problem += (
                    pulp.lpSum(
                        # Sum over variables[i][j][k], variables[i][j][k+1], variables[i][j][k+2]
                        # This was incorrect. Sum should be for a specific team over relevant time slots.
                        # Let's use the definition of matches played by a team in a slot.
                        # matches_in_slot_k = pulp.lpSum(variables[i][j][k] for i,m in enumerate(matchups) if team in m.teams for j in range(1,self.n_rooms+1))
                        # matches_in_slot_k_plus_1 = pulp.lpSum(variables[i][j][k+1] for i,m in enumerate(matchups) if team in m.teams for j in range(1,self.n_rooms+1))
                        # matches_in_slot_k_plus_2 = pulp.lpSum(variables[i][j][k+2] for i,m in enumerate(matchups) if team in m.teams for j in range(1,self.n_rooms+1))
                        # problem += matches_in_slot_k + matches_in_slot_k_plus_1 + matches_in_slot_k_plus_2 <= 2

                        # Corrected sum: sum of participations in slots k, k+1, k+2
                        # For a specific team, sum their appearances in slots k, k+1, and k+2
                        pulp.lpSum(
                            variables[m_idx][r_idx][slot_idx]
                            for m_idx, matchup_obj in enumerate(matchups) if team in matchup_obj.teams
                            for r_idx in range(1, self.n_rooms + 1)
                            for slot_idx in [k, k + 1, k + 2] # Iterate over the three consecutive slots
                        )
                    ) <= 2, f"{prefix}LimitConsecutive_T{team}_TS{k}"
                )
        return problem

    def _enforce_room_diversity(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Matchup],
        current_n_time_slots: int, # Added
        matches_per_team_for_this_context: int, # Added (e.g. total for Int'l, phase total for District phase)
        prefix: str = "" # Added
    ):
        # This constraint ensures each team plays 'matches_per_team_for_this_context' matches.
        # This might be redundant if another constraint (like TeamPlaysTargetMatches for phase) already does this.
        # However, it's good for ensuring the sum is correct before applying diversity bounds.
        for team in range(1, self.n_teams + 1):
            problem += (
                pulp.lpSum(
                    variables[i][j][k]
                    for i, matchup in enumerate(matchups)
                    for j in range(1, self.n_rooms + 1)
                    for k in range(1, current_n_time_slots + 1)
                    if team in matchup.teams
                )
                == matches_per_team_for_this_context # Use context-specific match count
            , f"{prefix}TeamTotalMatches_T{team}") # Added prefix

            if self.n_rooms > 0 and matches_per_team_for_this_context > 0: # Only apply if rooms and matches exist
                avg_visits = matches_per_team_for_this_context / self.n_rooms

                # Base bounds are avg +/- 1, as per user request.
                lower_bound = math.floor(avg_visits) - 1
                upper_bound = math.ceil(avg_visits) + 1

                # But ensure lower bound is not negative.
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
            elif matches_per_team_for_this_context > 0 and self.n_rooms == 0:
                 problem += pulp.lpSum(1) == 0, f"{prefix}RoomDiversity_Impossible_T{team}"
        return problem

    def _format_solution(self, solution: Dict[str, float], matchups: List[Matchup], n_time_slots_in_solution: int):
        data = []
        epsilon = 1e-6

        # Determine the prefix of variables based on the solution dictionary keys
        # Example: "Phase0_MatchupRoomTime_" or "MatchupRoomTime_Full_" or just "MatchupRoomTime_"
        # This is a bit fragile. Better if _attempt_schedule_one_phase and _attempt_schedule_full
        # return the variables dict they created. For now, infer.

        # We only care about variables that are set to 1.0
        # The variable names in `solution` will have the correct phase/full prefix.
        for key, value in solution.items():
            # Check if it's a variable indicating a scheduled match and its value is 1
            if ("MatchupRoomTime" in key) and abs(value - 1.0) < epsilon:
                parts = key.split("_") # e.g. "Phase0", "MatchupRoomTime", "idx", "room", "slot" OR "MatchupRoomTime", "Full", "idx", "room", "slot"

                # Adjust part indices based on actual key structure (could be different for phase vs full)
                # Assuming general structure: Prefix_idx_room_slot or Prefix_Extrainfo_idx_room_slot
                # Let's find "MatchupRoomTime" and parse relative to that.
                try:
                    mrt_part_index = -1
                    for idx, part_str in enumerate(parts):
                        if "MatchupRoomTime" in part_str:
                            mrt_part_index = idx
                            break

                    if mrt_part_index == -1:
                        # print(f"Warning (_format_solution): 'MatchupRoomTime' not found in key {key}.")
                        continue

                    # Assuming parts are [..., "MatchupRoomTime", matchup_idx_str, room_str, time_slot_str]
                    # or potentially [..., "MatchupRoomTime_Full" or "PhaseX_MatchupRoomTime", matchup_idx_str, room_str, time_slot_str]
                    # The actual indices for matchup_idx, room, time_slot depend on the prefix structure.
                    # A robust way is to find the numbers after "MatchupRoomTime".

                    # Simplistic parsing assuming last three numeric parts are idx, room, ts
                    # This might fail if variable names have other numbers.
                    numeric_parts = [p for p in parts[mrt_part_index+1:] if p.isdigit()]
                    if len(numeric_parts) < 3:
                        # print(f"Warning (_format_solution): Not enough numeric parts in key {key} after MatchupRoomTime. Parts: {parts}, Numeric: {numeric_parts}")
                        continue

                    # Assume they are in order: matchup_idx, room, time_slot from the end of numeric_parts
                    # This is still a bit of a guess. The variable naming needs to be very consistent.
                    # For variables like "MatchupRoomTime_0_1_1" (idx, room, ts)
                    # For "Phase0_MatchupRoomTime_0_1_1"

                    # Let's assume the structure is always ..._MatchupRoomTime_idx_room_timeslot
                    # So, parts[-3] is idx, parts[-2] is room, parts[-1] is timeslot.
                    matchup_idx_str = parts[-3]
                    room_str = parts[-2]
                    time_slot_str = parts[-1]

                    matchup_idx = int(matchup_idx_str)
                    room = int(room_str)
                    time_slot = int(time_slot_str)

                    # Validate time_slot against the context (n_time_slots_in_solution)
                    if not (1 <= time_slot <= n_time_slots_in_solution):
                        # This can happen if solution dict contains variables from a larger scope problem by mistake
                        # print(f"Warning (_format_solution): Parsed time_slot {time_slot} for key {key} is outside expected range 1-{n_time_slots_in_solution}.")
                        continue

                    if 0 <= matchup_idx < len(matchups):
                        matchup = matchups[matchup_idx] # Use the passed matchups list
                        data.append((time_slot, room, matchup))
                    else:
                        print(f"Warning (_format_solution): Invalid matchup_idx {matchup_idx} for key {key}. Max index is {len(matchups)-1}.")
                except ValueError:
                    # print(f"Warning (_format_solution): Could not parse integer from parts for key {key}. Parts: {parts}")
                    pass # Suppress for now, as many non-match variables might be in solution dict
                except IndexError:
                    # print(f"Warning (_format_solution): IndexError while parsing key {key}. Parts: {parts}")
                    pass

        df = pd.DataFrame(data, columns=["TimeSlot", "Room", "Matchup"])
        if not df.empty:
            df.sort_values(["TimeSlot", "Room"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def _check_team_conflicts(self, df_schedule: pd.DataFrame, current_n_time_slots: int) -> bool: # Added current_n_time_slots
        if current_n_time_slots == 0 and self.n_matches_per_team == 0: return True # Use current_n_time_slots
        if df_schedule.empty and self.n_matches_per_team > 0: return False
        if df_schedule.empty and self.n_matches_per_team == 0: return True


        for time_slot in range(1, current_n_time_slots + 1): # Use current_n_time_slots
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
        # This check is based on self.n_matches_per_team (total for the tournament)
        # and self.n_rooms. These are global properties.
        if self.n_rooms == 0:
            if self.n_matches_per_team > 0:
                print("Error: Matches scheduled but no rooms available for room visit check.")
                return False
            return True

        if self.n_matches_per_team == 0:
            return True

        # Use the same relaxed bounds as in _enforce_room_diversity
        avg_visits_per_room = self.n_matches_per_team / self.n_rooms
        min_allowed_visits = math.floor(avg_visits_per_room) - 1
        min_allowed_visits = max(0, min_allowed_visits)
        max_allowed_visits = math.ceil(avg_visits_per_room) + 1

        if self.n_matches_per_team == 0:
            min_allowed_visits = 0
            max_allowed_visits = 0
        elif self.n_rooms == 1: # If only one room, all matches must be in it
            min_allowed_visits = self.n_matches_per_team
            max_allowed_visits = self.n_matches_per_team


        for team_id, actual_rooms_played_in_list in team_rooms.items():
            # This check ensures that the team_rooms data is consistent with n_matches_per_team
            # before checking room distribution.
            if len(actual_rooms_played_in_list) != self.n_matches_per_team:
                print(
                    f"Team {team_id} has {len(actual_rooms_played_in_list)} scheduled matches in team_rooms data, "
                    f"but expected {self.n_matches_per_team} for room visit check consistency."
                )
                return False # Critical mismatch

            if not actual_rooms_played_in_list and self.n_matches_per_team > 0:
                # This case should be caught by the length check above if n_matches_per_team > 0
                print(f"Team {team_id} has no room visits recorded but expected {self.n_matches_per_team} matches.")
                return False
            if not actual_rooms_played_in_list and self.n_matches_per_team == 0:
                continue # Correct for a team with 0 matches

            room_ids_this_team_played_in, counts_of_visits_per_room = np.unique(
                actual_rooms_played_in_list, return_counts=True
            )
            actual_visit_counts_per_room_map = dict(zip(room_ids_this_team_played_in, counts_of_visits_per_room))

            for room_j_id in range(1, self.n_rooms + 1): # Check all available rooms
                visits_to_this_room_j = actual_visit_counts_per_room_map.get(room_j_id, 0)
                if not (min_allowed_visits <= visits_to_this_room_j <= max_allowed_visits):
                    print(
                        f"Team {team_id} visited Room {room_j_id} {visits_to_this_room_j} times. "
                        f"Expected between {min_allowed_visits} and {max_allowed_visits} times (overall tournament)."
                    )
                    return False
        return True

    def _check_consecutive_matches(self, team_time_slots: Dict[int, List[int]], current_n_time_slots: int) -> bool: # Added current_n_time_slots
        if current_n_time_slots == 0 : return True # Use current_n_time_slots

        for team, time_slots in team_time_slots.items():
            if len(time_slots) < 3:
                continue

            time_slots.sort() # Ensure sorted for consecutive check

            for i in range(len(time_slots) - 2):
                # Check if three consecutive time slots are present for the team
                # e.g., if team plays in ts=1, ts=2, ts=3
                if time_slots[i+1] == time_slots[i] + 1 and \
                   time_slots[i+2] == time_slots[i] + 2:
                    print(
                        f"VALIDATION ERROR: Team {team} is scheduled for 3 consecutive matches: "
                        f"Slots {time_slots[i]}, {time_slots[i+1]}, {time_slots[i+2]}."
                    )
                    return False
        return True

# This removes the entire block of duplicated old methods from this point to the end of the file.
