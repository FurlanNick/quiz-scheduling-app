import itertools
import math # Added for floor and ceil
from typing import List, Tuple
import pulp
import numpy as np


class MatchupSolver:
    def __init__(self, n_teams: int, n_matches_per_team: int, tournament_type: str = "international"):
        self.n_teams = n_teams
        self.n_matches_per_team = n_matches_per_team
        self.tournament_type = tournament_type # Store tournament type

    def generate_all_possible_matchups(self) -> List[Tuple[int, int, int]]:
        teams = list(range(1, self.n_teams + 1))
        all_triples = list(itertools.combinations(teams, 3))
        possible_matchups = []
        for triple in all_triples:
            for perm in itertools.permutations(triple):
                possible_matchups.append(perm)
        return possible_matchups

    def find_matchup_solutions(
        self, matchups: List[Tuple[int, int, int]], max_solutions: int = 10
    ) -> List[np.ndarray]:
        """
        Generates matchup solutions based on the number of teams, the number of matches played by
        each team, and the matchup constraints.

        Args:
            matchups: A list of tuples representing possible team matchups.
            max_solutions: Maximum number of solutions to return.

        Returns:
            A list of numpy arrays where each array contains a valid set of matchups that satisfy
            the given constraints.
        """

        self._validate_inputs() # Ensure inputs are valid before proceeding

        solutions = []
        problem = pulp.LpProblem("Quiz_Scheduling", sense=pulp.LpMaximize)
        variables = pulp.LpVariable.dicts("Matchup", (range(len(matchups))), cat=pulp.LpBinary)

        self.enforce_constraints(problem, variables, matchups)

        while len(solutions) < max_solutions:
            problem.solve()
            if pulp.LpStatus[problem.status] == "Optimal":
                solution = [i for i in range(len(matchups)) if variables[i].varValue == 1]
                selected_matchups = [matchups[i] for i in solution]
                solutions.append(np.array(selected_matchups))
                problem += pulp.lpSum(variables[i] for i in solution) <= len(solution) - 1
            else:
                break

        return solutions

    def check_matchups(self, solution: np.array) -> bool:
        # solution is np.array of matchups, e.g. np.array([[1,2,3], [1,4,5]])
        print(solution)
        is_solution = True

        # Team-specific checks
        for team in range(1, self.n_teams + 1):
            team_filter = (solution == team) # boolean array where team is present

            # Check to ensure each team has exactly n_matches_per_team
            if team_filter.sum() != self.n_matches_per_team:
                print(
                    f"Team {team} has {team_filter.sum()} matches, expected "
                    f"{self.n_matches_per_team}."
                )
                is_solution = False

            # Check bench constraints (seat positions)
            base_visits = self.n_matches_per_team // 3
            bench_counts = team_filter.sum(axis=0) # Counts occurrences of `team` in each column (seat)

            if self.n_matches_per_team % 3 == 0:
                if not np.all(bench_counts == base_visits):
                    print(
                        f"Team {team} seat distribution {bench_counts} not equal to {base_visits} for each seat."
                    )
                    is_solution = False
            else: # n_matches_per_team % 3 != 0
                if (bench_counts < base_visits).any() or \
                   (bench_counts > base_visits + 1).any():
                    print(
                        f"Team {team} seat distribution {bench_counts} not within range "
                        f"[{base_visits}, {base_visits + 1}] for each seat. Base visits: {base_visits}"
                    )
                    is_solution = False

        # Pairwise opponent check
        if self.n_teams >= 2:
            expected_min_meetings_check = 0
            # For District, max is not strictly checked here beyond what other constraints impose.
            # For International, it's ceil(lambda).
            expected_max_meetings_check = float('inf')
            lambda_log_str = "N/A"

            if self.tournament_type == "district":
                if self.n_matches_per_team > 0:
                    expected_min_meetings_check = 1
                # For District, the check is only concerned with meeting at least once.
                lambda_log_str = ">= 1 (District)"
            else:  # International mode or default
                if self.n_teams - 1 > 0:
                    lambda_float = (2 * self.n_matches_per_team) / (self.n_teams - 1)
                    expected_min_meetings_check = math.floor(lambda_float)
                    expected_max_meetings_check = math.ceil(lambda_float)
                    lambda_log_str = f"~{lambda_float:.2f} (Intl)"
                else: # Should ideally not be reached due to n_teams >= 3 validation
                    expected_min_meetings_check = 0
                    expected_max_meetings_check = 0

                if self.n_matches_per_team == 0: # If no matches, no meetings expected
                    expected_min_meetings_check = 0
                    expected_max_meetings_check = 0

            for team1 in range(1, self.n_teams + 1):
                for team2 in range(team1 + 1, self.n_teams + 1):
                    actual_meetings = 0
                    for match_tuple in solution:  # solution is a np.array of matchups
                        if team1 in match_tuple and team2 in match_tuple:
                            actual_meetings += 1

                    passes_check = False
                    if self.tournament_type == "district":
                        if actual_meetings >= expected_min_meetings_check:
                            passes_check = True
                    else: # International
                        if expected_min_meetings_check <= actual_meetings <= expected_max_meetings_check:
                            passes_check = True

                    if not passes_check:
                        is_solution = False
                        if self.tournament_type == "district":
                            print(
                                f"Pair ({team1}, {team2}) met {actual_meetings} times. For District mode, "
                                f"expected >= {expected_min_meetings_check}."
                            )
                        else: # International
                            print(
                                f"Pair ({team1}, {team2}) met {actual_meetings} times. For International mode, "
                                f"expected between {expected_min_meetings_check} and {expected_max_meetings_check} (lambda: {lambda_log_str})."
                            )

        print(f"Valid Matchups?: {is_solution}")
        print()
        return is_solution

    def enforce_constraints(
        self,
        problem: pulp.LpProblem,
        variables: pulp.LpVariable,
        matchups: List[Tuple[int, int, int]],
    ) -> None:
        problem = self._enforce_each_team_in_exactly_n_matches_per_team(
            problem, variables, matchups
        )
        problem = self._enforce_unique_opponents_constraint(problem, variables, matchups)
        problem = self._enforce_bench_constraints(problem, variables, matchups)

    def _enforce_each_team_in_exactly_n_matches_per_team(
        self, problem: pulp.LpProblem, variables: pulp.LpVariable, matchups: list
    ) -> pulp.LpProblem:
        for team in range(1, self.n_teams + 1):
            problem += (
                pulp.lpSum(variables[i] for i, M in enumerate(matchups) if team in M)
                == self.n_matches_per_team
            )
        return problem

    def _enforce_bench_constraints(
        self, problem: pulp.LpProblem, variables: pulp.LpVariable, matchups: list
    ) -> pulp.LpProblem:
        for team in range(1, self.n_teams + 1):
            for position in range(3):
                if self.n_matches_per_team < 3: # Less than 3 matches
                    # Original logic: <=1 visit per bench. This seems fine.
                    # If n_matches_per_team is 1 or 2.
                    # Base_visits will be 0. So it means it can be 0 or 1.
                    # This is consistent with the general base_visits / base_visits+1 logic below.
                    # So this special if condition might not be strictly needed if the else handles it.
                    # Let's test: n_matches_per_team = 2. base_visits = 0.
                    # Constraints: sum >= 0, sum <= 1. Correct.
                    # n_matches_per_team = 1. base_visits = 0.
                    # Constraints: sum >= 0, sum <= 1. Correct.
                    # So, the n_matches_per_team < 3 condition can be removed and covered by the general else.
                    # However, keeping it explicit might be for clarity or a subtle case. For now, let's keep it.
                    problem += (
                        pulp.lpSum(
                            variables[i] for i, M in enumerate(matchups) if M[position] == team
                        )
                        <= 1
                    )
                elif self.n_matches_per_team % 3 == 0: # Exactly multiple of 3
                    problem += pulp.lpSum(
                        variables[i] for i, M in enumerate(matchups) if M[position] == team
                    ) == (self.n_matches_per_team // 3)
                else: # Not a multiple of 3, and >= 3
                    base_visits = self.n_matches_per_team // 3
                    problem += (
                        pulp.lpSum(
                            variables[i] for i, M in enumerate(matchups) if M[position] == team
                        )
                        >= base_visits
                    )
                    problem += (
                        pulp.lpSum(
                            variables[i] for i, M in enumerate(matchups) if M[position] == team
                        )
                        <= base_visits + 1
                    )
        return problem

    def _enforce_unique_opponents_constraint( # Name could be updated, e.g., _enforce_pairwise_meeting_constraint
        self, problem: pulp.LpProblem, variables: pulp.LpVariable, matchups: list
    ) -> pulp.LpProblem:
        if self.n_teams < 2: # No pairs to constrain if less than 2 teams
            return problem

        # Calculate lambda (average number of times pairs meet)
        # self.n_teams - 1 will not be zero due to _validate_inputs (n_teams >= 3)

        if self.tournament_type == "district":
            # For District mode, ensure each pair meets at least once. No upper bound from this constraint.
            lower_bound_meetings = 0
            if self.n_matches_per_team > 0 and self.n_teams > 1: # Need at least 2 teams to form a pair
                lower_bound_meetings = 1

            if lower_bound_meetings > 0: # Only add constraint if it's meaningful
                for team1 in range(1, self.n_teams + 1):
                    for team2 in range(team1 + 1, self.n_teams + 1):
                        sum_matchups_for_pair = pulp.lpSum(
                            variables[i] for i, M in enumerate(matchups) if team1 in M and team2 in M
                        )
                        problem += sum_matchups_for_pair >= lower_bound_meetings, \
                                   f"DistrictOpponentMinMet_T{team1}_T{team2}"
            # No explicit upper bound constraint for District mode here. Max meetings implicitly handled by total matches.

        else: # International mode or default
            # Use lambda-based floor/ceil for both lower and upper bounds for more even distribution
            lambda_float = 0.0 # Default if n_teams <= 1, though validated n_teams >= 3
            if self.n_teams - 1 > 0:
                 lambda_float = (2 * self.n_matches_per_team) / (self.n_teams - 1)

            lower_bound_meetings = math.floor(lambda_float)
            upper_bound_meetings = math.ceil(lambda_float)

            # If no matches are to be played, bounds should be 0
            if self.n_matches_per_team == 0:
                lower_bound_meetings = 0
                upper_bound_meetings = 0

            for team1 in range(1, self.n_teams + 1):
                for team2 in range(team1 + 1, self.n_teams + 1):
                    sum_matchups_for_pair = pulp.lpSum(
                        variables[i] for i, M in enumerate(matchups) if team1 in M and team2 in M
                    )
                    problem += sum_matchups_for_pair >= lower_bound_meetings, \
                               f"InternationalOpponentMinMet_T{team1}_T{team2}"
                    problem += sum_matchups_for_pair <= upper_bound_meetings, \
                               f"InternationalOpponentMaxMet_T{team1}_T{team2}"
        return problem

    def _validate_inputs(self):
        assert self.n_teams >= 3, \
            f"Number of teams must be at least 3. Got {self.n_teams}."

        assert self.n_matches_per_team >= 0, \
            f"Number of matches per team must be non-negative. Got {self.n_matches_per_team}."

        # The product of total teams and matches per team must be divisible by 3,
        # as each match involves 3 teams.
        assert (self.n_teams * self.n_matches_per_team) % 3 == 0, \
            f"The product of teams and matches_per_team ({self.n_teams} * {self.n_matches_per_team} = " \
            f"{self.n_teams * self.n_matches_per_team}) must be divisible by 3."

        # Previous assertions like `self.n_teams > 2 * self.n_matches_per_team` or
        # `self.n_teams % 3 == 0 if self.n_matches_per_team % 3 != 0` have been removed
        # as they were too restrictive or based on the old opponent uniqueness model.
        # The current set of constraints (team/match counts, bench balance, pair meetings)
        # along with basic divisibility rules should define feasibility.
