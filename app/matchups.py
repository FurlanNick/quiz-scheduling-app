import itertools
import math # Added for floor and ceil
from typing import List, Tuple
import pulp
import numpy as np


class MatchupSolver:
    def __init__(self, n_teams: int, n_matches_per_team: int):
        self.n_teams = n_teams
        self.n_matches_per_team = n_matches_per_team

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
            if self.n_teams - 1 == 0: # Should be caught by _validate_inputs (n_teams >=3)
                lambda_float = 0
            else:
                lambda_float = (2 * self.n_matches_per_team) / (self.n_teams - 1)

            lower_bound_meetings = math.floor(lambda_float)
            upper_bound_meetings = math.ceil(lambda_float)

            for team1 in range(1, self.n_teams + 1):
                for team2 in range(team1 + 1, self.n_teams + 1):
                    actual_meetings = 0
                    for match_tuple in solution: # solution is a list/array of tuples
                        if team1 in match_tuple and team2 in match_tuple:
                            actual_meetings += 1

                    if not (lower_bound_meetings <= actual_meetings <= upper_bound_meetings):
                        print(
                            f"Pair ({team1}, {team2}) met {actual_meetings} times, "
                            f"expected between {lower_bound_meetings} and {upper_bound_meetings} (lambda={lambda_float:.2f})."
                        )
                        is_solution = False

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
        lambda_float = (2 * self.n_matches_per_team) / (self.n_teams - 1)

        lower_bound_meetings = math.floor(lambda_float)
        upper_bound_meetings = math.ceil(lambda_float)

        # Ensure that if matches are played, teams meet at least once if lambda is low but positive.
        # Example: n_teams=10, n_matches_per_team=3. lambda = 6/9 = 0.66. floor=0, ceil=1.
        # This allows pairs to meet 0 or 1 times.
        # User requirement: "Each Team needs to quiz against a new team first." (implies at least once)
        # This is hard to enforce strictly for *all pairs* if lambda < 1.
        # The current floor/ceil approach distributes meetings as evenly as possible.
        # If n_matches_per_team = 0, then lambda_float = 0, lb=0, ub=0. Correct.

        for team1 in range(1, self.n_teams + 1):
            for team2 in range(team1 + 1, self.n_teams + 1):
                sum_matchups_for_pair = pulp.lpSum(
                    variables[i] for i, M in enumerate(matchups) if team1 in M and team2 in M
                )
                problem += sum_matchups_for_pair >= lower_bound_meetings
                problem += sum_matchups_for_pair <= upper_bound_meetings
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
