# Quiz Scheduling App

## What is the Quiz Scheduling App?
The Quiz Scheduling App is a tool designed to generate schedules for quiz tournaments. Its primary purpose is to create valid and fair schedules by taking into account various constraints. 

In this quiz format, matches involve 3 teams. During each match, teams are assigned to one of three different bench positions. The app's main goal is to find schedules that optimally balance various logistical considerations (like venue availability) and fairness constraints. These fairness constraints include ensuring teams face unique opponents, distributing bench positions and room assignments as evenly as possible, and minimizing consecutive (back-to-back) games for any team.

A simple app to build schedules for Quiz Meet tournaments. Where possible, this tool generates *valid* schedules that satisfy several constraints described below.

## Problem Description and Solution Approach

A friend of mine approached me and asked if I would be willing to help him with his Quiz Meet tournament scheduling problem and I said yes. He informed me of the following facts:

1. In a Quiz Meet, matches consistent of 3 teams.
2. In each match, teams are assigned 1 of 3 possible bench positions.
3. In a typical Quiz Meet, teams play 3 matches, but this could potentially vary.
4. Matches are typically held in different rooms and of course different times.

He also informed me that he wanted a schedule that would meet several constraints. The constraints fall into two categories:

1. Matchup Constraints
    - No team should face the same opponent more than once.
    - Teams should visit bench positions as evenly as possible.

2. Scheduling Constraints
    - No team should have more than 1 back-to-back match (i.e. a team can play 2 consecutive matches, but not 3).
    - A team's matches should be scheduled across rooms as evenly as possible.

My friend considers a schedule that meets the above constraints to be a *valid* and *fair* schedule.

Given the nature of the constraints, I realized that the scheduling problem can be considered two different problems. The first problem is to generate a set of *valid* matchups given a number of teams and number of matches per team (valid in the sense of the constraints above). Here's an example of a *valid* matchup given `n_teams = 8` and `n_matches_per_team = 3`:

```python
[[7 2 1]
 [3 1 4]
 [1 8 6]
 [2 4 8]
 [5 6 2]
 [8 5 3]
 [6 3 7]
 [4 7 5]]
```

A *matchup* is essentially an ordered triple of integers, where the integers represent team numbers or ids, and the position of the integer in the triple represents the bench position. In the example above, where `n_matches_per_team = 3`, we can see that each integer (or team) appears in each column (bench position) exactly once, each integer appears exactly 3 times throughout the array, and each team faces unique opponents in each of their matchups (no repeat opponents).

Given a *valid* set of *matchups*, the second problem is to schedule the *matchups* in accordance with the scheduling constraints above, thereby producing a *valid* schedule. The additional input parameters for the scheduling problem are the number of rooms available to host matches at the venue and the number of time slots. My friend considers a *valid* schedule, meaning one that meets all the above constraints, to be as fair and balanced as possible.

The approach I took so solve each of these problems is to consider them binary linear programming problems. I made use of the PuLP library to define each problem, the associated variables, the constraints, and generate possible solutions.

## Some Notes on the Solution Space
In working on this problem, I've observed several key factors that affect whether a valid solution is possible. These observations helped shape the logic and constraints used in both the matchup and schedule generation phases of the app. While the app works for a wide range of configurations, certain edge cases remain infeasible based on the constraints above.

### Matchups
*Valid* solutions to the matchup problem only exist for certain special combinations of `n_teams` and `n_matches_per_team`. The pattern is best captured in the following table, where the rows indicate `n_matches_per_team` and the columns indicate `n_teams`.

|                    |1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|
|--------------------|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| 1                  |x|x|o|x|x|o|x|x|o|x|x|o|x|x|o| | | | |
| 2                  |x|x|x|x|x|o|x|x|o|x|x|o|x|x|o| | | | |
| 3                  |x|x|x|x|x|x|o|o|o|o|o|o|o|o|o|o|o|o|o|
| 4                  |x|x|x|x|x|x|x|x|o|x|x|o|x|x|o| | | | |
| 5                  |x|x|x|x|x|x|x|x|x|x|x|o|x|x|o| | | | |
| 6                  |x|x|x|x|x|x|x|x|x|x|x|x|o|o|o| | | | |
| 7                  | | | | | | | | | | | | | | |o| | | | |
| 8                  | | | | | | | | | | | | | | |x| | | | |
| 9                  | | | | | | | | | | | | | | | | | | |o|
| 10                 | | | | | | | | | | | | | | | | | | | |
- `x` represents a solution combination that was tested and did not work.
- `o` represents a solution combination for which a valid solution was found to exist.

Obviously this table is just a empirical tool to understand the solution space to the theoretical matchups problem. In other words, we don't really care about `n_matches_per_team >= 4` for most applications, but the tables helps us understand the solution space. Empirically, I observed that:

- If `n_teams` is a multiple of 3, then valid matchups only exist if `n_teams / n_matches_per_team >= 2` (look at the columns).

- If `n_matches_per_team` **is not** a multiple of 3, then valid matchups only exist if `n_teams` is a multiple of 3 which is also `> 2 x n_matches_per_team` (look at the rows that aren't indexed by multiples of 3).

- If `n_matches_per_team` **is** a multiple of 3, then valid matchups exist for any value of `n_teams` when `n_teams > 2 x n_matches_per_team` (look at the rows that are indexed by a multiple of 3).

### Scheduling
I've found it a little harder to pin down exactly what combinations of `n_rooms` and `n_time_slots` allow for *valid* schedules given a set of matchups. But here are a few observations:

1. Room Diversity:
    - If the number of rooms is less than the number of matches per team, it's challenging to guarantee that each team plays in a different room for each match.

    - When `n_rooms >= n_matches_per_team`, the schedule is typically feasible, and room diversity is maintained. Otherwise, I implemented a constraint relaxation where room diversity can be compromised if necessary to generate a schedule.

2. Minimum Number of Time Slots:
    - The minimum number of time slots is closely related to the number of teams and rooms available. In general, if you don't have enough time slots, the scheduling constraints (e.g., avoiding back-to-back matches) cannot be satisfied.

    - Empirically, I've found that a good rule of thumb for the number of time slots is: `n_time_slots >= ceil(n_teams / n_rooms)`. This is the absolute minimum, but often more time slots are required to fully satisfy the constraints.

3. Back-to-Back Matches:

    - Limiting consecutive matches for a team is one of the trickiest constraints. With too few time slots or too few rooms, this constraint becomes impossible to meet.

    - Adding more time slots helps prevent consecutive matches for individual teams, but the exact number depends on the specific configuration of rooms and teams.



## Features
- Generate valid team matchups for a quiz meet.
- Schedule matches across multiple rooms and time slots.
- Ensure no team has consecutive matches and respects room diversity constraints.
- Customize the number of teams, matches per team, rooms, and time slots.
- Randomize generated schedules with customizable inputs.
- Simple frontend interface with live schedule generation.
- Easily accessible REST API for generating matchups and schedules.

## Technologies Used

- Python: Core programming language for building the logic and backend.
- FastAPI: The web framework used to create the API.
- Pulp: A linear programming library used to optimize quiz scheduling.
- Uvicorn: ASGI server for running the FastAPI application.
- Vercel: Used for deployment of the frontend and API.
- HTML/CSS/JavaScript: For building the frontend interface.

## Installation

To run the app locally, follow these steps:

1. Clone the Repository:

```bash
git clone https://github.com/s-g-mo/quiz-scheduling-app.git
cd quiz-scheduling-app
```

2. Create a Virtual Environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # For Windows use `venv\Scripts\activate`
```

3. Install Dependencies: Install all required dependencies by running:

```bash
pip install -r requirements.txt
```

4. Run the App: You can run the app locally using Uvicorn:

```bash
uvicorn app.main:app --reload
```
 - Visit http://127.0.0.1:8000/static/index.html to access the frontend.

5. Access the API Documentation. You can visit the interactive docs at:

```bash
http://127.0.0.1:8000/docs
```

## Usage

Users interact with the Quiz Scheduling App by providing several key parameters: the number of teams participating, the desired number of matches each team should play, the number of available rooms for matches, and the number of available time slots. These inputs can be provided either through an intuitive web interface (an HTML page) or programmatically by calling the available API endpoints.

Based on these inputs, the application processes the request and attempts to generate a valid and fair schedule. The output is typically the generated schedule itself. If the constraints and parameters make it impossible to find a valid schedule, the application will indicate this, sometimes suggesting which constraints might have been relaxed to produce a near-valid schedule.

### Frontend

1. Open the app in your browser by navigating to the URL (e.g., http://127.0.0.1:8000/static/index.html for local use or the deployed Vercel URL).

2. Fill out the form by entering:
    - Number of Teams
    - Matches per Team
    - Number of Rooms
    - Number of Time Slots

3. Click "Generate Schedule" to create the quiz schedule.

4. The generated schedule will be displayed in a table format, along with any relaxed constraints (if applicable).

## API Endpoints

1. Generate Matchups
    - Endpoint: /generate-matchups/
    - Method: POST
    - Request Body:
    ```json
    {
      "n_teams": 30,
      "n_matches_per_team": 3,
      "n_matchup_solutions": 2
    }
    ```
    - Response:
    ```json
    {
      "solutions": {
        "solution_set_1": [
          { "teams": [30, 5, 3] },
          { "teams": [1, 13, 12] },
          ...
        ]
      }
    }
    ```

2. Generate Schedule
    - Endpoint: /generate-schedule/
    - Method: POST
    - Request Body:
    ```json
    {
      "n_teams": 30,
      "n_matches_per_team": 3,
      "n_rooms": 5,
      "n_time_slots": 6
    }
    ```
    - Response:
    ```json
    {
      "schedule": [
        { "TimeSlot": 1, "Room": 1, "Matchup": { "teams": [30, 5, 3] } },
        ...
     ],
     "constraints_relaxed": ["room_diversity"]
    }
    ```

## Deployment
This app can be easily deployed using Vercel or any other hosting service. Follow the steps below to deploy on Vercel:

1. Sign up for a Vercel account.
2. Link your GitHub repository to Vercel.
3. Ensure your project includes a requirements.txt file and a Procfile.
4. Deploy the project. Vercel will handle the deployment process and give you a URL for your app.

## Future Enhancements
- Include more customizable constraints for the scheduling process.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions, issues, or suggestions, please reach out via the GitHub repository
