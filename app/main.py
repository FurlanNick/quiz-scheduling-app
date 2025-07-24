import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.matchups import MatchupSolver
from app.scheduler import ScheduleSolver
from app.models import (
    MatchupsRequest,
    MatchupsResponse,
    Matchup,
    ScheduleRequest,
    ScheduleResponse,
    ScheduleItem,
)

app = FastAPI(
    title="Quiz Schedule Generator API",
    description="API to generate valid quiz schedules based on input parameters",
    version="1.0.0",
    contact={
        "name": "Stephen Mosher",
        "url": "https://github.com/s-g-mo",
        "contact": "email@example.com",
    },
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join("app/static", "index.html"))


@app.post("/generate-matchups/", tags=["Matchups"])
async def generate_matchups(request: MatchupsRequest) -> MatchupsResponse:
    try:
        matchups_solver = MatchupSolver(
            n_teams=request.n_teams, 
            n_matches_per_team=request.n_matches_per_team,
            tournament_type=request.tournament_type # Pass tournament_type
        )

        all_possible_matchups = matchups_solver.generate_all_possible_matchups()
        matchup_solutions = matchups_solver.find_matchup_solutions(
            matchups=all_possible_matchups, max_solutions=request.n_matchup_solutions
        )
        if matchup_solutions:
            for matchup_solution in matchup_solutions:
                matchups_solver.check_matchups(matchup_solution)
            solutions = {
                f"solution_set_{i+1}": [Matchup(teams=tuple(matchup)) for matchup in solution]
                for i, solution in enumerate(matchup_solutions)
            }

            return MatchupsResponse(solutions=solutions)
        else:
            return MatchupsResponse(solutions={})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-schedule/", tags=["Schedule"])
async def generate_schedule(request: ScheduleRequest) -> ScheduleResponse:
    try:
        schedule_solver = ScheduleSolver(
            n_teams=request.n_teams,
            n_matches_per_team=request.n_matches_per_team,
            n_rooms=request.n_rooms,
            # n_time_slots is now calculated internally by ScheduleSolver
            tournament_type=request.tournament_type,
            phase_buffer_slots=request.phase_buffer_slots, 
            international_buffer_slots=request.international_buffer_slots,
            matches_per_day=request.matches_per_day # Pass the new field
        )
        # MatchupsRequest does not need tournament_type or buffer slots
        matchups_request = MatchupsRequest(
            n_teams=request.n_teams,
            n_matches_per_team=request.n_matches_per_team,
            n_matchup_solutions=1,
            tournament_type=request.tournament_type # Ensure tournament_type is passed here
        )
        matchups_response = await generate_matchups(request=matchups_request)

        for proposed_matchups in matchups_response.solutions.values():
            schedule, constraints_relaxed = schedule_solver.schedule_matches(proposed_matchups)
            if schedule is not None:
                schedule_solver.check_schedule(schedule)
                schedule_items = [
                    ScheduleItem(
                        TimeSlot=row["TimeSlot"],
                        Room=row["Room"],
                        Matchup=row["Matchup"], # Matchup is already a Pydantic model instance from ScheduleSolver
                    )
                    for _, row in schedule.iterrows() # schedule is the DataFrame
                ]

                # Transform DataFrame for grid display
                grid_data = {}
                # max_sched_ts is now determined by the solver's calculated n_time_slots
                # max_sched_room is still determined by iterating through the actual schedule data
                # or could also be self.n_rooms from the solver if we assume all rooms up to n_rooms might be used.
                # For now, let's keep max_sched_room based on actual data, and max_sched_ts from calculated.
                
                actual_max_room_in_schedule = 0
                actual_max_ts_in_schedule = 0 # To see if schedule uses fewer slots than calculated

                if not schedule.empty:
                    for _, row in schedule.iterrows():
                        ts = int(row["TimeSlot"])
                        room = int(row["Room"])
                        teams_in_match = list(row["Matchup"].teams) # (T_seat1, T_seat2, T_seat3)

                        actual_max_ts_in_schedule = max(actual_max_ts_in_schedule, ts)
                        actual_max_room_in_schedule = max(actual_max_room_in_schedule, room)

                        ts_key = f"ts_{ts}"
                        room_key = f"room_{room}"

                        if ts_key not in grid_data:
                            grid_data[ts_key] = {}
                        
                        # Ensure teams_in_match is a list of 3 (it should be)
                        # If a matchup could have fewer than 3 teams (not current case) or if a seat can be empty,
                        # this might need padding with a placeholder like "---" or None.
                        # Assuming row["Matchup"].teams always provides 3 team IDs.
                        grid_data[ts_key][room_key] = teams_in_match
                
                return ScheduleResponse(
                    schedule=schedule_items, 
                    constraints_relaxed=constraints_relaxed,
                    grid_schedule=grid_data,
                    max_sched_timeslot=schedule_solver.n_time_slots, # Use calculated total n_time_slots
                    max_sched_room=actual_max_room_in_schedule if not schedule.empty else request.n_rooms 
                                      # Use actual max room from schedule, or input n_rooms if schedule is empty but valid
                )
        # If loop completes without returning a schedule (e.g. matchups_response was empty)
        raise HTTPException(status_code=404, detail="No valid matchups found to generate a schedule.")
    except ValueError as ve: # Catch specific validation errors from scheduler
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(ve)) # Return as Bad Request
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
