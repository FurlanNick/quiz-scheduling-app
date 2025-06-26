from pydantic import BaseModel, Field
from typing import Dict, List, Tuple


class MatchupsRequest(BaseModel):
    n_teams: int = Field(30, description="Number of teams participating", example=30)
    n_matches_per_team: int = Field(3, description="Number of matches each team plays", example=3)
    n_matchup_solutions: int = Field(
        2, description="Number of matchup solutions to generate", example=2
    )


class Matchup(BaseModel):
    teams: Tuple[int, int, int]


class MatchupsResponse(BaseModel):
    solutions: Dict[str, List[Matchup]]


class ScheduleRequest(BaseModel):
    n_teams: int = Field(30, description="Number of teams participating", example=30)
    n_matches_per_team: int = Field(3, description="Number of matches each team plays", example=3)
    n_rooms: int = Field(5, description="Number of available rooms for matches", example=5)
    n_time_slots: int = Field(6, description="Number of time slots for scheduling", example=6)
    # For future, consider: from typing import Literal; tournament_type: Literal['international', 'district']
    tournament_type: str = Field(
        "international",
        description="Type of tournament: 'international' or 'district'. Determines scheduling rules.",
        example="district"
    )


class ScheduleItem(BaseModel):
    TimeSlot: int
    Room: int
    Matchup: Matchup


from typing import Union # Added for Union type

class ScheduleResponse(BaseModel):
    schedule: List[ScheduleItem] # Original list-based schedule
    constraints_relaxed: List[str]

    # New fields for grid display
    grid_schedule: Dict[str, Dict[str, List[Union[int, str]]]] = Field(
        default_factory=dict,
        description="Schedule data formatted for a grid. Keyed by timeslot (e.g., 'ts_1'), then room (e.g., 'room_1'), with a list of team IDs for seats [seat1, seat2, seat3]. Empty seats can be represented by a placeholder like '---'."
    )
    max_sched_timeslot: int = Field(0, description="The highest timeslot number present in the generated schedule.")
    max_sched_room: int = Field(0, description="The highest room number present in the generated schedule.")
