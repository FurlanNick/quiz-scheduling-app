// Function to update form visibility based on tournament type
function updateFormForTournamentType() {
    const tournamentType = document.getElementById('tournament_type').value;
    const matchesPerDayContainer = document.getElementById('matches_per_day_container');

    if (tournamentType === 'district') {
        matchesPerDayContainer.style.display = 'block';
    } else { // International
        matchesPerDayContainer.style.display = 'none';
    }
}

// Add event listener to tournament_type select
document.getElementById('tournament_type').addEventListener('change', updateFormForTournamentType);

// Call it once on page load to set initial state
updateFormForTournamentType();


async function submitForm(event) {
    event.preventDefault();

    // Set up AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
        controller.abort();
        console.log("Request timed out after 15 minutes.");
    }, 900000); // 15 minutes in milliseconds

    const n_teams = document.getElementById('n_teams').value;
    const n_matches_per_team = document.getElementById('n_matches_per_team').value;
    const n_rooms = document.getElementById('n_rooms').value;
    const tournament_type = document.getElementById('tournament_type').value;
    const matches_per_day = document.getElementById('matches_per_day').value;

    // Hardcode buffer slots to 0 as per user request
    const phase_buffer_slots = 0;
    const international_buffer_slots = 0;

    // Show the loading spinner and clear previous errors
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('error-message').textContent = '';

    try {
        console.log("Sending request...");

        const response = await fetch('/generate-schedule/', {
            method: 'POST',
            signal: controller.signal, // Pass the signal to the fetch request
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                n_teams: parseInt(n_teams),
                n_matches_per_team: parseInt(n_matches_per_team),
                n_rooms: parseInt(n_rooms),
                tournament_type: tournament_type,
                phase_buffer_slots: phase_buffer_slots,
                international_buffer_slots: international_buffer_slots,
                matches_per_day: parseInt(matches_per_day)
            })
        });

        clearTimeout(timeoutId); // Clear the timeout if the request completes in time

        console.log("Response received...");
        document.getElementById('spinner').style.display = 'none';

        if (response.ok) {
            const data = await response.json();
            console.log("Data received (in submitForm): ", JSON.parse(JSON.stringify(data))); 
            
            if (data && typeof data === 'object') {
                displaySchedule(data);
            } else {
                displayError("Failed to parse or receive valid schedule data from server.");
            }
        } else {
            const errorData = await response.json().catch(() => ({ detail: "Failed to parse error response from server." })); 
            displayError(errorData.detail || "An unknown error occurred on the server.");
        }
    } catch (error) {
        clearTimeout(timeoutId); // Also clear timeout on any other error
        document.getElementById('spinner').style.display = 'none';

        if (error.name === 'AbortError') {
            displayError('Schedule generation timed out after 15 minutes. Please try again with different parameters.');
        } else {
            console.log("Error occurred during fetch or processing: ", error);
            displayError('An unexpected error occurred. Please check console for details.');
        }
    }
}

function displayError(message) {
    console.log("Displaying error: ", message);
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';

    // Clear previous results table
    const resultTable = document.getElementById('resultTable');
    if (resultTable) {
        const tableHead = resultTable.querySelector('thead');
        const tableBody = resultTable.querySelector('tbody');
        if (tableHead) tableHead.innerHTML = '';
        if (tableBody) tableBody.innerHTML = '';
    }
}

function displaySchedule(responseData) { 
    if (!responseData || typeof responseData !== 'object') {
        console.error("app.js displaySchedule - ERROR: responseData is null, undefined, or not an object. Received:", responseData);
        displayError("Invalid data received by display function.");
        return;
    }

    console.log("app.js displaySchedule - Entry. Data:", JSON.parse(JSON.stringify(responseData)));
    const { grid_schedule, max_sched_timeslot, max_sched_room, constraints_relaxed } = responseData;

    const resultTable = document.getElementById('resultTable');
    const tableHead = resultTable.querySelector('thead');
    const tableBody = resultTable.querySelector('tbody');

    tableHead.innerHTML = '';
    tableBody.innerHTML = '';

    if (!grid_schedule || max_sched_timeslot === 0 || max_sched_room === 0) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.textContent = "No schedule data to display.";
        td.colSpan = (max_sched_room || 0) + 2; 
        tr.appendChild(td);
        tableBody.appendChild(tr);
        return;
    }

    const headerRow = document.createElement('tr');
    const thTimeSlot = document.createElement('th');
    thTimeSlot.textContent = "TimeSlot";
    headerRow.appendChild(thTimeSlot);

    const thSeatPos = document.createElement('th');
    thSeatPos.textContent = "Seat Position";
    headerRow.appendChild(thSeatPos);

    for (let r = 1; r <= max_sched_room; r++) {
        const thRoom = document.createElement('th');
        thRoom.textContent = `Room ${r}`;
        headerRow.appendChild(thRoom);
    }
    tableHead.appendChild(headerRow);

    for (let ts = 1; ts <= max_sched_timeslot; ts++) {
        const ts_key = `ts_${ts}`;
        for (let seat_idx = 0; seat_idx < 3; seat_idx++) {
            const tr = document.createElement('tr');

            if (seat_idx === 0) {
                const tdTimeSlot = document.createElement('td');
                tdTimeSlot.textContent = ts;
                tdTimeSlot.rowSpan = 3;
                tr.appendChild(tdTimeSlot);
            }

            const tdSeatPos = document.createElement('td');
            tdSeatPos.textContent = seat_idx + 1;
            tr.appendChild(tdSeatPos);

            for (let r = 1; r <= max_sched_room; r++) {
                const room_key = `room_${r}`;
                const tdRoom = document.createElement('td');
                let teamId = "---"; 

                if (grid_schedule && grid_schedule[ts_key] && grid_schedule[ts_key][room_key]) {
                    const teamsInMatch = grid_schedule[ts_key][room_key]; 
                    if (teamsInMatch && teamsInMatch.length > seat_idx && teamsInMatch[seat_idx] !== null && teamsInMatch[seat_idx] !== undefined) {
                        teamId = teamsInMatch[seat_idx];
                    }
                }
                tdRoom.textContent = teamId;
                tr.appendChild(tdRoom);
            }
            tableBody.appendChild(tr);
        }
    }

    if (constraints_relaxed && constraints_relaxed.length > 0) {
        const relaxedMessage = `The following constraints were relaxed: ${constraints_relaxed.join(', ')}.`;
        displayError(relaxedMessage);
    }
}
