document.getElementById('scheduleForm').addEventListener('submit', submitForm);

async function submitForm(event) {
    event.preventDefault();
    const n_teams = document.getElementById('n_teams').value;
    const n_matches_per_team = document.getElementById('n_matches_per_team').value;
    const n_rooms = document.getElementById('n_rooms').value;
    const n_time_slots = document.getElementById('n_time_slots').value;

    // Show the loading spinner
    document.getElementById('spinner').style.display = 'block';

    try {
        console.log("Sending request...");

        const response = await fetch('/generate-schedule/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                n_teams: parseInt(n_teams),
                n_matches_per_team: parseInt(n_matches_per_team),
                n_rooms: parseInt(n_rooms),
                n_time_slots: parseInt(n_time_slots)
            })
        });

        console.log("Response received...");

        // Hide the loading spinner
        document.getElementById('spinner').style.display = 'none';

        // Clear the table and error message regardless of success or failure
        const resultTable = document.getElementById('resultTable').querySelector('tbody');
        resultTable.innerHTML = '';  // Clear table
        document.getElementById('error-message').textContent = '';  // Clear error message

        // Check for a successful response
        if (response.ok) {
            const data = await response.json();
            console.log("Data received: ", data);
            displaySchedule(data.schedule, data.constraints_relaxed); // Display the new schedule
        } else {
            const errorData = await response.json();
            console.log("Error response received: ", errorData);
            displayError(errorData.detail);  // Display the error message
        }
    } catch (error) {
        // Hide the loading spinner
        document.getElementById('spinner').style.display = 'none';

        console.log("Error occurred: ", error);

        // Clear the table and display error message
        document.getElementById('resultTable').querySelector('tbody').innerHTML = '';
        displayError('An unexpected error occurred. Please try again.');
    }
}

function displayError(message) {
    console.log("Displaying error: ", message);
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';  // Show the error message
}

function displaySchedule(responseData) { // responseData is the full ScheduleResponse object
    console.log("Displaying new grid schedule...");
    const { grid_schedule, max_sched_timeslot, max_sched_room, constraints_relaxed } = responseData;

    const resultTable = document.getElementById('resultTable');
    const tableHead = resultTable.querySelector('thead');
    const tableBody = resultTable.querySelector('tbody');

    // Clear existing table content
    tableHead.innerHTML = '';
    tableBody.innerHTML = '';

    if (max_sched_timeslot === 0 || max_sched_room === 0) {
        // Display a message if the schedule is empty or invalid
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.textContent = "No schedule data to display.";
        td.colSpan = 2; // Span TimeSlot and Seat Position
        tr.appendChild(td);
        tableBody.appendChild(tr);
        return;
    }

    // Generate Table Headers
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

    // Generate Table Body
    for (let ts = 1; ts <= max_sched_timeslot; ts++) {
        const ts_key = `ts_${ts}`;
        for (let seat_idx = 0; seat_idx < 3; seat_idx++) { // 0 for Seat 1, 1 for Seat 2, 2 for Seat 3
            const tr = document.createElement('tr');

            // TimeSlot cell (with rowspan for the first seat position row of a timeslot)
            if (seat_idx === 0) {
                const tdTimeSlot = document.createElement('td');
                tdTimeSlot.textContent = ts;
                tdTimeSlot.rowSpan = 3;
                tr.appendChild(tdTimeSlot);
            }

            // Seat Position cell
            const tdSeatPos = document.createElement('td');
            tdSeatPos.textContent = seat_idx + 1;
            tr.appendChild(tdSeatPos);

            // Room cells
            for (let r = 1; r <= max_sched_room; r++) {
                const room_key = `room_${r}`;
                const tdRoom = document.createElement('td');
                let teamId = "---"; // Default for empty slot

                if (grid_schedule && grid_schedule[ts_key] && grid_schedule[ts_key][room_key]) {
                    const teamsInMatch = grid_schedule[ts_key][room_key]; // Should be [T_seat1, T_seat2, T_seat3]
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

    // If any constraints were relaxed, display the information
    if (constraints_relaxed && constraints_relaxed.length > 0) {
        const relaxedMessage = `The following constraints were relaxed: ${constraints_relaxed.join(', ')}.`;
        displayError(relaxedMessage);
    } else {
        // Hide error message if schedule is successfully loaded and no constraints relaxed
        document.getElementById('error-message').style.display = 'none';
    }
}
