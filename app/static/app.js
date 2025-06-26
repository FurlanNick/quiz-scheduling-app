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

        // Clear previous error messages
        document.getElementById('error-message').textContent = '';
        // Note: Table clearing is now done at the beginning of displaySchedule

        // Check for a successful response
        if (response.ok) {
            const data = await response.json();
            // Robust log of the received data object
            console.log("Data received (in submitForm): ", JSON.parse(JSON.stringify(data)));

            if (data && typeof data === 'object') {
                displaySchedule(data); // Call with the full data object
            } else {
                console.error("submitForm: Data from response.json() is null, undefined, or not an object. Received:", data);
                displayError("Failed to parse or receive valid schedule data from server.");
                // Ensure table is cleared or shows error
                const resultTableEl = document.getElementById('resultTable');
                if (resultTableEl) {
                    const tableHead = resultTableEl.querySelector('thead');
                    const tableBody = resultTableEl.querySelector('tbody');
                    if (tableHead) tableHead.innerHTML = '';
                    if (tableBody) tableBody.innerHTML = '<tr><td colspan="2">Error: No valid data from server.</td></tr>';
                }
            }
        } else {
            const errorData = await response.json().catch(() => ({ detail: "Failed to parse error response from server." }));
            console.log("Error response received: ", errorData);
            displayError(errorData.detail || "An unknown error occurred on the server.");
             // Clear table on error too
            const resultTableEl = document.getElementById('resultTable');
            if (resultTableEl) {
                const tableHead = resultTableEl.querySelector('thead');
                const tableBody = resultTableEl.querySelector('tbody');
                if (tableHead) tableHead.innerHTML = '';
                if (tableBody) tableBody.innerHTML = '';
            }
        }
    } catch (error) {
        // Hide the loading spinner
        document.getElementById('spinner').style.display = 'none';

        console.log("Error occurred during fetch or processing: ", error);

        // Clear the table and display error message
        const resultTableEl = document.getElementById('resultTable');
        if (resultTableEl) {
            const tableHead = resultTableEl.querySelector('thead');
            const tableBody = resultTableEl.querySelector('tbody');
            if (tableHead) tableHead.innerHTML = '';
            if (tableBody) tableBody.innerHTML = '';
        }
        displayError('An unexpected error occurred. Please check console for details.');
    }
}

function displayError(message) {
    console.log("Displaying error: ", message);
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';  // Show the error message
}

function displaySchedule(responseData) {
    if (!responseData || typeof responseData !== 'object') {
        console.error("app.js displaySchedule - ERROR: responseData is null, undefined, or not an object. Received:", responseData);
        const resultTable = document.getElementById('resultTable');
        if (resultTable) {
            const tableBody = resultTable.querySelector('tbody');
            const tableHead = resultTable.querySelector('thead'); // Also clear head
            if (tableHead) tableHead.innerHTML = '';
            if (tableBody) {
                // Attempt to set colspan to a reasonable number if max_sched_room is unavailable
                let colSpan = 2; // Default for TimeSlot, Seat Position
                // Try to get max_sched_room if responseData is an object, otherwise use default
                if (responseData && typeof responseData.max_sched_room === 'number') {
                    colSpan += responseData.max_sched_room;
                }
                tableBody.innerHTML = `<tr><td colspan="${colSpan}">Error: Invalid data received by display function.</td></tr>`;
            }
        }
        return;
    }

    console.log("app.js displaySchedule - Entry. Data:", JSON.parse(JSON.stringify(responseData)));
    const { grid_schedule, max_sched_timeslot, max_sched_room, constraints_relaxed } = responseData;

    const resultTable = document.getElementById('resultTable');
    if (!resultTable) { console.error("app.js displaySchedule - ERROR: resultTable element not found!"); return; }
    const tableHead = resultTable.querySelector('thead');
    if (!tableHead) { console.error("app.js displaySchedule - ERROR: tableHead element not found!"); return; }
    const tableBody = resultTable.querySelector('tbody');
    if (!tableBody) { console.error("app.js displaySchedule - ERROR: tableBody element not found!"); return; }

    console.log("app.js displaySchedule - Clearing table head and body.");
    tableHead.innerHTML = '';
    tableBody.innerHTML = '';

    if (!grid_schedule || max_sched_timeslot === 0 || max_sched_room === 0) {
        console.log("app.js displaySchedule - No schedule data or zero dimensions. MaxTS:", max_sched_timeslot, "MaxRoom:", max_sched_room, "GridData:", grid_schedule);
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.textContent = "No schedule data to display.";
        // Calculate colspan: TimeSlot, Seat Position, plus one for each room. Default to 2 if max_sched_room is 0.
        td.colSpan = (max_sched_room || 0) + 2;
        tr.appendChild(td);
        tableBody.appendChild(tr);
        console.log("app.js displaySchedule - 'No data' message appended.");
        return;
    }

    // Generate Table Headers
    console.log("app.js displaySchedule - Generating headers. Max rooms:", max_sched_room);
    const headerRow = document.createElement('tr');

    const thTimeSlot = document.createElement('th');
    thTimeSlot.textContent = "TimeSlot";
    headerRow.appendChild(thTimeSlot);

    const thSeatPos = document.createElement('th');
    thSeatPos.textContent = "Seat Position";
    headerRow.appendChild(thSeatPos);
    console.log("app.js displaySchedule - Appended TimeSlot & SeatPos headers.");

    for (let r = 1; r <= max_sched_room; r++) {
        const thRoom = document.createElement('th');
        thRoom.textContent = `Room ${r}`;
        headerRow.appendChild(thRoom);
    }
    tableHead.appendChild(headerRow);
    console.log("app.js displaySchedule - All headers appended. Current tableHead HTML:", tableHead.innerHTML);

    // Generate Table Body
    console.log("app.js displaySchedule - Generating body. Max timeslots:", max_sched_timeslot);
    for (let ts = 1; ts <= max_sched_timeslot; ts++) {
        // console.log(`app.js displaySchedule - Processing timeslot: ${ts}`); // Can be too verbose
        const ts_key = `ts_${ts}`;
        for (let seat_idx = 0; seat_idx < 3; seat_idx++) { // 0 for Seat 1, 1 for Seat 2, 2 for Seat 3
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
            // console.log(`app.js displaySchedule - Appended row for TS: ${ts}, Seat: ${seat_idx + 1}`);
        }
    }
    console.log("app.js displaySchedule - All body rows appended. Current tableBody HTML (first 500 chars):", tableBody.innerHTML.substring(0, 500));

    if (constraints_relaxed && constraints_relaxed.length > 0) {
        const relaxedMessage = `The following constraints were relaxed: ${constraints_relaxed.join(', ')}.`;
        displayError(relaxedMessage);
    } else {
        document.getElementById('error-message').style.display = 'none';
    }
}
