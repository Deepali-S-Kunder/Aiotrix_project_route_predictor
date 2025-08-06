document.addEventListener("DOMContentLoaded", () => {
    const startBtn = document.getElementById("start");
    const mlInputData = document.getElementById("mlInputData");
    const overallPrediction = document.getElementById("overallPrediction");
    const incomingMessage = document.getElementById("incomingMessage");
    const routingDecisionSpan = document.getElementById("routingDecision");
    const laneStatusCardsContainer = document.getElementById("laneStatusCardsContainer");
    const messageProcessingResult = document.getElementById("messageProcessingResult");

    // Simulation elements
    const simulationBox = document.getElementById('simulationBox');
    const route1 = document.getElementById('route1');
    const route2 = document.getElementById('route2');
    const route3 = document.getElementById('route3');
    const signalDot = document.getElementById('signalDot');

    // THIS IS CRUCIAL: Map backend lane names to frontend route IDs
    // Ensure these keys EXACTLY match the lane names from your main.py (e.g., "Route_A" from backend)
    const laneToRouteIdMap = {
        "Route A": "route1",
        "Route B": "route2",
        "Route C": "route3"
    };

    let websocket;

    startBtn.addEventListener("click", () => {
        // Close existing connection if any before opening a new one
        if (websocket && (websocket.readyState === WebSocket.OPEN || websocket.readyState === WebSocket.CONNECTING)) {
            websocket.close();
        }

        startBtn.disabled = true;
        startBtn.textContent = "Streaming Live...";

        // Correct WebSocket URL
        websocket = new WebSocket(`ws://${location.host}/ws`);

        websocket.onopen = () => {
            console.log("WebSocket connection established.");
            mlInputData.innerHTML = "";
            overallPrediction.textContent = "Connecting...";
            incomingMessage.innerHTML = "";
            routingDecisionSpan.textContent = "";
            laneStatusCardsContainer.innerHTML = "";
            messageProcessingResult.innerHTML = "";
            signalDot.style.display = "none"; // Hide dot initially
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("Received data (script.js):", data); // Log the full received data

            // Display ML Input Features (fix for this issue)
            mlInputData.innerHTML = ""; // Clear previous data
            if (data.ml_input) { // Check if ml_input object exists
                for (const [key, value] of Object.entries(data.ml_input)) {
                    const li = document.createElement("li");
                    // Format numbers to 2 decimal places if they are numbers
                    li.textContent = `${key}: ${typeof value === "number" ? value.toFixed(2) : value}`;
                    mlInputData.appendChild(li);
                }
            } else {
                mlInputData.innerHTML = "<li>No ML input data received.</li>";
            }


            // Overall prediction
            overallPrediction.textContent = data.predicted_traffic_state;

            // Incoming message
            incomingMessage.innerHTML = `
                <li>ID: ${data.message_id || 'N/A'}</li>
                <li>Type: ${data.message_type_display || 'N/A'}</li>
            `;

            // Routing decision
            routingDecisionSpan.textContent = data.predicted_route;
            routingDecisionSpan.style.color = data.predicted_route === "DROPPED" ? 'red' : 'lime';
            // --- ADD THIS SECTION FOR DROPPED MESSAGES ---
    if (data.dropped_messages !== undefined) {
        document.getElementById('dropped-messages-count').textContent = Number(data.dropped_messages).toFixed(0);
    } else {
        // Optional: Set to 'N/A' or maintain last value if data is missing
        // document.getElementById('dropped-messages-count').textContent = 'N/A';
    }
            // Message processing result
            // This part assumes data.Latency and data['Server load'] are direct properties,
            // or you might want to adjust if they come from a different nested object.
            messageProcessingResult.innerHTML = `
                <li>Route: ${data.predicted_route || 'N/A'}</li>
                <li>Latency: ${data.Latency !== undefined ? data.Latency.toFixed(2) + 'ms' : 'N/A'}</li>
                <li>Load: ${data['Server load'] !== undefined ? data['Server load'] : 'N/A'}</li>
            `;

            // Lane statuses
            laneStatusCardsContainer.innerHTML = "";
            // First, reset all route visuals (remove optimal and congested classes)
            [route1, route2, route3].forEach(routeEl => {
                routeEl.classList.remove("congested", "optimal");
            });

            data.lane_statuses.forEach(lane => {
                const card = document.createElement("div");
                card.className = "lane-status-card";

                let statusColor = 'green';
                if (lane.utilization_percent > 80) statusColor = 'orange';
                if (lane.is_overloaded) statusColor = 'red';

                card.innerHTML = `
                    <h5>${lane.name}</h5>
                    <ul>
                        <li>Current Load: <span style="color: ${statusColor};">${lane.current_load} / ${lane.max_capacity}</span></li>
                        <li>Utilization: <span style="color: ${statusColor};">${lane.utilization_percent.toFixed(2)}%</span></li>
                        <li>Avg Latency: ${lane.avg_latency_ms.toFixed(2)}ms</li>
                        <li>Predicted Latency: ${lane.predicted_latency_based_on_load !== undefined ? lane.predicted_latency_based_on_load.toFixed(2) + 'ms' : 'N/A'}</li>
                    </ul>
                `;
                laneStatusCardsContainer.appendChild(card);
            });

            // Trigger single-shot animation for the routed message
            if (data.predicted_route && data.predicted_route !== "DROPPED") {
                const chosenLaneName = data.predicted_route;
                const chosenRouteId = laneToRouteIdMap[chosenLaneName];
                const latencyForAnimation = data.Latency || 200; // Use data.Latency from ML input for animation speed, or default

                if (chosenRouteId) {
                    const chosenRouteEl = document.getElementById(chosenRouteId);
                    if (chosenRouteEl) {
                        chosenRouteEl.classList.add("optimal"); // Highlight the chosen route
                        animateDotAlongRoute(chosenRouteEl, latencyForAnimation);
                    } else {
                         console.error(`Route element with ID "${chosenRouteId}" not found in HTML.`);
                    }
                } else {
                    console.error(`No route mapping found for lane: "${chosenLaneName}". Check laneToRouteIdMap.`);
                }
            } else {
                signalDot.style.display = "none"; // Hide dot if message dropped
            }
        };

        websocket.onclose = () => {
            console.log("WebSocket connection closed.");
            startBtn.disabled = false;
            startBtn.textContent = "Start Streaming";
            signalDot.style.display = "none"; // Ensure dot is hidden on close
        };

        websocket.onerror = (err) => {
            console.error("WebSocket error:", err);
            startBtn.disabled = false;
            startBtn.textContent = "Start Streaming (Error)";
            signalDot.style.display = "none"; // Ensure dot is hidden on error
        };
    });

    // Function for single-shot animation along a specific route
    function animateDotAlongRoute(routeEl, latencyMs) {
        signalDot.style.display = "block"; // Make the dot visible

        const boxRect = simulationBox.getBoundingClientRect();
        const routeRect = routeEl.getBoundingClientRect();

        signalDot.style.left = `${routeRect.left - boxRect.left - signalDot.offsetWidth / 2}px`;
        signalDot.style.top = `${routeRect.top - boxRect.top + routeRect.height / 2 - signalDot.offsetHeight / 2}px`;

        signalDot.style.transition = 'none';
        signalDot.style.transform = 'translateX(0px)';
        void signalDot.offsetWidth;

        const distanceToTravel = routeRect.width;

        const animationSpeedFactor = 3;
        const minAnimationDurationMs = 500;

        let effectiveLatencyMs = latencyMs * animationSpeedFactor;
        effectiveLatencyMs = Math.max(effectiveLatencyMs, minAnimationDurationMs);

        const animationDurationSeconds = effectiveLatencyMs / 1000;

        signalDot.style.transition = `transform ${animationDurationSeconds}s linear`;
        signalDot.style.transform = `translateX(${distanceToTravel}px)`;

        setTimeout(() => {
            signalDot.style.display = "none";
            signalDot.style.transform = 'translateX(0px)';
            routeEl.classList.remove("optimal");
        }, effectiveLatencyMs);
    }
});