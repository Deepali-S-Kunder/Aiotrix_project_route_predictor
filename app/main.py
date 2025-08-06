import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import asyncio
import random
import time
import json 
from collections import deque

# --- Global variables for loaded models and encoders ---
rf_optimal_route_model = None
optimal_route_scaler = None
optimal_route_label_encoders = None
traffic_state_label_encoder = None

# --- NEW GLOBAL VARIABLES ---
last_predicted_route = "Initializing..." # Initial state for display
initial_prediction_made = False # Flag to ensure initial prediction happens only once

# --- Paths to your models and encoders ---
# These paths are now correct for main.py being at the project root
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "rf_optimal_route_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "optimal_route_scaler.joblib")
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, "optimal_route_label_encoders.joblib")
LE_TRAFFIC_STATE_PATH = os.path.join(MODELS_DIR, "le_traffic_state.joblib")

# --- Attempt to load models and encoders once at startup ---
def load_resources():
    global rf_optimal_route_model, optimal_route_scaler, optimal_route_label_encoders, traffic_state_label_encoder
    print(f"Attempting to load resources from {os.path.abspath(MODELS_DIR)}")
    try:
        if os.path.exists(MODEL_PATH):
            rf_optimal_route_model = joblib.load(MODEL_PATH)
            print(f"Loaded ML model from: {MODEL_PATH}")
        else:
            print(f"ML model not found at: {MODEL_PATH}")

        if os.path.exists(SCALER_PATH):
            optimal_route_scaler = joblib.load(SCALER_PATH)
            print(f"Loaded scaler from: {SCALER_PATH}")
        else:
            print(f"Scaler not found at: {SCALER_PATH}")

        if os.path.exists(LABEL_ENCODERS_PATH):
            optimal_route_label_encoders = joblib.load(LABEL_ENCODERS_PATH)
            print(f"Loaded label encoders from: {LABEL_ENCODERS_PATH}")
            # ADDED: Print keys of the loaded label encoders for debugging
            if optimal_route_label_encoders is not None and isinstance(optimal_route_label_encoders, dict):
                print(f"Keys in optimal_route_label_encoders (after loading): {optimal_route_label_encoders.keys()}")
            else:
                print("optimal_route_label_encoders is None or not a dictionary after loading.")
        else:
            print(f"Label encoders not found at: {LABEL_ENCODERS_PATH}")
            
        if os.path.exists(LE_TRAFFIC_STATE_PATH):
            traffic_state_label_encoder = joblib.load(LE_TRAFFIC_STATE_PATH)
            print(f"Loaded traffic state label encoder from: {LE_TRAFFIC_STATE_PATH}")
        else:
            print(f"Traffic state label encoder not found at: {LE_TRAFFIC_STATE_PATH}")

    except Exception as e:
        print(f"Error loading resources: {e}")
        # Optionally, re-raise or handle more gracefully if resources are critical

# Load resources when the script starts
load_resources()

# --- FastAPI App Setup ---
app = FastAPI()

# CORRECTED PATHS FOR STATIC AND TEMPLATES - NOW RELATIVE TO MAIN.PY AT PROJECT ROOT
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# --- TrafficLane and TrafficManager Classes ---
class TrafficLane:
    def __init__(self, name, max_capacity):
        self.name = name
        self.max_capacity = max_capacity
        self.current_load = 0
        self.message_queue = deque() # Stores (message_id, arrival_time)
        self.avg_latency_ms = 0.0
        self.utilization_percent = 0.0
        self.is_overloaded = False
        self.predicted_latency_based_on_load = 0.0

    def add_message(self, message_id):
        self.message_queue.append((message_id, time.time()))
        self.current_load = len(self.message_queue)
        self.update_stats()

    def process_message(self):
        if self.message_queue:
            message_id, arrival_time = self.message_queue.popleft()
            processing_time = random.uniform(50, 200) # Simulate processing time
            latency = (time.time() - arrival_time) * 1000 + processing_time
            self.current_load = len(self.message_queue)
            self.update_stats()
            return message_id, latency
        return None, None

    def update_stats(self):
        self.utilization_percent = (self.current_load / self.max_capacity) * 100
        self.is_overloaded = self.current_load > self.max_capacity # True if load exceeds capacity

        # Simple moving average for latency (placeholder, can be improved)
        if self.current_load > 0:
            # Estimate latency based on current load, higher load means higher latency
            # This is a simple model; a real system would have more complex calculations or ML
            self.predicted_latency_based_on_load = 50 + (self.current_load / self.max_capacity) * 150
        else:
            self.predicted_latency_based_on_load = 50 # Base latency

        # If no messages processed yet, use predicted latency for display
        if self.avg_latency_ms == 0 and self.predicted_latency_based_on_load > 0:
             self.avg_latency_ms = self.predicted_latency_based_on_load
        elif self.predicted_latency_based_on_load > 0:
            # Smooth the displayed average latency
            self.avg_latency_ms = (self.avg_latency_ms * 0.9) + (self.predicted_latency_based_on_load * 0.1)


class TrafficManager:
    def __init__(self):
        self.lanes = {
            "Route_A": TrafficLane("Route A", max_capacity=20),
            "Route_B": TrafficLane("Route B", max_capacity=15),
            "Route_C": TrafficLane("Route C", max_capacity=30),
        }
        self.congestion_event = None # {'active': True/False, 'route': 'Route_X', 'start_time': time, 'duration': seconds}
        self.input_data_history = deque(maxlen=10) # To store recent input data for ML
        self.messages_processed_count = 0
        self.dropped_messages_count = 0 # Initialize dropped messages count

    def get_lane_statuses(self):
        statuses = []
        for name, lane in self.lanes.items():
            lane.update_stats() # Ensure stats are fresh
            statuses.append({
                "name": lane.name,
                "current_load": lane.current_load,
                "max_capacity": lane.max_capacity,
                "utilization_percent": lane.utilization_percent,
                "avg_latency_ms": lane.avg_latency_ms,
                "is_overloaded": lane.is_overloaded,
                "predicted_latency_based_on_load": lane.predicted_latency_based_on_load
            })
        return statuses
    
    def induce_congestion(self, route_name, duration_seconds):
        if route_name in self.lanes:
            print(f"Inducing congestion on {route_name} for {duration_seconds} seconds.")
            self.congestion_event = {
                'active': True,
                'route': route_name,
                'start_time': time.time(),
                'duration': duration_seconds
            }
            # Temporarily reduce capacity or increase load to simulate congestion
            self.lanes[route_name].max_capacity = 5 # Reduce capacity significantly
            # Or add some dummy messages to increase load quickly
            for _ in range(self.lanes[route_name].max_capacity + 5): # make it overloaded
                self.lanes[route_name].add_message(f"CONGEST_MSG_{_}")
        
    def clear_congestion(self):
        current_congestion_event = self.congestion_event # Capture it locally for safety
        if current_congestion_event and current_congestion_event.get('active'):
            route_to_clear = current_congestion_event.get('route')
            if route_to_clear and route_to_clear in self.lanes: # Ensure route_to_clear is not None and valid
                print(f"Clearing congestion on {route_to_clear}.")
                self.lanes[route_to_clear].max_capacity = {
                    "Route_A": 20, "Route_B": 15, "Route_C": 30
                }[route_to_clear] # Restore original capacity
                # Optionally clear messages added due to congestion, or let them process
                self.lanes[route_to_clear].message_queue.clear() 
                self.lanes[route_to_clear].current_load = 0
                self.lanes[route_to_clear].update_stats()
            else:
                print(f"Warning: Attempted to clear congestion on invalid/unknown route or missing 'route' key: {route_to_clear}")
            self.congestion_event = None # Reset after processing


traffic_manager = TrafficManager()

# --- Helper function for ML Prediction ---
def predict_traffic_and_route(current_input_data):
    global rf_optimal_route_model, optimal_route_scaler, optimal_route_label_encoders, traffic_state_label_encoder, last_predicted_route, initial_prediction_made

    if not rf_optimal_route_model or not optimal_route_scaler or not optimal_route_label_encoders or not traffic_state_label_encoder:
        print("ML models or encoders not loaded. Cannot make predictions.")
        return "UNKNOWN", "DROPPED" # Default if models are not ready

    # --- DEBUGGING: Check state of optimal_route_label_encoders here ---
    #print(f"DEBUG inside predict_traffic_and_route: optimal_route_label_encoders type: {type(optimal_route_label_encoders)}")


    # Create a DataFrame for prediction
    try:
        input_df = pd.DataFrame([current_input_data])

        # Apply label encoding for categorical features
        # Note: 'Traffic state' and 'Message type' are part of optimal_route_label_encoders
        categorical_cols = [col for col in optimal_route_label_encoders.keys() if col in input_df.columns]
        for column in categorical_cols:
            le = optimal_route_label_encoders[column]
            if column in input_df.columns:
                # Handle unseen labels by transforming them to a default/first seen category (-1 or 0)
                # It's crucial that your model training handled these as well, e.g., by imputing
                input_df[column] = input_df[column].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else (0 if len(le.classes_) > 0 else -1))


        # Scale numerical features
        numerical_cols = [col for col in optimal_route_scaler.feature_names_in_ if col in input_df.columns]
        if numerical_cols: # Ensure there are numerical columns to scale
            input_df[numerical_cols] = optimal_route_scaler.transform(input_df[numerical_cols])
        
        # Predict optimal route
        # Ensure the DataFrame has all features expected by the model in the correct order
        # rf_optimal_route_model.feature_names_in_ contains the exact feature names the model was trained on
        missing_features = set(rf_optimal_route_model.feature_names_in_) - set(input_df.columns)
        if missing_features:
            print(f"Error: Missing features for ML prediction: {missing_features}")
            return "UNKNOWN", "DROPPED"
        
        # Ensure the order of columns matches the model's expected order
        input_df = input_df[rf_optimal_route_model.feature_names_in_]

        predicted_route_encoded = rf_optimal_route_model.predict(input_df)[0]
        
        # Decode the predicted route - Using the correct key 'Optimal Route'
        route_le = None
        if 'Optimal Route' in optimal_route_label_encoders: # Corrected key based on latest terminal output
            route_le = optimal_route_label_encoders['Optimal Route']
        elif 'Route' in optimal_route_label_encoders: # Fallback to previously found key
            route_le = optimal_route_label_encoders['Route']
        elif 'predicted_route' in optimal_route_label_encoders: # Further fallback
            route_le = optimal_route_label_encoders['predicted_route']
        elif 'optimal_route' in optimal_route_label_encoders: # Even further fallback
            route_le = optimal_route_label_encoders['optimal_route']
        
        if route_le is None:
            print("Error: Label encoder for the predicted route ('Optimal Route' or alternatives) not found in optimal_route_label_encoders. Please ensure your optimal_route_label_encoders.joblib file contains the correct key for the route.")
            return "UNKNOWN", "DROPPED" # Cannot decode route without its label encoder

        predicted_route = route_le.inverse_transform([predicted_route_encoded])[0]

        # --- FIX for "Encoded traffic state '0' not in traffic state label encoder classes" warning ---
        # The `predicted_traffic_state` should just be the original input 'Traffic state' string,
        # as the ML model itself only predicts the optimal route.
        traffic_state_str = current_input_data["Traffic state"] # Use the original string value
        
        last_predicted_route = predicted_route # Update global state
        initial_prediction_made = True
        return traffic_state_str, predicted_route # Return original traffic state string and decoded route

    except Exception as e:
        print(f"Error during ML prediction: {e}")
        return "UNKNOWN", "DROPPED" # Return default values in case of error

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected.")
    
    global last_predicted_route, initial_prediction_made

    try:
        while True:
            # 1. Simulate Incoming Message Data
            message_id = f"MSG-{random.randint(10000, 99999)}"
            message_type_options = ["text", "image", "video", "audio"]
            message_type = random.choice(message_type_options)
            
            # For display purposes (e.g., "Video Call", "Image Share")
            message_type_display = {
                "text": "Text Message",
                "image": "Image Share",
                "video": "Video Call",
                "audio": "Audio Call"
            }.get(message_type, "Unknown")

            # 2. Generate Realistic Input Features for ML Model
            current_input_data = {
                "Active users": random.randint(10000, 100000),
                "New users": random.randint(0, 50),
                "Message rate": round(random.uniform(1000, 10000), 2),
                "Media sharing": round(random.uniform(0, 1), 2), # Proportion of media messages
                "Spam ratio": round(random.uniform(0, 0.1), 2),
                "User sentiment": random.choice(["positive", "neutral", "negative"]),
                "Server load": random.randint(10, 90), # Percentage
                "Time of day": random.choice(["Morning", "Afternoon", "Evening", "Night"]),
                "Latency": random.randint(50, 300), # ms
                "Bandwidth usage": round(random.uniform(100, 1000), 2), # Mbps
                "Traffic state": random.choice(["Low", "Medium", "High"]), # Current perceived traffic state
                "Message type": message_type # The actual message type for ML
            }
            
            # --- CRITICAL FIX: Add dynamic lane statuses to ML input data ---
            lane_statuses = traffic_manager.get_lane_statuses()
            for status in lane_statuses:
                route_name_prefix = status['name'].replace(' ', '_') # e.g., 'Route_A'
                current_input_data[f'{route_name_prefix}_current_load'] = status['current_load']
                current_input_data[f'{route_name_prefix}_utilization_percent'] = status['utilization_percent']
                current_input_data[f'{route_name_prefix}_avg_latency_ms'] = status['avg_latency_ms']
                current_input_data[f'{route_name_prefix}_is_overloaded'] = int(status['is_overloaded']) # Convert boolean to 0 or 1

            traffic_manager.input_data_history.append(current_input_data)
            
            # 3. Make Prediction using ML Model
            predicted_traffic_state, predicted_route = "UNKNOWN", "DROPPED"
            if rf_optimal_route_model: # Only predict if model is loaded
                predicted_traffic_state, predicted_route = predict_traffic_and_route(current_input_data)
            
            # --- ADD THIS LOGIC FOR DROPPED MESSAGES ---
            if predicted_traffic_state == "High" or predicted_traffic_state == "Congested": # Added "Congested" for consistency
                # Simulate a chance of dropping messages when traffic is high or congested
                if random.random() < 0.15: # 15% chance to drop a message in high traffic
                    traffic_manager.dropped_messages_count += 1
                    print(f"!!! Message dropped due to high traffic! Total dropped: {traffic_manager.dropped_messages_count}")
                    predicted_route = "DROPPED_BY_CONGESTION" # Indicate explicitly that it was dropped due to congestion
            # --- END OF DROPPED MESSAGES LOGIC ---

            # If the predicted route is 'DROPPED', don't assign it to a lane
            if predicted_route != "DROPPED" and predicted_route != "DROPPED_BY_CONGESTION":
                # --- FIX for "Predicted route 'Route C' not found in traffic manager lanes" warning ---
                # Convert predicted_route to match the lane keys (e.g., "Route A" -> "Route_A")
                lane_key = predicted_route.replace(" ", "_") 
                lane_to_route_message = traffic_manager.lanes.get(lane_key) # Use the converted key
                if lane_to_route_message:
                    lane_to_route_message.add_message(message_id)
                    processed_message_id, processed_latency = lane_to_route_message.process_message()
                    # You might want to use processed_latency instead of current_input_data["Latency"] for display
                    current_input_data["Latency"] = processed_latency if processed_latency is not None else current_input_data["Latency"]
                else:
                    print(f"Warning: Predicted route '{predicted_route}' (converted to '{lane_key}') not found in traffic manager lanes.")
                    predicted_route = "DROPPED" # Fallback if route invalid
            
            # 4. Get Current Lane Statuses
            lane_statuses_for_frontend = traffic_manager.get_lane_statuses()

            # 5. Prepare Data for Frontend (INCLUDING NESTED ML_INPUT)
            data_to_send = {
                "predicted_traffic_state": predicted_traffic_state,
                "predicted_route": predicted_route,
                "lane_statuses": lane_statuses_for_frontend,
                "message_id": message_id,
                "message_type_display": message_type_display,
                "dropped_messages_count": traffic_manager.dropped_messages_count, # Include dropped count
                # --- ADD THIS LINE FOR DROPPED MESSAGES ---
                "dropped_messages": traffic_manager.dropped_messages_count,
                # --- END OF DROPPED MESSAGES LINE ---
                # --- This ensures ml_input is nested as script.js expects ---
                "ml_input": {
                    "Active users": current_input_data["Active users"],
                    "New users": current_input_data["New users"],
                    "Message rate": current_input_data["Message rate"],
                    "Media sharing": current_input_data["Media sharing"],
                    "Spam ratio": current_input_data["Spam ratio"],
                    "User sentiment": current_input_data["User sentiment"],
                    "Server load": current_input_data["Server load"],
                    "Time of day": current_input_data["Time of day"],
                    "Latency": current_input_data["Latency"],
                    "Bandwidth usage": current_input_data["Bandwidth usage"]
                }
            }
            
            # print(f"DEBUG - Data being sent to frontend: {json.dumps(data_to_send, indent=2)}") # Uncomment for full debug

            await websocket.send_json(data_to_send)

            # Check for congestion events
            if traffic_manager.congestion_event and traffic_manager.congestion_event['active']:
                if time.time() - traffic_manager.congestion_event['start_time'] > traffic_manager.congestion_event['duration']:
                    traffic_manager.clear_congestion()
            else:
                if random.random() < 0.05: # Increased probability to 5%
                    traffic_manager.induce_congestion(random.choice(["Route_A", "Route_B", "Route_C"]), duration_seconds=random.randint(15,30))

            await asyncio.sleep(1) # Send updates every 1 second
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # This finally block ensures congestion is cleared if the WebSocket closes unexpectedly
        # The clear_congestion function has been made more robust
        traffic_manager.clear_congestion()


# --- HTML Template Route ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})