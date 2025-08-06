import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib # For saving and loading models

def train_and_predict_telecom_traffic(data_filepath="telecom_traffic_data.csv"):
    """
    Loads the telecom traffic dataset, preprocesses it, trains two Random Forest
    models (one for Traffic State, one for Optimal Route), and evaluates them.

    Args:
        data_filepath (str): The path to the CSV file containing the dataset.
    """

    print(f"Loading dataset from: {data_filepath}")
    try:
        df = pd.read_csv(data_filepath)
        print("Dataset loaded successfully.")
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\nDataset Info:")
        df.info()
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {data_filepath}")
        print("Please ensure 'telecom_traffic_data.csv' is in the same directory or provide the correct path.")
        return

    print("\n--- Preprocessing Data ---")

    # Identify features (X) and target variables (y)
    # Features are all columns except the target variables
    features = df.drop(columns=["Traffic state", "Optimal Route"])

    # Target variables
    target_traffic_state = df["Traffic state"]
    target_optimal_route = df["Optimal Route"]

    # --- Feature Engineering / Encoding Categorical Features ---
    # Apply One-Hot Encoding to categorical features in X
    # 'Time of day', 'User sentiment' are categorical features
    # (Other features like active_users, message_rate etc. are numerical and don't need encoding)

    # Let's verify which columns are indeed categorical and need encoding
    categorical_cols = features.select_dtypes(include=['object']).columns
    print(f"Categorical columns to be One-Hot Encoded: {list(categorical_cols)}")

    X_encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=True) # drop_first avoids multicollinearity

    print("\nFeatures after One-Hot Encoding (X_encoded.head()):")
    print(X_encoded.head())
    print(f"Shape of X_encoded: {X_encoded.shape}")


    # --- Label Encoding for Target Variables (y) ---
    # Random Forest Classifier works best with numerical labels
    print("\n--- Encoding Target Variables ---")
    
    label_encoder_traffic = LabelEncoder()
    y_traffic_encoded = label_encoder_traffic.fit_transform(target_traffic_state)
    print(f"Original Traffic States: {label_encoder_traffic.classes_}")
    print(f"Encoded Traffic States (first 5): {y_traffic_encoded[:5]}")

    label_encoder_route = LabelEncoder()
    y_route_encoded = label_encoder_route.fit_transform(target_optimal_route)
    print(f"Original Optimal Routes: {label_encoder_route.classes_}")
    print(f"Encoded Optimal Routes (first 5): {y_route_encoded[:5]}")


    # --- Model Training for Traffic State Prediction ---
    print("\n--- Training Model for Traffic State Prediction ---")
    X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(
        X_encoded, y_traffic_encoded, test_size=0.2, random_state=42, stratify=y_traffic_encoded
    )

    rf_model_traffic = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model_traffic.fit(X_train_ts, y_train_ts)

    print("\nEvaluating Traffic State Prediction Model:")
    y_pred_ts = rf_model_traffic.predict(X_test_ts)
    print(f"Accuracy: {accuracy_score(y_test_ts, y_pred_ts):.4f}")
    print("Classification Report:")
    print(classification_report(y_test_ts, y_pred_ts, target_names=label_encoder_traffic.classes_))

    # Save the Traffic State model and its LabelEncoder
    joblib.dump(rf_model_traffic, 'rf_traffic_state_model.joblib')
    joblib.dump(label_encoder_traffic, 'le_traffic_state.joblib')
    print("Traffic State Model and LabelEncoder saved as 'rf_traffic_state_model.joblib' and 'le_traffic_state.joblib'")


    # --- Model Training for Optimal Route Prediction ---
    print("\n--- Training Model for Optimal Route Prediction ---")
    X_train_or, X_test_or, y_train_or, y_test_or = train_test_split(
        X_encoded, y_route_encoded, test_size=0.2, random_state=42, stratify=y_route_encoded
    )

    rf_model_route = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model_route.fit(X_train_or, y_train_or)

    print("\nEvaluating Optimal Route Prediction Model:")
    y_pred_or = rf_model_route.predict(X_test_or)
    print(f"Accuracy: {accuracy_score(y_test_or, y_pred_or):.4f}")
    print("Classification Report:")
    print(classification_report(y_test_or, y_pred_or, target_names=label_encoder_route.classes_))

    # Save the Optimal Route model and its LabelEncoder
    joblib.dump(rf_model_route, 'rf_optimal_route_model.joblib')
    joblib.dump(label_encoder_route, 'le_optimal_route.joblib')
    print("Optimal Route Model and LabelEncoder saved as 'rf_optimal_route_model.joblib' and 'le_optimal_route.joblib'")

    # Return trained models and encoders for potential future use (e.g., real-time prediction)
    return rf_model_traffic, label_encoder_traffic, rf_model_route, label_encoder_route, X_train_ts.columns


# --- How to use the trained models for a new prediction ---
def predict_new_data_point(
    new_data,
    traffic_model_path='rf_traffic_state_model.joblib',
    le_traffic_path='le_traffic_state.joblib',
    route_model_path='rf_optimal_route_model.joblib',
    le_route_path='le_optimal_route.joblib',
    feature_columns=None # Pass the feature columns from training
):
    """
    Loads the trained models and encoders, then makes predictions for a new data point.

    Args:
        new_data (dict): A dictionary representing a single new data point.
                         Keys should match the original feature column names.
        traffic_model_path (str): Path to the saved traffic state model.
        le_traffic_path (str): Path to the saved traffic state label encoder.
        route_model_path (str): Path to the saved optimal route model.
        le_route_path (str): Path to the saved optimal route label encoder.
        feature_columns (pd.Index): The feature columns (including one-hot encoded)
                                    that the models were trained on. Crucial for
                                    consistent input.

    Returns:
        tuple: Predicted traffic state (str), Predicted optimal route (str)
    """
    try:
        # Load models and encoders
        loaded_rf_traffic = joblib.load(traffic_model_path)
        loaded_le_traffic = joblib.load(le_traffic_path)
        loaded_rf_route = joblib.load(route_model_path)
        loaded_le_route = joblib.load(le_route_path)
        print("\nModels and LabelEncoders loaded successfully for prediction.")

        # Convert new_data to DataFrame
        new_df = pd.DataFrame([new_data])

        # Apply the same one-hot encoding as during training
        # IMPORTANT: Ensure consistency in columns
        # Get categorical columns from new_df (assuming same structure as training)
        categorical_cols_new = new_df.select_dtypes(include=['object']).columns
        
        # Use reindex to align columns with training data, filling missing with 0
        new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols_new, drop_first=True)
        
        # This step is CRUCIAL: Align new data's columns with the training data's columns
        # Any column in feature_columns not in new_df_encoded will be added as 0.
        # Any column in new_df_encoded not in feature_columns will be dropped.
        if feature_columns is not None:
             new_df_aligned = new_df_encoded.reindex(columns=feature_columns, fill_value=0)
        else:
             print("Warning: feature_columns not provided. Column alignment may be incorrect.")
             new_df_aligned = new_df_encoded # Proceed without explicit alignment if not provided

        # Predict Traffic State
        pred_traffic_encoded = loaded_rf_traffic.predict(new_df_aligned)
        predicted_traffic_state = loaded_le_traffic.inverse_transform(pred_traffic_encoded)[0]

        # Predict Optimal Route
        pred_route_encoded = loaded_rf_route.predict(new_df_aligned)
        predicted_optimal_route = loaded_le_route.inverse_transform(pred_route_encoded)[0]

        return predicted_traffic_state, predicted_optimal_route

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

if __name__ == "__main__":
    # First, train the models and get the feature columns for consistent prediction
    rf_traffic, le_traffic, rf_route, le_route, trained_feature_columns = train_and_predict_telecom_traffic()

    if trained_feature_columns is not None:
        print("\n--- Demonstrating Prediction for a New Data Point ---")
        # Example new data point (keys must match the original column names from the CSV)
        example_new_data = {
            "Active users": 15000,
            "New users": 150,
            "Message rate": 3000,
            "Media sharing": 0.35,
            "Spam ratio": 0.04,
            "User sentiment": "positive",
            "Server load": 60,
            "Time of day": "Afternoon",
            "Latency": 120,
            "Bandwidth usage": 500.25
        }

        # Predict using the loaded models and encoders
        pred_traffic, pred_route = predict_new_data_point(
            example_new_data,
            feature_columns=trained_feature_columns # Pass the feature columns from training
        )

        if pred_traffic and pred_route:
            print(f"\nFor the new data point:\n{example_new_data}")
            print(f"Predicted Traffic State: {pred_traffic}")
            print(f"Predicted Optimal Route: {pred_route}")

        # Another example: High server load, negative sentiment
        example_new_data_2 = {
            "Active users": 45000,
            "New users": 20,
            "Message rate": 8000,
            "Media sharing": 0.10,
            "Spam ratio": 0.18,
            "User sentiment": "negative",
            "Server load": 95,
            "Time of day": "Evening",
            "Latency": 800,
            "Bandwidth usage": 900.50
        }

        pred_traffic_2, pred_route_2 = predict_new_data_point(
            example_new_data_2,
            feature_columns=trained_feature_columns
        )

        if pred_traffic_2 and pred_route_2:
            print(f"\nFor a second new data point:\n{example_new_data_2}")
            print(f"Predicted Traffic State: {pred_traffic_2}")
            print(f"Predicted Optimal Route: {pred_route_2}")
    else:
        print("\nModel training failed, cannot perform prediction demonstration.")