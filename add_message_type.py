import pandas as pd
import numpy as np # For random choices

input_csv_path = "telecom_traffic_data.csv"
output_csv_path = "telecom_traffic_data.csv" # Overwrite the original, or save to a new name

print(f"Loading data from: {input_csv_path}")
try:
    df = pd.read_csv(input_csv_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: {input_csv_path} not found. Make sure it's in the same directory.")
    exit()

# Define possible message types
message_types = ['text', 'image', 'video', 'spam']
weights = [0.6, 0.2, 0.1, 0.1] # Assign probabilities

print("Adding 'Message type' column...")
# Assign random message types to the new column based on weights
df['Message type'] = np.random.choice(message_types, size=len(df), p=weights)

print("First 5 rows with new 'Message type' column:")
print(df.head())

print(f"Saving updated data to: {output_csv_path}")
df.to_csv(output_csv_path, index=False) # index=False prevents writing the DataFrame index as a column
print("Data saved successfully. 'Message type' column added.")