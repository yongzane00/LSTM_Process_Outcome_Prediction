import asyncio
import csv
import numpy as np
import time
from asyncua import Client
from joblib import load
import warnings
from collections import deque
from datetime import datetime

# Suppress the specific UserWarning
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but EllipticEnvelope was fitted with feature names"
)

# OPC UA server connection details
url = "opc.tcp://192.168.0.20:4840"

# Get the current date in YYYYMMDD format and add the date prefix to the file name
date_prefix = datetime.now().strftime("%Y%m%d")
output_file = f"../data/{date_prefix}_glass.csv"

# Load the saved Elliptic Envelope model
elliptic_env = load("../trained_model/elliptic_env_glass.joblib")

# Define the variable nodes to read
variable_nodes = [
    ("ABB_X_Pos", "ns=4;s=ABB_TESTPOS_X"),
    ("ABB_Y_Pos", "ns=4;s=ABB_TESTPOS_Y"),
    ("ABB_Z_Pos", "ns=4;s=ABB_TESTPOS_Z"),
    ("Chamber_Temp", "ns=4;s=Heater_PowerSupply_HumidA"),
    ("Chamber_Pressure", "ns=4;s=MainGas_ChamberPressureA"),
    ("Main_Gas_Flow_Rate", "ns=4;s=MainGas_FlowRate"),
    ("PF1_Pressure", "ns=4;s=Powder_Feeder_Gas_Pressure"),
    ("PF1_Gas_Flow_Rate", "ns=4;s=Powder_Feeder_Gas_FlowRate"),
    ("PF2_Pressure", "ns=4;s=Powder_Feeder2_Gas_Pressure"),
    ("PF2_Gas_Flow_Rate", "ns=4;s=Powder_Feeder2_Gas_FlowRate"),
]

async def read_variable(client, node_id):
    """Asynchronous function to read a single variable."""
    try:
        node = client.get_node(node_id)
        value = await node.read_data_value()
        return value.Value.Value, value.SourceTimestamp
    except Exception as e:
        print(f"Error reading node {node_id}: {e}")
        return None, None

async def main():
    print(f"Connecting to {url} ...")
    async with Client(url=url) as client:
        # Open the CSV file in append mode
        with open(output_file, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write the header if the file is empty
            if csvfile.tell() == 0:
                header = ["TimeStamp"] + [name for name, _ in variable_nodes]
                writer.writerow(header)

            previous_second = None
            predictions_queue = deque(maxlen=50)

            while True:
                # Create tasks to read variables concurrently
                tasks = [read_variable(client, node_id) for _, node_id in variable_nodes]
                results = await asyncio.gather(*tasks)
                row = []
                timestamp = None
                for (name, _), (extracted_value, source_timestamp) in zip(variable_nodes, results):
                    # Handle specific processing for certain variables
                    if extracted_value is not None:
                        if name in ["Chamber_Pressure", "PF1_Pressure", "PF2_Pressure"]:
                            extracted_value = extracted_value / 1000
                        elif name in ["PF1_Gas_Flow_Rate", "PF2_Gas_Flow_Rate"]:
                            extracted_value = extracted_value / 10

                    # Capture the timestamp from the first successful read
                    if timestamp is None and source_timestamp is not None:
                        timestamp = source_timestamp

                    row.append(extracted_value)

                # Format and add the timestamp
                formatted_timestamp = (
                    timestamp.strftime("%Y%m%dT%H:%M:%S") + f".{int(timestamp.microsecond / 1000):03d}"
                    if timestamp
                    else "N/A"
                )
                current_second = datetime.strptime(formatted_timestamp.split('.')[0], "%Y%m%dT%H:%M:%S")

                # Skip if the current second matches the previous second
                if current_second == previous_second:
                    print("Skipping row: Current second matches the previous second.")
                    continue

                # Update the previous second
                previous_second = current_second
                row.insert(0, formatted_timestamp)

                # Write the row to the CSV
                writer.writerow(row)
                print(f"Saved row: {row}")

                # Predict anomalies using "Chamber_Pressure", "Main_Gas_Flow" and "PF1_Pressure"
                row = np.array(row[1:], dtype=float).reshape(1, -1)
                new_data = row[:, [4, 5, 6, 7]]
                dist = elliptic_env.decision_function(new_data)
                outlier_threshold = 0  # Adjust based on your data
                heavy_outlier = -5  # Adjust based on your data

                predictions = (
                    "Heavy Anomaly" if dist <= heavy_outlier
                    else "Slight Anomaly" if heavy_outlier < dist < outlier_threshold
                    else "Normal"
                )

                # Add the prediction to the deque
                predictions_queue.append(predictions)

                # Calculate the percentage of "Normal" predictions
                num_normal = predictions_queue.count("Normal")
                percentage_normal = (num_normal / 50) * 100 if len(predictions_queue) == 50 else 0
                print(f"Number of data in Queue: {len(predictions_queue)}")
                print(f"Decision Score: {dist[0]:.2f}, Prediction: {predictions}")
                print(f"Cold Spray Machine is {percentage_normal}% ready.")

                # Flush the file to ensure data is saved
                csvfile.flush()

                # Wait 1 second before reading again
                elapsed_time = (1000 - int(formatted_timestamp.split('.')[-1])) / 1000
                await asyncio.sleep(max(0, elapsed_time))

if __name__ == "__main__":
    asyncio.run(main())
