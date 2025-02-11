import asyncio
import csv
import numpy as np
import time
import torch
import torch.nn as nn
from asyncua import Client
from collections import deque
from datetime import datetime

# OPC UA server connection details
url = "opc.tcp://192.168.0.20:4840"

# Get the current date in YYYYMMDD format for the file name
date_prefix = datetime.now().strftime("%Y%m%d")
output_file = f"../data/{date_prefix}_polymer.csv"

# Load the trained LSTM model
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)  # 30% dropout rate

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.dense(lstm_out[:, -1, :])
        return output

# Load the trained LSTM model
model = LSTMAnomalyDetector(input_dim=4, hidden_dim=64, output_dim=4)
model.load_state_dict(torch.load("../trained_model/lstm_polymer_anomaly.pth", weights_only=True))
model.eval()  # Set to evaluation mode

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

# Sliding window buffer for sequential LSTM inference
sequence_window = deque(maxlen=60)

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

            while True:
                # Create tasks to read variables concurrently
                tasks = [read_variable(client, node_id) for _, node_id in variable_nodes]
                results = await asyncio.gather(*tasks)

                # Process the results
                row = []
                timestamp = None
                for (name, _), (extracted_value, source_timestamp) in zip(variable_nodes, results):
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
                    if timestamp else "N/A"
                )

                # Extract the seconds part of the current timestamp
                current_second = datetime.strptime(formatted_timestamp.split('.')[0], "%Y%m%dT%H:%M:%S")

                # Skip if the current second matches the previous second
                if current_second == previous_second:
                    print("Skipping row: Current second matches the previous second.")
                    continue

                previous_second = current_second
                row.insert(0, formatted_timestamp)
                writer.writerow(row)
                print(f"Saved row: {row}")

                # Extract features for LSTM model
                new_data = np.array(row[1:], dtype=float)[[4, 5, 8, 9]]  # Select key process variables
                sequence_window.append(new_data)

                # Run LSTM inference when enough timesteps are collected (60-sequence window)
                if len(sequence_window) == 60:
                    input_seq = torch.tensor(np.array(sequence_window), dtype=torch.float32).unsqueeze(0)  # Shape: (1, 60, 4)
                    
                    model.eval()  # Ensure model is in evaluation mode
                    with torch.no_grad():
                        reconstructed = model(input_seq)  # Get model's reconstructed output
                        reconstruction_error = torch.mean(torch.abs(input_seq[:, -1, :] - reconstructed), axis=1).item()  # Compute error

                    # Define anomaly classification thresholds
                    slight_abnormal_threshold = 0.1609
                    heavy_abnormal_threshold = 0.2111

                    # Classify anomaly based on reconstruction error
                    if reconstruction_error >= heavy_abnormal_threshold:
                        anomaly_status = "Heavy Anomaly"
                    elif slight_abnormal_threshold <= reconstruction_error < heavy_abnormal_threshold:
                        anomaly_status = "Slight Anomaly"
                    else:
                        anomaly_status = "Normal"

                    print(f"Reconstruction Error: {reconstruction_error:.4f}, Prediction: {anomaly_status}")


                # Flush the file to ensure data is saved
                csvfile.flush()

                # Wait 1 second before reading again
                elapsed_time = (1000 - int(formatted_timestamp.split('.')[-1])) / 1000
                await asyncio.sleep(max(0, elapsed_time))

if __name__ == "__main__":
    asyncio.run(main())
