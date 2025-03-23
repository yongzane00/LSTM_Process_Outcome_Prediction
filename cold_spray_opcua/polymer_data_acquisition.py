import asyncio
import csv
import numpy as np
import time
from asyncua import Client, ua
from collections import deque
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch

# OPC UA server connection details
url = "opc.tcp://192.168.0.20:4840"

# Get the current date in YYYYMMDD format and add the date prefix to the file name
date_prefix = datetime.now().strftime("%Y%m%d")
output_file = f"../data/{date_prefix}_polymer.csv"

# Initialize LTSM Model
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)  # 30% dropout rate

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)  # Apply dropout to LSTM output
        output = self.dense(lstm_out[:, -1, :])
        return output
    
# Load model
input_dim = 4
hidden_dim = 64
output_dim = input_dim
model = LSTMAnomalyDetector(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("../trained_model/lstm_polymer_anomaly.pth"))
model.eval()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

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

write_nodes = {
    "AIO_Chamber_Pressure": "ns=4;s=AIO_Chamber_Pressure",
    "AIO_Main_Gas_Flow_Rate": "ns=4;s=AIO_Main_Gas_Flow_Rate",
    "AIO_PF1_Pressure": "ns=4;s=AIO_PF1_Pressure",
    "AIO_PF1_Gas_Flow_Rate": "ns=4;s=AIO_PF1_Gas_Flow_Rate",
    "AIO_PF2_Pressure": "ns=4;s=AIO_PF2_Pressure",
    "AIO_PF2_Gas_Flow_Rate": "ns=4;s=AIO_PF2_Gas_Flow_Rate",
    "AIO_Classification": "ns=4;s=AIO_Classification",
    "AIO_Prediction_Error": "ns=4;s=AIO_Prediction_Error",
}

async def read_variable(client, node_id):
    """Asynchronous function to read a single variable."""
    try:
        node = client.get_node(node_id)
        value = await node.read_data_value()
        return value.Value.Value, value.SourceTimestamp
    except Exception as e:
        print(f"Error reading node {node_id}: {e}")
        return None, None

async def write_variable(client, node_id, value, value_type):
    """Writes a value to an OPC UA node asynchronously with explicit type conversion."""
    try:
        node = client.get_node(node_id)
        data_value = ua.DataValue(ua.Variant(value, value_type))
        await node.write_value(data_value)
        print(f"Written {value} to {node_id} as {value_type.name}")
    except Exception as e:
        print(f"Error writing to node {node_id}: {e}")

async def main():
    print(f"Connecting to {url} ...")
    async with Client(url=url) as client:
        # Initiate the session with writing all prediction to zero
        write_tasks = [
            write_variable(client, "ns=4;s=AIO_Chamber_Pressure", 0, ua.VariantType.Float),
            write_variable(client, "ns=4;s=AIO_Main_Gas_Flow_Rate", 0, ua.VariantType.UInt16),
            write_variable(client, "ns=4;s=AIO_PF2_Pressure", 0, ua.VariantType.Float),
            write_variable(client, "ns=4;s=AIO_PF2_Gas_Flow_Rate", 0, ua.VariantType.Float),
            write_variable(client, "ns=4;s=AIO_Prediction_Error", 0, ua.VariantType.Float),
            write_variable(client, "ns=4;s=AIO_Classification", 0, ua.VariantType.Int16),
        ]

        await asyncio.gather(*write_tasks)  # Execute all writes asynchronously
        # Open the CSV file in append mode
        with open(output_file, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write the header if the file is empty
            if csvfile.tell() == 0:
                header = ["TimeStamp"] + [name for name, _ in variable_nodes]
                writer.writerow(header)

            previous_second = None
            data_queue = deque(maxlen=60)
            classifier_queue = deque(maxlen=60)

            min_values = None
            max_values = None

            while True:
                # Read variables concurrently
                tasks = [read_variable(client, node_id) for _, node_id in variable_nodes]
                results = await asyncio.gather(*tasks)
                row = []
                timestamp = None

                for (name, _), (extracted_value, source_timestamp) in zip(variable_nodes, results):
                    if extracted_value is not None:
                        if name in ["Chamber_Pressure", "PF1_Pressure", "PF2_Pressure"]:
                            extracted_value /= 1000
                        elif name in ["PF1_Gas_Flow_Rate", "PF2_Gas_Flow_Rate"]:
                            extracted_value /= 10

                    if timestamp is None and source_timestamp is not None:
                        timestamp = source_timestamp

                    row.append(extracted_value)

                formatted_timestamp = (
                    timestamp.strftime("%Y%m%dT%H:%M:%S") + f".{int(timestamp.microsecond / 1000):03d}"
                    if timestamp else "N/A"
                )
                current_second = datetime.strptime(formatted_timestamp.split('.')[0], "%Y%m%dT%H:%M:%S")

                if current_second == previous_second:
                    print("Skipping row: Current second matches the previous second.")
                    continue

                previous_second = current_second
                row.insert(0, formatted_timestamp)
                writer.writerow(row)
                print(f"Saved row: {row}")

                # Extract relevant data for anomaly detection
                row_data = np.array(row[1:], dtype=float).reshape(1, -1)
                selected_data = row_data[:, [4, 5, 8, 9]] # [Chamber_Pressure, Main_Gas_Flow_Rate, PF2_Pressure, PF2_Gas_Flow_Rate

                # Assuming chamber_temp is always 400
                chamber_temp = 400

                # Actual Cold Spray Maching running
                selected_data = row_data[:, [3]]

                if chamber_temp == 400:
                    selected_data = np.array(selected_data, dtype=np.float32)

                    # Initialize or update min/max values
                    if min_values is None or max_values is None:
                        min_values = selected_data.copy()
                        max_values = selected_data.copy()
                    else:
                        min_values = np.minimum(min_values, selected_data)
                        max_values = np.maximum(max_values, selected_data)

                    # Fit scaler dynamically
                    scaler.fit(np.vstack([min_values, max_values]))

                    # Add new data to queue
                    data_queue.append(selected_data)

                    if len(data_queue) == 60:
                        # Prepare data for model inference
                        evaluation_data = np.array(data_queue).reshape(1, 60, 4)
                        scaled_data = scaler.transform(evaluation_data.reshape(-1, evaluation_data.shape[-1])).reshape(evaluation_data.shape)

                        with torch.no_grad():
                            actual = torch.tensor(scaled_data, dtype=torch.float32)
                            prediction = model(actual)
                            reconstruction_error = torch.mean(torch.abs(actual[:, -1, :] - prediction), axis=1)

                        print("LSTM Prediction (scaled)  :", prediction.numpy())

                        # Convert prediction back to original values (inverse transform)
                        if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
                            original_prediction = scaler.inverse_transform(prediction.numpy())
                            print("LSTM Prediction (original):", original_prediction)

                            # Anomaly Classification Mapping
                            slight_abnormal_threshold = 0.1683
                            heavy_abnormal_threshold = 0.2114

                            if reconstruction_error >= heavy_abnormal_threshold:
                                anomaly_classifier = 2  # Heavy Anomaly
                            elif slight_abnormal_threshold < reconstruction_error < heavy_abnormal_threshold:
                                anomaly_classifier = 1  # Slight Anomaly
                            else:
                                anomaly_classifier = 0  # Normal

                            print(f"Reconstruction Error Score: {reconstruction_error.item():.4f}, Classification: {anomaly_classifier}")

                            # Ensure classification is an integer (0, 1, or 2)
                            anomaly_classifier = int(anomaly_classifier)

                            # Prepare tasks to write values
                            write_tasks = [
                                write_variable(client, "ns=4;s=AIO_Chamber_Pressure", float(original_prediction[0, 0]), ua.VariantType.Float),
                                write_variable(client, "ns=4;s=AIO_Main_Gas_Flow_Rate", int(original_prediction[0, 1]), ua.VariantType.UInt16),
                                write_variable(client, "ns=4;s=AIO_PF2_Pressure", float(original_prediction[0, 2]), ua.VariantType.Float),
                                write_variable(client, "ns=4;s=AIO_PF2_Gas_Flow_Rate", float(original_prediction[0, 3]), ua.VariantType.Float),
                                write_variable(client, "ns=4;s=AIO_Prediction_Error", float(reconstruction_error.item()), ua.VariantType.Float),
                                write_variable(client, "ns=4;s=AIO_Classification", int(anomaly_classifier), ua.VariantType.Int16),
                            ]

                            await asyncio.gather(*write_tasks)  # Execute all writes asynchronously

                        else:
                            print("Scaler has not been fitted properly. Ensure min/max values were set.")
                            
                # Flush CSV to ensure data is saved
                csvfile.flush()

                # Wait 1 second before reading again
                elapsed_time = max(0, (1000 - int(formatted_timestamp.split('.')[-1])) / 1000)
                await asyncio.sleep(elapsed_time)


if __name__ == "__main__":
    asyncio.run(main())
