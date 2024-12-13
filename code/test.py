import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

# Simulated data
data = np.sin(np.linspace(0, 4 * np.pi, 300)) + np.random.normal(0, 0.1, 300)  # Example signal
rolling_window_size = 60  # Rolling window size
rolling_data = pd.Series(data).rolling(window=rolling_window_size).mean()

# Generate frames for the GIF
frames = []
output_folder = "frames"

# Create output folder if it doesn't exist
import os
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(len(data) - rolling_window_size + 1):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="Original Data", alpha=0.7)
    plt.plot(range(rolling_window_size - 1, i + rolling_window_size), rolling_data.iloc[rolling_window_size - 1 : i + rolling_window_size], label="Rolling Average", color='red')
    plt.axvline(i + rolling_window_size - 1, color='gray', linestyle='--', alpha=0.5, label="Current Window End")
    plt.legend()
    plt.title(f"Rolling Window Visualization (Window ending at {i + rolling_window_size - 1})")
    plt.xlabel("Timestamp")
    plt.ylabel("Gas Flow Rate")
    
    # Save frame
    frame_path = f"{output_folder}/frame_{i}.png"
    plt.savefig(frame_path)
    plt.close()
    frames.append(frame_path)

# Combine frames into a GIF
gif_path = "rolling_window.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:  # duration sets frame speed
    for frame_path in frames:
        writer.append_data(imageio.imread(frame_path))

# Clean up temporary files
for frame_path in frames:
    os.remove(frame_path)

print(f"GIF saved as {gif_path}")
