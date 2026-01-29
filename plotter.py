import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("ppg_stream_1.csv")

# Skip first 400 data points
df = df.iloc[400:]

# Extract signal
y = df["ir_filtered"].to_numpy()
x = df["timestamp"].to_numpy()

# Normalize (z-score)
y = (y - np.mean(y)) / (np.std(y) + 1e-6)

# Plot
plt.figure()
plt.plot(x, y)
plt.xlabel("Timestamp")
plt.ylabel("Normalized IR Filtered")
plt.title("Timestamp vs Normalized IR Filtered")
plt.show()
