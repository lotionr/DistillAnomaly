import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Required for cloud servers without a monitor
import matplotlib.pyplot as plt

# 1. Setup Directories
output_dir = "adbench_images"
os.makedirs(os.path.join(output_dir, "normal"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "anomaly"), exist_ok=True)

print("Directories created. Preparing data...")

# 2. Generate/Load Data
# We simulate the ADBench tabular structure here
X_normal = np.random.normal(loc=0, scale=1, size=(80, 50))
X_anomaly = np.random.normal(loc=0, scale=1, size=(20, 50))
X_anomaly[:, 20:25] += 5.0 # The "Anomaly" spike

X = np.vstack([X_normal, X_anomaly])
y = np.hstack([np.zeros(80), np.ones(20)])

# 3. Conversion Loop
print(f"Converting {len(X)} ADBench rows to chart images...")

for i in range(len(X)):
    feature_vector = X[i]
    label = int(y[i])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(feature_vector, color='blue', linewidth=2)
    ax.set_title(f"ADBench Sample {i}")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    folder = "anomaly" if label == 1 else "normal"
    save_path = os.path.join(output_dir, folder, f"sample_{i:04d}.png")
    
    plt.savefig(save_path)
    plt.close(fig)

print(f"Done! Check the '{output_dir}' folder.")
