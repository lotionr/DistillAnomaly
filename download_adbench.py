import os
import numpy as np
import pandas as pd
from adbench.datasets.ad_dataset import ADDataset

save_dir = "./adbench_data"
os.makedirs(save_dir, exist_ok=True)

# Dataset list from the RFuni paper
datasets = ["Cardiotocography", "SpamBase", "Satellite", "Shuttle", "Thyroid"]

print(f"Downloading to {os.path.abspath(save_dir)}...")
loader = ADDataset()

for name in datasets:
    try:
        print(f"Fetching {name}...")
        X, y = loader.load_data(name)
        np.savez(os.path.join(save_dir, f"{name}.npz"), X=X, y=y)
        print(f"Saved {name}")
    except Exception as e:
        print(f"Error on {name}: {e}")
