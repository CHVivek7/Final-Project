import pandas as pd
import numpy as np

data = pd.read_csv("qm9.csv")

print("Original:", data.shape)

ml = pd.DataFrame()

# features
ml["atoms"] = (data["r2"] / 10).round().astype(int)

ml["electrons"] = (data["alpha"] * 2).round().astype(int)

ml["qubits"] = ml["atoms"] * 2

ml["basis"] = ml["atoms"] * 5

ml["complex"] = (abs(data["homo"]) + data["gap"]) / 2

# label
ml["label"] = np.where(ml["qubits"] > 10, 1, 0)

ml = ml.dropna()

print("ML dataset:", ml.shape)

ml.to_csv("data.csv", index=False)

print("Saved → data.csv")