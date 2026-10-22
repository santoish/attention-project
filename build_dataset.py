"""
build_dataset.py
Turns the recorded CSV logs (data/) plus labels.csv into dataset.csv.
Each row = one 3-second window of one student, with summary features.
Feature names/order are identical to attention_core.window_features().
"""
import os
import sys

import pandas as pd

WINDOW_S, STEP_S = 3.0, 1.0
FEATS = ["ear", "pitch", "yaw", "roll"]

if not os.path.exists("labels.csv"):
    sys.exit("labels.csv not found. Record clips with a label in the app first.")

rows = []
labels = pd.read_csv("labels.csv")

for _, r in labels.iterrows():
    path = os.path.join("data", r["file"])
    if not os.path.exists(path):
        print("missing:", path)
        continue
    df = pd.read_csv(path)
    for sid, g in df.groupby("student_id"):
        g = g.sort_values("time")
        start, end = g["time"].iloc[0], g["time"].iloc[-1]
        while start + WINDOW_S <= end:
            w = g[(g["time"] >= start) & (g["time"] < start + WINDOW_S)]
            if len(w) >= 10:
                f = {}
                for c in FEATS:
                    f[c + "_mean"] = w[c].mean()
                    f[c + "_std"] = w[c].std()
                    f[c + "_min"] = w[c].min()
                    f[c + "_max"] = w[c].max()
                f["ear_closed_ratio"] = (w["ear"] < 0.21).mean()
                f["yaw_abs_mean"] = w["yaw"].abs().mean()
                f["pitch_abs_mean"] = w["pitch"].abs().mean()
                f["label"] = r["label"]
                f["source"] = r["file"]
                rows.append(f)
            start += STEP_S

out = pd.DataFrame(rows).fillna(0)
out.to_csv("dataset.csv", index=False)
print("windows per label:")
print(out["label"].value_counts())
print("saved dataset.csv")
