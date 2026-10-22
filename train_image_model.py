"""
train_image_model.py
Train a Random Forest from image_dataset.csv and save attention_image_rf.joblib.
"""
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

DATASET = "image_dataset.csv"
MODEL_OUT = "attention_image_rf.joblib"
FEATURES = ["ear", "pitch", "yaw", "roll"]

def main():
    try:
        df = pd.read_csv(DATASET)
    except FileNotFoundError:
        sys.exit("image_dataset.csv not found. Run: python build_image_dataset.py")

    required = FEATURES + ["label", "source"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit("Missing columns in image_dataset.csv: " + ", ".join(missing))

    df = df.dropna(subset=required).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["attentive", "inattentive"])]

    if df["label"].nunique() < 2:
        sys.exit("Need images in BOTH images/attentive and images/inattentive.")

    if df["source"].nunique() < 4:
        sys.exit(
            "Please add at least 4 different image files (preferably many more) "
            "across both classes before training. Similar crops from one photo "
            "do not count as independent examples."
        )

    X = df[FEATURES].astype(float)
    y = df["label"]
    groups = df["source"]

    splitter = GroupShuffleSplit(n_splits=20, test_size=0.25, random_state=42)
    split = None
    for tr, te in splitter.split(X, y, groups=groups):
        if y.iloc[tr].nunique() == 2 and y.iloc[te].nunique() == 2:
            split = (tr, te)
            break

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )

    if split is None:
        print(
            "Warning: could not create a train/test split containing both labels. "
            "Metrics may be incomplete. Add more independent photos of both classes."
        )
        clf.fit(X, y)
        joblib.dump(clf, MODEL_OUT)
        print(f"\nTrained on all {len(df)} face rows; saved {MODEL_OUT}")
        print("No reliable held-out evaluation was possible.")
        return

    tr, te = split
    clf.fit(X.iloc[tr], y.iloc[tr])
    pred = clf.predict(X.iloc[te])

    print("Evaluation on held-out image files:")
    print(classification_report(y.iloc[te], pred, zero_division=0))
    print("Confusion matrix (label order: attentive, inattentive):")
    print(confusion_matrix(y.iloc[te], pred, labels=["attentive", "inattentive"]))

    # Refit on all labelled photos for the final model.
    clf.fit(X, y)
    joblib.dump(clf, MODEL_OUT)
    print(f"\nSaved {MODEL_OUT}")
    print("In the app, Settings -> Trained model -> Load model...")
    print(f"Select {MODEL_OUT}")

if __name__ == "__main__":
    main()
