"""
train_rf.py
Trains a Random Forest on dataset.csv and saves attention_rf.joblib.
The split is by source clip (not by row) so results are not inflated.
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("dataset.csv")
X = df.drop(columns=["label", "source"])
y = df["label"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr, te = next(gss.split(X, y, groups=df["source"]))

if y.iloc[te].nunique() < 2:
    print("Warning: the test set has only one class. Record more clips "
          "of both labels for a meaningful evaluation.")

clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                             random_state=42)
clf.fit(X.iloc[tr], y.iloc[tr])
pred = clf.predict(X.iloc[te])

print(classification_report(y.iloc[te], pred))
print("Confusion matrix:")
print(confusion_matrix(y.iloc[te], pred))

imp = pd.Series(clf.feature_importances_, index=X.columns)
print("\nTop features:")
print(imp.sort_values(ascending=False).head(10))

joblib.dump(clf, "attention_rf.joblib")
print("\nSaved attention_rf.joblib")
