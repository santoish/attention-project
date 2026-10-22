"""
attention_core.py  -  detection logic (no GUI)
------------------------------------------------
Face Mesh -> head pose + EAR -> per-student tracking -> rules or trained model.
Used by attention_app.py (the interface). Can also be imported on its own.
"""
import collections
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

WINDOW_S = 3.0             # window length used for the trained model
SCORE_ALPHA = 0.1          # smoothing for the attention score
MAX_MATCH_DIST = 120       # px, matching faces between frames
MAX_MISSING_SECONDS = 2.0  # drop a track after this long without a match

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
POSE_IDS = [1, 152, 33, 263, 61, 291]   # nose, chin, eye corners, mouth corners

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # nose tip
    (0.0, -330.0, -65.0),      # chin
    (-225.0, 170.0, -135.0),   # right eye outer corner
    (225.0, 170.0, -135.0),    # left eye outer corner
    (-150.0, -150.0, -125.0),  # right mouth corner
    (150.0, -150.0, -125.0),   # left mouth corner
], dtype=np.float64)

COLORS_BGR = {"Attentive": (0, 200, 0), "Distracted": (0, 165, 255),
              "Inattentive": (0, 165, 255), "Drowsy": (0, 0, 255)}
TARGET_SCORE = {"Attentive": 100.0, "Distracted": 20.0,
                "Inattentive": 20.0, "Drowsy": 0.0}


@dataclass
class Config:
    ear_thresh: float = 0.21     # eye closed below this
    yaw_thresh: float = 30.0     # degrees from the calibrated board direction
    pitch_thresh: float = 22.0
    drowsy_s: float = 1.5        # eyes closed this long -> drowsy
    away_s: float = 2.0          # head away this long -> distracted
    use_model: bool = False      # False = rules, True = trained model


# ------------------------------------------------------------------ features
def eye_aspect_ratio(pts):
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def head_pose(landmarks_px, frame_w, frame_h):
    """(pitch, yaw, roll) in degrees or None."""
    image_pts = landmarks_px[POSE_IDS].astype(np.float64)
    cam = np.array([[frame_w, 0, frame_w / 2],
                    [0, frame_w, frame_h / 2],
                    [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_pts, cam, np.zeros((4, 1)),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles
    pitch = wrap180(pitch)
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch, wrap180(yaw), wrap180(roll)


def window_features(arr):
    """arr columns: time, ear, pitch, yaw, roll.
    Feature names/order match build_dataset.py in the project manual."""
    f = {}
    for i, c in enumerate(["ear", "pitch", "yaw", "roll"], start=1):
        v = arr[:, i]
        f[c + "_mean"] = float(v.mean())
        f[c + "_std"] = float(v.std(ddof=1))
        f[c + "_min"] = float(v.min())
        f[c + "_max"] = float(v.max())
    f["ear_closed_ratio"] = float((arr[:, 1] < 0.21).mean())
    f["yaw_abs_mean"] = float(np.abs(arr[:, 3]).mean())
    f["pitch_abs_mean"] = float(np.abs(arr[:, 2]).mean())
    return f


# ------------------------------------------------------------------ students
class Student:

    def __init__(self, student_id, centroid, now):
        self.id = student_id
        self.centroid = np.asarray(centroid, dtype=np.float64)
        self.last_seen = now

        self.ear = 0.0
        self.pose = None
        self.hist = collections.deque()
        self.last_pred_t = 0.0
        self.model_label = "Attentive"

        self.baseline = (0.0, 0.0)
        self.eyes_closed_since = None
        self.away_since = None

        self.status = "Attentive"
        self.score = 100.0

    def model_status(self, clf, now):
        """Predict using either a photo model or the existing temporal model."""
        cols = list(getattr(clf, "feature_names_in_", []))

        # Photo model: one row of features from the current frame.
        image_cols = ["ear", "pitch", "yaw", "roll"]
        if cols == image_cols:
            if self.pose is not None:
                values = {
                    "ear": float(self.ear),
                    "pitch": float(self.pose[0]),
                    "yaw": float(self.pose[1]),
                    "roll": float(self.pose[2]),
                }
                X = pd.DataFrame([values], columns=image_cols)
                pred = str(clf.predict(X)[0]).lower()
                self.model_label = (
                    "Attentive" if pred.startswith("att") else "Inattentive"
                )
            return self.model_label

        # Original video model: use a 3-second feature window.
        if now - self.last_pred_t >= 0.5 and len(self.hist) >= 10:
            arr = np.array(self.hist, dtype=np.float64)
            arr = arr[arr[:, 0] >= now - WINDOW_S]
            if len(arr) >= 10 and arr[-1, 0] - arr[0, 0] >= 2.0:
                X = pd.DataFrame([window_features(arr)])
                if cols:
                    X = X[cols]
                pred = str(clf.predict(X)[0]).lower()
                self.model_label = (
                    "Attentive" if pred.startswith("att") else "Inattentive"
                )
            self.last_pred_t = now
        return self.model_label

    def observe(self, now, ear, pose):
        self.ear, self.pose = ear, pose
        self.hist.append((now, ear, pose[0], pose[1], pose[2]))
        while self.hist and now - self.hist[0][0] > WINDOW_S + 0.5:
            self.hist.popleft()

    def rules_status(self, cfg, now):
        pitch, yaw, _ = self.pose
        dyaw = wrap180(yaw - self.baseline[1])
        dpitch = pitch - self.baseline[0]

        if self.ear < cfg.ear_thresh:
            self.eyes_closed_since = self.eyes_closed_since or now
        else:
            self.eyes_closed_since = None
        drowsy = (self.eyes_closed_since is not None
                  and now - self.eyes_closed_since >= cfg.drowsy_s)

        if abs(dyaw) > cfg.yaw_thresh or abs(dpitch) > cfg.pitch_thresh:
            self.away_since = self.away_since or now
        else:
            self.away_since = None
        away = (self.away_since is not None
                and now - self.away_since >= cfg.away_s)

        return "Drowsy" if drowsy else ("Distracted" if away else "Attentive")

    '''def model_status(self, clf, now):
        """Predict from the last 3 s of features (twice per second)."""
        if now - self.last_pred_t >= 0.5 and len(self.hist) >= 10:
            arr = np.array(self.hist, dtype=np.float64)
            arr = arr[arr[:, 0] >= now - WINDOW_S]
            if len(arr) >= 10 and arr[-1, 0] - arr[0, 0] >= 2.0:
                X = pd.DataFrame([window_features(arr)])
                cols = getattr(clf, "feature_names_in_", None)
                if cols is not None:
                    X = X[list(cols)]
                pred = str(clf.predict(X)[0]).lower()
                self.model_label = ("Attentive" if pred.startswith("att")
                                    else "Inattentive")
            self.last_pred_t = now
        return self.model_label'''

    def update_score(self):
        self.score += SCORE_ALPHA * (TARGET_SCORE[self.status] - self.score)


class Tracker:
    """Nearest-centroid tracker: every face keeps an ID."""

    def __init__(self):
        self.students = {}
        self.next_id = 1

    def update(self, centroids, now):
        assigned, free = {}, set(self.students)
        for idx, c in enumerate(centroids):
            best, best_d = None, MAX_MATCH_DIST
            for sid in free:
                d = np.linalg.norm(self.students[sid].centroid - c)
                if d < best_d:
                    best, best_d = sid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.students[best] = Student(best, c, now)
            else:
                free.discard(best)
            s = self.students[best]
            s.centroid, s.last_seen = c, now
            assigned[idx] = s
        for sid in [k for k, s in self.students.items()
                    if now - s.last_seen > MAX_MISSING_SECONDS]:
            del self.students[sid]
        return assigned


# ------------------------------------------------------------------ engine
class Engine:
    def __init__(self, max_faces=15):
        self.max_faces = max_faces
        self.mesh = None
        self.tracker = Tracker()

    def reset(self):
        self.close()
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe version has no 'solutions' (Face Mesh) API.\n"
                "Fix:  pip install mediapipe==0.10.14   (use Python 3.9 - 3.11)")
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.max_faces, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.tracker = Tracker()

    def close(self):
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None

    def calibrate(self):
        """Use every tracked student's current pose as 'looking at the board'."""
        n = 0
        for s in self.tracker.students.values():
            if s.pose is not None:
                s.baseline = (s.pose[0], s.pose[1])
                n += 1
        return n

    def process(self, frame, now, cfg, clf=None):
        """Analyse one BGR frame. Draws on it and returns
        (frame, rows_for_logging, status_change_events)."""
        if self.mesh is None:
            self.reset()
        h, w = frame.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = [np.array([[p.x * w, p.y * h] for p in fl.landmark])
                 for fl in (res.multi_face_landmarks or [])]
        assigned = self.tracker.update([p[1] for p in faces], now)

        rows, events = [], []
        use_model = cfg.use_model and clf is not None
        for idx, pts in enumerate(faces):
            s = assigned[idx]
            pose = head_pose(pts, w, h)
            if pose is None:
                continue
            ear = (eye_aspect_ratio(pts[RIGHT_EYE]) +
                   eye_aspect_ratio(pts[LEFT_EYE])) / 2.0
            s.observe(now, ear, pose)
            rule_status = s.rules_status(cfg, now)   # keeps timers updated
            status = s.model_status(clf, now) if use_model else rule_status
            if status != s.status:
                events.append((now, s.id, status))
            s.status = status
            s.update_score()

            x1, y1 = pts.min(axis=0).astype(int)
            x2, y2 = pts.max(axis=0).astype(int)
            color = COLORS_BGR[status]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{s.id} {status} {s.score:.0f}",
                        (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2)
            rows.append({"student_id": s.id, "ear": ear, "pitch": pose[0],
                         "yaw": pose[1], "roll": pose[2], "status": status,
                         "score": s.score})
        return frame, rows, events

    def summary(self):
        """(class_average, counts) or None when nobody is tracked."""
        st = list(self.tracker.students.values())
        if not st:
            return None
        counts = {"attentive": 0, "inattentive": 0, "drowsy": 0}
        for s in st:
            if s.status == "Attentive":
                counts["attentive"] += 1
            elif s.status == "Drowsy":
                counts["drowsy"] += 1
            else:
                counts["inattentive"] += 1
        return float(np.mean([s.score for s in st])), counts
