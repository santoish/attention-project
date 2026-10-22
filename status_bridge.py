"""
status_bridge.py - shared helpers between the attention app and the voice assistant
-----------------------------------------------------------------------------------
No GUI and no LiveKit imports here, so both programs can use it (and it is easy
to test). The app writes live_status.json every second; the voice assistant reads
it. The assistant can also ask the app to calibrate via voice_command.json.
"""
import json
import os
import time

STATUS_FILE = os.environ.get("ATTENTION_STATUS_FILE", "live_status.json")
COMMAND_FILE = os.environ.get("ATTENTION_COMMAND_FILE", "voice_command.json")
STALE_AFTER_S = 5.0

NOT_RUNNING = ("The attention app is not running right now. Ask the teacher to "
               "start the webcam or open a video in the app first.")


def read_status():
    """Latest status dict, or None if the app is not running or the data is old."""
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not data.get("running") or time.time() - data.get("updated", 0) > STALE_AFTER_S:
        return None
    return data


def describe_class(data):
    if data is None:
        return NOT_RUNNING
    score = data.get("class_score")
    if score is None:
        return "The app is running, but no student faces are visible yet."
    c = data["counts"]
    level = "good" if score >= 70 else ("moderate" if score >= 40 else "low")
    mode = "a trained model" if data.get("mode") == "model" else "rule based detection"
    return (f"Class attention is {score} percent, which is {level}. "
            f"Attentive students: {c['attentive']}. Not attentive: {c['inattentive']}. "
            f"Drowsy: {c['drowsy']}. The app is using {mode}.")


def describe_student(data, student_id):
    if data is None:
        return NOT_RUNNING
    for s in data["students"]:
        if s["id"] == student_id:
            return (f"Student {student_id} is {s['status'].lower()} with an attention "
                    f"score of {s['score']} out of 100.")
    return f"I cannot see student number {student_id} right now."


def describe_alerts(data):
    if data is None:
        return NOT_RUNNING
    alerts = data.get("recent_alerts", [])[:5]
    if not alerts:
        return "There are no alerts yet. Everyone has been attentive so far."
    return "Most recent alerts, newest first: " + ". ".join(alerts)


def request_calibration():
    """Ask the app to use every student's current head direction as 'attentive'."""
    if read_status() is None:
        return NOT_RUNNING
    try:
        with open(COMMAND_FILE, "w", encoding="utf-8") as fh:
            json.dump({"command": "calibrate"}, fh)
    except OSError:
        return "I could not send the calibration request."
    return "Calibration requested. Ask the students to look at the board."
