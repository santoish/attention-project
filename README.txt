Student Attention Detection - college project
=============================================
Setup (Python 3.9 - 3.11):
    pip install -r requirements.txt
    (Linux only: sudo apt install python3-tk)

Run:
    python attention_app.py        interface (rules mode works immediately)

Train your own model:
    1. In the app, Data collection tab: record 20-30 clips, labeled
       attentive / inattentive (saved to data/ and labels.csv).
    2. python build_dataset.py     -> dataset.csv
    3. python train_rf.py          -> attention_rf.joblib + metrics
    4. Restart the app, Settings tab -> choose "Trained model".

RUN STEPS
=========

1. In the project folder, create:
       images/attentive/
       images/inattentive/

2. Put labelled photos in those folders. Start with clear photos and use many
   different images and people/poses; do not put the same photo or near-identical
   crops in both folders.

3. Activate your existing virtual environment, then run:
       python build_image_dataset.py
       python train_image_model.py

4. Start the app:
       python attention_app.py

5. In Settings, choose Trained model -> Load model... and select:
       attention_image_rf.joblib

Do NOT rename it to attention_rf.joblib unless you have backed up the old model.

NOTES
=====
- Each image's folder label is applied to all faces detected in that image.
- The feature model learns visible eye/head pose patterns, not a student's
  actual mental state. Looking down to write notes may be misclassified.
- A single image cannot show how long someone has had their eyes closed.
- This model is a basic baseline, not a reliable measure of student attention.

Files:
    attention_core.py   detection logic (face mesh, head pose, EAR, rules, model)
    attention_app.py    desktop interface (run this)
    build_dataset.py    recorded logs -> dataset.csv
    train_rf.py         Random Forest training
    requirements.txt    libraries (mediapipe is pinned to 0.10.14 on purpose)

Voice assistant (LiveKit)
=========================
The assistant runs as a separate program from the app, in its own environment,
and talks to the app through live_status.json (numbers only, never video).

One-time setup:
    python -m venv venv_voice
    venv_voice\Scripts\activate            (Linux/Mac: source venv_voice/bin/activate)
    pip install -r requirements_voice.txt
    1. Create a free LiveKit Cloud project at cloud.livekit.io
    2. Copy env_example.txt to .env.local and fill in the 3 values

Every time:
    Terminal 1 (app environment):    python attention_app.py   -> Start Webcam
    Terminal 2 (voice environment):  python voice_agent.py console

Ask: "How is the class doing?", "Is student 2 paying attention?",
     "Any alerts?", "Calibrate the students."

New files: voice_agent.py, status_bridge.py, requirements_voice.txt, env_example.txt
(attention_app.py was updated to write live_status.json and show a Voice tab.)

Training from videos you already recorded
=========================================
    videos/attentive/     one behavior per video
    videos/inattentive/
    python process_videos.py     -> data/*.csv and labels.csv
    python build_dataset.py      -> dataset.csv
    python train_rf.py           -> attention_rf.joblib
