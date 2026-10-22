"""
attention_app.py  -  Classroom Attention Detection (desktop interface)
-----------------------------------------------------------------------
Run:  python attention_app.py
Needs attention_core.py in the same folder.

Features
  - Live webcam or video file, annotated video with student IDs
  - Class attention score, counts, per-student table, alert log, live chart
  - Rule-based mode or trained-model mode (attention_rf.joblib)
  - Adjustable thresholds, per-student calibration
  - Data collection: record CSV logs and auto-update labels.csv
"""
import collections
import csv
import datetime
import json
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from attention_core import Config, Engine
from status_bridge import COMMAND_FILE, STATUS_FILE

try:
    import joblib
except ImportError:      # model mode will be unavailable
    joblib = None

VIDEO_W, VIDEO_H = 640, 480
STATUS_COLORS = {"Attentive": "#1a8f1a", "Distracted": "#e07b00",
                 "Inattentive": "#e07b00", "Drowsy": "#d02020"}
LOG_HEADER = ["time", "student_id", "ear", "pitch", "yaw", "roll",
              "status", "score"]


class App:
    def __init__(self, root):
        self.root = root
        root.title("Classroom Attention Detection")
        root.geometry("1180x740")
        root.minsize(1100, 700)

        self.engine = Engine()
        self.cap = None
        self.running = False
        self.is_file = False
        self.fps_src = 30.0
        self.frame_idx = 0
        self.now_last = 0.0
        self.clf = None
        self.model_name = "none"
        self.rec = None
        self.score_hist = collections.deque(maxlen=300)
        self.last_sample = None
        self.fps, self.fps_t = 0.0, time.time()
        self.alert_log = collections.deque(maxlen=20)   # for the voice assistant
        self.last_export = 0.0

        self.v_ear = tk.DoubleVar(value=0.21)
        self.v_yaw = tk.DoubleVar(value=30.0)
        self.v_pitch = tk.DoubleVar(value=22.0)
        self.v_drowsy = tk.DoubleVar(value=1.5)
        self.v_away = tk.DoubleVar(value=2.0)
        self.v_mode = tk.StringVar(value="rules")
        self.v_label = tk.StringVar(value="none")
        self.v_status = tk.StringVar(value="Ready. Start the webcam or open a video.")

        self.build_ui()
        self.try_autoload_model()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------------------------------------------- layout
    def build_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="n")
        right = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(self.root, textvariable=self.v_status, relief="sunken",
                  anchor="w", padding=4).grid(row=1, column=0, columnspan=2,
                                              sticky="ew")

        # --- video
        self.blank = ImageTk.PhotoImage(Image.new("RGB", (VIDEO_W, VIDEO_H), "black"))
        self.video = tk.Label(left, image=self.blank, text="No video",
                              compound="center", fg="white", bg="black")
        self.video.pack()

        bar = ttk.Frame(left)
        bar.pack(fill="x", pady=6)
        ttk.Button(bar, text="Start Webcam", command=self.start_webcam).pack(side="left", padx=2)
        ttk.Button(bar, text="Open Video...", command=self.open_video).pack(side="left", padx=2)
        ttk.Button(bar, text="Stop", command=self.stop).pack(side="left", padx=2)
        ttk.Button(bar, text="Calibrate", command=self.calibrate).pack(side="left", padx=(14, 2))
        ttk.Button(bar, text="Snapshot", command=self.snapshot).pack(side="left", padx=2)

        # --- chart
        fig = Figure(figsize=(6.4, 1.9), dpi=80)
        self.ax = fig.add_subplot(111)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 60)
        self.ax.set_title("Class attention over time", fontsize=9)
        self.ax.set_xlabel("seconds", fontsize=8)
        self.ax.tick_params(labelsize=8)
        self.ax.grid(alpha=0.3)
        (self.line,) = self.ax.plot([], [], color="#1f3a5f", linewidth=1.6)
        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=left)
        self.canvas.get_tk_widget().pack(fill="x")

        # --- right: tabs
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        live = ttk.Frame(nb, padding=8)
        sett = ttk.Frame(nb, padding=8)
        data = ttk.Frame(nb, padding=8)
        nb.add(live, text="Live")
        nb.add(sett, text="Settings")
        nb.add(data, text="Data collection")
        voice = ttk.Frame(nb, padding=8)
        nb.add(voice, text="Voice assistant")
        self.build_voice(voice)
        self.build_live(live)
        self.build_settings(sett)
        self.build_data(data)

    def build_live(self, f):
        ttk.Label(f, text="Class attention").pack(anchor="w")
        self.lbl_score = tk.Label(f, text="--", font=("Helvetica", 40, "bold"))
        self.lbl_score.pack(anchor="w")
        self.bar = ttk.Progressbar(f, maximum=100)
        self.bar.pack(fill="x", pady=4)

        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        self.cnt = {}
        for key, text, color in [("attentive", "Attentive", "#1a8f1a"),
                                 ("inattentive", "Not attentive", "#e07b00"),
                                 ("drowsy", "Drowsy", "#d02020")]:
            box = ttk.Frame(row)
            box.pack(side="left", expand=True)
            self.cnt[key] = tk.Label(box, text="0", font=("Helvetica", 20, "bold"), fg=color)
            self.cnt[key].pack()
            ttk.Label(box, text=text).pack()

        ttk.Label(f, text="Students").pack(anchor="w", pady=(8, 0))
        cols = ("id", "status", "score", "ear", "pitch", "yaw")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=7)
        for c, wd in zip(cols, (45, 90, 55, 55, 55, 55)):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=wd, anchor="center")
        for st, col in STATUS_COLORS.items():
            self.tree.tag_configure(st, foreground=col)
        self.tree.pack(fill="x")

        ttk.Label(f, text="Alerts").pack(anchor="w", pady=(8, 0))
        af = ttk.Frame(f)
        af.pack(fill="both", expand=True)
        self.alerts = tk.Listbox(af, height=8, activestyle="none")
        sb = ttk.Scrollbar(af, command=self.alerts.yview)
        self.alerts.configure(yscrollcommand=sb.set)
        self.alerts.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def build_settings(self, f):
        box = ttk.LabelFrame(f, text="Detection mode", padding=6)
        box.pack(fill="x")
        ttk.Radiobutton(box, text="Rules (no training needed)", value="rules",
                        variable=self.v_mode, command=self.on_mode).pack(anchor="w")
        ttk.Radiobutton(box, text="Trained model", value="model",
                        variable=self.v_mode, command=self.on_mode).pack(anchor="w")
        mrow = ttk.Frame(box)
        mrow.pack(fill="x", pady=(4, 0))
        self.lbl_model = ttk.Label(mrow, text="Model: none")
        self.lbl_model.pack(side="left")
        ttk.Button(mrow, text="Load model...", command=self.load_model_dialog).pack(side="right")

        th = ttk.LabelFrame(f, text="Rule thresholds", padding=6)
        th.pack(fill="x", pady=10)
        self.slider(th, "Eye closed (EAR <)", self.v_ear, 0.10, 0.35, 0.01)
        self.slider(th, "Head turn (yaw deg)", self.v_yaw, 10, 60, 1)
        self.slider(th, "Head down (pitch deg)", self.v_pitch, 10, 60, 1)
        self.slider(th, "Drowsy after (s)", self.v_drowsy, 0.5, 5, 0.1)
        self.slider(th, "Distracted after (s)", self.v_away, 0.5, 6, 0.1)

        ttk.Label(f, wraplength=380, justify="left", text=(
            "Calibration: ask students to look at the board, then click "
            "Calibrate. Each student's current head direction becomes "
            "'attentive'.")).pack(anchor="w")

    def slider(self, parent, text, var, lo, hi, res):
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text=text, width=21).pack(side="left")
        tk.Scale(row, from_=lo, to=hi, resolution=res, orient="horizontal",
                 variable=var, length=170, bd=0,
                 highlightthickness=0).pack(side="left", fill="x", expand=True)

    def build_voice(self, f):
        ttk.Label(f, wraplength=400, justify="left", text=(
            "The voice assistant (LiveKit) runs as a separate program:\n\n"
            "    python voice_agent.py console\n\n"
            "While the webcam or a video is running, this app writes the live "
            "numbers to live_status.json every second. The assistant reads that "
            "file to answer your questions. Only numbers are shared, never video.\n\n"
            "Try asking:\n"
            "  - How is the class doing?\n"
            "  - Is student 2 paying attention?\n"
            "  - Any alerts?\n"
            "  - Calibrate the students.")).pack(anchor="w")

    def build_data(self, f):
        ttk.Label(f, wraplength=380, justify="left", text=(
            "Record labeled clips for training:\n"
            "1. Start the webcam or open a video.\n"
            "2. Choose the label below.\n"
            "3. Click Start Recording, then Stop Recording.\n\n"
            "Each recording is saved to data/ and added to labels.csv "
            "automatically, ready for build_dataset.py.")).pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=12)
        ttk.Label(row, text="Label:").pack(side="left")
        ttk.Combobox(row, textvariable=self.v_label, state="readonly", width=14,
                     values=["none", "attentive", "inattentive"]).pack(side="left", padx=6)
        self.btn_rec = ttk.Button(row, text="Start Recording", command=self.toggle_record)
        self.btn_rec.pack(side="left", padx=6)
        self.lbl_rec = ttk.Label(f, text="Not recording", foreground="#555555")
        self.lbl_rec.pack(anchor="w")

    # -------------------------------------------------------------- sources
    def start_webcam(self):
        self.open_source(0, False)

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Open video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"),
                                           ("All files", "*.*")])
        if path:
            self.open_source(path, True)

    def open_source(self, src, is_file):
        self.stop()
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open the camera / video.\n"
                                          "Close other apps using the camera, "
                                          "or choose a different file.")
            return
        self.cap, self.is_file = cap, is_file
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps_src = fps if 1 <= fps <= 240 else 30.0
        self.frame_idx = 0
        self.last_sample = None
        self.score_hist.clear()
        self.line.set_data([], [])
        self.ax.set_xlim(0, 60)
        self.canvas.draw_idle()
        self.engine.reset()
        self.running = True
        self.v_status.set("Running: " + ("video file" if is_file else "webcam"))
        self.tick()

    def stop(self):
        if self.rec:
            self.stop_recording()
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.export_status(stopped=True)

    # -------------------------------------------------------------- voice assistant
    def export_status(self, stopped=False):
        """Write the live numbers to live_status.json for the voice assistant."""
        summ = None if stopped else self.engine.summary()
        students = []
        if not stopped:
            for sid, s in self.engine.tracker.students.items():
                p = s.pose
                students.append({"id": sid, "status": s.status,
                                 "score": round(s.score), "ear": round(s.ear, 2),
                                 "pitch": round(p[0]) if p else None,
                                 "yaw": round(p[1]) if p else None})
        data = {"updated": time.time(),
                "running": bool(self.running and not stopped),
                "mode": self.v_mode.get(),
                "class_score": round(summ[0]) if summ else None,
                "counts": summ[1] if summ else
                {"attentive": 0, "inattentive": 0, "drowsy": 0},
                "students": students,
                "recent_alerts": list(self.alert_log)[:10]}
        try:
            tmp = STATUS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, STATUS_FILE)
        except OSError:
            pass

    def poll_commands(self):
        """Run a command requested by the voice assistant (voice_command.json)."""
        if not os.path.exists(COMMAND_FILE):
            return
        try:
            with open(COMMAND_FILE, encoding="utf-8") as fh:
                cmd = json.load(fh).get("command")
            os.remove(COMMAND_FILE)
        except (OSError, ValueError):
            return
        if cmd == "calibrate":
            self.calibrate()

    # -------------------------------------------------------------- main loop
    def make_config(self):
        return Config(ear_thresh=self.v_ear.get(), yaw_thresh=self.v_yaw.get(),
                      pitch_thresh=self.v_pitch.get(), drowsy_s=self.v_drowsy.get(),
                      away_s=self.v_away.get(),
                      use_model=(self.v_mode.get() == "model"))

    def tick(self):
        if not self.running or self.cap is None:
            return
        try:
            ok, frame = self.cap.read()
            if not ok:
                self.stop()
                self.v_status.set("Video ended." if self.is_file else "Camera read failed.")
                return
            self.frame_idx += 1
            now = self.frame_idx / self.fps_src if self.is_file else time.time()
            self.now_last = now

            frame, rows, events = self.engine.process(
                frame, now, self.make_config(), self.clf)

            if self.rec:
                for r in rows:
                    self.rec["w"].writerow([
                        f"{now - self.rec['t0']:.2f}", r["student_id"],
                        f"{r['ear']:.3f}", f"{r['pitch']:.1f}", f"{r['yaw']:.1f}",
                        f"{r['roll']:.1f}", r["status"], f"{r['score']:.1f}"])
            for ev in events:
                self.add_alert(*ev)

            self.last_frame = frame
            self.show(frame)
            self.update_panels(now)
        except Exception as exc:                       # show errors, don't freeze
            self.stop()
            messagebox.showerror("Processing error", str(exc))
            return
        self.root.after(1, self.tick)

    def show(self, frame):
        h, w = frame.shape[:2]
        s = min(VIDEO_W / w, VIDEO_H / h)
        img = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(img)
        self.video.configure(image=self.photo, text="")

    def update_panels(self, now):
        t = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(t - self.fps_t, 1e-6)
        self.fps_t = t

        summ = self.engine.summary()
        if summ is None:
            self.lbl_score.configure(text="--", fg="#555555")
            self.bar["value"] = 0
            for k in self.cnt:
                self.cnt[k].configure(text="0")
        else:
            avg, counts = summ
            color = "#1a8f1a" if avg >= 70 else ("#e07b00" if avg >= 40 else "#d02020")
            self.lbl_score.configure(text=f"{avg:.0f}%", fg=color)
            self.bar["value"] = avg
            for k, v in counts.items():
                self.cnt[k].configure(text=str(v))
            if self.last_sample is None:
                self.last_sample = now
            if now - self.last_sample >= 1.0:
                self.last_sample = now
                self.score_hist.append(avg)
                xs = list(range(len(self.score_hist)))
                self.line.set_data(xs, list(self.score_hist))
                self.ax.set_xlim(0, max(60, len(xs)))
                self.canvas.draw_idle()

        if self.frame_idx % 5 == 0:
            self.update_tree()
        if t - self.last_export >= 1.0:
            self.last_export = t
            self.export_status()
            self.poll_commands()
        rec = f" | REC {self.rec['name']}" if self.rec else ""
        self.v_status.set(f"Running | {self.fps:.0f} FPS | mode: {self.v_mode.get()}{rec}")

    def update_tree(self):
        existing = set(self.tree.get_children())
        for sid, s in self.engine.tracker.students.items():
            p = s.pose
            vals = (f"#{sid}", s.status, f"{s.score:.0f}", f"{s.ear:.2f}",
                    f"{p[0]:+.0f}" if p else "-", f"{p[1]:+.0f}" if p else "-")
            iid = str(sid)
            if iid in existing:
                self.tree.item(iid, values=vals, tags=(s.status,))
                existing.discard(iid)
            else:
                self.tree.insert("", "end", iid=iid, values=vals, tags=(s.status,))
        for iid in existing:
            self.tree.delete(iid)

    def add_alert(self, t, sid, status):
        if status == "Attentive":
            return
        if self.is_file:
            stamp = f"{int(t // 60):02d}:{int(t % 60):02d}"
        else:
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.alerts.insert(0, f"{stamp}  Student #{sid}: {status}")
        self.alert_log.appendleft(f"{stamp}  Student #{sid}: {status}")
        self.alerts.itemconfig(0, fg=STATUS_COLORS.get(status, "black"))
        if self.alerts.size() > 200:
            self.alerts.delete(200, "end")

    # -------------------------------------------------------------- actions
    def calibrate(self):
        n = self.engine.calibrate()
        self.v_status.set(f"Calibrated {n} student(s)." if n else
                          "No faces to calibrate. Make sure students are visible.")

    def snapshot(self):
        frame = getattr(self, "last_frame", None)
        if frame is None:
            messagebox.showinfo("Snapshot", "Nothing to save yet.")
            return
        os.makedirs("snapshots", exist_ok=True)
        path = os.path.join("snapshots", datetime.datetime.now().strftime(
            "snap_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, frame)
        self.v_status.set(f"Saved {path}")

    def toggle_record(self):
        if self.rec:
            self.stop_recording()
            return
        if not self.running:
            messagebox.showinfo("Record", "Start the webcam or open a video first.")
            return
        os.makedirs("data", exist_ok=True)
        name = datetime.datetime.now().strftime("rec_%Y%m%d_%H%M%S.csv")
        fh = open(os.path.join("data", name), "w", newline="")
        w = csv.writer(fh)
        w.writerow(LOG_HEADER)
        self.rec = {"f": fh, "w": w, "t0": self.now_last, "name": name,
                    "label": self.v_label.get()}
        self.btn_rec.configure(text="Stop Recording")
        self.lbl_rec.configure(text=f"Recording: data/{name}  (label: {self.rec['label']})")

    def stop_recording(self):
        rec, self.rec = self.rec, None
        if rec is None:
            return
        rec["f"].close()
        msg = f"Saved data/{rec['name']}"
        if rec["label"] != "none":
            new = not os.path.exists("labels.csv")
            with open("labels.csv", "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["file", "label"])
                w.writerow([rec["name"], rec["label"]])
            msg += f" and added to labels.csv as '{rec['label']}'"
        self.btn_rec.configure(text="Start Recording")
        self.lbl_rec.configure(text=msg)
        self.v_status.set(msg)

    # -------------------------------------------------------------- model
    def try_autoload_model(self):
        if joblib is not None and os.path.exists("attention_rf.joblib"):
            self.load_model("attention_rf.joblib", quiet=True)

    def load_model_dialog(self):
        path = filedialog.askopenfilename(
            title="Load trained model", filetypes=[("Model", "*.joblib *.pkl"),
                                                   ("All files", "*.*")])
        if path:
            self.load_model(path)

    def load_model(self, path, quiet=False):
        if joblib is None:
            messagebox.showerror("Model", "Install joblib and scikit-learn first.")
            return False
        try:
            clf = joblib.load(path)
            if not hasattr(clf, "predict"):
                raise ValueError("file is not a scikit-learn model")
        except Exception as exc:
            if not quiet:
                messagebox.showerror("Model", f"Could not load model:\n{exc}")
            return False
        self.clf = clf
        self.model_name = os.path.basename(path)
        self.lbl_model.configure(text=f"Model: {self.model_name}")
        return True

    def on_mode(self):
        if self.v_mode.get() == "model" and self.clf is None:
            if not messagebox.askyesno(
                    "Trained model", "No model loaded yet. Load one now?\n"
                    "(Train it first with the training script.)"):
                self.v_mode.set("rules")
                return
            self.load_model_dialog()
            if self.clf is None:
                self.v_mode.set("rules")

    def on_close(self):
        self.stop()
        self.engine.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
