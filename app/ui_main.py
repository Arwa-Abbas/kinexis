"""
Kinexis — GUI Launcher  (app/ui_main.py)

A Tkinter-based popup GUI that lets the user:
  - Choose webcam (live) or pick a video file
  - Set optional ground-truth exercise + rep count
  - Start / Stop the inference session
  - Watch live metrics (exercise, confidence, reps, form score, joint angles,
    form feedback) update in real time without blocking the main thread

The GUI polls gui_state every 100 ms and updates all panels
without any cv2 window interaction.

The cv2 window (cv2.imshow / cv2.waitKey) will still open as before —
the GUI and the OpenCV window coexist.  If we'd want headless mode (no
cv2 window) we can wrap the imshow calls like this:

    if gui_state is None:
        cv2.imshow('Kinexis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

Run:
    python app/ui_main.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import sys
import os

# ── make the project root importable ─────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# lazy-import so the GUI can open even without torch/mediapipe installed
_inference_module = None
def _get_inference():
    global _inference_module
    if _inference_module is None:
        import inference as _m
        _inference_module = _m
    return _inference_module


# ═══════════════════════════════════════════════════════════════════════════════
#  Colour palette & fonts
# ═══════════════════════════════════════════════════════════════════════════════
BG        = "#0d0f14"
CARD      = "#13161e"
BORDER    = "#1e2130"
ACCENT    = "#00e5ff"
ACCENT2   = "#7c3aed"
GOOD      = "#22c55e"
WARN      = "#f59e0b"
BAD       = "#ef4444"
FG        = "#e2e8f0"
FG_DIM    = "#64748b"
FG_MID    = "#94a3b8"

FONT_HEAD  = ("Courier New", 15, "bold")
FONT_LABEL = ("Courier New", 10)
FONT_SMALL = ("Courier New", 9)
FONT_VALUE = ("Courier New", 11, "bold")
FONT_TITLE = ("Courier New", 16, "bold")

EXERCISES = ["auto-detect", "squats", "push_ups", "pull_ups",
             "bench_press", "sit_ups", "jumping_jacks", "jump_rope"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared state passed between inference thread ↔ GUI
# ═══════════════════════════════════════════════════════════════════════════════
class SessionState:
    """Thread-safe bag of live metrics written by the inference thread."""
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "exercise":   "—",
            "confidence": 0.0,
            "reps":       0,
            "rep_state":  "INIT",
            "form_score": 100.0,
            "feedback":   [],
            "angles":     {},
            "deviations": {},
            "frame_count": 0,
            "running":    False,
            "error":      None,
        }

    def update(self, **kwargs):
        with self._lock:
            self._data.update(kwargs)

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def snapshot(self):
        with self._lock:
            return dict(self._data)


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference thread wrapper
# ═══════════════════════════════════════════════════════════════════════════════
class InferenceThread(threading.Thread):
    """
    Runs inference.run_inference() in a background thread, pushing metric
    updates into a SessionState object so the GUI can poll them safely.
    """

    def __init__(self, state: SessionState, source, output_path=None,
                 max_frames=None, ground_truth=None, ground_truth_reps=None):
        super().__init__(daemon=True)
        self.state            = state
        self.source           = source
        self.output_path      = output_path
        self.max_frames       = max_frames
        self.ground_truth     = ground_truth
        self.ground_truth_reps = ground_truth_reps
        self._stop_event      = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        self.state.update(running=True, error=None)
        try:
            inf = _get_inference()
            inf.run_inference(
                source             = self.source,
                output_path        = self.output_path,
                max_frames         = self.max_frames,
                ground_truth       = self.ground_truth,
                ground_truth_reps  = self.ground_truth_reps,
                gui_state          = self.state,          # ← NEW hook
                stop_event         = self._stop_event,    # ← NEW hook
            )
        except Exception as exc:
            self.state.update(error=str(exc))
        finally:
            self.state.update(running=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Reusable widgets
# ═══════════════════════════════════════════════════════════════════════════════
class MetricCard(tk.Frame):
    """A dark-background card that shows a label + large value."""
    def __init__(self, parent, label, unit="", **kwargs):
        super().__init__(parent, bg=CARD, bd=0, highlightthickness=1,
                         highlightbackground=BORDER, **kwargs)
        tk.Label(self, text=label, bg=CARD, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w", padx=10, pady=(8, 0))
        self._var = tk.StringVar(value="—")
        self._lbl = tk.Label(self, textvariable=self._var, bg=CARD, fg=ACCENT,
                             font=FONT_VALUE)
        self._lbl.pack(anchor="w", padx=10, pady=(0, 8))
        self._unit = unit

    def set(self, value, color=None):
        display = f"{value} {self._unit}".strip() if self._unit else str(value)
        self._var.set(display)
        if color:
            self._lbl.config(fg=color)


class AngleRow(tk.Frame):
    """One row in the angle table: joint name | value | deviation bar."""
    def __init__(self, parent, joint_name):
        super().__init__(parent, bg=CARD)
        short = joint_name.replace("_", " ")
        tk.Label(self, text=f"{short:<22}", bg=CARD, fg=FG_MID,
                 font=FONT_SMALL, width=22, anchor="w").pack(side="left")
        self._val = tk.StringVar(value="  —   ")
        tk.Label(self, textvariable=self._val, bg=CARD, fg=FG,
                 font=FONT_SMALL, width=10).pack(side="left")
        # deviation indicator
        self._canvas = tk.Canvas(self, bg=CARD, width=60, height=12,
                                 highlightthickness=0)
        self._canvas.pack(side="left", padx=(4, 0))

    def update(self, value, deviation=None, is_ratio=False):
        if is_ratio:
            self._val.set(f"{value:+.3f}  ")
        else:
            self._val.set(f"{value:5.1f}°  ")
        self._canvas.delete("all")
        if deviation is not None and deviation > 0:
            fill = BAD if deviation > 20 else WARN
            w = min(60, int(deviation / 45 * 60))
            self._canvas.create_rectangle(0, 2, w, 10, fill=fill, outline="")
        else:
            self._canvas.create_rectangle(0, 2, 4, 10, fill=GOOD, outline="")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════════════════════════════════════
class KinexisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kinexis — Motion Analysis")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(800, 600)

        self._state  = SessionState()
        self._thread: InferenceThread | None = None
        self._angle_rows: dict[str, AngleRow] = {}

        self._build_ui()
        self._poll()  # start the 100ms refresh loop

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── title bar ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(18, 8))
        tk.Label(hdr, text="KINEXIS", bg=BG, fg=ACCENT,
                 font=FONT_TITLE).pack(side="left")
        tk.Label(hdr, text=" // motion analysis", bg=BG, fg=FG_DIM,
                 font=FONT_LABEL).pack(side="left", pady=4)
        # status dot
        self._status_canvas = tk.Canvas(hdr, width=12, height=12, bg=BG,
                                         highlightthickness=0)
        self._status_canvas.pack(side="right", padx=(0, 4))
        self._status_dot = self._status_canvas.create_oval(
            1, 1, 11, 11, fill=FG_DIM, outline="")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20)

        # ── two-column body ───────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=14)
        body.columnconfigure(0, weight=3, minsize=300)
        body.columnconfigure(1, weight=2, minsize=260)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=BG)
        tk.Label(f, text=title, bg=BG, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w", pady=(0, 4))
        return f

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        # ── Source selection ──────────────────────────────────────────────────
        sec = self._section(left, "─── INPUT SOURCE")
        sec.pack(fill="x", pady=(0, 12))

        self._source_var = tk.StringVar(value="webcam")
        row = tk.Frame(sec, bg=BG)
        row.pack(fill="x")

        for text, val in [("Webcam (realtime)", "webcam"),
                           ("Video file",        "file")]:
            rb = tk.Radiobutton(row, text=text, variable=self._source_var,
                                value=val, bg=BG, fg=FG, selectcolor=BG,
                                activebackground=BG, activeforeground=ACCENT,
                                font=FONT_LABEL, command=self._on_source_change)
            rb.pack(side="left", padx=(0, 16))

        # webcam index
        wc_row = tk.Frame(sec, bg=BG)
        wc_row.pack(fill="x", pady=(6, 0))
        tk.Label(wc_row, text="Camera index:", bg=BG, fg=FG_MID,
                 font=FONT_SMALL).pack(side="left")
        self._cam_var = tk.IntVar(value=0)
        tk.Spinbox(wc_row, from_=0, to=9, textvariable=self._cam_var,
                   width=4, bg=CARD, fg=FG, insertbackground=FG,
                   buttonbackground=CARD, relief="flat",
                   font=FONT_SMALL).pack(side="left", padx=6)

        # file picker
        file_row = tk.Frame(sec, bg=BG)
        file_row.pack(fill="x", pady=(6, 0))
        self._file_path = tk.StringVar(value="")
        self._file_entry = tk.Entry(file_row, textvariable=self._file_path,
                                    bg=CARD, fg=FG, insertbackground=FG,
                                    relief="flat", font=FONT_SMALL, width=30,
                                    state="disabled")
        self._file_entry.pack(side="left")
        self._browse_btn = tk.Button(file_row, text="Browse…",
                                      bg=BORDER, fg=FG_MID,
                                      relief="flat", font=FONT_SMALL,
                                      activebackground=ACCENT2,
                                      activeforeground=FG,
                                      command=self._browse,
                                      state="disabled")
        self._browse_btn.pack(side="left", padx=(6, 0))

        # ── Optional settings ─────────────────────────────────────────────────
        sec2 = self._section(left, "─── OPTIONS")
        sec2.pack(fill="x", pady=(0, 12))

        opt_grid = tk.Frame(sec2, bg=BG)
        opt_grid.pack(fill="x")
        opt_grid.columnconfigure(1, weight=1)
        opt_grid.columnconfigure(3, weight=1)

        tk.Label(opt_grid, text="Ground truth:", bg=BG, fg=FG_MID,
                 font=FONT_SMALL).grid(row=0, column=0, sticky="w", pady=3)
        self._gt_var = tk.StringVar(value="auto-detect")
        gt_cb = ttk.Combobox(opt_grid, textvariable=self._gt_var,
                             values=EXERCISES, state="readonly",
                             font=FONT_SMALL, width=16)
        gt_cb.grid(row=0, column=1, sticky="w", padx=(8, 24), pady=3)
        self._style_combobox(gt_cb)

        tk.Label(opt_grid, text="Expected reps:", bg=BG, fg=FG_MID,
                 font=FONT_SMALL).grid(row=0, column=2, sticky="w", pady=3)
        self._reps_var = tk.StringVar(value="")
        tk.Entry(opt_grid, textvariable=self._reps_var, width=6,
                 bg=CARD, fg=FG, insertbackground=FG, relief="flat",
                 font=FONT_SMALL).grid(row=0, column=3, sticky="w", padx=(8, 0))

        tk.Label(opt_grid, text="Save output to:", bg=BG, fg=FG_MID,
                 font=FONT_SMALL).grid(row=1, column=0, sticky="w", pady=3)
        self._out_path = tk.StringVar(value="")
        tk.Entry(opt_grid, textvariable=self._out_path, width=28,
                 bg=CARD, fg=FG, insertbackground=FG, relief="flat",
                 font=FONT_SMALL).grid(row=1, column=1, columnspan=2,
                                        sticky="ew", padx=(8, 8), pady=10)
        tk.Button(opt_grid, text="…", bg=BORDER, fg=FG_MID, relief="flat",
                  font=FONT_SMALL, command=self._browse_out,
                  activebackground=ACCENT2).grid(row=1, column=3, sticky="w")

        tk.Label(opt_grid, text="Max frames:", bg=BG, fg=FG_MID,
                 font=FONT_SMALL).grid(row=2, column=0, sticky="w", pady=3)
        self._maxf_var = tk.StringVar(value="")
        tk.Entry(opt_grid, textvariable=self._maxf_var, width=8,
                 bg=CARD, fg=FG, insertbackground=FG, relief="flat",
                 font=FONT_SMALL).grid(row=2, column=1, sticky="w", padx=(8, 0))
        tk.Label(opt_grid, text="(blank = unlimited)", bg=BG, fg=FG_DIM,
                 font=FONT_SMALL).grid(row=2, column=2, columnspan=2,
                                        sticky="w", padx=(4, 0))

        # ── Control buttons ───────────────────────────────────────────────────
        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=(4, 0))

        self._start_btn = tk.Button(
            btn_row, text="▶  START SESSION", font=FONT_HEAD,
            bg=ACCENT, fg=BG, relief="flat", padx=16, pady=10,
            activebackground=ACCENT2, activeforeground=FG,
            command=self._start)
        self._start_btn.pack(side="left", padx=(0, 10))

        self._stop_btn = tk.Button(
            btn_row, text="■  STOP", font=FONT_HEAD,
            bg=BORDER, fg=FG_DIM, relief="flat", padx=16, pady=10,
            activebackground=BAD, activeforeground=FG,
            command=self._stop, state="disabled")
        self._stop_btn.pack(side="left")

        # ── Live metrics grid ─────────────────────────────────────────────────
        sec3 = self._section(left, "─── LIVE METRICS")
        sec3.pack(fill="x", pady=(16, 0))

        metrics_grid = tk.Frame(sec3, bg=BG)
        metrics_grid.pack(fill="x")
        for c in range(4):
            metrics_grid.columnconfigure(c, weight=1)

        self._mc_exercise  = MetricCard(metrics_grid, "EXERCISE")
        self._mc_exercise.grid(row=0, column=0, columnspan=2, sticky="nsew",
                               padx=(0, 4), pady=(0, 4))

        self._mc_conf  = MetricCard(metrics_grid, "CONFIDENCE", "%")
        self._mc_conf.grid(row=0, column=2, sticky="nsew", padx=(0, 4), pady=(0, 4))

        self._mc_reps  = MetricCard(metrics_grid, "REPS")
        self._mc_reps.grid(row=0, column=3, sticky="nsew", pady=(0, 4))

        self._mc_form  = MetricCard(metrics_grid, "FORM SCORE", "%")
        self._mc_form.grid(row=1, column=0, columnspan=2, sticky="nsew",
                           padx=(0, 4), pady=(0, 4))

        self._mc_state = MetricCard(metrics_grid, "REP STATE")
        self._mc_state.grid(row=1, column=2, sticky="nsew", padx=(0, 4), pady=(0, 4))

        self._mc_frames = MetricCard(metrics_grid, "FRAMES")
        self._mc_frames.grid(row=1, column=3, sticky="nsew", pady=(0, 4))

        # ── Feedback log ──────────────────────────────────────────────────────
        sec4 = self._section(left, "─── FORM FEEDBACK")
        sec4.pack(fill="x", pady=(12, 0))
        self._feedback_frame = tk.Frame(sec4, bg=CARD, bd=0,
                                         highlightthickness=1,
                                         highlightbackground=BORDER)
        self._feedback_frame.pack(fill="x")
        self._feedback_labels = []
        for _ in range(3):
            lbl = tk.Label(self._feedback_frame, text="", bg=CARD, fg=FG_MID,
                           font=FONT_SMALL, anchor="w", padx=10, pady=4)
            lbl.pack(fill="x")
            self._feedback_labels.append(lbl)

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        sec = self._section(right, "─── JOINT ANGLES & DEVIATIONS")
        sec.pack(fill="both", expand=True)

        # header row
        hdr = tk.Frame(sec, bg=CARD, bd=0, highlightthickness=1,
                        highlightbackground=BORDER)
        hdr.pack(fill="x")
        for txt, w in [("Joint", 22), ("Value", 10), ("Deviation", 10)]:
            tk.Label(hdr, text=txt, bg=CARD, fg=ACCENT, font=FONT_SMALL,
                     width=w, anchor="w").pack(side="left", padx=(10 if txt=="Joint" else 0, 0),
                                                pady=4)

        # scrollable area
        container = tk.Frame(sec, bg=CARD, bd=0, highlightthickness=1,
                              highlightbackground=BORDER)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg=CARD, highlightthickness=0)
        sb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self._angle_inner = tk.Frame(canvas, bg=CARD)
        canvas.create_window((0, 0), window=self._angle_inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._angle_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # create one row per angle key
        ALL_ANGLE_KEYS = [
            "left_elbow", "right_elbow", "left_knee", "right_knee",
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "left_wrist_hip_off", "right_wrist_hip_off",
            "left_wrist_chest", "right_wrist_chest",
            "left_wrist_above_sho", "right_wrist_above_sho",
            "_torso_incline",
        ]
        for key in ALL_ANGLE_KEYS:
            row = AngleRow(self._angle_inner, key)
            row.pack(fill="x", padx=8, pady=1)
            self._angle_rows[key] = row

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _style_combobox(cb):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TCombobox",
                         fieldbackground=CARD, background=CARD,
                         foreground=FG, selectbackground=CARD,
                         selectforeground=ACCENT,
                         arrowcolor=FG_DIM)

    def _on_source_change(self):
        is_file = self._source_var.get() == "file"
        self._file_entry.config(state="normal" if is_file else "disabled")
        self._browse_btn.config(state="normal" if is_file else "disabled")

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            initialdir="/home/b2l/Desktop/kinexis/videos",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")])
        if path:
            self._file_path.set(path)

    def _browse_out(self):
        path = filedialog.asksaveasfilename(
            title="Save output video",
            initialdir="/home/b2l/Desktop/kinexis/outputs",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")])
        if path:
            self._out_path.set(path)

    # ── Session control ───────────────────────────────────────────────────────
    def _start(self):
        # resolve source
        if self._source_var.get() == "webcam":
            source = self._cam_var.get()
        else:
            p = self._file_path.get().strip()
            if not p:
                messagebox.showerror("No file", "Please select a video file.")
                return
            source = p

        gt = self._gt_var.get()
        if gt == "auto-detect":
            gt = None

        reps_str = self._reps_var.get().strip()
        gt_reps  = int(reps_str) if reps_str.isdigit() else None

        maxf_str = self._maxf_var.get().strip()
        max_frames = int(maxf_str) if maxf_str.isdigit() else None

        out_path = self._out_path.get().strip() or None

        # reset state
        self._state.update(
            exercise="—", confidence=0.0, reps=0, rep_state="INIT",
            form_score=100.0, feedback=[], angles={}, deviations={},
            frame_count=0, error=None
        )

        self._thread = InferenceThread(
            state             = self._state,
            source            = source,
            output_path       = out_path,
            max_frames        = max_frames,
            ground_truth      = gt,
            ground_truth_reps = gt_reps,
        )
        self._thread.start()

        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal", bg=BAD, fg=FG)

    def _stop(self):
        if self._thread and self._thread.is_alive():
            self._thread.stop()
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled", bg=BORDER, fg=FG_DIM)

    # ── Poll loop (every 100 ms) ───────────────────────────────────────────────
    def _poll(self):
        snap = self._state.snapshot()

        # status dot
        running = snap["running"]
        self._status_canvas.itemconfig(
            self._status_dot, fill=GOOD if running else FG_DIM)

        # if session ended, re-enable start
        if not running and self._stop_btn["state"] == "normal":
            self._stop()
            err = snap.get("error")
            if err:
                messagebox.showerror("Inference error", err)

        # metric cards
        ex = snap["exercise"]
        self._mc_exercise.set(ex.replace("_", " ").upper() if ex else "—")
        conf = snap["confidence"]
        self._mc_conf.set(f"{conf*100:.0f}",
                          color=GOOD if conf >= 0.65 else WARN if conf >= 0.45 else BAD)
        self._mc_reps.set(snap["reps"], color=ACCENT)
        fs = snap["form_score"]
        self._mc_form.set(f"{fs:.0f}",
                          color=GOOD if fs >= 75 else WARN if fs >= 50 else BAD)
        self._mc_state.set(snap["rep_state"])
        self._mc_frames.set(snap["frame_count"])

        # feedback
        fb = snap["feedback"]
        sev_color = {"good": GOOD, "warning": WARN, "error": BAD}
        for i, lbl in enumerate(self._feedback_labels):
            if i < len(fb):
                sev, msg = fb[i]
                lbl.config(text=f"  {'●'} {msg}",
                           fg=sev_color.get(sev, FG_MID))
            else:
                lbl.config(text="", fg=FG_DIM)

        # joint angles
        angles    = snap["angles"]
        devs      = snap["deviations"]
        ratio_keys = {"left_wrist_hip_off", "right_wrist_hip_off",
                      "left_wrist_chest",   "right_wrist_chest",
                      "left_wrist_above_sho", "right_wrist_above_sho"}
        for key, row in self._angle_rows.items():
            if key in angles:
                row.update(angles[key],
                           deviation=devs.get(key),
                           is_ratio=key in ratio_keys)

        self.after(100, self._poll)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = KinexisApp()
    app.mainloop()
