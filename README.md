# Kinexis GUI — Setup & Usage Guide

A polished Tkinter popup that wraps your existing `inference.py` with a
live dashboard: video source selector, options, and real-time metrics.

---

## Final folder structure

```
kinexis/
├── inference.py          ← your existing file (patch it per INTEGRATION_PATCH.py)
├── requirements.txt
├── models/
│   ├── best_model.pth
│   └── model_meta.json
├── videos/               ← put test videos here
├── outputs/
├── tests/
│   └── test_dryrun.py
└── app/
    └── ui_main.py        ← the GUI launcher
```

---

## Install dependencies

Everything the GUI needs is already in your `requirements.txt`.
Tkinter ships with Python; no extra install needed.

```bash
pip install -r requirements.txt
```

> On Ubuntu/Debian, if `tkinter` is missing:
> ```bash
> sudo apt install python3-tk
> ```

---

## Run the GUI

```bash
# from the project root
python app/ui_main.py
```

The popup opens immediately — no model loading until you click **START**.

---

## GUI walkthrough

### Input Source panel
| Control | Purpose |
|---------|---------|
| **Webcam (realtime)** | Uses `cv2.VideoCapture(index)`. Set camera index (0, 1, …) in the spinner. |
| **Video file** | Click **Browse…** to pick any `.mp4 / .avi / .mov / .mkv`. |

### Options panel
| Field | Purpose |
|-------|---------|
| Ground truth | Lock a known exercise for accuracy tracking (or leave on *auto-detect*). |
| Expected reps | Feed your known rep count for rep-accuracy reporting. |
| Save output to | Optional path — annotated video written to disk. |
| Max frames | Stop early after N frames (blank = run until end / Q pressed). |

### Buttons
- **▶ START SESSION** — validates inputs, spawns the inference thread, lights the green status dot.
- **■ STOP** — sets the stop event; the inference loop exits cleanly after the current frame.

### Live Metrics panel (updates every 100 ms)
| Card | Description |
|------|-------------|
| EXERCISE | Detected exercise label |
| CONFIDENCE | Model confidence 0–100 % (green ≥65 %, amber ≥45 %, red <45 %) |
| REPS | Current rep count |
| FORM SCORE | Deviation from ideal angle ranges (green ≥75 %, amber ≥50 %, red <50 %) |
| REP STATE | INIT / DOWN / UP — which phase of the rep cycle |
| FRAMES | Total frames processed |

### Form Feedback panel
Up to 3 live coaching messages with severity colour:
- 🟢 good — correct technique
- 🟡 warning — minor deviation
- 🔴 error — significant form issue

### Joint Angles & Deviations panel (right column)
Every angle computed by `get_angles()` is listed:
- **Degrees** for elbow / knee / hip / shoulder
- **Ratio** (dimensionless) for wrist-hip offset, wrist-chest distance, wrist-above-shoulder

The orange/red bar shows how far each angle deviates from the ideal range
for the current exercise (defined in `IDEAL_ANGLES` in inference.py).

---

## Next steps (optional)

- **Headless mode** — skip `cv2.imshow` when `gui_state` is not None so
  the OpenCV window doesn't appear alongside the Tkinter window.
- **Session history** — log each session's metrics to a JSON file and
  add a "History" tab that plots reps and form score over time.
- **Rep timeline chart** — embed a small `matplotlib` figure (or use
  `canvas` drawing) to show reps-per-minute as a sparkline.
- **Audio cues** — use `winsound` / `playsound` to beep on each rep or
  when form drops below 50 %.
- **Packaging** — `pyinstaller --onefile app/ui_main.py` bundles
  everything into a single executable.
