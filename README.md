# Kinexis — Real-Time Exercise Tracking

Kinexis is a desktop app that uses your webcam or a video file to track exercise reps, analyse your form, and give you live coaching feedback — all in a single window.

---

## Requirements

- Python 3.8 or later
- A webcam (optional — you can also use pre-recorded video files)

---

## Installation

**1. Clone or download the project**, then open a terminal in the project folder.

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

> **Ubuntu/Debian users:** If you see an error about `tkinter`, run:
> ```bash
> sudo apt install python3-tk
> ```

---

## Starting the app

```bash
python app/ui_main.py
```

The dashboard opens immediately. Nothing loads until you press **START** — so startup is instant.

---

## Using the dashboard

### Step 1 — Choose your input source

In the **Input Source** panel, pick one of two options:

- **Webcam (realtime)** — streams directly from your camera. Use the number spinner to select your camera (0 is usually the built-in webcam; try 1 or 2 for external cameras).
- **Video file** — click **Browse…** to load a `.mp4`, `.avi`, `.mov`, or `.mkv` file.

---

### Step 2 — Configure options (all optional)

| Setting | What it does |
|---------|-------------|
| Ground truth | Lock the app to a specific exercise for more accurate tracking. Leave on *auto-detect* if you're not sure. |
| Expected reps | Enter your target rep count to get a rep-accuracy score at the end. |
| Save output to | Choose a file path to save an annotated video of your session. |
| Max frames | Set a frame limit to stop the session early (leave blank to run until the video ends or you press STOP). |

---

### Step 3 — Start your session

Click **▶ START SESSION**. The status dot turns green and the live metrics update in real time.

Click **■ STOP** at any time to end the session cleanly.

---

## Reading the metrics

### Live Metrics

| Card | What it shows |
|------|--------------|
| **EXERCISE** | The exercise Kinexis has detected |
| **CONFIDENCE** | How certain the model is — green ≥ 65 %, amber ≥ 45 %, red < 45 % |
| **REPS** | Your current rep count |
| **FORM SCORE** | How closely your movement matches ideal form — green ≥ 75 %, amber ≥ 50 %, red < 50 % |
| **REP STATE** | Which phase of the rep you're in: INIT → DOWN → UP |
| **FRAMES** | Total frames analysed so far |

### Form Feedback

Up to three coaching messages appear here as you move:

- 🟢 **Good** — your technique is correct
- 🟡 **Warning** — a minor deviation detected
- 🔴 **Error** — a significant form issue that should be corrected

### Joint Angles & Deviations

The right-hand column shows every joint angle being measured (elbows, knees, hips, shoulders, wrist position). An orange or red bar appears next to any angle that falls outside the ideal range for your current exercise.

---

## Folder layout

```
kinexis/
├── app/
│   └── ui_main.py      ← the app you launch
├── inference.py
├── models/
│   ├── best_model.pth
│   └── model_meta.json
├── videos/             ← place your test videos here
└── outputs/            ← saved session videos appear here
```

---

## Troubleshooting

**The app opens but no video appears**
Check that the correct camera index is selected, or that your video file path is valid.

**Confidence stays red the whole session**
Make sure the camera can see your full body. Step back so your head, torso, and legs are all visible.

**The app crashes on startup**
Run `pip install -r requirements.txt` again to make sure all dependencies are installed.

**Webcam not detected**
Try changing the camera index from 0 to 1 (or 2) in the spinner.
