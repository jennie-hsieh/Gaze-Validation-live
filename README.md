# Gaze-Validation-live
# Live comparison: CV vs Tobii vs EOG

Compare three horizontal gaze signals during recording and compute their correlations in real time:

1. **CV**: a webcam records the face, and MediaPipe detects the iris to track the horizontal pupil position.
2. **Tobii**: gaze-x from a Tobii Pro Nano eye tracker.
3. **EOG**: BioSemi horizontal EOG, HEOG = HR - HL (EXG5 - EXG1), band-passed 0.05-20 Hz.

All three streams are converted to the LSL clock, then plotted, saved to CSV, and used to compute sliding-window correlations.

[中文版 / Chinese version](README.md)

## Display

```
+----------------------+--------------------------------------+
|                      |  CV gaze-x       (webcam pupil)       |
|  live webcam view    +--------------------------------------+
|  (eye corners/iris   |  Tobii gaze-x                         |
|   marked, mirrored)  +--------------------------------------+
|                      |  HEOG (uV)                            |
+----------------------+--------------------------------------+
|  r(CV,EOG)  r(Tobii,EOG)  r(CV,Tobii)   valid fraction   REC  |
+---------------------------------------------------------------+
```

- All gaze-x values use the subject's point of view: **0 = left, 1 = right**. Looking right makes the curve go up.
- HEOG is positive when looking right.
- The traces on the right are a scrolling window; the x axis is seconds relative to now.

## Installation

Requires Python 3.10 or later.

```
python -m pip install -r requirements.txt
```

Notes:

- scipy is deliberately not used; filtering is implemented in plain numpy.
- Newer mediapipe versions (0.10.30 and later) no longer have `solutions`. The program then switches to the Tasks API automatically, and **the first run downloads `face_landmarker.task` (about 4 MB)** next to the script. If you are offline, download it manually and pass its path with `--face-model`:
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- If `tobii-research` is not installed or no eye tracker is found, the program prints a message and skips Tobii; the other two streams keep working.

## Quick start

Check the display and workflow first, without any hardware:

```
python gaze_validation_live.py --simulate
```

With real hardware, in this order:

1. Turn on the BioSemi LSL output and confirm the stream name is `BioSemi` (otherwise pass it with `--eeg-stream`).
2. Calibrate the Tobii with Tobii Pro Eye Tracker Manager (the program does not calibrate).
3. Make sure the webcam has a clear view of the participant's face.
4. Run:

```
python gaze_validation_live.py
```

On startup the terminal prints the LSL stream's channel list; check that `EXG1` and `EXG5` are present. Press **Esc** or **Q** to quit.

Start the BioSemi LSL output before this program. At startup the program waits only about 5 seconds for the EOG stream (then 3 more seconds looking for a stream of type EEG). If it is not found, the EOG row stays empty and you need to restart the program.

## Options

| Option | Default | Description |
|---|---|---|
| `--simulate` | off | Use simulated signals; no hardware needed |
| `--camera` | 0 | Webcam index |
| `--cv-delay` | 0.0 | Fixed webcam latency in seconds, subtracted from CV timestamps; typical webcams are about 0.05-0.10 |
| `--cv-gain` | 1.0 | Scale CV gaze-x around 0.5; affects the display scale only, not the correlations |
| `--invert-cv` | off | Add this if the CV curve goes down when looking right (camera mirrored by the driver) |
| `--face-model` | next to script | Path to `face_landmarker.task` (only needed with newer mediapipe) |
| `--no-tobii` | off | Do not use Tobii |
| `--tobii-address` | first found | Eye tracker address, e.g. `tobii-ttp://TPNA1-xxxx` |
| `--tobii-either-eye` | off | Compute gaze-x when either eye is valid; by default both eyes must be valid, matching the offline analysis |
| `--no-eog` | off | Do not use EOG |
| `--eeg-stream` | `BioSemi` | BioSemi LSL stream name |
| `--hr` | `EXG5` | Channel label (or 0-based index) of the HR electrode |
| `--hl` | `EXG1` | Channel label (or 0-based index) of the HL electrode |
| `--bp LOW HIGH` | `0.05 20` | HEOG band-pass frequencies (Hz) |
| `--window` | 10 | Length of the scrolling display window (seconds) |
| `--corr-window` | 5 | Length of the sliding correlation window (seconds) |
| `--corr-interval` | 0.5 | Correlation update interval (seconds) |
| `--out` | `recordings` | Root folder for saved files |
| `--no-record` | off | Do not save anything; display only |
| `--duration` | 0 | Auto-quit after this many seconds; 0 means run until you quit |

## Output files

Every run records by default into `recordings/<YYYYmmdd_HHMMSS>/`. All `lsl_time` values are seconds on the LSL clock, so the three streams can be aligned directly.

| File | Columns |
|---|---|
| `cv_gaze.csv` | `lsl_time, frame, gaze_x, ratio_right_eye, ratio_left_eye` |
| `tobii_gaze.csv` | `lsl_time, device_ts_us, l_x, l_y, l_valid, r_x, r_y, r_valid, gaze_x` |
| `eog.csv` | `lsl_time, hr_raw, hl_raw, heog_raw, heog_filt` |
| `corr.csv` | `lsl_time, r_cv_eog, r_tobii_eog, r_cv_tobii` |
| `meta.json` | The arguments of this run, plus the wall-clock and LSL time at start |

- In `cv_gaze.csv`, `gaze_x` includes `--cv-gain`; the raw per-eye ratios are in `ratio_*_eye`. It is NaN when no face is detected.
- In `eog.csv`, `heog_raw` is the unfiltered HR - HL; `heog_filt` is the real-time filtered result, with NaN for saturated samples.

## How it works

**CV gaze-x**: for each eye, take the position of the iris center between the two eye corners, average the two eyes, and convert to the subject's point of view (0 = left, 1 = right). The range is only about 0.4-0.6, much smaller than Tobii's, so correlations are comparable but the amplitudes cannot be compared directly with Tobii.

**Time alignment**
- CV: the LSL time is taken when each frame is read, then `--cv-delay` is subtracted.
- Tobii: `system_time_stamp` is converted to LSL time; the clock offset is the minimum over the most recent ~600 callbacks.
- EOG: the LSL inlet enables clock sync, dejitter and monotonize.

**EOG processing**: HEOG = HR - HL, passed through a causal 0.05-20 Hz band-pass (one second-order Butterworth high-pass and one low-pass). Samples with absolute value of 199999 or more are treated as electrode detachment or saturation: the filter holds the last valid value through them, and the output is set to NaN. Compared with the zero-phase FFT filtering used in the offline analysis, the live filter adds a little phase delay, roughly ten-odd milliseconds, mostly from the 20 Hz low-pass.

**Correlation**: computed every 0.5 s over the most recent 5 s. All three signals are interpolated onto a common 100 Hz grid, then Pearson r is computed. Gaps where neighbouring valid samples are more than 0.2 s apart (blinks, lost tracking) are treated as missing. When less than 40% of the data is valid, the display shows a dash. The bottom line also shows the valid fraction for each of the three comparisons.

## Troubleshooting

| Symptom | What to do |
|---|---|
| CV curve goes down when looking right | Add `--invert-cv` |
| CV always shows `no face` | Improve lighting and face position, or check the `--camera` index |
| `face_landmarker.task` not found / download fails | Download it manually and pass it with `--face-model` |
| Terminal says the EOG stream is not found | Make sure BioSemi LSL is on, set `--eeg-stream` if needed, and restart the program |
| Channel label not found | Check the channel list printed at startup, then use `--hr` / `--hl` (label or index) |
| Tobii row is empty | Check the eye tracker is connected, calibrated, and not held exclusively by another program |
| r shows a dash | Less than 40% valid data; check for tracking loss or detached electrodes |
| r between CV and EOG is low | The webcam has a fixed latency; adjust `--cv-delay` and see whether r rises |

## Current limitations

- The display, correlations and saving have been tested with simulated signals (`--simulate`) only. **It has not yet been tested with a real webcam, Tobii or BioSemi.**
- The newer mediapipe Tasks API path has not been verified on a real face. On first use, check that eye-corner and iris markers appear on the video.
- If you run it alongside the PsychoPy experiment, both programs subscribe to the same Tobii. The Tobii SDK is expected to allow multiple programs to read gaze data at once, but this is untested; try it first without running the experiment.
- The Tobii time conversion (`system_time_stamp` to LSL) has been tested on simulated data only. On first use with hardware, check that the lag between Tobii and EOG is reasonable with your offline cross-correlation analysis.
