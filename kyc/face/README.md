# 🎯 Face Platform

A production-quality Python platform for **face detection**, **68-point landmark extraction**, and **face recognition** — supporting every common image format.

---

## Features

| Capability | Detail |
|---|---|
| **Detection** | Multi-scale Haar Cascade + profile detection with NMS |
| **Landmarks** | 68-point LBF model (jaw, brows, eyes, nose, lips) |
| **MediaPipe landmarks** | 478-point FaceLandmarker model with iris landmarks |
| **Metrics** | Eye distance, face width, yaw angle, mouth-open ratio |
| **Recognition** | Fusion of LBPH + HOG-cosine (no GPU, no DL framework needed) |
| **Image formats** | JPEG, PNG, BMP, TIFF, WEBP, GIF, HEIC, RAW (NEF/CR2/ARW…) |
| **Output** | Annotated image, JSON report, Python API |

---

## Quick Start

```bash
make -C kyc/face setup
```

### Analyze an image (detect + landmarks)
```bash
python -m kyc.face.cli --landmark-mode mediapipe \
    --mediapipe-model kyc/models/face_landmarker.task \
    analyze photo.jpg \
    --output annotated.jpg \
    --json result.json \
    --metrics
```

With Make:

```bash
make -C kyc/face analyze IMAGE=tests/obama1.jpg
```

### Detect faces and crop them

Draw square boxes:

```bash
python -m kyc.face.cli --landmark-mode mediapipe detect kyc/face/tests/obama1.jpg \
    --overlay boxes \
    --output /tmp/obama_boxes.jpg \
    --json /tmp/obama_faces.json
```

By default the CLI uses YuNet when `kyc/models/face_detection_yunet_2023mar.onnx` is available. That is much stricter than the Haar/multiscale fallback and avoids common false positives. To force another detector, pass `--detection-mode multiscale` or `--detection-mode haar`.

Draw landmark points and square boxes:

```bash
python -m kyc.face.cli --landmark-mode mediapipe detect kyc/face/tests/obama1.jpg \
    --overlay both \
    --output /tmp/obama_points_boxes.jpg
```

Crop detected faces:

```bash
python -m kyc.face.cli detect kyc/face/tests/obama1.jpg \
    --overlay none \
    --crop-dir /tmp/face_crops
```

With Make:

```bash
make -C kyc/face detect IMAGE=tests/obama1.jpg OVERLAY=both
make -C kyc/face crop IMAGE=tests/obama1.jpg CROP_DIR=/tmp/face_crops
```

### Detect all images and store landmark points

Process every image in a folder:

```bash
python -m kyc.face.cli --landmark-mode mediapipe detect-all kyc/face/tests \
    --output-dir /tmp/face_points \
    --overlay both \
    --crop
```

Output layout:

```text
/tmp/face_points/
├── points/      # one *.faces.json file per source image, including 478-point landmarks
├── overlays/    # box/point overlays
├── crops/       # cropped faces grouped by source image
└── summary.json
```

With Make:

```bash
make -C kyc/face detect-all FOLDER=tests POINTS_DIR=/tmp/face_points OVERLAY=both
```

### Search for a face in a folder

```bash
make -C kyc/face search QUERY=tests/test-1.webp FOLDER=tests
```

Direct CLI:

```bash
python -m kyc.face.cli search kyc/face/tests/test-1.webp kyc/face/tests \
    --overlay-dir /tmp/search_overlays \
    --crop-dir /tmp/search_crops \
    --verbose \
    --json /tmp/search.json
```

Search an explicit image list:

```bash
python -m kyc.face.cli search kyc/face/tests/test-1.webp /path/to/images.txt
```

Verbose mode explains each decision:

```text
MATCH/no_match
cosine score and threshold margin
closest landmark regions with facial-position details
most different landmark regions with facial-position details
geometry-only note for features such as moles or skin marks
```

The SFace cosine score decides the identity match. Landmark differences are shown as supporting context, because expression, pose, and crop quality can move mouth/eye/nose/chin/face-oval points even for the same person. Texture-level features such as moles are not facial landmarks, so verbose mode calls them out as not measured instead of treating them as proof.

`images.txt` should contain one image path per line. Relative paths are resolved relative to the list file first, then relative to the current working directory.

### Enroll a person and recognize them
```bash
# Enroll from a folder of photos
python -m kyc.face.cli enroll "Alice" photos/alice/ --db mydb

# Recognize in a new photo
python -m kyc.face.cli recognize group_photo.jpg --db mydb --output out.jpg
```

### Batch-process a folder
```bash
python -m kyc.face.cli batch photos/ --output-dir results/
```

---

## Python API

```python
from kyc.face import FacePlatform

# Initialize
platform = FacePlatform(
    mediapipe_model_path="kyc/models/face_landmarker.task",  # 478-point MediaPipe
    lbf_model_path="kyc/models/lbfmodel.yaml",   # 68-point landmarks
    detection_mode="multiscale",       # 'haar' | 'multiscale' | 'yunet'
    landmark_mode="auto",              # 'auto' | 'mediapipe' | 'lbf' | 'region'
    recognition_enabled=True,
)

# ── Enrollment ──────────────────────────────────────────────────────────────
platform.enroll_from_folder("Alice", "photos/alice/")
platform.enroll_from_image("Bob", "bob_headshot.jpg")
platform.save_database("faces_db")    # persist for reuse

# ── Analysis ─────────────────────────────────────────────────────────────────
result = platform.analyze(
    "group_photo.jpg",
    save_annotated="output.jpg",
    draw_metrics=True,
)

result.print_summary()

# ── Per-face results ─────────────────────────────────────────────────────────
for i, face in enumerate(result.faces):
    print(f"Face #{i+1}  bbox={face.bbox}  confidence={face.confidence:.2f}")

    if i < len(result.landmarks):
        lm = result.landmarks[i]
        print(f"  {len(lm.points)} landmarks detected ({lm.mode} mode)")
        print(f"  Eye distance  : {lm.eye_distance():.1f} px")
        print(f"  Face width    : {lm.face_width():.1f} px")
        print(f"  Yaw estimate  : {lm.yaw_estimate():+.1f}°")
        print(f"  Mouth open    : {lm.mouth_open_ratio():.2f}")

        # Access specific landmark groups (68-point mode)
        right_eye_pts = lm.groups["right_eye"]   # shape (6, 2)
        left_eye_pts  = lm.groups["left_eye"]
        jaw_pts       = lm.groups["jaw"]          # shape (17, 2)
        lip_pts       = lm.groups["outer_lips"]   # shape (12, 2)
        if lm.mode == "mediapipe_478":
            right_iris = lm.groups["right_iris"]  # shape (5, 2)
            left_iris  = lm.groups["left_iris"]   # shape (5, 2)

    if i < len(result.recognitions):
        rec = result.recognitions[i]
        print(f"  Identity      : {rec.label}  ({rec.confidence:.1%})")

# ── Raw JSON export ──────────────────────────────────────────────────────────
import json
print(json.dumps(result.to_dict(), indent=2))

# ── Batch processing ─────────────────────────────────────────────────────────
results = platform.analyze_batch(
    ["photo1.jpg", "photo2.png", "photo3.heic"],
    output_folder="results/",
    draw_metrics=True,
)
```

---

## Landmark Groups (68-point)

```
Points 0–16   → jaw line (17 pts)
Points 17–21  → right eyebrow (5 pts)
Points 22–26  → left eyebrow (5 pts)
Points 27–30  → nose bridge (4 pts)
Points 31–35  → nose tip (5 pts)
Points 36–41  → right eye (6 pts)
Points 42–47  → left eye (6 pts)
Points 48–59  → outer lips (12 pts)
Points 60–67  → inner lips (8 pts)
```

---

## Architecture

```
face/
├── engine.py         ← FacePlatform — main entry point
├── detector.py       ← Multi-scale face detection (Haar + NMS)
├── landmarks.py      ← LBF 68-pt + region-based landmark extraction
├── recognizer.py     ← LBPH + HOG fusion recognizer
├── image_loader.py   ← Universal image loader (all formats)
├── visualizer.py     ← Annotated image rendering
├── cli.py            ← Command-line interface
└── demo.py           ← Demo & test script
```

---

## Supported Image Formats

| Format | Extension | Notes |
|---|---|---|
| JPEG | `.jpg`, `.jpeg` | All variants |
| PNG | `.png` | With transparency |
| BMP | `.bmp` | |
| TIFF | `.tiff`, `.tif` | Multi-page |
| WebP | `.webp` | |
| GIF | `.gif` | First frame |
| HEIC/HEIF | `.heic`, `.heif` | Requires `pillow-heif` |
| Camera RAW | `.nef`, `.cr2`, `.arw`, `.dng` | Requires `rawpy` |

---

## Optional Enhanced Model (YuNet)

For even higher detection accuracy, download the YuNet ONNX model:

```bash
curl -L "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
     -o yunet.onnx
```

```python
platform = FacePlatform(
    yunet_model_path="yunet.onnx",
    detection_mode="yunet",
)
```

---

## Recognition Accuracy

The fusion recognizer (LBPH + HOG cosine) achieves the best results when:
- ✅ At least 5 enrollment images per person
- ✅ Varied lighting in enrollment photos
- ✅ Clear, unobstructed face in the query image
- ✅ Face resolution ≥ 80×80 pixels

> **Note:** No face recognition system is literally "100% accurate" under all conditions — that applies to human vision too. This platform maximizes accuracy using an ensemble approach with per-method confidence scores so you can threshold appropriately for your use case.

---

## Requirements

```
opencv-python>=4.8
opencv-contrib-python>=4.8   # for LBF facemark and LBPH
Pillow>=9.0
numpy>=1.24
```
