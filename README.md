# Computer Vision-Based Indoor Navigation

## Overview
This project explores indoor navigation using computer vision, object detection, and visual landmark matching. It combines YOLO-based object detection with ORB feature matching and RANSAC geometric verification to identify indoor landmarks from uploaded images.

The current implementation supports image upload, indoor object detection, and landmark localization against stored reference images. Route planning and turn-by-turn navigation are planned future enhancements.

## Features
- Upload indoor images through a React web interface.
- Detect indoor objects using YOLO models.
- Display detected object labels and confidence scores.
- Compare uploaded images with stored indoor landmark images using ORB features.
- Verify feature correspondences using RANSAC.
- Display the predicted landmark, feature-match counts, verified matches, inlier ratio, and visual match results.
- REST API built using FastAPI.
- Interactive frontend built using React and Vite.

## Methodology

### Step 1 — Image Input
The user uploads an indoor image through the web interface. The current implementation uses uploaded images instead of live webcam or phone-camera frames.

### Step 2 — YOLO Object Detection
YOLO processes the uploaded image and detects objects such as chairs, windows, doors, and tables. The application returns object labels and confidence scores.

### Step 3 — ORB Feature Matching and Verification
ORB extracts visual features from the uploaded image and stored landmark reference images. Feature matching identifies candidate correspondences, and RANSAC performs geometric verification to reject inconsistent matches.

The system ranks reference landmarks and reports the best match when the configured verification thresholds are satisfied.

**Steps 4–7 are planned enhancements:** predefined-map localization, weighted graph construction, Dijkstra route planning, and navigation directions with obstacle alerts.

## Models
1. **Kaggle Indoor Model:** Detects indoor object classes supported by the trained model, including doors, windows, chairs, and tables.
2. **Original Indoor Model:** Uses the original indoor object detection model.
3. **YOLOv8 General Model:** Uses a pretrained general-purpose object detection model.

The exact classes depend on the selected model's trained or pretrained class list.

## Technology Stack
- Python
- FastAPI
- Ultralytics YOLO
- OpenCV
- ORB feature detection and matching
- RANSAC geometric verification
- NumPy
- Pillow
- React
- Vite
- JavaScript and CSS

## Project Structure
```text
indoor-navigation/
├── backend/
│   ├── main.py
│   ├── localization.py
│   ├── landmarks/
│   │   ├── entrance.jpeg
│   │   ├── openspace.jpeg
│   │   └── studyarea.jpeg
│   ├── requirements.txt
│   ├── best.pt
│   ├── best_old.pt
│   └── yolov8n.pt
├── web/
│   ├── src/
│   │   ├── App.jsx
│   │   └── App.css
│   ├── package.json
│   └── ...
├── .gitignore
└── README.md
```

## Run Locally

### Prerequisites
- Python 3.11 recommended
- Node.js and npm
- Git

### Backend

Open a terminal in the project root:

```bash
cd backend
```

Create and activate a virtual environment if you do not already have one:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
python -m uvicorn main:app --reload --port 8000
```

The backend API is available at `http://127.0.0.1:8000`.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Frontend

Open a second terminal in the project root:

```bash
cd web
npm install
npm run dev
```

Open the local URL displayed by Vite, commonly `http://localhost:5173`. If that port is already occupied, Vite may select another port.

Keep both the backend and frontend running while using the application.

## Landmark Reference Images
The `backend/landmarks/` directory contains the reference images used by the localization endpoint. The system compares uploaded query images against these references.

Localization quality depends on reference image coverage, image quality, viewpoint, lighting, and visual similarity. Using the same or nearly identical image as a reference is not sufficient to establish general localization accuracy.

## Current Limitations
- The current interface accepts uploaded images rather than continuous live camera frames.
- Localization is limited to the stored reference landmarks.
- ORB and RANSAC matching does not provide precise coordinates or continuous indoor tracking.
- Object detection confidence and performance vary with image conditions and object classes.
- Detected objects are not yet integrated into route planning or obstacle avoidance.
- A predefined indoor map, weighted walkway graph, Dijkstra route planning, and navigation instructions have not yet been implemented.
- This project is a prototype and is not intended to be used as a reliable standalone navigation system.

## Future Work
1. Estimate the user's location on a predefined indoor map.
2. Represent indoor locations and walkways as graph nodes and weighted edges.
3. Implement Dijkstra's algorithm for route planning.
4. Display navigation directions and relevant obstacle alerts.
5. Evaluate localization and object detection performance using different images and viewpoints.

## Dataset and Model Weights
Training datasets are not included because of their size. Model weights are required for the corresponding YOLO detection options. Ensure the required model files are available in the backend directory before running the application.
