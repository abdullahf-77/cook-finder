# Cook Finder

An ingredient-recognition recipe recommender. Point a camera or upload a
photo of your ingredients, and Cook Finder detects them with a custom-trained
YOLO model and suggests matching recipes ranked by how many ingredients you
already have.

## Overview

Cook Finder is a two-part application:

- **Backend** (`main.py`) — a FastAPI service that runs a fine-tuned
  Ultralytics YOLO model (87 ingredient classes) over an uploaded image,
  returns the detected ingredients plus an annotated preview image, and
  matches them against a local recipe database.
- **Frontend** (`CookFinder_Website.html`) — a single-page, right-to-left
  Arabic UI that captures/uploads a photo and displays the matched recipes
  returned by the API.

## Features

- Real-time ingredient detection from a photo via a custom YOLO model (`last.pt`, 87 classes)
- Annotated result image (bounding boxes) returned as base64 alongside detections
- Recipe matching engine that scores 200 recipes by ingredient overlap
- Ingredient alias/normalization map (e.g. matches "Chili Pepper (Khursani)" to "chili", "pepper")
- Simple REST API: `/predict`, `/health`, `/classes`
- Standalone HTML/CSS/JS frontend — no build step required
- Recipe set spans multiple cuisines (Nepali, Indian, Asian, Western, Arabian, Fusion, and more)

## Tech Stack

| Category   | Technology                          |
|------------|--------------------------------------|
| Backend    | Python, FastAPI, Uvicorn             |
| ML / CV    | Ultralytics YOLO, OpenCV (headless), NumPy |
| Frontend   | HTML, CSS, vanilla JavaScript (RTL, Arabic) |
| Data       | Local JSON recipe database (200 recipes) |

## Architecture

```
Browser (CookFinder_Website.html)
        │  photo upload
        ▼
FastAPI backend (main.py)
        │
        ├── Ultralytics YOLO (last.pt) ──► detected ingredient classes
        │
        └── Recipe matcher ──► scores recipes.json against detections
        │
        ▼
JSON response: detected ingredients + annotated image + ranked recipes
```

## Folder Structure

```
CookFinder/
├── main.py                  # FastAPI backend + YOLO inference + recipe matching
├── CookFinder_Website.html  # Frontend (Arabic, RTL)
├── recipes.json             # Recipe database (200 recipes)
├── last.pt                  # Trained YOLO model weights (87 ingredient classes)
├── requirements.txt         # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

- Python 3.9+
- A modern browser (for the frontend)

## Installation

```bash
git clone <this-repo-url>
cd CookFinder
pip install -r requirements.txt
```

## Configuration

The backend reads optional environment variables (all have sensible defaults):

| Variable       | Default          | Description                              |
|----------------|-------------------|-------------------------------------------|
| `MODEL_PATH`   | `last.pt`         | Path to the YOLO weights file             |
| `RECIPES_PATH` | `recipes.json`    | Path to the recipe database               |
| `CONF_THRESH`  | `0.25`            | YOLO detection confidence threshold        |
| `MAX_RECIPES`  | `6`               | Max number of recipes returned per request |

## Usage

1. Start the backend:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Open `CookFinder_Website.html` in a browser (double-click it, or serve it
   with any static file server). By default the frontend talks to the API at
   `http://localhost:8000`.

3. Upload or capture a photo of your ingredients and view the matched recipes.

## API Documentation

### `POST /predict`

Multipart form upload, field name `image`. Returns:

```json
{
  "detected": ["Onion", "Tomato", "..."],
  "annotated_image": "<base64 JPEG>",
  "recipes": [
    {
      "name": "Dal Bhat",
      "cuisine": "indian",
      "category": "Main Course",
      "total_time": "45",
      "servings": 4,
      "description": "...",
      "ingredients": ["..."],
      "steps": ["..."],
      "matched": ["..."],
      "match_count": 3,
      "score": 62
    }
  ]
}
```

### `GET /health`

Returns service status plus loaded recipe/class counts.

### `GET /classes`

Returns the full list of ingredient classes the model can detect.

## Future Improvements

- Make the frontend's backend URL configurable instead of hardcoded `localhost:8000`.
- Containerize the backend with Docker for easier deployment.
- Add automated tests for the recipe-matching logic.
- Serve the frontend and backend from the same origin (or add a small static file route) to avoid CORS entirely.
- Add pagination/filtering (cuisine, prep time) to the recipe results.

## Screenshots

_Add UI screenshots / example detections here._

```
docs/
└── ui_preview.png
└── detection_example.png
```

## License

Distributed under the [MIT License](./LICENSE).

## Author

**Abdullah**
