# Tennis Analyzer

An AI-powered tennis match analysis system that tracks players and ball movements, detects court lines, calculates player statistics, and visualizes everything on a mini court overlay.

## Features

- **Player Tracking**: Detects and tracks tennis players throughout the match using YOLOv8
- **Ball Tracking**: Custom-trained YOLO model for tennis ball detection with interpolation for smooth tracking
- **Court Line Detection**: Identifies court keypoints and lines for accurate position mapping
- **Mini Court Visualization**: Real-time top-down view showing player and ball positions
- **Player Statistics**: Calculates and displays:
  - Shot speed (ball velocity in km/h)
  - Player movement speed
  - Average shot speed per player
  - Number of shots per player
  - Last shot and movement speeds

## Output Video

![Tennis Analysis Demo](output_videos/demo.gif)

> **Note**: Convert a sample clip to GIF using: `ffmpeg -i output_video.avi -vf "fps=10,scale=640:-1" -t 10 demo.gif`

The output video includes:
- Bounding boxes around players and ball
- Court keypoint detection overlays
- Mini court visualization with player/ball positions
- Real-time player statistics overlay
- Frame counter

## Project Structure

```
Tennis_analyzer/
├── main.py                      # Main execution script
├── constants.py                 # Configuration constants
├── utils/                       # Utility functions
├── trackers/                    # Player and ball tracking modules
│   ├── player_tracker.py
│   └── ball_tracker.py
├── court_line_detector/         # Court detection module
├── mini_court/                  # Mini court visualization
│   └── mini_court.py
├── models/                      # Trained models (gitignored)
│   ├── yolov8x.pt              # Player detection model
│   ├── training_runs_detect_train5_weights_last.pt  # Ball detection
│   └── training_keypoint_model.pth  # Court line detection
├── input_videos/                # Input tennis videos
├── output_videos/               # Generated output videos
├── tracker_stubs/               # Cached detection results (.pkl)
└── training/                    # Model training notebooks

```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install ultralytics opencv-python pandas numpy torch
```

3. Ensure you have the trained models in the `models/` directory

## Usage

```bash
python main.py
```

The script will:
1. Read the input video from `input_videos/input_video.mp4`
2. Detect players and ball across all frames
3. Identify court lines and keypoints
4. Calculate player statistics
5. Generate annotated output video at `output_videos/output_video.avi`

### Using Cached Detections

By default, the script uses cached detections stored in `tracker_stubs/` for faster processing. To force re-detection, set `read_from_stubs=False` in main.py.

## Technical Details

### Player Detection
- Uses YOLOv8x pretrained model
- Filters to 2 main players using court position analysis
- Tracks foot positions for accurate court mapping

### Ball Detection
- Custom-trained YOLO model on tennis ball dataset
- Interpolates missing detections for smooth tracking
- Detects ball hit frames using trajectory analysis

### Court Line Detection
- Deep learning model for keypoint detection
- Identifies 14 key court points (corners, service lines, etc.)
- Used for perspective transformation to mini court

### Statistics Calculation
- Ball speed: Distance traveled between shots / time (converted to km/h)
- Player speed: Movement distance / time between shots
- Calculated using mini court coordinates for real-world accuracy
- Forward-filled across all frames for consistent display

## Configuration

Edit `constants.py` to adjust:
- Court dimensions (in meters)
- Player heights
- Frame rate
- Model paths

## Dependencies

- Python 3.8+
- ultralytics (YOLO)
- opencv-python
- pandas
- numpy
- torch

## Performance Notes

- First run: Detections are cached to `.pkl` files
- Subsequent runs: Uses cached detections for 10x faster processing
- Processing time: ~2-5 minutes for a 1-minute video (with caching)

## License

MIT License

## Acknowledgments

- YOLOv8 by Ultralytics
- Tennis ball detection dataset
- Court line detection model training
