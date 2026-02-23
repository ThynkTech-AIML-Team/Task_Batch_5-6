# Image Processing Task (OpenCV)

## What this project does
- Loads an image from this folder
- Runs Canny edge detection
- Applies binary and adaptive thresholding
- Performs augmentations: horizontal flip, vertical flip, rotate 90/180, brightness increase/decrease
- Displays all results using matplotlib with titles

## Requirements
- Python 3.9+
- `opencv-python`
- `matplotlib`
- `numpy`

## Install
```bash
pip install opencv-python matplotlib numpy
```

## Run
```bash
python image_processing_opencv.py
```

## Notes
- Keep at least one image file (`.jpg`, `.png`, `.jpeg`, etc.) in this same folder.
- The script automatically picks the first image file it finds.
- You can adjust Canny and brightness values at the top of `image_processing_opencv.py`.
