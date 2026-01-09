# The remaining files of yolo with .pt extension are availabel at (https://drive.google.com/drive/folders/1OFDtYIZjsVFBLCr0bpExHNX6cWRdZT6S?usp=drive_link) download them and arrange them based on the file structure below


# Advanced Lane & Object Detection System

## Project Overview
A comprehensive computer vision application for real-time lane detection and object tracking in road scenes. The system combines lane marking detection with YOLO-based object detection and tracking for autonomous driving assistance and traffic analysis.

## Features
- **Lane Detection**: Advanced algorithms for detecting road lanes and markings
- **Object Detection**: YOLOv8 models for detecting vehicles, pedestrians, traffic signs, and other road objects
- **Object Tracking**: ByteTrack algorithm for robust multi-object tracking
- **Multiple Processing Modes**: Image, video, and live webcam processing
- **3D Visualization**: Interactive 3D view of detected objects in road scenes
- **Performance Monitoring**: Real-time processing statistics and performance metrics
- **User-Friendly GUI**: PyQt5-based interface with dark theme

## Project Structure
```
Lane and object detection Final/
├── main.py                    # Main application file
├── project.py                 # Additional project code
├── test_images/               # Directory for test images
├── __pycache__/              # Python cache files
├── .idea/                    # IDE configuration
├── .vscode/                  # VS Code configuration
├── CameraCalibration.py      # Camera calibration module
├── Thresholding.py          # Image thresholding module
├── PerspectiveTransformation.py # Perspective transformation module
├── LaneLines.py             # Lane line detection module
├── yolov8n.pt              # YOLOv8 Nano model
├── yolov8s.pt              # YOLOv8 Small model
├── yolov8l.pt              # YOLOv8 Large model
├── yolov8x.pt              # YOLOv8 X-Large model
├── yolov8x6.pt             # YOLOv8 Ultra-Large model
├── yolov5s.pt              # YOLOv5 Small model
├── yolov8n-seg.pt          # YOLOv8 Nano segmentation model
├── left_turn.png           # Sample images
├── right_turn.png
├── straight.png
└── tmpkojl966u            # Temporary file
```

## Prerequisites

### Python Version
Python 3.8 or higher

### Required Libraries
Install the required packages using:

```bash
pip install numpy opencv-python pillow matplotlib scikit-learn pyqt5 qdarkstyle moviepy docopt ultralytics
```

### Additional Dependencies
For optimal performance, ensure you have:
- CUDA-compatible GPU (optional, for faster processing)
- Webcam (for live feed processing)

## Installation

1. Clone or extract the project to your desired directory
2. Install the required dependencies:
```bash
cd "Lane and object detection Final"
pip install -r requirements.txt
```

*Note: If you don't have a requirements.txt file, use the pip install command above with all the listed libraries.*

## Running the Application

### Method 1: GUI Application (Recommended)
Run the main application with GUI interface:
```bash
python main.py
```

### Method 2: Command Line Interface
For batch processing without GUI:
```bash
python main.py INPUT_PATH OUTPUT_PATH [options]
```

#### Command Line Options:
- `INPUT_PATH`: Path to input image or video
- `OUTPUT_PATH`: Path for output file
- `--video`: Process as video file
- `--tracking TRACKING`: Tracking method (default: bytetrack)
- `--model MODEL`: Model size (n, s, m, l, x, ul)
- `--conf CONF`: Confidence threshold (0.1-1.0)

#### Example command:
```bash
python main.py test_image.jpg output.jpg --model l --conf 0.6
```

## Usage Instructions

### GUI Application Workflow:
1. **Launch Application**: Run `python main.py`
2. **Select Mode**: Choose Image or Video processing mode
3. **Open File**: Click "Open File" and select an image/video
4. **Configure Settings**:
   - Select model size (Nano to Ultra Large)
   - Adjust confidence threshold
5. **Process**: Click "Process" button
6. **Save Output**: Save processed results when complete
7. **Optional**: Use webcam mode for live processing

### Model Selection Guide:
- **Nano (yolov8n.pt)**: Fastest, least accurate - for low-power devices
- **Small (yolov8s.pt)**: Balanced speed/accuracy - recommended for most uses
- **Large (yolov8l.pt)**: High accuracy - default choice
- **X-Large (yolov8x.pt)**: Maximum accuracy - for critical applications
- **Ultra Large (yolov8x6.pt)**: Extended detection - requires significant resources

## File Organization

### Input Files:
Place your input images/videos in the `test_images` directory or use the "Open File" dialog.

### Default Directories:
The application creates default directories at:
- Input: `Documents/LaneAndObjectDetection/input/`
- Output: `Documents/LaneAndObjectDetection/output/`

### Model Files:
YOLO model files (.pt) should remain in the project root directory.

## Troubleshooting

### Common Issues:

1. **"Module not found" errors**:
   ```bash
   pip install missing-module-name
   ```

2. **Slow performance**:
   - Use smaller model (Nano or Small)
   - Reduce video resolution
   - Ensure GPU acceleration is available

3. **Webcam not working**:
   - Check webcam permissions
   - Verify webcam is not being used by another application

4. **Memory errors**:
   - Use smaller model size
   - Process smaller images/videos
   - Close other memory-intensive applications

### System Requirements:
- **Minimum**: 4GB RAM, Dual-core processor
- **Recommended**: 8GB RAM, Quad-core processor, GPU with 4GB VRAM
- **Optimal**: 16GB RAM, GPU with 8GB+ VRAM for large models

## Notes

- The application automatically creates necessary directories on first run
- Processing times vary based on model size and input resolution
- For best results with lane detection, use clear road images with visible lane markings
- The system supports various video formats (MP4, AVI, MOV)
- Output is saved with lane overlays and bounding boxes around detected objects

## Performance Tips

1. **For real-time processing**: Use Nano or Small models
2. **For maximum accuracy**: Use X-Large or Ultra Large models
3. **Adjust confidence threshold**: Higher values reduce false positives
4. **Use appropriate input resolution**: HD (1280x720) provides good balance

