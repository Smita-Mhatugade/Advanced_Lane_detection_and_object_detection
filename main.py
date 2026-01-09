import numpy as np
import cv2
import sys
import os
import time
import json
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
from docopt import docopt
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                            QWidget, QPushButton, QFileDialog, QComboBox, QSlider,
                            QProgressBar, QGroupBox, QTabWidget, QSizePolicy, QFrame,
                            QStackedWidget, QGraphicsDropShadowEffect, QSpacerItem, 
                            QTextEdit, QScrollArea, QDockWidget, QMenuBar, QMenu,
                            QMessageBox, QAction, QDialog, QStyle)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QSize, QObject, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor, QPalette, QLinearGradient, QPainter
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
import qdarkstyle

DEFAULT_DIR = Path.home() / "Documents" / "LaneAndObjectDetection"
DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
(DEFAULT_DIR / "input").mkdir(exist_ok=True)
(DEFAULT_DIR / "output").mkdir(exist_ok=True)

class VideoProcessor(QThread):
    update_progress = pyqtSignal(int)
    processing_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    frame_processed = pyqtSignal(np.ndarray, list)

    def __init__(self, processor, input_path, output_path):
        super().__init__()
        self.processor = processor
        self.input_path = input_path
        self.output_path = output_path
        self._is_running = True

    def run(self):
        try:
            clip = VideoFileClip(self.input_path)
            total_frames = int(clip.fps * clip.duration)
            processed_frames = 0
            
            def process_frame_with_progress(frame):
                nonlocal processed_frames
                if not self._is_running:
                    return frame
                
                processed_frames += 1
                progress = int((processed_frames / total_frames) * 100)
                self.update_progress.emit(progress)
                
                processed_frame = self.processor.process_frame(frame)
                tracks = list(self.processor.tracker.tracks.values())
                self.frame_processed.emit(processed_frame, tracks)
                return processed_frame
            
            out_clip = clip.fl_image(process_frame_with_progress)
            out_clip.write_videofile(self.output_path, audio=False, threads=4, verbose=False)
            self.processing_complete.emit(self.output_path)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self._is_running = False
        self.wait()

class VideoPlayer(QThread):
    frame_ready = pyqtSignal(np.ndarray, list)

    def __init__(self, video_path, processor):
        super().__init__()
        self.video_path = video_path
        self.processor = processor
        self._is_running = True
        self._speed = 1.0

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps) / self._speed
        
        while self._is_running and cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.processor.process_frame(frame)
            tracks = list(self.processor.tracker.tracks.values())
            self.frame_ready.emit(processed_frame, tracks)
            
            elapsed = (time.time() - start_time) * 1000
            sleep_time = max(0, (frame_delay - elapsed) / 1000)
            time.sleep(sleep_time)
            
        cap.release()

    def set_speed(self, speed):
        self._speed = speed

    def stop(self):
        self._is_running = False
        self.wait()

class BaseTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.colors = np.random.randint(0, 255, (1000, 3))
        self.max_age = 5  # Frames to keep lost tracks

    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        raise NotImplementedError

    def _iou(self, box1: List[float], box2: List[float]) -> float:
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area != 0 else 0

class ByteTrackTracker(BaseTracker):
    def __init__(self):
        super().__init__()
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.kalman_filters = {}

    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        self.frame_count += 1
        active_tracks = {}
        
        predicted_tracks = {}
        for track_id, track in self.tracks.items():
            if track_id in self.track_history and len(self.track_history[track_id]) >= 2:
                xs = [p[0] for p in self.track_history[track_id]]
                ys = [p[1] for p in self.track_history[track_id]]
                
                if len(xs) >= 2:
                    x_pred = 2 * xs[-1] - xs[-2]
                    y_pred = 2 * ys[-1] - ys[-2]
                    predicted_tracks[track_id] = {
                        'bbox': self._predict_bbox(track['bbox'], x_pred, y_pred),
                        'class': track['class'],
                        'conf': track['conf'],
                        'age': track['age']
                    }
            else:
                predicted_tracks[track_id] = track

        high_conf_dets = [d for d in detections if d['conf'] > 0.7]
        remaining_dets = [d for d in detections if d['conf'] <= 0.7]

        matched_pairs = self._match_detections_to_tracks(high_conf_dets, predicted_tracks)
        
        for det_idx, track_id in matched_pairs:
            det = high_conf_dets[det_idx]
            center = ((det['bbox'][0] + det['bbox'][2]) // 2, (det['bbox'][1] + det['bbox'][3]) // 2)
            self.track_history[track_id].append(center)
            
            active_tracks[track_id] = {
                'bbox': det['bbox'],
                'class': det['class'],
                'conf': det['conf'],
                'age': 0
            }

        for det_idx, det in enumerate(high_conf_dets):
            if det_idx not in [p[0] for p in matched_pairs]:
                track_id = self.next_id
                center = ((det['bbox'][0] + det['bbox'][2]) // 2, (det['bbox'][1] + det['bbox'][3]) // 2)
                self.track_history[track_id].append(center)
                
                active_tracks[track_id] = {
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'conf': det['conf'],
                    'age': 0
                }
                self.next_id += 1

        matched_pairs = self._match_detections_to_tracks(remaining_dets, predicted_tracks)
        
        for det_idx, track_id in matched_pairs:
            det = remaining_dets[det_idx]
            center = ((det['bbox'][0] + det['bbox'][2]) // 2, (det['bbox'][1] + det['bbox'][3]) // 2)
            self.track_history[track_id].append(center)
            
            active_tracks[track_id] = {
                'bbox': det['bbox'],
                'class': det['class'],
                'conf': det['conf'],
                'age': 0
            }

        self.tracks = {k: v for k, v in active_tracks.items() if v['age'] <= self.max_age}
        return self.tracks

    def _predict_bbox(self, bbox, x_pred, y_pred):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return [x_pred - w//2, y_pred - h//2, x_pred + w//2, y_pred + h//2]

    def _match_detections_to_tracks(self, detections, tracks):
        matched_pairs = []
        available_tracks = list(tracks.keys())
        
        for det_idx, det in enumerate(detections):
            best_iou = 0.3
            best_track = None
            
            for track_id in available_tracks:
                iou = self._iou(det['bbox'], tracks[track_id]['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track_id
            
            if best_track:
                matched_pairs.append((det_idx, best_track))
                available_tracks.remove(best_track)
        
        return matched_pairs

class EnhancedLaneDetection:
    
    def __init__(self, tracking_method: str = 'bytetrack', model_size: str = 'l', conf_thresh: float = 0.6):
        # Lane detection components
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()
        
        self.model_size = model_size
        self.conf_thresh = conf_thresh
        self.object_model = self._load_model(model_size)
        self.object_classes = self.object_model.names
        
        self.tracking_method = tracking_method
        self.tracker = ByteTrackTracker()  # Using only ByteTrack for best results
        
        self.detection_history = defaultdict(list)
        self.processing_times = deque(maxlen=100)
        self.frame_count = 0
        
        # Visualization
        self.class_colors = {
            'car': (0, 255, 0),         # Green
            'person': (255, 0, 0),       # Red
            'traffic light': (0, 0, 255), # Blue
            'stop sign': (255, 255, 0),  # Yellow
            'truck': (255, 165, 0),      # Orange
            'bus': (0, 255, 255),        # Cyan
            'motorcycle': (255, 0, 255), # Magenta
            'bicycle': (0, 128, 255),    # Light Blue
            'traffic sign': (255, 128, 0), # Orange
            'fire hydrant': (255, 0, 128), # Pink
            'parking meter': (128, 0, 255), # Purple
            'bench': (0, 255, 128),      # Teal
            'pole': (128, 128, 128),     # Gray
            'construction': (255, 128, 128), # Light Orange
            'tree': (0, 128, 0),         # Dark Green
            'building': (128, 0, 0),     # Dark Red
            'fence': (192, 192, 192),    # Light Gray
            'billboard': (255, 0, 255),  # Magenta
            'street sign': (255, 255, 128), # Light Yellow
            'road': (100, 100, 100),     # Road color
            'lane': (255, 255, 255),     # Lane markings
            'vehicle': (0, 200, 200),    # General vehicles
            'sign': (200, 200, 0),       # General signs
            'roadside': (50, 50, 50),    # Roadside objects
            'signboard': (0, 255, 255),  # For sign boards
            'street-name': (255, 0, 255), # For street name signs
            'speed-limit': (0, 0, 255),  # For speed limit signs
            'information': (255, 255, 0) # For informational signs
        }

    def _load_model(self, model_size: str) -> YOLO:
        model_map = {
            'n': 'yolov8n.pt',
            's': 'yolov8s.pt',
            'm': 'yolov8m.pt',
            'l': 'yolov8l.pt',
            'x': 'yolov8x.pt',
            'ul': 'yolov8x6.pt'  # Ultra large model
        }
        return YOLO(model_map.get(model_size, 'yolov8l.pt'))

    def process_frame(self, img: np.ndarray) -> np.ndarray:
        start_time = time.time()
        self.frame_count += 1
        
        try:
            lane_img = np.copy(img)
            lane_img = self.calibration.undistort(lane_img)
            lane_img = self.transform.forward(lane_img)
            lane_img = self.thresholding.forward(lane_img)
            lane_img = self.lanelines.forward(lane_img)
            lane_img = self.transform.backward(lane_img)
            
            output_img = cv2.addWeighted(img, 1, lane_img, 0.6, 0)
            output_img = self.lanelines.plot(output_img)
            
            detections = self._detect_objects(img)
            
            tracks = self.tracker.update(detections)
            
            output_img = self._draw_relevant_objects(output_img, tracks)
            
            self.processing_times.append(time.time() - start_time)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            output_img = img  
        
        return output_img

    def _detect_objects(self, img: np.ndarray) -> List[Dict[str, Any]]:
        results = self.object_model(img, stream=True, conf=self.conf_thresh, iou=0.5)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.object_classes[cls]
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'class': cls,
                        'class_name': class_name.lower()
                    })
                    
                    if 'sign' in class_name.lower() or 'board' in class_name.lower():
                        sign_roi = img[y1:y2, x1:x2]
                        if sign_roi.size > 0:
                            gray = cv2.cvtColor(sign_roi, cv2.COLOR_RGB2GRAY)
                            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                x, y, w, h = cv2.boundingRect(cnt)
                                if w > 10 and h > 10:  # Filter small regions
                                    detections.append({
                                        'bbox': [x1+x, y1+y, x1+x+w, y1+y+h],
                                        'conf': conf,
                                        'class': cls,
                                        'class_name': 'sign-text'
                                    })
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
        
        return detections

    def _draw_relevant_objects(self, img: np.ndarray, tracks: Dict[int, Dict[str, Any]]) -> np.ndarray:
        for track_id, track in tracks.items():
            try:
                bbox = track['bbox']
                class_name = track.get('class_name', self.object_classes[track['class']]).lower()
                color = self.class_colors.get(class_name, (0, 255, 0))
                
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                label = f"{class_name} {track['conf']:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (bbox[0], bbox[1] - label_height - baseline), 
                            (bbox[0] + label_width, bbox[1]), color, cv2.FILLED)
                cv2.putText(img, label, (bbox[0], bbox[1] - baseline), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
            except Exception as e:
                print(f"Error drawing track {track_id}: {e}")
                continue
        
        return img

    def process_image(self, input_path: str, output_path: str) -> None:

        try:
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Could not read image from {input_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out_img = self.process_frame(img)
            cv2.imwrite(output_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error processing image: {e}")
            raise

    def process_video(self, input_path: str, output_path: str) -> None:

        try:
            clip = VideoFileClip(input_path)
            out_clip = clip.fl_image(self.process_frame)
            out_clip.write_videofile(output_path, audio=False, threads=4, verbose=False)
        except Exception as e:
            print(f"Error processing video: {e}")
            raise

class StatsCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=120)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        
        self.ax.set_title('Processing Performance', color='white', pad=20, fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Frame', color='white', fontsize=10)
        self.ax.set_ylabel('Time (ms)', color='white', fontsize=10)
        self.ax.tick_params(axis='x', colors='white', labelsize=8)
        self.ax.tick_params(axis='y', colors='white', labelsize=8)
        
        self.ax.grid(True, color='#444444', linestyle='--', alpha=0.7)
        self.fig.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        
        self.line, = self.ax.plot([], [], 'c-', linewidth=2, label='Processing Time', marker='o', 
                                markersize=4, markerfacecolor='cyan', markeredgecolor='white')
        
        self.ax.legend(facecolor='#3a3a3a', edgecolor='none', labelcolor='white', 
                     fontsize=9, loc='upper right')
        
        self.fig.tight_layout()

    def update_stats(self, processing_times):

        if processing_times:
            x = range(len(processing_times))
            y = [t*1000 for t in processing_times]
            self.line.set_data(x, y)
            
            self.ax.set_xlim(0, max(1, len(processing_times)-1))
            self.ax.set_ylim(0, max(10, max(y)*1.2))
            
            avg_time = np.mean(processing_times)*1000 if processing_times else 0
            min_time = min(processing_times)*1000 if processing_times else 0
            max_time = max(processing_times)*1000 if processing_times else 0
            self.ax.set_title(f'Processing Time\nAvg: {avg_time:.1f}ms | Min: {min_time:.1f}ms | Max: {max_time:.1f}ms', 
                            color='white', fontsize=10)
            

            if hasattr(self, 'avg_line'):
                self.avg_line.remove()
            self.avg_line = self.ax.axhline(avg_time, color='magenta', linestyle='--', alpha=0.7)
            
            self.draw()

class Object3DView(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=120)
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        
        self.fig.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        
        self.ax.xaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
        self.ax.yaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
        self.ax.zaxis.set_pane_color((0.17, 0.17, 0.17, 1.0))
        
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        self.ax.tick_params(axis='x', colors='white', labelsize=8)
        self.ax.tick_params(axis='y', colors='white', labelsize=8)
        self.ax.tick_params(axis='z', colors='white', labelsize=8)
        
        self.ax.set_title('3D Road Scene Visualization', color='white', pad=20, fontsize=12, fontweight='bold')
        self.ax.set_xlabel('X Position (Distance)', color='white', fontsize=10)
        self.ax.set_ylabel('Y Position (Lateral)', color='white', fontsize=10)
        self.ax.set_zlabel('Object Type', color='white', fontsize=10)
        
        self.class_mapping = {
            'road': 1, 'lane': 2, 'car': 3, 'truck': 4, 'bus': 5,
            'motorcycle': 6, 'bicycle': 7, 'person': 8, 'traffic light': 9,
            'stop sign': 10, 'traffic sign': 11, 'pole': 12, 'building': 13,
            'tree': 14, 'fence': 15, 'billboard': 16, 'street sign': 17,
            'construction': 18, 'roadside': 19, 'vehicle': 20, 'sign': 21,
            'signboard': 22, 'street-name': 23, 'speed-limit': 24, 'information': 25
        }
        
        self.colors = plt.cm.tab20(np.linspace(0, 1, len(self.class_mapping)))
        
        self.scatter = self.ax.scatter([], [], [], c=[], s=[], alpha=0.8, 
                                      depthshade=True, edgecolors='white', linewidths=0.5)
        
        self.road_surface = None
        
        legend_elements = []
        for class_name, z_value in self.class_mapping.items():
            color = self.colors[z_value-1]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          label=class_name.capitalize(),
                                          markerfacecolor=color, markersize=8))
        
        self.ax.legend(handles=legend_elements, loc='center left', 
                     bbox_to_anchor=(1.05, 0.5), facecolor='#3a3a3a', 
                     edgecolor='none', labelcolor='white', fontsize=8)
        
        self.ax.view_init(elev=25, azim=-60)
        
        self.fig.tight_layout()

    def update_objects(self, objects):

        if objects:

            if self.road_surface is not None:
                self.road_surface.remove()
            

            x_range = np.linspace(0, 1000, 10)
            y_range = np.linspace(-200, 200, 10)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.ones_like(X) * 1  # Road is at z=1
            
            self.road_surface = self.ax.plot_surface(X, Y, Z, color='#555555', alpha=0.3)
            
            
            xs = []
            ys = []
            zs = []
            colors = []
            sizes = []
            
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                size = (x2 - x1) * (y2 - y1)
                
                class_name = obj.get('class_name', 'car').lower()
                z_value = self.class_mapping.get(class_name, 3)  # Default to car
                color = self.colors[z_value-1]
                
                vis_x = center_x * 2  # Scale x to represent distance
                vis_y = (center_y - 360) / 2  # Center and scale y
                
                if class_name in ['road', 'lane']:
                    vis_z = 1  # On the road surface
                elif class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    vis_z = 2  # Slightly above road
                elif class_name in ['person']:
                    vis_z = 8  # Higher for pedestrians
                elif 'sign' in class_name or 'board' in class_name:
                    vis_z = 5  # Elevated for signs
                else:
                    vis_z = z_value  # Use class mapping for height
                
                xs.append(vis_x)
                ys.append(vis_y)
                zs.append(vis_z)
                colors.append(color)
                sizes.append(np.sqrt(size) / 5)  # Adjust size scaling
            
            if len(xs) > 0:

                xs = np.array(xs)
                ys = np.array(ys)
                zs = np.array(zs)
                colors = np.array(colors)
                sizes = np.array(sizes)
                
                self.scatter._offsets3d = (xs, ys, zs)
                self.scatter.set_color(colors)
                self.scatter.set_sizes(sizes)
                
                if len(xs) > 0:
                    self.ax.set_xlim(min(xs)-50, max(xs)+50)
                    self.ax.set_ylim(min(ys)-50, max(ys)+50)
                    self.ax.set_zlim(0, len(self.class_mapping)+1)
                
                lane_x = np.linspace(0, max(xs)+50, 10)
                lane_y1 = np.ones_like(lane_x) * -5  # Left lane
                lane_y2 = np.ones_like(lane_x) * 5   # Right lane
                lane_z = np.ones_like(lane_x) * 1.1  # Slightly above road
                
                self.ax.plot(lane_x, lane_y1, lane_z, 'w-', linewidth=2, alpha=0.8)
                self.ax.plot(lane_x, lane_y2, lane_z, 'w-', linewidth=2, alpha=0.8)
            
            self.draw()

class DocumentationView(QWidget):
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        content = QWidget()
        scroll.setWidget(content)
        
        vbox = QVBoxLayout()
        content.setLayout(vbox)
        
        overview = QGroupBox("System Overview")
        overview_layout = QVBoxLayout()
        overview_text = QTextEdit()
        overview_text.setReadOnly(True)
        overview_text.setPlainText(
            "Advanced Lane & Object Detection System\n\n"
            "This system provides real-time detection and tracking of:\n"
            "- Lane markings and road boundaries\n"
            "- Vehicles (cars, trucks, buses, motorcycles, bicycles)\n"
            "- Traffic signs and signals (traffic lights, stop signs)\n"
            "- Pedestrians and other road objects\n\n"
            "Key Features:\n"
            "- High-precision detection using YOLOv8 models\n"
            "- Robust object tracking with ByteTrack algorithm\n"
            "- Real-time performance monitoring\n"
            "- 3D visualization of detected objects\n"
            "- Video processing with playback controls"
        )
        overview_layout.addWidget(overview_text)
        overview.setLayout(overview_layout)
        vbox.addWidget(overview)
        
        usage = QGroupBox("Usage Instructions")
        usage_layout = QVBoxLayout()
        usage_text = QTextEdit()
        usage_text.setReadOnly(True)
        usage_text.setPlainText(
            "How to use the system:\n\n"
            "1. File Operations:\n"
            "   - Click 'Open File' to select an image or video\n"
            "   - Choose processing mode (Image or Video)\n"
            "   - Click 'Process' to analyze the file\n"
            "   - Save the processed output when complete\n\n"
            "2. Webcam Mode:\n"
            "   - Click 'Start Webcam' for live processing\n"
            "   - The system will analyze live video feed\n\n"
            "3. Playback Controls:\n"
            "   - For processed videos, use the playback controls\n"
            "   - Adjust speed with the playback speed slider\n\n"
            "4. Detection Settings:\n"
            "   - Select model size (larger = more accurate but slower)\n"
            "   - Adjust confidence threshold for detections\n"
            "   - View results in real-time"
        )
        usage_layout.addWidget(usage_text)
        usage.setLayout(usage_layout)
        vbox.addWidget(usage)
        
        models = QGroupBox("Model Information")
        models_layout = QVBoxLayout()
        models_text = QTextEdit()
        models_text.setReadOnly(True)
        models_text.setPlainText(
            "Available Models:\n\n"
            "1. Nano (yolov8n.pt):\n"
            "   - Fastest but least accurate\n"
            "   - Good for low-power devices\n\n"
            "2. Small (yolov8s.pt):\n"
            "   - Balanced speed and accuracy\n"
            "   - Recommended for most applications\n\n"
            "3. Medium (yolov8m.pt):\n"
            "   - Improved accuracy with moderate speed\n\n"
            "4. Large (yolov8l.pt):\n"
            "   - High accuracy with slower speed\n"
            "   - Recommended for high-performance systems\n\n"
            "5. X-Large (yolov8x.pt):\n"
            "   - Most accurate but slowest\n"
            "   - For applications requiring maximum precision\n\n"
            "6. Ultra Large (yolov8x6.pt):\n"
            "   - Highest accuracy with extended detection\n"
            "   - Detects more object classes\n"
            "   - Requires significant computational resources"
        )
        models_layout.addWidget(models_text)
        models.setLayout(models_layout)
        vbox.addWidget(models)
        
        classes = QGroupBox("Detectable Object Classes")
        classes_layout = QVBoxLayout()
        classes_text = QTextEdit()
        classes_text.setReadOnly(True)
        classes_text.setPlainText(
            "The system can detect and track the following objects:\n\n"
            "- Vehicles: car, truck, bus, motorcycle, bicycle, trailer, van, suv\n"
            "- Traffic Objects: traffic light, stop sign, traffic sign, speed limit\n"
            "- Road Infrastructure: fire hydrant, parking meter, pole, guard rail\n"
            "- Road Features: road, lane, crosswalk, highway, overpass, bridge\n"
            "- Roadside Objects: tree, building, fence, billboard, street sign\n"
            "- Construction: road work, road barrier, pothole, construction zone\n"
            "- Pedestrians: person, pedestrian crossing\n"
            "- Miscellaneous: traffic cone, barrier, road marking, road divider\n\n"
            "Each object class is displayed with a distinct color\n"
            "in the visualization for easy identification."
        )
        classes_layout.addWidget(classes_text)
        classes.setLayout(classes_layout)
        vbox.addWidget(classes)
        
        vbox.addStretch()

class GradientLabel(QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumHeight(40)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, QColor(41, 128, 185))
        gradient.setColorAt(1, QColor(142, 68, 173))
        painter.fillRect(self.rect(), gradient)
        painter.setPen(Qt.white)
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())
        painter.end()

class AnimatedButton(QPushButton):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._animation = QPropertyAnimation(self, b"geometry")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.OutQuad)
        self._animation.finished.connect(self._reset_animation)
        self._original_geometry = None
        
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QColor(0, 0, 0, 150))
        self.shadow.setOffset(3, 3)
        self.setGraphicsEffect(self.shadow)
        
    def enterEvent(self, event):
        if self._original_geometry is None:
            self._original_geometry = self.geometry()
            
        self._animation.stop()
        self._animation.setStartValue(self.geometry())
        self._animation.setEndValue(self.geometry().adjusted(-5, -5, 5, 5))
        self._animation.start()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self._animation.stop()
        self._animation.setStartValue(self.geometry())
        self._animation.setEndValue(self._original_geometry)
        self._animation.start()
        super().leaveEvent(event)
        
    def _reset_animation(self):

        if self._original_geometry:
            self.setGeometry(self._original_geometry)

class VideoControls(QWidget):

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignCenter) 
        self.setLayout(layout)
        
        layout.addStretch()
        
        self.play_button = AnimatedButton()
        self.play_button.setCheckable(True)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setIconSize(QSize(32, 32))
        self.play_button.setToolTip("Play")
        self.play_button.setFixedSize(50, 50)
        layout.addWidget(self.play_button)
        
        self.pause_button = AnimatedButton()
        self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pause_button.setIconSize(QSize(32, 32))
        self.pause_button.setToolTip("Pause")
        self.pause_button.setFixedSize(50, 50)
        layout.addWidget(self.pause_button)
        
        self.stop_button = AnimatedButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setIconSize(QSize(32, 32))
        self.stop_button.setToolTip("Stop")
        self.stop_button.setFixedSize(50, 50)
        layout.addWidget(self.stop_button)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 5)
        self.speed_slider.setValue(3)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setFixedWidth(150)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4a4a8a;
                height: 8px;
                background: #2b2b2b;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a4a8a;
                border: 1px solid #6a6aaa;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #4a4a8a;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("Speed: 1x")
        self.speed_label.setStyleSheet("color: white;")
        layout.addWidget(self.speed_label)
        
        layout.addStretch()

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Lane & Object Detection System")
        self.setWindowIcon(QIcon('icon.png'))  # Add your icon file
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        self.tracking_method = 'bytetrack'
        self.model_size = 'l'
        self.conf_thresh = 0.6
        self.processor = None
        self.video_processor = None
        self.video_player = None
        self.current_video_path = ""
        
        self._setup_ui()
        
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.webcam_active = False
        self.input_path = ""
        self.output_image = None
        
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        
    def _setup_ui(self):
        """Setup the premium UI components"""

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self._create_menu_bar()
        
        header = GradientLabel("ADVANCED LANE & OBJECT DETECTION SYSTEM")
        header.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            qproperty-alignment: AlignCenter;
            padding: 15px;
        """)
        main_layout.addWidget(header)
        
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        control_panel = QGroupBox("CONTROLS")
        control_panel.setMinimumWidth(350)
        control_panel.setMaximumWidth(400)
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        content_layout.addWidget(control_panel, 1)
        
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        self.btn_open = AnimatedButton("📁 Open File")
        self.btn_open.setIcon(QIcon.fromTheme("document-open"))
        self.btn_open.setStyleSheet("font-size: 14px; padding: 10px;")
        self.btn_open.clicked.connect(self.open_file)
        file_layout.addWidget(self.btn_open)
        
        self.btn_save = AnimatedButton("💾 Save Output")
        self.btn_save.setIcon(QIcon.fromTheme("document-save"))
        self.btn_save.setStyleSheet("font-size: 14px; padding: 10px;")
        self.btn_save.clicked.connect(self.save_output)
        self.btn_save.setEnabled(False)
        file_layout.addWidget(self.btn_save)
        
        default_dir_layout = QHBoxLayout()
        self.btn_open_default = AnimatedButton("📂 Open Default Input")
        self.btn_open_default.setIcon(QIcon.fromTheme("folder"))
        self.btn_open_default.setStyleSheet("font-size: 12px; padding: 8px;")
        self.btn_open_default.clicked.connect(lambda: self.open_default_dir("input"))
        default_dir_layout.addWidget(self.btn_open_default)
        
        self.btn_save_default = AnimatedButton("💾 Save to Default Output")
        self.btn_save_default.setIcon(QIcon.fromTheme("folder"))
        self.btn_save_default.setStyleSheet("font-size: 12px; padding: 8px;")
        self.btn_save_default.clicked.connect(lambda: self.save_to_default_dir())
        self.btn_save_default.setEnabled(False)
        default_dir_layout.addWidget(self.btn_save_default)
        
        file_layout.addLayout(default_dir_layout)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QVBoxLayout()
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Image", "Video"])
        self.combo_mode.setStyleSheet("font-size: 14px; padding: 8px;")
        mode_layout.addWidget(self.combo_mode)
        
        self.btn_webcam = AnimatedButton("📷 Start Webcam")
        self.btn_webcam.setIcon(QIcon.fromTheme("camera-web"))
        self.btn_webcam.setStyleSheet("font-size: 14px; padding: 10px;")
        self.btn_webcam.clicked.connect(self.toggle_webcam)
        mode_layout.addWidget(self.btn_webcam)
        
        mode_group.setLayout(mode_layout)
        control_layout.addWidget(mode_group)
        
        model_group = QGroupBox("Detection Settings")
        model_layout = QVBoxLayout()
        
        model_size_layout = QHBoxLayout()
        model_size_layout.addWidget(QLabel("Model Size:"))
        
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Nano (Fastest)", "Small", "Medium", "Large (Recommended)", "X-Large (Best)", "Ultra Large (Max Detection)"])
        self.combo_model.setCurrentIndex(3)
        self.combo_model.setStyleSheet("font-size: 14px; padding: 8px;")
        self.combo_model.currentIndexChanged.connect(self.set_model_size)
        model_size_layout.addWidget(self.combo_model)
        
        model_layout.addLayout(model_size_layout)
        
        self.conf_label = QLabel("Confidence Threshold: 0.60")
        self.conf_label.setStyleSheet("font-size: 14px;")
        model_layout.addWidget(self.conf_label)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(30, 90)
        self.conf_slider.setValue(60)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.valueChanged.connect(self.set_conf_threshold)
        model_layout.addWidget(self.conf_slider)
        
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)
        
        self.btn_process = AnimatedButton("⚙️ Process")
        self.btn_process.setIcon(QIcon.fromTheme("system-run"))
        self.btn_process.setStyleSheet("""
            font-size: 16px; 
            padding: 12px; 
            background-color: #2e7d32; 
            color: white;
        """)
        self.btn_process.clicked.connect(self.process_file)
        self.btn_process.setEnabled(False)
        control_layout.addWidget(self.btn_process)
        

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #4a4a8a;
                border-radius: 5px;
                text-align: center;
                background: #16213e;
                height: 20px;
            }
            QProgressBar::chunk { 
                background-color: #4CAF50; 
                width: 10px; 
                border-radius: 5px;
            }
        """)
        control_layout.addWidget(self.progress_bar)
        
        display_panel = QTabWidget()
        display_panel.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #4a4a8a;
                border-radius: 4px;
                margin-top: 5px;
            }
            QTabBar::tab {
                background: #2b2b2b;
                color: white;
                padding: 8px;
                border: 1px solid #4a4a8a;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #4a4a8a;
                border-color: #4a4a8a;
            }
            QTabBar::tab:hover {
                background: #5656a3;
            }
        """)
        content_layout.addWidget(display_panel, 3)
        
        video_tab = QWidget()
        video_layout = QVBoxLayout()
        

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: black; 
            border: 2px solid #4a4a8a; 
            border-radius: 5px;
        """)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.video_label, 1)
        
        self.video_controls = VideoControls()
        self.video_controls.play_button.clicked.connect(self.play_video)
        self.video_controls.pause_button.clicked.connect(self.pause_video)
        self.video_controls.stop_button.clicked.connect(self.stop_video)
        self.video_controls.speed_slider.valueChanged.connect(self.set_playback_speed)
        self.video_controls.setVisible(False)
        
        video_layout.addWidget(self.video_controls)
        video_tab.setLayout(video_layout)
        
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        self.stats_canvas = StatsCanvas()
        stats_layout.addWidget(self.stats_canvas)
        stats_tab.setLayout(stats_layout)
        

        vis3d_tab = QWidget()
        vis3d_layout = QVBoxLayout()
        self.vis3d_canvas = Object3DView()
        vis3d_layout.addWidget(self.vis3d_canvas)
        vis3d_tab.setLayout(vis3d_layout)
        
        display_panel.addTab(video_tab, "Video")
        display_panel.addTab(stats_tab, "Performance")
        display_panel.addTab(vis3d_tab, "3D Visualization")
        
        self.status_bar = QLabel()
        self.status_bar.setStyleSheet("""
            background-color: #16213e; 
            color: #e6e6e6; 
            padding: 10px; 
            border-top: 2px solid #4a4a8a; 
            font-size: 14px;
        """)
        self.status_bar.setText("System ready | Select a file to begin processing")
        main_layout.addWidget(self.status_bar)
        
    def _create_menu_bar(self):

        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        
        open_action = QAction(QIcon.fromTheme("document-open"), "Open File", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction(QIcon.fromTheme("document-save"), "Save Output", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_output)
        file_menu.addAction(save_action)
        
        exit_action = QAction(QIcon.fromTheme("application-exit"), "Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        help_menu = menubar.addMenu("Help")
        
        docs_action = QAction(QIcon.fromTheme("help-contents"), "Documentation", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
        
        about_action = QAction(QIcon.fromTheme("help-about"), "About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def show_documentation(self):

        doc_dialog = QDialog(self)
        doc_dialog.setWindowTitle("Documentation")
        doc_dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        doc_view = DocumentationView()
        layout.addWidget(doc_view)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(doc_dialog.close)
        layout.addWidget(close_button)
        
        doc_dialog.setLayout(layout)
        doc_dialog.exec_()
        
    def show_about(self):

        about_text = """
        <h2>Advanced Lane & Object Detection System</h2>
        <p>Version 1.0</p>
        <p>This application provides real-time detection and tracking of:</p>
        <ul>
            <li>Lane markings and road boundaries</li>
            <li>Vehicles (cars, trucks, buses, motorcycles)</li>
            <li>Traffic signs and signals</li>
            <li>Pedestrians and other road objects</li>
        </ul>
        <p>Developed using Python, OpenCV, PyQt5, and YOLOv8</p>
        <p>© 2025 Advanced Vision Systems By Smita Mhatugade</p>
        """
        
        QMessageBox.about(self, "About", about_text)
        
    def set_model_size(self, index):

        sizes = ['n', 's', 'm', 'l', 'x', 'ul']
        self.model_size = sizes[index]
        if self.processor:
            self.processor.model_size = self.model_size
            self.processor.object_model = self.processor._load_model(self.model_size)
    
    def set_conf_threshold(self, value):

        self.conf_thresh = value / 100
        self.conf_label.setText(f"Confidence Threshold: {self.conf_thresh:.2f}")
        if self.processor:
            self.processor.conf_thresh = self.conf_thresh
    
    def open_default_dir(self, dir_type):

        path = DEFAULT_DIR / dir_type
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            os.system(f"open {path}")
        else:
            os.system(f"xdg-open {path}")
    
    def save_to_default_dir(self):

        if not hasattr(self, 'output_image') or not self.output_image:
            return
            
        is_image = self.combo_mode.currentText() == "Image"
        ext = ".jpg" if is_image else ".mp4"
        default_path = str(DEFAULT_DIR / "output" / f"output_{int(time.time())}{ext}")
        
        try:
            if is_image:
                cv2.imwrite(default_path, cv2.cvtColor(self.output_image, cv2.COLOR_RGB2BGR))
            else:
                pass
            self.status_bar.setText(f"Saved output to {default_path}")
        except Exception as e:
            self.status_bar.setText(f"Error saving file: {str(e)}")
    
    def open_file(self):

        options = QFileDialog.Options()
        is_image = self.combo_mode.currentText() == "Image"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image" if is_image else "Open Video", 
            str(DEFAULT_DIR / "input"), 
            "Image Files (*.png *.jpg *.jpeg *.bmp);;Video Files (*.mp4 *.avi *.mov)" if is_image 
            else "Video Files (*.mp4 *.avi *.mov)", 
            options=options)
        
        if file_path:
            self.input_path = file_path
            self.btn_process.setEnabled(True)
            self.btn_save.setEnabled(False)
            self.btn_save_default.setEnabled(False)
            self.video_controls.setVisible(False)
            
            self.processor = EnhancedLaneDetection(
                tracking_method=self.tracking_method,
                model_size=self.model_size,
                conf_thresh=self.conf_thresh
            )
            
            if is_image:
                img = cv2.imread(file_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.display_image(img)
                    self.status_bar.setText(f"Loaded image: {os.path.basename(file_path)}")
            else:
                self.current_video_path = file_path
                self.cap = cv2.VideoCapture(file_path)
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.display_image(frame)
                    self.status_bar.setText(f"Loaded video: {os.path.basename(file_path)}")
                    self.video_controls.setVisible(True)
                self.cap.release()
    
    def save_output(self):
        """Save the processed output"""
        if not hasattr(self, 'output_image') or not self.output_image:
            return
            
        is_image = self.combo_mode.currentText() == "Image"
        options = QFileDialog.Options()
        
        if is_image:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Processed Image",
                str(DEFAULT_DIR / "output"),
                "Image Files (*.png *.jpg *.jpeg)",
                options=options)
            
            if file_path:
                try:
                    cv2.imwrite(file_path, cv2.cvtColor(self.output_image, cv2.COLOR_RGB2BGR))
                    self.status_bar.setText(f"Saved processed image to {file_path}")
                except Exception as e:
                    self.status_bar.setText(f"Error saving image: {str(e)}")
        else:

            pass
    
    def play_video(self):

        if not self.current_video_path or not self.processor:
            return
            
        if hasattr(self, 'video_player') and self.video_player:
            self.video_player.stop()
            
        self.video_player = VideoPlayer(self.current_video_path, self.processor)
        self.video_player.frame_ready.connect(self.update_video_frame)
        self.video_player.start()
        self.video_controls.play_button.setChecked(True)
        self.status_bar.setText(f"Playing video: {os.path.basename(self.current_video_path)}")
    
    def pause_video(self):

        if hasattr(self, 'video_player') and self.video_player:
            self.video_player.stop()
            self.video_controls.play_button.setChecked(False)
            self.status_bar.setText(f"Video paused: {os.path.basename(self.current_video_path)}")
    
    def stop_video(self):

        if hasattr(self, 'video_player') and self.video_player:
            self.video_player.stop()
            self.video_controls.play_button.setChecked(False)
            
            # Show first frame
            cap = cv2.VideoCapture(self.current_video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_image(frame)
            cap.release()
            self.status_bar.setText(f"Video stopped: {os.path.basename(self.current_video_path)}")
    
    def set_playback_speed(self, value):

        speeds = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.5, 5: 2.0}
        self.video_controls.speed_label.setText(f"Speed: {speeds[value]}x")
        if hasattr(self, 'video_player') and self.video_player:
            self.video_player.set_speed(speeds[value])
    
    def update_video_frame(self, frame, tracks):

        self.display_image(frame)
        
        if hasattr(self.processor, 'processing_times'):
            self.stats_canvas.update_stats(self.processor.processing_times)
        
        if tracks:
            self.vis3d_canvas.update_objects(tracks)
    
    def process_file(self):

        if not self.processor or not self.input_path:
            return
            
        is_image = self.combo_mode.currentText() == "Image"
        
        if is_image:
            try:
                output_path, _ = QFileDialog.getSaveFileName(
                    self, 
                    "Save Processed Image", 
                    str(DEFAULT_DIR / "output"), 
                    "Image Files (*.png *.jpg *.jpeg)")
                
                if output_path:
                    self.btn_process.setEnabled(False)
                    self.progress_bar.setValue(0)
                    
                    processing_thread = QThread()
                    worker = ImageProcessor(self.processor, self.input_path, output_path)
                    worker.moveToThread(processing_thread)
                    processing_thread.started.connect(worker.process)
                    worker.finished.connect(processing_thread.quit)
                    worker.finished.connect(worker.deleteLater)
                    processing_thread.finished.connect(processing_thread.deleteLater)
                    worker.progress.connect(self.progress_bar.setValue)
                    worker.completed.connect(lambda path: self.image_processing_complete(path))
                    worker.error.connect(lambda msg: self.show_error(msg))
                    processing_thread.start()
            except Exception as e:
                self.show_error(f"Error processing image: {e}")
                self.btn_process.setEnabled(True)
        else:
            output_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Processed Video", 
                str(DEFAULT_DIR / "output"), 
                "Video Files (*.mp4)")
            
            if output_path:
                self.btn_process.setEnabled(False)
                self.progress_bar.setValue(0)
                
                if hasattr(self, 'video_processor') and self.video_processor:
                    self.video_processor.stop()
                    
                self.video_processor = VideoProcessor(self.processor, self.input_path, output_path)
                self.video_processor.update_progress.connect(self.progress_bar.setValue)
                self.video_processor.processing_complete.connect(self.video_processing_complete)
                self.video_processor.error_occurred.connect(self.show_error)
                self.video_processor.frame_processed.connect(self.update_processing_frame)
                self.video_processor.start()
    
    def update_processing_frame(self, frame, tracks):

        self.display_image(frame)
        
        if hasattr(self.processor, 'processing_times'):
            self.stats_canvas.update_stats(self.processor.processing_times)
        
        if tracks:
            self.vis3d_canvas.update_objects(tracks)
    
    def image_processing_complete(self, output_path):

        self.btn_process.setEnabled(True)
        self.progress_bar.setValue(100)
        self.btn_save.setEnabled(True)
        self.btn_save_default.setEnabled(True)
        
        img = cv2.imread(output_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.output_image = img
            self.display_image(img)
            self.status_bar.setText(f"Processing complete. Saved to {output_path}")
    
    def video_processing_complete(self, output_path):

        self.btn_process.setEnabled(True)
        self.progress_bar.setValue(100)
        self.btn_save.setEnabled(True)
        self.btn_save_default.setEnabled(True)
        self.video_controls.setVisible(True)
        self.current_video_path = output_path
        
        cap = cv2.VideoCapture(output_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.output_image = frame
            self.display_image(frame)
        cap.release()
        self.status_bar.setText(f"Processing complete. Saved to {output_path}")
    
    def show_error(self, message):
        """Show error message"""
        self.btn_process.setEnabled(True)
        self.status_bar.setText(f"Error: {message}")
    
    def toggle_webcam(self):
        """Toggle webcam feed"""
        if not self.webcam_active:

            self.processor = EnhancedLaneDetection(
                tracking_method=self.tracking_method,
                model_size='n',  # Use nano model for webcam
                conf_thresh=self.conf_thresh
            )
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.show_error("Could not open webcam")
                return
                
            self.webcam_active = True
            self.btn_webcam.setText("📷 Stop Webcam")
            self.btn_webcam.setStyleSheet("""
                font-size: 14px; 
                padding: 10px; 
                background-color: #c62828; 
                color: white;
            """)
            self.timer.start(30)  # ~30 FPS
            self.status_bar.setText("Webcam active - Processing live feed")
        else:
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.webcam_active = False
            self.btn_webcam.setText("📷 Start Webcam")
            self.btn_webcam.setStyleSheet("font-size: 14px; padding: 10px;")
            self.video_label.clear()
            self.status_bar.setText("Webcam stopped")
    
    def update_frame(self):

        if self.webcam_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = self.processor.process_frame(frame)
                self.display_image(processed_frame)
                
                if self.processor.frame_count % 10 == 0:
                    self.stats_canvas.update_stats(self.processor.processing_times)
                
                if hasattr(self.processor, 'tracker') and self.processor.tracker.tracks:
                    objects = list(self.processor.tracker.tracks.values())
                    self.vis3d_canvas.update_objects(objects)
    
    def display_image(self, img):
        """Display image on the QLabel with proper aspect ratio handling"""
        if isinstance(img, QImage):
            pixmap = QPixmap.fromImage(img)
        else:
            h, w, ch = img.shape
            bytes_per_line = ch * w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
        
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)    

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)

        if hasattr(self, 'video_label') and self.video_label.pixmap():
            pixmap = self.video_label.pixmap()
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        """Clean up on window close"""
        if self.timer.isActive():
            self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'video_processor') and self.video_processor and self.video_processor.isRunning():
            self.video_processor.stop()
        if hasattr(self, 'video_player') and self.video_player and self.video_player.isRunning():
            self.video_player.stop()
        event.accept()

class ImageProcessor(QObject):
    """Worker for processing images in background"""
    progress = pyqtSignal(int)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, processor, input_path, output_path):
        super().__init__()
        self.processor = processor
        self.input_path = input_path
        self.output_path = output_path

    def process(self):
        try:
            self.progress.emit(10)
            img = cv2.imread(self.input_path)
            if img is None:
                raise ValueError(f"Could not read image from {self.input_path}")
                
            self.progress.emit(30)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            self.progress.emit(50)
            out_img = self.processor.process_frame(img)
            
            self.progress.emit(80)
            cv2.imwrite(self.output_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            
            self.progress.emit(100)
            self.completed.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
            self.deleteLater()  # Add


def main():

    if len(sys.argv) == 1:
        app = QApplication(sys.argv)
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        
        font = QFont()
        font.setFamily("Segoe UI" if platform.system() == "Windows" else "Arial")
        font.setPointSize(10)
        app.setFont(font)
        
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    else:
        args = docopt(__doc__)
        input_path = args['INPUT_PATH']
        output_path = args['OUTPUT_PATH']
        tracking_method = args['--tracking']
        model_size = args['--model']
        conf_thresh = float(args['--conf'])
        
        processor = EnhancedLaneDetection(
            tracking_method=tracking_method,
            model_size=model_size,
            conf_thresh=conf_thresh
        )
        
        try:
            if args['--video']:
                processor.process_video(input_path, output_path)
            else:
                processor.process_image(input_path, output_path)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()