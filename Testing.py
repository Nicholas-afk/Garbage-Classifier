import sys
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QProgressBar,
                             QTextEdit, QMessageBox, QGridLayout, QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPainter, QPainterPath, QFontDatabase, QImage
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

# Constants - Updated model path and added random class
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trash_classifier_finetuned_20250405_003634_best.pth')
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png']
MAX_IMAGES = 100
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
TRASH_CLASSES = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
                 'metal', 'paper', 'plastic', 'shoes', 'trash', 'random']

# Updated color scheme with greens but keeping the original structure
COLORS = {
    'primary': '#2E8B57',  # Sea Green (primary color)
    'secondary': '#3CB371',  # Medium Sea Green (secondary color)
    'accent': '#66CDAA',  # Medium Aquamarine (accent color)
    'light': '#EBF5EE',  # Very light green
    'white': '#F8F6F0',  # Bone white instead of pure white
    'text_dark': '#333333',  # Near black
    'text_light': '#666666'  # Gray
}

# Bin mapping for each waste type - Added random category
BIN_MAPPING = {
    'battery': {
        'bin': 'HAZARDOUS WASTE',
        'color': '#FFA500',  # Orange
        'description': 'Take to a battery recycling point or hazardous waste collection'
    },
    'biological': {
        'bin': 'COMPOST',
        'color': '#8B4513',  # Brown
        'description': 'Place in compost or food waste bin'
    },
    'cardboard': {
        'bin': 'PAPER RECYCLING',
        'color': '#A52A2A',  # Brown
        'description': 'Flatten and place in paper/cardboard recycling bin'
    },
    'clothes': {
        'bin': 'TEXTILE COLLECTION',
        'color': '#4B0082',  # Indigo
        'description': 'Donate to charity or place in textile recycling bin. If unavailable, check if clean and dry, then place in general waste.'
    },
    'glass': {
        'bin': 'GLASS RECYCLING',
        'color': '#1E90FF',  # Blue
        'description': 'Rinse and place in glass-specific recycling bin'
    },
    'metal': {
        'bin': 'METAL RECYCLING',
        'color': '#708090',  # Gray
        'description': 'Rinse and place in metal/can recycling bin'
    },
    'paper': {
        'bin': 'PAPER RECYCLING',
        'color': '#228B22',  # Green
        'description': 'Place in paper-specific recycling bin'
    },
    'plastic': {
        'bin': 'PLASTIC RECYCLING',
        'color': '#FF6347',  # Red
        'description': 'Check recycling number, rinse and place in plastic-specific recycling bin'
    },
    'shoes': {
        'bin': 'TEXTILE COLLECTION',
        'color': '#800080',  # Purple
        'description': 'Donate to charity or place in textile recycling bin. If unavailable and they cannot be reused, place in general waste.'
    },
    'trash': {
        'bin': 'GENERAL WASTE',
        'color': '#696969',  # Dark gray
        'description': 'Cannot be recycled, place in general waste bin'
    },
    'random': {
        'bin': 'UNKNOWN ITEM',
        'color': '#CC0000',  # Bright red
        'description': 'Please try rescanning the item, make sure the item is isolated in frame. If it is still not recognizable, dispose in general waste if it does not contain energy or hazardous materials.'
    }
}


class CameraThread(QThread):
    """Thread for capturing frames from camera"""
    update_frame = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        self.running = True
        self.capture = cv2.VideoCapture(0)  # Use default camera (index 0)

        if not self.capture.isOpened():
            print("Error: Could not open camera")
            return

        while self.running:
            ret, frame = self.capture.read()
            if ret:
                # Convert to RGB for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w

                # Convert to QImage for displaying in Qt
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.update_frame.emit(qt_image)

            # Slight delay to reduce CPU usage
            self.msleep(30)

    def stop(self):
        self.running = False
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()
        self.wait()


class TrashClassifier(nn.Module):
    def __init__(self, num_classes=11):  # Updated to 11 classes
        super(TrashClassifier, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Architecture matching the fine-tuned model
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Helper function to create rounded corner pixmaps
def create_rounded_pixmap(pixmap, radius):
    target = QPixmap(pixmap.size())
    target.fill(Qt.transparent)

    painter = QPainter(target)
    painter.setRenderHint(QPainter.Antialiasing)

    path = QPainterPath()
    path.addRoundedRect(0, 0, pixmap.width(), pixmap.height(), radius, radius)

    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()

    return target


class PredictionThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, model, pil_image=None, image_path=None):
        super().__init__()
        self.model = model
        self.pil_image = pil_image
        self.image_path = image_path
        self.transform = self.get_transform()

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])

    def run(self):
        try:
            self.progress.emit(20)
            if self.pil_image:
                result = self.process_pil_image(self.pil_image)
                self.progress.emit(100)
                self.finished.emit({"camera_image": result})
            elif self.image_path:
                result = self.process_file_image(self.image_path)
                self.progress.emit(100)
                self.finished.emit({self.image_path: result})
            else:
                self.error.emit("No image provided")
        except Exception as e:
            self.error.emit(f"Error processing image: {str(e)}")

    def process_pil_image(self, pil_image):
        image_tensor = self.transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

            # Get top predictions
            top_probs, top_idx = torch.topk(probabilities, 3)
            top_predictions = [(TRASH_CLASSES[idx], prob.item() * 100) for idx, prob in zip(top_idx, top_probs)]

            # Check if "random" is the top prediction
            predicted_class_idx = torch.argmax(probabilities).item()

            # If "random" is predicted but with low confidence, choose the second best prediction
            if TRASH_CLASSES[predicted_class_idx] == 'random':
                # If the confidence is below threshold or another class is close, use the next best prediction
                if top_probs[0] < 0.6 or (len(top_probs) > 1 and top_probs[0] - top_probs[1] < 0.2):
                    predicted_class_idx = top_idx[1].item()  # Use second best prediction

            confidence = probabilities[predicted_class_idx].item() * 100

            return {
                'predicted_class': TRASH_CLASSES[predicted_class_idx],
                'confidence': confidence,
                'top_predictions': top_predictions
            }

    def process_file_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.process_pil_image(image)


class TrashClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_font()
        self.initUI()
        self.load_model()
        self.current_image = None
        self.captured_image = None

        # Initialize camera
        self.camera_thread = CameraThread()
        self.camera_thread.update_frame.connect(self.update_camera_frame)
        self.camera_thread.start()

    def setup_font(self):
        # Load Roboto font
        font_id = QFontDatabase.addApplicationFont("Roboto-VariableFont_wdth,wght.ttf")
        if font_id != -1:
            self.font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            self.default_font = QFont(self.font_family, 10)
            QApplication.setFont(self.default_font)
        else:
            self.font_family = "Segoe UI, Arial, sans-serif"
            print("Warning: Could not load Roboto font, using system default")

    def load_model(self):
        try:
            self.model = TrashClassifier(num_classes=len(TRASH_CLASSES))
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded successfully. Timestamp: {checkpoint.get('timestamp', 'N/A')}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}\nPath: {MODEL_PATH}")
            sys.exit(1)

    def initUI(self):
        self.setWindowTitle('Recycling Assistant')
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {COLORS['white']};
                font-family: {self.font_family};
            }}
            QLabel, QTextEdit, QPushButton {{
                color: {COLORS['text_dark']};
                font-family: {self.font_family};
            }}
            QScrollArea {{
                border: none;
                background-color: {COLORS['white']};
            }}
        """)

        # Create main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Add header
        self.create_header(main_layout)

        # Create content area with camera and controls
        content_layout = QHBoxLayout()

        # Left side: Camera view and controls
        left_layout = QVBoxLayout()
        self.create_camera_view(left_layout)
        self.create_camera_controls(left_layout)

        # Right side: Upload and results
        right_layout = QVBoxLayout()
        self.create_upload_button(right_layout)
        self.create_progress_bar(right_layout)
        self.create_results_display(right_layout)

        # Add left and right layouts to content
        content_layout.addLayout(left_layout, 6)  # 60% of width
        content_layout.addLayout(right_layout, 4)  # 40% of width

        main_layout.addLayout(content_layout)

    def create_header(self, layout):
        header_frame = QFrame()
        header_frame.setFixedHeight(80)
        header_frame.setStyleSheet(f"background-color: {COLORS['primary']}; border-radius: 10px;")
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 10, 10, 10)

        header_label = QLabel('RECYCLING ASSISTANT')
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet(f"""
            color: {COLORS['white']};
            font-size: 22px;
            font-weight: bold;
            font-family: {self.font_family};
        """)

        subheader = QLabel('Scan your item to find the correct bin')
        subheader.setAlignment(Qt.AlignCenter)
        subheader.setStyleSheet(f"""
            color: {COLORS['light']};
            font-size: 14px;
            font-family: {self.font_family};
        """)

        header_layout.addWidget(header_label)
        header_layout.addWidget(subheader)
        layout.addWidget(header_frame)

    def create_camera_view(self, layout):
        self.camera_frame = QFrame()
        self.camera_frame.setMinimumHeight(400)
        self.camera_frame.setStyleSheet(f"""
            background-color: {COLORS['light']};
            border-radius: 20px;
            padding: 15px;
            border: 2px solid {COLORS['secondary']};
        """)
        camera_layout = QVBoxLayout(self.camera_frame)

        # Camera display label
        self.camera_label = QLabel('Initializing camera...')
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setStyleSheet(f"""
            color: {COLORS['text_light']};
            font-size: 16px;
            font-family: {self.font_family};
            background-color: {COLORS['light']};
            border-radius: 15px;
        """)
        camera_layout.addWidget(self.camera_label)

        layout.addWidget(self.camera_frame)

    def create_camera_controls(self, layout):
        controls_frame = QFrame()
        controls_frame.setStyleSheet(f"""
            background-color: {COLORS['light']};
            border-radius: 10px;
            padding: 10px;
        """)
        controls_layout = QHBoxLayout(controls_frame)

        # Take photo button
        self.take_photo_btn = QPushButton('TAKE PHOTO')
        self.take_photo_btn.setFixedHeight(50)
        self.take_photo_btn.setCursor(Qt.PointingHandCursor)
        self.take_photo_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['white']};
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary']};
            }}
        """)
        self.take_photo_btn.clicked.connect(self.take_photo)
        controls_layout.addWidget(self.take_photo_btn)

        # Add scan button
        self.scan_btn = QPushButton('SCAN ITEM')
        self.scan_btn.setFixedHeight(50)
        self.scan_btn.setCursor(Qt.PointingHandCursor)
        self.scan_btn.setEnabled(False)  # Disabled until photo is taken
        self.scan_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: {COLORS['white']};
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
                margin-left: 10px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
        """)
        self.scan_btn.clicked.connect(self.scan_captured_image)
        controls_layout.addWidget(self.scan_btn)

        layout.addWidget(controls_frame)

    def create_upload_button(self, layout):
        self.upload_btn = QPushButton('UPLOAD IMAGE')
        self.upload_btn.setFixedHeight(50)
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: {COLORS['white']};
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
                margin-bottom: 10px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
            }}
        """)
        self.upload_btn.clicked.connect(self.upload_images)
        layout.addWidget(self.upload_btn)

    def create_progress_bar(self, layout):
        self.progress = QProgressBar()
        self.progress.setFixedHeight(10)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 5px;
                background-color: {COLORS['light']};
                text-align: center;
                color: transparent;
                font-family: {self.font_family};
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent']};
                border-radius: 5px;
            }}
        """)
        layout.addWidget(self.progress)

    def create_results_display(self, layout):
        self.results_frame = QFrame()
        self.results_frame.setStyleSheet(f"""
            background-color: {COLORS['light']};
            border-radius: 10px;
        """)
        results_layout = QVBoxLayout(self.results_frame)

        results_header = QLabel('DISPOSAL INSTRUCTIONS')
        results_header.setAlignment(Qt.AlignCenter)
        results_header.setStyleSheet(f"""
            color: {COLORS['primary']};
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
            font-family: {self.font_family};
        """)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(300)
        self.results_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.results_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.results_text.setStyleSheet(f"""
            border: none;
            background-color: {COLORS['white']};
            color: {COLORS['text_dark']};
            font-size: 16px;
            padding: 15px;
            border-radius: 8px;
            font-family: {self.font_family};
        """)

        results_layout.addWidget(results_header)
        results_layout.addWidget(self.results_text)
        layout.addWidget(self.results_frame, 1)

    def update_camera_frame(self, qt_image):
        """Update the camera preview with the latest frame"""
        if not self.captured_image:
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(qt_image)
            # Scale to fit in the label while maintaining aspect ratio
            pixmap = pixmap.scaled(self.camera_label.width(), self.camera_label.height(),
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # Apply rounded corners
            rounded_pixmap = create_rounded_pixmap(pixmap, 15)
            self.camera_label.setPixmap(rounded_pixmap)

    def take_photo(self):
        """Capture an image from the camera stream without analyzing it immediately"""
        if hasattr(self, 'camera_thread') and self.camera_thread.running:
            # If we already have a captured image, reset it first
            if self.captured_image:
                self.captured_image = None
                self.take_photo_btn.setText('TAKE PHOTO')
                # Allow the camera to update before capturing
                QTimer.singleShot(100, self.take_photo)
                return

            # Get the current frame from the camera
            pixmap = self.camera_label.pixmap()
            if pixmap:
                # Convert QPixmap to QImage
                qimage = pixmap.toImage()
                width, height = qimage.width(), qimage.height()

                # Convert QImage to numpy array
                ptr = qimage.constBits()
                ptr.setsize(qimage.byteCount())
                arr = np.array(ptr).reshape(height, width, 4)  # 4 channels (RGBA)

                # Convert to RGB
                rgb_arr = arr[:, :, 0:3]

                # Convert to PIL Image
                self.captured_image = Image.fromarray(rgb_arr)

                # Update UI to indicate photo is captured
                self.take_photo_btn.setText('RETAKE PHOTO')
                self.scan_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", "Failed to capture image from camera")

    def scan_captured_image(self):
        """Process the captured image with the model"""
        if self.captured_image:
            self.progress.setValue(0)
            self.results_text.clear()

            self.scan_btn.setEnabled(False)
            self.scan_btn.setText('SCANNING...')

            self.prediction_thread = PredictionThread(self.model, pil_image=self.captured_image)
            self.prediction_thread.finished.connect(self.handle_results)
            self.prediction_thread.error.connect(self.handle_error)
            self.prediction_thread.progress.connect(self.progress.setValue)
            self.prediction_thread.start()
        else:
            QMessageBox.warning(self, "Error", "Please take a photo first")

    def upload_images(self):
        """Upload an image from file system"""
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self,
            "Select Image",
            "",
            "Image Files (*.jpg *.jpeg *.png)"
        )

        if file_paths:
            self.captured_image = None  # Clear any captured image
            self.current_image = file_paths[0]  # Take only the first image
            self.display_uploaded_image()

            # Enable the scan button but don't scan automatically
            self.take_photo_btn.setText('TAKE NEW PHOTO')
            self.scan_btn.setEnabled(True)

    def display_uploaded_image(self):
        """Display the uploaded image in the camera area"""
        if not self.current_image:
            return

        pixmap = QPixmap(self.current_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.width(), self.camera_label.height(),
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Create rounded corners
        rounded_pixmap = create_rounded_pixmap(scaled_pixmap, 15)

        self.camera_label.setPixmap(rounded_pixmap)

        # Load image into PIL format for processing
        self.captured_image = Image.open(self.current_image).convert('RGB')

    def handle_results(self, results):
        """Display the prediction results"""
        self.results_text.clear()
        for _, result in results.items():
            predicted_class = result['predicted_class']

            # Get bin information
            bin_info = BIN_MAPPING[predicted_class]
            bin_name = bin_info['bin']
            bin_color = bin_info['color']
            bin_description = bin_info['description']

            # Check if the item is unknown/random
            if predicted_class == 'random':
                # Special formatting for unknown items
                result_text = f"""
                <div style='margin-bottom: 15px; font-family: {self.font_family};'>
                    <div style='background-color: {bin_color}; color: white; padding: 12px; border-radius: 8px; margin: 15px 0; font-size: 20px; font-weight: bold; text-align: center;'>
                        {bin_name}
                    </div>

                    <div style='color: {COLORS['text_dark']}; font-style: italic; margin-top: 10px; font-size: 16px;'>
                        {bin_description}
                    </div>
                </div>
                """
            else:
                # Normal display for recognized items
                result_text = f"""
                <div style='margin-bottom: 15px; font-family: {self.font_family};'>
                    <div style='background-color: {bin_color}; color: white; padding: 12px; border-radius: 8px; margin: 15px 0; font-size: 20px; font-weight: bold; text-align: center;'>
                        DISPOSE IN: {bin_name}
                    </div>

                    <div style='color: {COLORS['text_dark']}; font-style: italic; margin-top: 10px; font-size: 16px;'>
                        {bin_description}
                    </div>
                </div>
                """

            self.results_text.append(result_text)

        # Reset buttons
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText('SCAN ITEM')

    def keyPressEvent(self, event):
        """Hidden keyboard shortcuts for manual classification"""
        if not self.captured_image:
            super().keyPressEvent(event)
            return

        key_map = {
            Qt.Key_M: 'metal',  # Metal
            Qt.Key_P: 'plastic',  # Plastic
            Qt.Key_G: 'glass',  # Glass
            Qt.Key_H: 'battery',  # Hazardous
            Qt.Key_F: 'clothes',  # Fabric/Clothes
            Qt.Key_C: 'cardboard',  # Cardboard
            Qt.Key_R: 'paper',  # Paper
            Qt.Key_B: 'biological',  # Biological
            Qt.Key_S: 'shoes',  # Shoes
            Qt.Key_T: 'trash'  # Trash
        }

        key = event.key()
        if key in key_map:
            waste_type = key_map[key]
            self.show_manual_classification(waste_type)
        else:
            super().keyPressEvent(event)

    def show_manual_classification(self, waste_type):
        """Display bin information based on manual classification"""
        if waste_type in BIN_MAPPING:
            self.results_text.clear()

            bin_info = BIN_MAPPING[waste_type]
            bin_name = bin_info['bin']
            bin_color = bin_info['color']
            bin_description = bin_info['description']

            result_text = f"""
            <div style='margin-bottom: 15px; font-family: {self.font_family};'>
                <div style='background-color: {bin_color}; color: white; padding: 12px; border-radius: 8px; margin: 15px 0; font-size: 20px; font-weight: bold; text-align: center;'>
                    DISPOSE IN: {bin_name}
                </div>

                <div style='color: {COLORS['text_dark']}; font-style: italic; margin-top: 10px; font-size: 16px;'>
                    {bin_description}
                </div>
            </div>
            """

            self.results_text.append(result_text)

    def handle_error(self, error_message):
        """Handle errors during prediction"""
        QMessageBox.warning(self, "Error", error_message)
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText('SCAN ITEM')

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = TrashClassifierGUI()
    ex.show()
    sys.exit(app.exec_())
