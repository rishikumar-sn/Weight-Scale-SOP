import sys
import time
import os
import serial
import serial.tools.list_ports
import re
import cv2
import json
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QFrame, QPushButton, QDialog, QComboBox,
                             QColorDialog, QFontComboBox, QSpinBox, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QFont, QColor

# --- CONFIGURATION ---
BAUD_RATE = 9600
REFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REFERENCE_DIR, "..", ".."))
CONFIG_FILE = os.path.join(REFERENCE_DIR, "config.json")
CAPTURES_DIR = os.path.join(REFERENCE_DIR, "captures")

if not os.path.exists(CAPTURES_DIR):
    os.makedirs(CAPTURES_DIR)

# --- STYLESHEET ---
MODERN_STYLE = """
QWidget {
    background-color: #1E1F22;
    color: #F3F4F6;
    font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
    font-size: 13px;
}
QDialog {
    background-color: #1E1F22;
}
QLabel {
    background: transparent;
}
QFrame#Card {
    background-color: #2B2D31;
    border: 1px solid #3F4248;
    border-radius: 12px;
}
QComboBox, QSpinBox, QFontComboBox {
    background-color: #1E1F22;
    border: 1px solid #3F4248;
    border-radius: 6px;
    color: #F3F4F6;
    padding: 6px;
}
QComboBox:drop-down {
    border-left: 1px solid #3F4248;
}
QPushButton {
    background-color: #313338;
    border: 1px solid #3F4248;
    color: #F3F4F6;
    padding: 8px 16px;
    font-weight: 600;
    border-radius: 6px;
}
QPushButton:hover { background-color: #3d4147; }
QPushButton:pressed { background-color: #242628; }

QPushButton#PrimaryBtn {
    background-color: #2563EB;
    border: none;
}
QPushButton#PrimaryBtn:hover { background-color: #3B82F6; }

QPushButton#DangerBtn {
    background-color: #DC2626;
    border: none;
}
QPushButton#DangerBtn:hover { background-color: #EF4444; }

QLabel#HeaderTitle {
    font-size: 22px;
    font-weight: 600;
}
QLabel#HeaderSub {
    font-size: 12px;
    color: #A7A7A7;
}
QLabel#StatusIndicator {
    border-radius: 6px;
}
"""

# --- HELPER FUNCTIONS ---
def load_config():
    default_config = {
        "com_port": None, 
        "camera_index": 0,
        "rotation": 0,
        "font_family": "Arial",
        "font_size": 120,
        "text_color": "#00FF00"
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                default_config.update(saved_config)
        except Exception:
            pass
            
    return default_config

def save_config(config_dict):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_dict, f, indent=4)

def create_shadow():
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(15)
    shadow.setColor(QColor(0, 0, 0, 80))
    shadow.setOffset(0, 4)
    return shadow


# --- BACKGROUND THREADS ---
class CameraWorker(QThread):
    frame_received = pyqtSignal(QImage)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latest_weight = 0.00

    def update_config(self, new_config):
        self.config = new_config

    def run(self):
        cap = cv2.VideoCapture(self.config.get("camera_index", 0), cv2.CAP_DSHOW)        
        while not self.isInterruptionRequested():
            ret, frame = cap.read()
            if ret:
                rot = self.config.get("rotation", 0)
                if rot == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rot == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                
                painter = QPainter(q_img)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                
                font = QFont(self.config.get("font_family", "Arial"))
                font.setPixelSize(self.config.get("font_size", 120))
                font.setBold(True)
                painter.setFont(font)
                
                text_color = QColor(self.config.get("text_color", "#00FF00"))
                shadow_color = QColor(0, 0, 0, 180)

                weight_text = f"{self.latest_weight:.2f} g"
                
                painter.setPen(shadow_color)
                painter.drawText(54, 550 + font.pixelSize() + 4, weight_text)
                
                painter.setPen(text_color)
                painter.drawText(50, 550 + font.pixelSize(), weight_text)

                ts_font = QFont(self.config.get("font_family", "Arial"))
                ts_font.setPixelSize(int(self.config.get("font_size", 120) * 0.5))
                ts_font.setBold(True)
                painter.setFont(ts_font)
                
                timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(timestamp_text)
                
                painter.setPen(shadow_color)
                painter.drawText(250, 600, timestamp_text)
                
                painter.setPen(text_color)
                painter.drawText(250, 607, timestamp_text)

                painter.end()
                self.frame_received.emit(q_img)
            else:
                time.sleep(0.1)
            
            time.sleep(0.03)
            
        cap.release()

class SerialWorker(QThread):
    weight_received = pyqtSignal(float)
    status_changed = pyqtSignal(str, str) 

    def __init__(self, port_name):
        super().__init__()
        self.port_name = port_name

    def run(self):
        while not self.isInterruptionRequested():
            try:
                with serial.Serial(self.port_name, BAUD_RATE, timeout=1) as ser:
                    self.status_changed.emit("active", f"PORT: {self.port_name}")
                    ser.reset_input_buffer()
                    last_valid_time = time.time()
                    
                    while not self.isInterruptionRequested():
                        if ser.in_waiting > 0:
                            if ser.in_waiting > 1024:
                                ser.reset_input_buffer()
                                continue

                            raw_data = ser.readline().decode('ascii', errors='ignore').strip()
                            if raw_data:
                                match = re.search(r'[-+]?\d*\.\d+|\d+', raw_data)
                                if match:
                                    try:
                                        weight_val = float(match.group())
                                        self.weight_received.emit(weight_val)
                                        last_valid_time = time.time() 
                                    except ValueError:
                                        pass
                        
                        if time.time() - last_valid_time > 3.0:
                            raise serial.SerialException("Watchdog Timeout")

                        time.sleep(0.05)
                        
            except serial.SerialException:
                self.status_changed.emit("error", f"RECONNECTING: {self.port_name}")
                for _ in range(20):
                    if self.isInterruptionRequested():
                        break
                    time.sleep(0.1)


# --- SETTINGS DIALOG ---
class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(450, 480)
        self.config = config.copy()
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        self.setLayout(layout)
        
        # --- HARDWARE CARD ---
        hw_card = QFrame()
        hw_card.setObjectName("Card")
        hw_layout = QVBoxLayout(hw_card)
        hw_layout.setContentsMargins(15, 15, 15, 15)
        
        hw_label = QLabel("Hardware Configuration")
        hw_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #F3F4F6; margin-bottom: 5px;")
        hw_layout.addWidget(hw_label)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("COM Port:"))
        self.port_combo = QComboBox()
        row1.addWidget(self.port_combo, 1)
        self.refresh_port_btn = QPushButton("Refresh")
        self.refresh_port_btn.clicked.connect(self.scan_ports)
        row1.addWidget(self.refresh_port_btn)
        hw_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        row2.addWidget(self.camera_combo, 1)
        self.refresh_cam_btn = QPushButton("Refresh")
        self.refresh_cam_btn.clicked.connect(self.scan_cameras)
        row2.addWidget(self.refresh_cam_btn)
        hw_layout.addLayout(row2)
        
        layout.addWidget(hw_card)

        # --- OVERLAY CARD ---
        ov_card = QFrame()
        ov_card.setObjectName("Card")
        ov_layout = QVBoxLayout(ov_card)
        ov_layout.setContentsMargins(15, 15, 15, 15)
        
        ov_label = QLabel("Text Overlay")
        ov_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #F3F4F6; margin-bottom: 5px;")
        ov_layout.addWidget(ov_label)
        
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font:"))
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(self.config.get("font_family", "Arial")))
        font_row.addWidget(self.font_combo, 1)
        ov_layout.addLayout(font_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size:"))
        self.size_spinner = QSpinBox()
        self.size_spinner.setRange(20, 300)
        self.size_spinner.setValue(self.config.get("font_size", 120))
        size_row.addWidget(self.size_spinner, 1)
        ov_layout.addLayout(size_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color:"))
        self.color_btn = QPushButton()
        self.current_color = self.config.get("text_color", "#00FF00")
        self.update_color_btn()
        self.color_btn.clicked.connect(self.pick_color)
        color_row.addWidget(self.color_btn, 1)
        ov_layout.addLayout(color_row)
        
        layout.addWidget(ov_card)
        
        layout.addStretch()
        
        # --- Save Button ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.save_btn = QPushButton("Save & Apply")
        self.save_btn.setObjectName("PrimaryBtn")
        self.save_btn.setMinimumWidth(120)
        self.save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        
        self.scan_ports()
        self.scan_cameras()

    def update_color_btn(self):
        self.color_btn.setStyleSheet(f"background-color: {self.current_color}; border: 1px solid #3F4248; color: #000;")
        self.color_btn.setText(self.current_color)

    def pick_color(self):
        color = QColorDialog.getColor(QColor(self.current_color), self, "Select Text Color")
        if color.isValid():
            self.current_color = color.name()
            self.update_color_btn()

    def scan_ports(self):
        self.port_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        if ports:
            self.port_combo.addItems(ports)
            if self.config.get("com_port") in ports:
                self.port_combo.setCurrentText(self.config["com_port"])
        else:
            self.port_combo.addItem("NO PORTS FOUND")

    def scan_cameras(self):
        self.camera_combo.clear()
        self.camera_combo.addItem("Scanning...", -1)
        QApplication.processEvents() 
        self.camera_combo.clear()

        found = False
        for i in range(4):
            cap = cv2.VideoCapture(i) 
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}", i)
                found = True
                cap.release()
        
        if not found:
            self.camera_combo.addItem("No Cameras Found", 0)
        else:
            saved_idx = self.config.get("camera_index", 0)
            index = self.camera_combo.findData(saved_idx)
            if index >= 0:
                self.camera_combo.setCurrentIndex(index)

    def get_updated_config(self):
        port = self.port_combo.currentText()
        if port == "NO PORTS FOUND": port = None
        cam_idx = self.camera_combo.currentData()
        if cam_idx is None or cam_idx == -1: cam_idx = 0
        
        self.config["com_port"] = port
        self.config["camera_index"] = cam_idx
        self.config["font_family"] = self.font_combo.currentFont().family()
        self.config["font_size"] = self.size_spinner.value()
        self.config["text_color"] = self.current_color
        
        return self.config


# --- MAIN APPLICATION ---
class WeightMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.serial_worker = None
        self.camera_worker = None
        self.latest_image = None  
        
        self.config = load_config()
        self.current_port = self.config.get("com_port")
        self.current_camera = self.config.get("camera_index", 0)
        
        self.init_ui()
        
        # UI Update Timer for Clock
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        
        self.start_camera()
        if self.current_port:
            self.start_connection(self.current_port)

    def init_ui(self):
        self.setWindowTitle("GOLD LOAN DIGITAL WEIGHING SYSTEM")
        self.resize(1024, 768)
        self.setStyleSheet(MODERN_STYLE)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        self.setLayout(main_layout)

        # --- HEADER SECTION ---
        header_layout = QHBoxLayout()
        
        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(2)
        app_title = QLabel("GOLD LOAN DIGITAL WEIGHING SYSTEM")
        app_title.setObjectName("HeaderTitle")
        app_subtitle = QLabel("")
        app_subtitle.setObjectName("HeaderSub")
        title_vbox.addWidget(app_title)
        title_vbox.addWidget(app_subtitle)
        header_layout.addLayout(title_vbox)
        
        header_layout.addStretch()

        self.logo_label = QLabel()
        pixmap = QPixmap(os.path.join(PROJECT_ROOT, "assets", "branding", "shriram.jpeg"))
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaledToHeight(100, Qt.TransformationMode.SmoothTransformation)
            scaled_pixmap.setDevicePixelRatio(2.0)
            self.logo_label.setPixmap(scaled_pixmap)
        header_layout.addWidget(self.logo_label)
        
        main_layout.addLayout(header_layout)

        # --- SUB-HEADER / INFO BAR ---
        info_card = QFrame()
        info_card.setObjectName("Card")
        info_card.setGraphicsEffect(create_shadow())
        info_layout = QHBoxLayout(info_card)
        info_layout.setContentsMargins(15, 10, 15, 10)
        
        self.connection_dot = QLabel()
        self.connection_dot.setObjectName("StatusIndicator")
        self.connection_dot.setFixedSize(12, 12)
        self.set_indicator_color(self.connection_dot, "#EF4444") # Red default
        
        self.top_status_label = QLabel("Disconnected")
        self.top_status_label.setStyleSheet("color: #A7A7A7; font-weight: bold;")
        
        self.com_label = QLabel(f"Port: {self.current_port if self.current_port else 'None'}")
        self.com_label.setStyleSheet("color: #A7A7A7;")
        
        self.cam_label = QLabel(f"Camera: {self.current_camera}")
        self.cam_label.setStyleSheet("color: #A7A7A7;")
        
        self.clock_label = QLabel(datetime.now().strftime("%d %b %Y %H:%M:%S"))
        self.clock_label.setStyleSheet("color: #A7A7A7;")
        
        info_layout.addWidget(self.connection_dot)
        info_layout.addWidget(self.top_status_label)
        info_layout.addSpacing(20)
        info_layout.addWidget(self.com_label)
        info_layout.addSpacing(20)
        info_layout.addWidget(self.cam_label)
        info_layout.addStretch()
        info_layout.addWidget(self.clock_label)
        
        main_layout.addWidget(info_card)

        # --- CENTRAL VIDEO DISPLAY ---
        video_container = QFrame()
        video_container.setObjectName("Card")
        video_container.setGraphicsEffect(create_shadow())
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(10, 10, 10, 10)
        
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; border-radius: 8px;")
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        
        main_layout.addWidget(video_container, 1) # Give it stretch factor 1

        # --- TOOLBAR SECTION ---
        toolbar_card = QFrame()
        toolbar_card.setObjectName("Card")
        toolbar_card.setGraphicsEffect(create_shadow())
        toolbar_layout = QHBoxLayout(toolbar_card)
        toolbar_layout.setContentsMargins(15, 10, 15, 10)
        toolbar_layout.setSpacing(10)
        
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.capture_image)
        toolbar_layout.addWidget(self.capture_btn)
        
        self.rotate_btn = QPushButton("Rotate")
        self.rotate_btn.clicked.connect(self.rotate_camera)
        toolbar_layout.addWidget(self.rotate_btn)
        
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        toolbar_layout.addWidget(self.settings_btn)
        
        toolbar_layout.addStretch()
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setObjectName("PrimaryBtn")
        self.connect_btn.setMinimumWidth(100)
        self.connect_btn.clicked.connect(self.toggle_connection)
        toolbar_layout.addWidget(self.connect_btn)
        
        main_layout.addWidget(toolbar_card)

        # --- FOOTER SECTION ---
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(5, 0, 5, 0)
        
        self.footer_status = QLabel("Status: Ready")
        self.footer_status.setStyleSheet("color: #A7A7A7;")
        
        self.time_label = QLabel("Last Weight Update: --")
        self.time_label.setStyleSheet("color: #A7A7A7;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        footer_layout.addWidget(self.footer_status)
        footer_layout.addStretch()
        footer_layout.addWidget(self.time_label)
        
        main_layout.addLayout(footer_layout)

    def set_indicator_color(self, label, color):
        label.setStyleSheet(f"background-color: {color}; border-radius: 6px;")

    def update_clock(self):
        self.clock_label.setText(datetime.now().strftime("%d %b %Y %H:%M:%S"))

    # --- IMAGE & CAMERA LOGIC ---
    def capture_image(self):
        if self.latest_image is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(CAPTURES_DIR, f"Capture_{timestamp}.jpg")
            success = self.latest_image.save(filename, "JPG", 90)
            
            if success:
                self.footer_status.setText(f"Saved: {filename}")
                self.footer_status.setStyleSheet("color: #22C55E;")
            else:
                self.footer_status.setText("Error saving image")
                self.footer_status.setStyleSheet("color: #EF4444;")

    def rotate_camera(self):
        current_rot = self.config.get("rotation", 0)
        new_rot = (current_rot + 90) % 360
        self.config["rotation"] = new_rot
        save_config(self.config)
        
        if self.camera_worker:
            self.camera_worker.update_config(self.config)

    def start_camera(self):
        if self.camera_worker is not None:
            self.camera_worker.requestInterruption()
            self.camera_worker.wait()
            
        self.camera_worker = CameraWorker(self.config)
        self.camera_worker.frame_received.connect(self.update_video_frame)
        self.camera_worker.start()

    def update_video_frame(self, image):
        self.latest_image = image 
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), 
                                      Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    # --- SERIAL & UI LOGIC ---
    def open_settings(self):
        if self.camera_worker is not None:
            self.camera_worker.requestInterruption()
            self.camera_worker.wait()
            self.camera_worker = None
            self.video_label.setText("Paused for Settings...")
            QApplication.processEvents() 
        
        dialog = SettingsDialog(self.config, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_config = dialog.get_updated_config()
            
            port_changed = (new_config["com_port"] != self.current_port)
            self.current_port = new_config["com_port"]
            self.current_camera = new_config["camera_index"]
            self.cam_label.setText(f"Camera: {self.current_camera}")
            
            self.config = new_config
            save_config(self.config)
            
            if port_changed and self.current_port:
                self.com_label.setText(f"Port: {self.current_port}")
                self.start_connection(self.current_port)

        self.video_label.setText("Initializing Camera...")
        self.start_camera()

    def start_connection(self, port_name):
        if self.serial_worker is not None:
            self.serial_worker.requestInterruption()
            self.serial_worker.wait()
            
        self.serial_worker = SerialWorker(port_name)
        self.serial_worker.weight_received.connect(self.update_weight_display)
        self.serial_worker.status_changed.connect(self.update_status_display)
        self.serial_worker.start()
        
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setObjectName("DangerBtn")
        self.style().unpolish(self.connect_btn)
        self.style().polish(self.connect_btn)

    def toggle_connection(self):
        if self.serial_worker is not None and self.serial_worker.isRunning():
            self.serial_worker.requestInterruption()
            self.serial_worker.wait()
            self.serial_worker = None
            
            self.connect_btn.setText("Connect")
            self.connect_btn.setObjectName("PrimaryBtn")
            self.style().unpolish(self.connect_btn)
            self.style().polish(self.connect_btn)
            
            self.set_indicator_color(self.connection_dot, "#EF4444")
            self.top_status_label.setText("Disconnected")
            self.top_status_label.setStyleSheet("color: #A7A7A7; font-weight: bold;")
        else:
            if self.current_port:
                self.start_connection(self.current_port)
            else:
                self.open_settings()

    def update_weight_display(self, weight):
        if self.camera_worker:
            self.camera_worker.latest_weight = weight
            
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(f"Last Weight Update: {current_time}")

    def update_status_display(self, status_type, message):
        if status_type == "active":
            self.set_indicator_color(self.connection_dot, "#22C55E")
            self.top_status_label.setText("Connected")
            self.top_status_label.setStyleSheet("color: #F3F4F6; font-weight: bold;")
        else:
            self.set_indicator_color(self.connection_dot, "#EF4444")
            self.top_status_label.setText(f"Error: {message}")
            self.top_status_label.setStyleSheet("color: #EF4444; font-weight: bold;")

    def closeEvent(self, event):
        if self.serial_worker is not None:
            self.serial_worker.requestInterruption()
            self.serial_worker.wait()
        if self.camera_worker is not None:
            self.camera_worker.requestInterruption()
            self.camera_worker.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WeightMonitorApp()
    window.show()
    sys.exit(app.exec())
