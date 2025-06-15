# AstraBharat Surveillance System

## Overview

This project is a comprehensive surveillance and object detection suite developed for AstraBharat Defense Systems. It features real-time video analytics, data logging, and a secure login interface, leveraging state-of-the-art YOLO object detection models and a user-friendly GUI.

---

## Features

- **Secure Login:** Modern login interface with splash screen ([gui_login.py](gui_login.py))
- **YOLOv8 Detection:** Snapdragon NPU-optimized YOLOv8 tracking ([detection.py](detection.py))
- **YOLOv8 Quantized:** Quantized YOLOv8 detection for efficient inference ([yolov8-quant.py](yolov8-quant.py))
- **Performance Graphs:** Live CPU and  NPU performance graphs
- **Detection Logs:** Save detection logs as CSV
- **Custom Branding:** Includes AstraBharat logos

---

## Directory Structure

```
.
├── best.pt                  # Custom YOLOv8 model weights
├── yolov8n.pt               # YOLOv8 nano model weights
├── detection.py             # Snapdragon NPU-optimized YOLOv8 GUI
├── gui_login.py             # Secure login and splash screen
├── tempCodeRunnerFile.py    # Main telemetry dashboard and detection GUI
├── yolo-nas.py              # YOLO-NAS detection GUI with quantization
├── yolov8-quant.py          # Quantized YOLOv8 detection GUI
├── requirements.txt         # Python dependencies
├── images/                  # GUI assets (icons, backgrounds)
├── Logos/                   # Branding logos
```

---

## Setup

1. **Clone the repository** and ensure all files and folders are present.
2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    *(You may need to manually install some packages like `ultralytics`, `torch`, `opencv-python`, `matplotlib`, `pillow`, `psutil`, etc.)*

3. **Model Weights:**
    - `best.pt` and `yolov8n.pt` should be present in the root directory.
    - For YOLO-NAS, pretrained weights are downloaded automatically.

---

## Usage

### 1. Secure Login

```sh
python gui_login.py
```
- Splash screen followed by a secure login page.
- Default credentials:  
  - **SECURITY ID:** `AstraBharat`  
  - **ACCESS CODE:** `12345`

### 2. Telemetry Dashboard & Detection

```sh
python tempCodeRunnerFile.py
```
- Real-time telemetry, detection logs, camera view, and performance graphs.

### 3. Snapdragon YOLOv8 Detection

```sh
python detection.py
```
- Optimized for Snapdragon NPU (runs on CPU if NPU unavailable).

### 4. YOLO-NAS Detection

```sh
python yolo-nas.py
```
- Advanced detection with quantization support.

### 5. YOLOv8 Quantized Detection

```sh
python yolov8-quant.py
```
- Efficient detection using quantized YOLOv8 models.

---

## Assets

- **Images:** GUI icons and backgrounds in [`images/`](images)
- **Logos:** AstraBharat and Qualcomm logos in [`Logos/`](Logos)

---

## Notes

- For camera features, ensure a webcam is connected.
- Serial telemetry requires a device on `COM5` (can be changed in code).
- Some features (like NPU usage) are simulated unless run on compatible hardware.

---

## License

© 2025 AstraBharat Defense Systems. All Rights Reserved.

---

## Authors

- AstraBharat Defense Systems