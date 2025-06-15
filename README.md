# AstraBharat Surveillance System (# Inspired by Operation Sindoor)

## Overview

AstraBharat is a secure, AI-powered surveillance system. It features a secure login, real-time stats, object detection using vision models, and analysis using LLM models for faster response in critical and emergency situations.

---

## Directory Structure

```
.
├── src/                    # Source code for GUI, login, and logic
│   ├── astra_gui_final.py  # Main dashboard and detection GUI
│   ├── gui_login.py        # Secure login and splash screen
│   ├── llm.py              # LLM integration (if present)
│   └── __pycache__/        # Python bytecode cache
├── Models/                 # Model weights for detection
│   ├── best.pt             # YOLOv8 PyTorch weights
│   └── best.onnx           # YOLOv8 ONNX weights
├── Logos/                  # Branding logos
│   ├── AstraBharat_logo.png
│   └── qualcomm_logo.png
├── images/                 # GUI assets (icons, backgrounds)
│   ├── astro_logo.png
│   ├── background1.png
│   ├── btn1.png
│   ├── hide.png
│   ├── in_space.png
│   ├── password_icon.png
│   ├── show.png
│   ├── splash.gif
│   ├── username_icon.png
│   └── vector.png
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── LICENSE                 # License information
```

---

## Setup

1. **Clone the repository** and ensure all files and folders are present.
2. **Python Version:**  
   This project requires **Python 3.12.5**.  
   You can download it from [python.org](https://www.python.org/downloads/release/python-3125/).
3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    - You may need to manually install some packages like `ultralytics`, `torch`, `opencv-python`, `matplotlib`, `pillow`, `psutil`

4. **Model Weights:**
    - Place `best.pt` and `best.onnx` in the `Models/` directory.

---

## Usage

### 1. Secure Login

```sh
python src/gui_login.py
```
- Splash screen followed by a secure login page.
- Default credentials:  
  - **SECURITY ID:** `AstraBharat`  
  - **ACCESS CODE:** `12345`

### 2. Dashboard, Detection and Analysis

```sh
python src/astra_gui_final.py
```
- Real-time detection, logs, camera view, and performance graphs.
- User interactive LLM Agent for faster and accurate analysis and feedback.

---

## Assets

- **Images:** GUI icons and backgrounds in [`images/`](images)
- **Logos:** AstraBharat and Qualcomm logos in [`Logos/`](Logos)
- **Models:** YOLOv8 weights in [`Models/`](Models)

---

## Notes

- For camera features, ensure a webcam is connected.
- Some features (like NPU usage) are simulated unless run on compatible hardware.

---

## License

© 2025 AstraBharat Defense Systems. All Rights Reserved.

---

## Authors

- Victor Azad, Piyush Kumar, Sohan Patidar, Ramesh Kumar