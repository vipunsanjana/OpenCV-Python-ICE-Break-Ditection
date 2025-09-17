# ❄️ OpenCV Ice Break Detection – Lake Michigan Analysis

This repository demonstrates an **advanced OpenCV-based system** for detecting and analyzing **ice break patterns** in **Lake Michigan** videos.
The system uses **computer vision techniques** to extract the main ice break line, generate **analytics**, and save results in multiple formats with **logging support**.

---

## 📌 Features

* Detect and track **main ice break lines** in videos
* Generate:

  * ✅ **Trend Visualizations** 📈
  * ✅ **Processed & Annotated Videos** 🎥
  * ✅ **Logs for Experiment Tracking** 📝
* Support for multiple **video output formats** (MP4, AVI, WMV, MOV)
* Professional **logging and structured output management**
* Fully modular with **`utils`** and **`lake_ice_break`** packages

---

## 📂 Project Structure

```
OPENCV-PYTHON-ICE-BREAK-DITECTION/
│── data/  
│   └── ice.mp4                          # Input video file
│
│── lake_ice_break/                      # Core detection module
│   ├─ __init__.py
│   └─ detection.py
│
│── logs/                                # Logs for runs
│   └─ ice_break_20250918_002013.log
│
│── output/                              # All generated outputs
│   ├─ lake_michigan_ice_break_analysis_20250918_002013_trend.png
│   ├─ lake_michigan_ice_break_analysis_20250918_002013.mp4
│   ├─ lake_michigan_ice_break_analysis_20250918_002013.avi
│   ├─ lake_michigan_ice_break_analysis_20250918_002013.mov
│   └─ lake_michigan_ice_break_analysis_20250918_002013.wmv
│
│── utils/                               # Utility functions
│   ├─ __init__.py
│   └─ logger.py
│
│── venv/                                # Virtual environment
│── main.py                              # Entry point script
│── requirements.txt                     # Python dependencies
│── README.md                            # Documentation
│── LICENSE
│── .gitignore
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/vipunsanjana/OpenCV-Python-ICE-Break-Ditection.git
cd OpenCV-Python-ICE-Break-Ditection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run Detection

```bash
python main.py --video data/ice.mp4 --output output/
```

### Example: Python Script

```python
from lake_ice_break.detection import detect_main_ice_break

video_path = "data/ice.mp4"
output_dir = "output"

detect_main_ice_break(video_path, output_dir)
```

---

## 📊 Example Outputs

```
- output/ → Annotated videos, analysis graphs, and trend images
- logs/   → Execution logs with timestamps
```

---

## 🛠️ Tech Stack

* Python 3.8+ 🐍
* OpenCV 🎥
* NumPy 🔢
* Matplotlib 📊
* Logging & Utilities 📝

---

## 👨‍💻 Author

**Vipun Sanjana**
*Software Engineer | AI & ML | Fullstack DevOps*


