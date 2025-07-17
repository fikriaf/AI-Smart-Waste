# AI Smart Waste

AI Smart Waste is a Python-based system designed to recognize, classify, and manage waste using computer vision techniques. It uses image input, either from a live camera or static images, to detect types of waste such as organic, inorganic, or recyclable items.

## Features

- Real-time waste detection with camera input (`smart-waste-realtime.py`)
- Deep learning-based image classification (`smart-waste.py`
- UI testing and interface logic (`ui_testing.py`, `interface_new.py`)
- Simulated hardware interaction (`hardware.py`)
- API endpoint integration (`api.py`)
- Jupyter Notebook for exploration and visualization (`main.ipynb`)
- Includes test images for validation (`tests/ folder`)

## Requirements

- Python 3.7+
- Required Python libraries (install via pip install -r requirements.txt):
  - OpenCV
  - TensorFlow or PyTorch
  - NumPy
  - Flask (for API)

## How to Run

### 1. Waste Classification (Standalone)
```bash
python smart-waste.py
```
### 2. Realtime Camera Detection
```bash
python smart-waste-realtime.py
```
### 3. API Mode
```bash
python api.py
```
### 4. UI Testing
```bash
python ui_testing.py
```
## File Structure

```bash
├── smart-waste.py # Main classification logic
├── smart-waste-realtime.py # Realtime camera detection
├── api.py # REST API for classification
├── hardware.py # Hardware interaction simulation
├── interface_new.py # UI Interface logic
├── ui_testing.py # UI testing logic
├── main.ipynb # Notebook for experiments
├── tests/
│   ├── image.png # Test image 1
│   └── testfoto.jpg # Test image 2
```
