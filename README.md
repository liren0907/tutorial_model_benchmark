# Ultralytics YOLO Model Benchmarking Tool

### 🚀 Overview

A comprehensive YOLOv8 model benchmarking tool that measures performance metrics including FPS, latency, memory usage, parameters, and FLOPs across different YOLOv8 model variants (nano, small, medium, large, xlarge).

### ✨ Features

- **Complete Model Coverage**: Tests all YOLOv8 variants (nano, small, medium, large, xlarge)
- **Comprehensive Metrics**: Measures FPS, latency, memory usage, parameters, and FLOPs
- **Professional Benchmarking**: Includes warm-up runs, progress tracking, and resource management
- **Flexible Usage**: Command-line interface with optional plotting
- **Data Export**: CSV output with comprehensive visualizations
- **Memory Management**: Proper cleanup and peak memory tracking

### 📊 Benchmarking Metrics
- **FPS (Frames Per Second)**: Average inference speed
- **Latency**: Time taken for single inference (min/avg/max)
- **Throughput**: Number of inferences per unit time
- **Parameters**: Total number of trainable weights (in millions)
- **FLOPs**: Floating Point Operations per inference (in giga)
- **Memory Usage**: Peak memory consumption during inference


### 🛠️ Installation

#### Prerequisites
- Python 3.11 or higher
- uv package manager

#### UV python package management installation
```bash
# On MacOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Setup
```bash
# Clone the repository
git clone https://github.com/liren0907/tutorial_model_benchmark.git
cd tutorial_model_benchmark

# Install/Sync dependencies using uv
uv sync
```

### 📦 Dependencies introduction

The project uses the following key dependencies:
- `ultralytics`: YOLOv8 model loading and inference
- `opencv-python`: Video processing
- `matplotlib`: Data visualization
- `pandas`: Data manipulation and CSV export
- `psutil`: Memory usage monitoring
- `thop`: FLOPs and parameter calculation
- `torch`: PyTorch backend

### 🎯 Usage

#### Basic Usage
```bash
# Benchmark all YOLOv8 models (CSV output only)
uv run main.py --video test_short.mp4

# Benchmark specific model
uv run main.py --video test_short.mp4 --model yolov8n.pt

# Benchmark all models with comprehensive plots
uv run main.py --video test_short.mp4 --plot

# Benchmark specific model with plots
uv run main.py --video test_short.mp4 --model yolov8n.pt --plot
```

#### Command Line Arguments

| Argument | Required | Description | Options |
|----------|----------|-------------|---------|
| `--video` | Yes | Path to video file for benchmarking | Any valid video file |
| `--model` | No | Specific YOLOv8 model to test | yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt |
| `--plot` | No | Generate comprehensive visualizations | Flag (no value needed) |

### 📈 Output Files

#### CSV Results
- **File**: `yolo_benchmark_results.csv`
- **Content**: All benchmark metrics for each model
- **Format**: Structured data with columns for each metric

#### Visualizations (with `--plot` flag)
- **File**: `comprehensive_benchmark_{video_name}.png`
- **Content**: 6-panel comprehensive chart including:
  - FPS vs Model Size
  - Average Latency vs Model Size
  - Memory Usage vs Model Size
  - Parameters vs Model Size
  - FLOPs vs Model Size
  - FPS vs Parameters (scatter plot)

### 📋 Example Output

```
🚀 YOLOv8 Comprehensive Benchmarking Tool
==================================================
🎯 Testing all YOLOv8 models: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
📹 Video file: test_short.mp4
📊 Generate plots: Yes
--------------------------------------------------

--- Testing Model: yolov8n.pt on Video: test_short.mp4 ---
✓ Model yolov8n.pt loaded successfully
  Parameters: 3.16M, FLOPs: 8.7G
  Video: 150 frames, 30.00 FPS
  Warming up model...
  Running benchmark...
    Processed 50/150 frames
    Processed 100/150 frames
    Processed 150/150 frames
  ✓ Benchmark completed:
    FPS: 45.23
    Avg Latency: 22.15ms
    Memory Peak: 125.67MB

✅ Benchmarking completed successfully!
📁 Results saved to: yolo_benchmark_results.csv
📊 Comprehensive chart saved to: comprehensive_benchmark_test_short.png
```

## 📚 Documentation

### [explanation.md](explanation.md)
Detailed function documentation and technical specifications for the YOLOv8 benchmarking tool. This comprehensive guide covers all functions, parameters, return values, and implementation details for developers who want to understand the codebase architecture and extend the tool's functionality.