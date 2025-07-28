# YOLOv8 Model Benchmarking Tool - Function Documentation

## 📋 Table of Contents
- [Module Imports](#module-imports)
- [Core Functions](#core-functions)
- [Helper Functions](#helper-functions)
- [Data Processing Functions](#data-processing-functions)
- [Visualization Functions](#visualization-functions)
- [Main Program Functions](#main-program-functions)

---

## Module Imports

### Standard Library Modules
```python
import cv2          # Video processing and computer vision
import time         # Time measurement and control
import os           # Operating system interface
import numpy as np  # Numerical computation
import argparse     # Command line argument parsing
import gc           # Garbage collection
```

### Third-party Modules
```python
import matplotlib.pyplot as plt  # Data visualization
import pandas as pd              # Data processing and CSV export
import psutil                    # System and process monitoring
import torch                     # PyTorch deep learning framework
from ultralytics import YOLO     # YOLOv8 model
from thop import profile         # Model complexity analysis
```

---

## Core Functions

### `get_model_info(model)`
**Function**: Get model complexity metrics (parameter count and FLOPs)

**Parameters**:
- `model`: YOLO model object

**Returns**:
- `params_millions`: Parameter count (millions)
- `flops_giga`: FLOPs count (billions)

**Detailed Description**:
```python
def get_model_info(model):
    # Create dummy input for FLOPs calculation
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Calculate parameters and FLOPs
    try:
        flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
        params_millions = params / 1e6
        flops_giga = flops / 1e9
    except:
        # Fallback if thop fails
        params_millions = sum(p.numel() for p in model.model.parameters()) / 1e6
        flops_giga = 0
    
    return params_millions, flops_giga
```

**Technical Details**:
- Uses `thop` library to calculate model complexity
- Creates 640x640 dummy input tensor
- Provides error handling mechanism
- Converts results to readable units (millions of parameters, billions of FLOPs)

---

## Helper Functions

### `measure_memory_usage()`
**Function**: Measure current memory usage

**Returns**:
- `memory_mb`: Memory usage (MB)

**Detailed Description**:
```python
def measure_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb
```

**Technical Details**:
- Uses `psutil` to get current process information
- Calculates RSS (Resident Set Size) memory
- Converts bytes to MB

### `get_peak_memory_usage()`
**Function**: Get peak memory usage during the process

**Returns**:
- `memory_mb`: Peak memory usage (MB)

**Detailed Description**:
```python
def get_peak_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb
```

**Note**: This function is identical to `measure_memory_usage()`, retained for future expansion

---

## Main Benchmarking Function

### `benchmark_model(model_name, video_path, device='cpu')`
**Function**: Benchmark a single YOLOv8 model

**Parameters**:
- `model_name`: Model file name (e.g., 'yolov8n.pt')
- `video_path`: Video file path
- `device`: Inference device ('cpu' or 'gpu', default is 'cpu')

**Returns**:
- Dictionary containing all benchmark results, or `None` if failed

**Detailed Description**:
```python
def benchmark_model(model_name, video_path, device='cpu'):
    # 1. Load model and set device
    # 2. Get model complexity metrics
    # 3. Open video file
    # 4. Perform warm-up runs
    # 5. Execute benchmark inference
    # 6. Calculate performance metrics
    # 7. Clean up resources
    # 8. Return results
```

**Execution Flow**:

#### 1. Model Loading and Device Setup
```python
model = YOLO(model_name)
if device == 'gpu':
    if torch.cuda.is_available():
        model.to('cuda:0')
    else:
        device = 'cpu'  # Fallback to CPU
else:
    model.to('cpu')
```

#### 2. Video Processing Setup
```python
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
```

#### 3. Warm-up Runs
```python
for _ in range(5):
    ret, frame = cap.read()
    if ret:
        _ = model.predict(frame, verbose=False)
```

#### 4. Benchmark Inference
```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Measure single inference latency
    inference_start = time.time()
    _ = model.predict(frame, verbose=False)
    inference_end = time.time()
    
    latency = (inference_end - inference_start) * 1000
    latencies.append(latency)
    
    # Track peak memory
    current_memory = measure_memory_usage()
    peak_memory = max(peak_memory, current_memory)
```

#### 5. Metric Calculation
```python
total_time = end_time - start_time
avg_fps = frame_count / total_time
avg_latency = np.mean(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)
memory_peak = max(0, peak_memory - memory_before)
```

#### 6. Resource Cleanup
```python
del model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

**Returned Result Dictionary Contains**:
- `model`: Model name
- `video`: Video path
- `device`: Used device
- `parameters_millions`: Parameter count (millions)
- `flops_giga`: FLOPs count (billions)
- `total_frames`: Total processed frames
- `total_time`: Total processing time
- `avg_fps`: Average FPS
- `avg_latency_ms`: Average latency (milliseconds)
- `min_latency_ms`: Minimum latency (milliseconds)
- `max_latency_ms`: Maximum latency (milliseconds)
- `memory_peak_mb`: Peak memory usage (MB)
- `memory_before_mb`: Memory usage before inference (MB)
- `memory_after_mb`: Memory usage after inference (MB)

---

## Data Processing Functions

### `save_results_to_csv(results, filename='yolo_benchmark_results.csv')`
**Function**: Save benchmark results to CSV file

**Parameters**:
- `results`: List of benchmark results
- `filename`: Output CSV file name (default is 'yolo_benchmark_results.csv')

**Returns**:
- `df`: pandas DataFrame object

**Detailed Description**:
```python
def save_results_to_csv(results, filename='yolo_benchmark_results.csv'):
    if not results:
        print("No results to save")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\n✓ Results saved to: {filename}")
    return df
```

**Technical Details**:
- Uses pandas to convert result list to DataFrame
- Saves in CSV format without index
- Provides error handling (empty results check)

---

## Visualization Functions

### `create_visualizations(df, video_name)`
**Function**: Create comprehensive visualization charts from benchmark results

**Parameters**:
- `df`: pandas DataFrame containing benchmark results
- `video_name`: Video name (for file naming)

**Detailed Description**:
```python
def create_visualizations(df, video_name):
    # Set plotting style
    plt.style.use('default')
    
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'YOLOv8 Benchmark Results - {video_name}', fontsize=16, fontweight='bold')
```

**Generated Charts Include**:

#### 1. FPS vs Model Size
```python
axes[0, 0].bar(model_names, df['avg_fps'], color='skyblue', alpha=0.8)
axes[0, 0].set_title('FPS vs Model Size')
axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('Average FPS')
```

#### 2. Average Latency vs Model Size
```python
axes[0, 1].bar(model_names, df['avg_latency_ms'], color='lightcoral', alpha=0.8)
axes[0, 1].set_title('Average Latency vs Model Size')
axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('Average Latency (ms)')
```

#### 3. Memory Usage vs Model Size
```python
axes[0, 2].bar(model_names, df['memory_peak_mb'], color='lightgreen', alpha=0.8)
axes[0, 2].set_title('Memory Usage vs Model Size')
axes[0, 2].set_xlabel('Model')
axes[0, 2].set_ylabel('Peak Memory (MB)')
```

#### 4. Parameters vs Model Size
```python
axes[1, 0].bar(model_names, df['parameters_millions'], color='gold', alpha=0.8)
axes[1, 0].set_title('Parameters vs Model Size')
axes[1, 0].set_xlabel('Model')
axes[1, 0].set_ylabel('Parameters (Millions)')
```

#### 5. FLOPs vs Model Size
```python
axes[1, 1].bar(model_names, df['flops_giga'], color='plum', alpha=0.8)
axes[1, 1].set_title('FLOPs vs Model Size')
axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('FLOPs (Giga)')
```

#### 6. FPS vs Parameters (Scatter Plot)
```python
axes[1, 2].scatter(df['parameters_millions'], df['avg_fps'], s=100, alpha=0.7, c='red')
axes[1, 2].set_title('FPS vs Parameters')
axes[1, 2].set_xlabel('Parameters (Millions)')
axes[1, 2].set_ylabel('Average FPS')
```

**Technical Details**:
- Uses matplotlib to create 2x3 subplot layout
- Each subplot displays different performance metrics
- Includes grid lines and appropriate labels
- Saves as high-resolution PNG file (300 DPI)
- Automatically cleans up chart objects to free memory

---

## Main Program Functions

### `parse_arguments()`
**Function**: Parse command line arguments

**Returns**:
- `args`: Parsed arguments object

**Supported Parameters**:

#### `--video` (Required)
- **Type**: String
- **Description**: Path to video file for benchmarking
- **Example**: `--video test_short.mp4`

#### `--model` (Optional)
- **Type**: String
- **Options**: `['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']`
- **Description**: Specific YOLOv8 model to test
- **Example**: `--model yolov8n.pt`

#### `--plot` (Optional)
- **Type**: Boolean flag
- **Description**: Generate comprehensive visualization charts
- **Example**: `--plot`

#### `--device` (Optional)
- **Type**: String
- **Options**: `['cpu', 'gpu']`
- **Default**: `'cpu'`
- **Description**: Device to use for inference
- **Example**: `--device gpu`

**Detailed Description**:
```python
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='YOLOv8 Comprehensive Benchmarking Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run main.py --video test_short.mp4                    # Benchmark all models\n"
            "  uv run main.py --video test_short.mp4 --model yolov8n.pt # Benchmark specific model\n"
            "  uv run main.py --video test_short.mp4 --plot             # Benchmark all models with plots\n"
            "  uv run main.py --video test_short.mp4 --device gpu       # Use GPU for inference\n"
        )
    )
    # Add various parameters...
    return parser.parse_args()
```

### `main()`
**Function**: Main benchmarking function

**Execution Flow**:

#### 1. Argument Parsing and Validation
```python
args = parse_arguments()
video_path = args.video
generate_plots = args.plot
device = args.device
```

#### 2. Model List Setup
```python
if args.model:
    model_names = [args.model]
else:
    model_names = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
```

#### 3. File and Device Checking
```python
# Check if video file exists
if not os.path.exists(video_path):
    print(f"❌ Error: Video file not found: {video_path}")
    return

# Check GPU availability
if device == 'gpu' and not torch.cuda.is_available():
    print(f"⚠️  Warning: GPU requested but CUDA not available, falling back to CPU")
    device = 'cpu'
```

#### 4. Execute Benchmarking
```python
results = []
for model_name in model_names:
    result = benchmark_model(model_name, video_path, device)
    if result:
        results.append(result)
```

#### 5. Result Processing and Output
```python
if results:
    # Save CSV results
    df = save_results_to_csv(results)
    
    # Generate visualizations (if requested)
    if generate_plots:
        video_name = os.path.basename(video_path).split('.')[0]
        create_visualizations(df, video_name)
    
    # Display summary
    best_fps = max(results, key=lambda x: x['avg_fps'])
    fastest_latency = min(results, key=lambda x: x['avg_latency_ms'])
    lowest_memory = min(results, key=lambda x: x['memory_peak_mb'])
    
    print(f"🏆 Best FPS: {best_fps['model'].replace('.pt', '')} ({best_fps['avg_fps']:.2f} FPS)")
    print(f"⚡ Fastest Latency: {fastest_latency['model'].replace('.pt', '')} ({fastest_latency['avg_latency_ms']:.2f}ms)")
    print(f"💾 Lowest Memory: {lowest_memory['model'].replace('.pt', '')} ({lowest_memory['memory_peak_mb']:.2f}MB)")
```

**Technical Details**:
- Complete error handling and validation
- Flexible model selection (single or all)
- Automatic device fallback (GPU → CPU)
- Progress indicators and status messages
- Result summary and best performance model identification

---

## Program Entry Point

### `if __name__ == "__main__":`
**Function**: Program execution entry point

**Description**:
```python
if __name__ == "__main__":
    main()
```

**Technical Details**:
- Ensures program only runs when executed directly
- Prevents main function execution during module import
- Standard Python program structure

---

## Summary

This YOLOv8 model benchmarking tool provides a complete function architecture:

1. **Core Functions**: Model complexity analysis and benchmarking
2. **Helper Functions**: Memory monitoring and system resource management
3. **Data Processing**: CSV export and result management
4. **Visualization**: Comprehensive chart generation
5. **User Interface**: Command line argument parsing and error handling
6. **Main Program Flow**: Coordination of all function execution

Each function is carefully designed with error handling, resource management, and clear output to ensure the reliability and usability of the benchmarking process. 