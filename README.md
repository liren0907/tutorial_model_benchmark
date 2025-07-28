# Ultralytics YOLO Model Benchmarking Tool

[English](#english) | [繁體中文](#traditional-chinese)

---

## English

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

#### Performance Metrics
- **FPS (Frames Per Second)**: Average inference speed
- **Latency**: Time taken for single inference (min/avg/max)
- **Throughput**: Number of inferences per unit time

#### Model Complexity Metrics
- **Parameters**: Total number of trainable weights (in millions)
- **FLOPs**: Floating Point Operations per inference (in giga)

#### Resource Usage Metrics
- **Memory Usage**: Peak memory consumption during inference
- **Memory Tracking**: Real-time memory monitoring

### 🛠️ Installation

#### Prerequisites
- Python 3.11 or higher
- uv package manager

#### Setup
```bash
# Clone the repository
git clone <repository-url>
cd yolo_model_benchmark

# Install dependencies using uv
uv sync
```

### 📦 Dependencies

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

### 🔧 Technical Details

#### Benchmarking Process
1. **Model Loading**: Automatic download if not present locally
2. **Warm-up Runs**: 5 inference cycles to avoid cold-start effects
3. **Real-time Monitoring**: Memory and latency tracking during inference
4. **Resource Cleanup**: Proper garbage collection and CUDA cache clearing
5. **Data Collection**: Comprehensive metrics recording

#### Memory Management
- **Peak Tracking**: Real-time memory usage monitoring
- **Non-negative Values**: Ensures accurate memory measurements
- **Resource Cleanup**: Automatic cleanup between models

#### Supported Models
| Model | Parameters | Typical FLOPs | Use Case |
|-------|------------|---------------|----------|
| YOLOv8n | ~3.2M | ~8.7G | Fast inference, edge devices |
| YOLOv8s | ~11.2M | ~28.6G | Balanced speed/accuracy |
| YOLOv8m | ~25.9M | ~78.9G | High accuracy applications |
| YOLOv8l | ~43.7M | ~165.2G | Professional applications |
| YOLOv8x | ~68.2M | ~257.8G | Maximum accuracy |

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

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Traditional Chinese

### 🚀 概述

一個全面的 YOLOv8 模型基準測試工具，可測量不同 YOLOv8 模型變體（nano、small、medium、large、xlarge）的性能指標，包括 FPS、延遲、記憶體使用量、參數數量和 FLOPs。

### ✨ 功能特色

- **完整模型覆蓋**: 測試所有 YOLOv8 變體（nano、small、medium、large、xlarge）
- **全面指標測量**: 測量 FPS、延遲、記憶體使用量、參數數量和 FLOPs
- **專業基準測試**: 包含熱身運行、進度追蹤和資源管理
- **靈活使用**: 命令列介面，可選繪圖功能
- **資料匯出**: CSV 輸出和全面視覺化
- **記憶體管理**: 適當清理和峰值記憶體追蹤

### 📊 基準測試指標

#### 性能指標
- **FPS (每秒幀數)**: 平均推理速度
- **延遲**: 單次推理所需時間（最小/平均/最大）
- **吞吐量**: 單位時間內的推理次數

#### 模型複雜度指標
- **參數**: 可訓練權重總數（以百萬計）
- **FLOPs**: 每次推理的浮點運算次數（以十億計）

#### 資源使用指標
- **記憶體使用量**: 推理期間的峰值記憶體消耗
- **記憶體追蹤**: 即時記憶體監控

### 🛠️ 安裝

#### 前置需求
- Python 3.11 或更高版本
- uv 套件管理器

#### 設定
```bash
# 複製儲存庫
git clone <repository-url>
cd yolo_model_benchmark

# 使用 uv 安裝依賴項
uv sync
```

### 📦 依賴項

專案使用以下關鍵依賴項：
- `ultralytics`: YOLOv8 模型載入和推理
- `opencv-python`: 影片處理
- `matplotlib`: 資料視覺化
- `pandas`: 資料操作和 CSV 匯出
- `psutil`: 記憶體使用量監控
- `thop`: FLOPs 和參數計算
- `torch`: PyTorch 後端

### 🎯 使用方法

#### 基本使用
```bash
# 基準測試所有 YOLOv8 模型（僅 CSV 輸出）
uv run main.py --video test_short.mp4

# 基準測試特定模型
uv run main.py --video test_short.mp4 --model yolov8n.pt

# 基準測試所有模型並生成全面圖表
uv run main.py --video test_short.mp4 --plot

# 基準測試特定模型並生成圖表
uv run main.py --video test_short.mp4 --model yolov8n.pt --plot
```

#### 命令列參數

| 參數 | 必需 | 描述 | 選項 |
|------|------|------|------|
| `--video` | 是 | 用於基準測試的影片檔案路徑 | 任何有效的影片檔案 |
| `--model` | 否 | 要測試的特定 YOLOv8 模型 | yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt |
| `--plot` | 否 | 生成全面視覺化 | 標誌（無需值） |

### 📈 輸出檔案

#### CSV 結果
- **檔案**: `yolo_benchmark_results.csv`
- **內容**: 每個模型的所有基準測試指標
- **格式**: 結構化資料，每列代表一個指標

#### 視覺化（使用 `--plot` 標誌）
- **檔案**: `comprehensive_benchmark_{video_name}.png`
- **內容**: 6 面板全面圖表，包括：
  - FPS vs 模型大小
  - 平均延遲 vs 模型大小
  - 記憶體使用量 vs 模型大小
  - 參數 vs 模型大小
  - FLOPs vs 模型大小
  - FPS vs 參數（散點圖）

### 🔧 技術細節

#### 基準測試流程
1. **模型載入**: 如果本地不存在則自動下載
2. **熱身運行**: 5 次推理循環以避免冷啟動效應
3. **即時監控**: 推理期間的記憶體和延遲追蹤
4. **資源清理**: 適當的垃圾回收和 CUDA 快取清理
5. **資料收集**: 全面指標記錄

#### 記憶體管理
- **峰值追蹤**: 即時記憶體使用量監控
- **非負值**: 確保準確的記憶體測量
- **資源清理**: 模型間自動清理

#### 支援的模型
| 模型 | 參數 | 典型 FLOPs | 使用場景 |
|------|------|------------|----------|
| YOLOv8n | ~3.2M | ~8.7G | 快速推理，邊緣裝置 |
| YOLOv8s | ~11.2M | ~28.6G | 平衡速度/準確性 |
| YOLOv8m | ~25.9M | ~78.9G | 高準確性應用 |
| YOLOv8l | ~43.7M | ~165.2G | 專業應用 |
| YOLOv8x | ~68.2M | ~257.8G | 最高準確性 |

### 📋 範例輸出

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

### 🤝 貢獻

1. Fork 儲存庫
2. 建立功能分支
3. 進行您的更改
4. 如果適用，添加測試
5. 提交拉取請求

### 📄 授權

本專案採用 MIT 授權條款 - 詳見 LICENSE 檔案。
