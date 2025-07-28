# YOLOv8 模型基準測試工具 - 函數說明文件

## 📋 目錄
- [導入模組](#導入模組)
- [核心函數](#核心函數)
- [輔助函數](#輔助函數)
- [資料處理函數](#資料處理函數)
- [視覺化函數](#視覺化函數)
- [主程式函數](#主程式函數)

---

## 導入模組

### 標準庫模組
```python
import cv2          # 影片處理和電腦視覺
import time         # 時間測量和控制
import os           # 作業系統介面
import numpy as np  # 數值計算
import argparse     # 命令列參數解析
import gc           # 垃圾回收
```

### 第三方模組
```python
import matplotlib.pyplot as plt  # 資料視覺化
import pandas as pd              # 資料處理和 CSV 匯出
import psutil                    # 系統和程序監控
import torch                     # PyTorch 深度學習框架
from ultralytics import YOLO     # YOLOv8 模型
from thop import profile         # 模型複雜度分析
```

---

## 核心函數

### `get_model_info(model)`
**功能**: 獲取模型複雜度指標（參數數量和 FLOPs）

**參數**:
- `model`: YOLO 模型物件

**返回值**:
- `params_millions`: 參數數量（百萬）
- `flops_giga`: FLOPs 數量（十億）

**詳細說明**:
```python
def get_model_info(model):
    # 創建虛擬輸入用於 FLOPs 計算
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # 計算參數和 FLOPs
    try:
        flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
        params_millions = params / 1e6
        flops_giga = flops / 1e9
    except:
        # 如果 thop 失敗，使用備用方法
        params_millions = sum(p.numel() for p in model.model.parameters()) / 1e6
        flops_giga = 0
    
    return params_millions, flops_giga
```

**技術細節**:
- 使用 `thop` 庫計算模型複雜度
- 創建 640x640 的虛擬輸入張量
- 提供錯誤處理機制
- 將結果轉換為易讀的單位（百萬參數、十億 FLOPs）

---

## 輔助函數

### `measure_memory_usage()`
**功能**: 測量當前記憶體使用量

**返回值**:
- `memory_mb`: 記憶體使用量（MB）

**詳細說明**:
```python
def measure_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb
```

**技術細節**:
- 使用 `psutil` 獲取當前程序資訊
- 計算 RSS（常駐集大小）記憶體
- 將位元組轉換為 MB

### `get_peak_memory_usage()`
**功能**: 獲取程序峰值記憶體使用量

**返回值**:
- `memory_mb`: 峰值記憶體使用量（MB）

**詳細說明**:
```python
def get_peak_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb
```

**注意**: 此函數與 `measure_memory_usage()` 相同，保留用於未來擴展

---

## 主要基準測試函數

### `benchmark_model(model_name, video_path, device='cpu')`
**功能**: 對單個 YOLOv8 模型進行基準測試

**參數**:
- `model_name`: 模型檔案名稱（如 'yolov8n.pt'）
- `video_path`: 影片檔案路徑
- `device`: 推理裝置（'cpu' 或 'gpu'，預設為 'cpu'）

**返回值**:
- 包含所有基準測試結果的字典，如果失敗則返回 `None`

**詳細說明**:
```python
def benchmark_model(model_name, video_path, device='cpu'):
    # 1. 載入模型並設定裝置
    # 2. 獲取模型複雜度指標
    # 3. 開啟影片檔案
    # 4. 執行熱身運行
    # 5. 進行基準測試推理
    # 6. 計算性能指標
    # 7. 清理資源
    # 8. 返回結果
```

**執行流程**:

#### 1. 模型載入和裝置設定
```python
model = YOLO(model_name)
if device == 'gpu':
    if torch.cuda.is_available():
        model.to('cuda:0')
    else:
        device = 'cpu'  # 降級到 CPU
else:
    model.to('cpu')
```

#### 2. 影片處理設定
```python
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
```

#### 3. 熱身運行
```python
for _ in range(5):
    ret, frame = cap.read()
    if ret:
        _ = model.predict(frame, verbose=False)
```

#### 4. 基準測試推理
```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 測量單次推理延遲
    inference_start = time.time()
    _ = model.predict(frame, verbose=False)
    inference_end = time.time()
    
    latency = (inference_end - inference_start) * 1000
    latencies.append(latency)
    
    # 追蹤峰值記憶體
    current_memory = measure_memory_usage()
    peak_memory = max(peak_memory, current_memory)
```

#### 5. 指標計算
```python
total_time = end_time - start_time
avg_fps = frame_count / total_time
avg_latency = np.mean(latencies)
min_latency = np.min(latencies)
max_latency = np.max(latencies)
memory_peak = max(0, peak_memory - memory_before)
```

#### 6. 資源清理
```python
del model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

**返回的結果字典包含**:
- `model`: 模型名稱
- `video`: 影片路徑
- `device`: 使用的裝置
- `parameters_millions`: 參數數量（百萬）
- `flops_giga`: FLOPs 數量（十億）
- `total_frames`: 處理的總幀數
- `total_time`: 總處理時間
- `avg_fps`: 平均 FPS
- `avg_latency_ms`: 平均延遲（毫秒）
- `min_latency_ms`: 最小延遲（毫秒）
- `max_latency_ms`: 最大延遲（毫秒）
- `memory_peak_mb`: 峰值記憶體使用量（MB）
- `memory_before_mb`: 推理前記憶體使用量（MB）
- `memory_after_mb`: 推理後記憶體使用量（MB）

---

## 資料處理函數

### `save_results_to_csv(results, filename='yolo_benchmark_results.csv')`
**功能**: 將基準測試結果儲存為 CSV 檔案

**參數**:
- `results`: 基準測試結果列表
- `filename`: 輸出 CSV 檔案名稱（預設為 'yolo_benchmark_results.csv'）

**返回值**:
- `df`: pandas DataFrame 物件

**詳細說明**:
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

**技術細節**:
- 使用 pandas 將結果列表轉換為 DataFrame
- 儲存為 CSV 格式，不包含索引
- 提供錯誤處理（空結果檢查）

---

## 視覺化函數

### `create_visualizations(df, video_name)`
**功能**: 從基準測試結果創建全面視覺化圖表

**參數**:
- `df`: 包含基準測試結果的 pandas DataFrame
- `video_name`: 影片名稱（用於檔案命名）

**詳細說明**:
```python
def create_visualizations(df, video_name):
    # 設定繪圖樣式
    plt.style.use('default')
    
    # 創建 2x3 子圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'YOLOv8 Benchmark Results - {video_name}', fontsize=16, fontweight='bold')
```

**生成的圖表包括**:

#### 1. FPS vs 模型大小
```python
axes[0, 0].bar(model_names, df['avg_fps'], color='skyblue', alpha=0.8)
axes[0, 0].set_title('FPS vs Model Size')
axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('Average FPS')
```

#### 2. 平均延遲 vs 模型大小
```python
axes[0, 1].bar(model_names, df['avg_latency_ms'], color='lightcoral', alpha=0.8)
axes[0, 1].set_title('Average Latency vs Model Size')
axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('Average Latency (ms)')
```

#### 3. 記憶體使用量 vs 模型大小
```python
axes[0, 2].bar(model_names, df['memory_peak_mb'], color='lightgreen', alpha=0.8)
axes[0, 2].set_title('Memory Usage vs Model Size')
axes[0, 2].set_xlabel('Model')
axes[0, 2].set_ylabel('Peak Memory (MB)')
```

#### 4. 參數 vs 模型大小
```python
axes[1, 0].bar(model_names, df['parameters_millions'], color='gold', alpha=0.8)
axes[1, 0].set_title('Parameters vs Model Size')
axes[1, 0].set_xlabel('Model')
axes[1, 0].set_ylabel('Parameters (Millions)')
```

#### 5. FLOPs vs 模型大小
```python
axes[1, 1].bar(model_names, df['flops_giga'], color='plum', alpha=0.8)
axes[1, 1].set_title('FLOPs vs Model Size')
axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('FLOPs (Giga)')
```

#### 6. FPS vs 參數（散點圖）
```python
axes[1, 2].scatter(df['parameters_millions'], df['avg_fps'], s=100, alpha=0.7, c='red')
axes[1, 2].set_title('FPS vs Parameters')
axes[1, 2].set_xlabel('Parameters (Millions)')
axes[1, 2].set_ylabel('Average FPS')
```

**技術細節**:
- 使用 matplotlib 創建 2x3 子圖佈局
- 每個子圖顯示不同的性能指標
- 包含網格線和適當的標籤
- 儲存為高解析度 PNG 檔案（300 DPI）
- 自動清理圖表物件以釋放記憶體

---

## 主程式函數

### `parse_arguments()`
**功能**: 解析命令列參數

**返回值**:
- `args`: 解析後的參數物件

**支援的參數**:

#### `--video` (必需)
- **類型**: 字串
- **說明**: 用於基準測試的影片檔案路徑
- **範例**: `--video test_short.mp4`

#### `--model` (可選)
- **類型**: 字串
- **選項**: `['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']`
- **說明**: 要測試的特定 YOLOv8 模型
- **範例**: `--model yolov8n.pt`

#### `--plot` (可選)
- **類型**: 布林值標誌
- **說明**: 生成全面視覺化圖表
- **範例**: `--plot`

#### `--device` (可選)
- **類型**: 字串
- **選項**: `['cpu', 'gpu']`
- **預設值**: `'cpu'`
- **說明**: 用於推理的裝置
- **範例**: `--device gpu`

**詳細說明**:
```python
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='YOLOv8 Comprehensive Benchmarking Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run main.py --video test_short.mp4                    # 基準測試所有模型\n"
            "  uv run main.py --video test_short.mp4 --model yolov8n.pt # 基準測試特定模型\n"
            "  uv run main.py --video test_short.mp4 --plot             # 基準測試所有模型並生成圖表\n"
            "  uv run main.py --video test_short.mp4 --device gpu       # 使用 GPU 進行推理\n"
        )
    )
    # 添加各個參數...
    return parser.parse_args()
```

### `main()`
**功能**: 主基準測試函數

**執行流程**:

#### 1. 參數解析和驗證
```python
args = parse_arguments()
video_path = args.video
generate_plots = args.plot
device = args.device
```

#### 2. 模型列表設定
```python
if args.model:
    model_names = [args.model]
else:
    model_names = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
```

#### 3. 檔案和裝置檢查
```python
# 檢查影片檔案是否存在
if not os.path.exists(video_path):
    print(f"❌ Error: Video file not found: {video_path}")
    return

# 檢查 GPU 可用性
if device == 'gpu' and not torch.cuda.is_available():
    print(f"⚠️  Warning: GPU requested but CUDA not available, falling back to CPU")
    device = 'cpu'
```

#### 4. 執行基準測試
```python
results = []
for model_name in model_names:
    result = benchmark_model(model_name, video_path, device)
    if result:
        results.append(result)
```

#### 5. 結果處理和輸出
```python
if results:
    # 儲存 CSV 結果
    df = save_results_to_csv(results)
    
    # 生成視覺化（如果要求）
    if generate_plots:
        video_name = os.path.basename(video_path).split('.')[0]
        create_visualizations(df, video_name)
    
    # 顯示摘要
    best_fps = max(results, key=lambda x: x['avg_fps'])
    fastest_latency = min(results, key=lambda x: x['avg_latency_ms'])
    lowest_memory = min(results, key=lambda x: x['memory_peak_mb'])
    
    print(f"🏆 Best FPS: {best_fps['model'].replace('.pt', '')} ({best_fps['avg_fps']:.2f} FPS)")
    print(f"⚡ Fastest Latency: {fastest_latency['model'].replace('.pt', '')} ({fastest_latency['avg_latency_ms']:.2f}ms)")
    print(f"💾 Lowest Memory: {lowest_memory['model'].replace('.pt', '')} ({lowest_memory['memory_peak_mb']:.2f}MB)")
```

**技術細節**:
- 完整的錯誤處理和驗證
- 靈活的模型選擇（單一或全部）
- 自動裝置降級（GPU → CPU）
- 進度指示和狀態訊息
- 結果摘要和最佳性能模型識別

---

## 程式入口點

### `if __name__ == "__main__":`
**功能**: 程式執行入口點

**說明**:
```python
if __name__ == "__main__":
    main()
```

**技術細節**:
- 確保程式只在直接執行時運行
- 避免在模組導入時執行主函數
- 標準的 Python 程式結構

---

## 總結

這個 YOLOv8 模型基準測試工具提供了完整的函數架構：

1. **核心功能**: 模型複雜度分析和基準測試
2. **輔助功能**: 記憶體監控和系統資源管理
3. **資料處理**: CSV 匯出和結果管理
4. **視覺化**: 全面的圖表生成
5. **使用者介面**: 命令列參數解析和錯誤處理
6. **主程式流程**: 協調所有功能的執行

每個函數都經過精心設計，提供錯誤處理、資源管理和清晰的輸出，確保基準測試過程的可靠性和易用性。 