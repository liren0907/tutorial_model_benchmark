import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import torch
from ultralytics import YOLO
from thop import profile
import gc
import argparse


def get_model_info(model):
    """
    Get model complexity metrics: parameters and FLOPs
    """
    # Create a dummy input for FLOPs calculation
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


def measure_memory_usage():
    """
    Measure current memory usage
    """
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


def get_peak_memory_usage():
    """
    Get peak memory usage during the process
    """
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


def benchmark_model(model_name, video_path, device='cpu'):
    """
    Benchmark a single YOLOv8 model
    """
    print(f"\n--- Testing Model: {model_name} on Video: {video_path} (Device: {device.upper()}) ---")

    # Load model
    try:
        model = YOLO(model_name)

        # Set device for the model
        if device == 'gpu':
            if torch.cuda.is_available():
                model.to('cuda:0')
                print(f"✓ Model {model_name} loaded successfully on GPU (cuda:0)")
            else:
                print(f"⚠️  GPU requested but CUDA not available, falling back to CPU")
                device = 'cpu'
        else:
            model.to('cpu')
            print(f"✓ Model {model_name} loaded successfully on CPU")

    except Exception as e:
        print(f"✗ Error loading model {model_name}: {e}")
        return None

    # Get model complexity metrics
    params_millions, flops_giga = get_model_info(model)
    print(f"  Parameters: {params_millions:.2f}M, FLOPs: {flops_giga:.2f}G")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Error: Cannot open video file {video_path}")
        return None

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total_frames} frames, {fps_video:.2f} FPS")

    # Memory before inference
    memory_before = measure_memory_usage()

    # Warm-up runs (discard first few inferences)
    print("  Warming up model...")
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            _ = model.predict(frame, verbose=False)

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Benchmark inference
    print("  Running benchmark...")
    frame_count = 0
    total_inference_time = 0
    latencies = []
    peak_memory = memory_before  # Track peak memory during inference

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Measure single inference latency
        inference_start = time.time()
        _ = model.predict(frame, verbose=False)
        inference_end = time.time()

        latency = (inference_end - inference_start) * 1000  # Convert to milliseconds
        latencies.append(latency)
        total_inference_time += latency

        # Track peak memory during inference
        current_memory = measure_memory_usage()
        peak_memory = max(peak_memory, current_memory)

        frame_count += 1

        # Progress indicator
        if frame_count % 50 == 0:
            print(f"    Processed {frame_count}/{total_frames} frames")

    end_time = time.time()
    cap.release()

    # Calculate metrics
    total_time = end_time - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_latency = np.mean(latencies) if latencies else 0
    min_latency = np.min(latencies) if latencies else 0
    max_latency = np.max(latencies) if latencies else 0

    # Memory after inference
    memory_after = measure_memory_usage()
    memory_peak = max(0, peak_memory - memory_before)  # Ensure non-negative peak memory

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"  ✓ Benchmark completed:")
    print(f"    FPS: {avg_fps:.2f}")
    print(f"    Avg Latency: {avg_latency:.2f}ms")
    print(f"    Memory Peak: {memory_peak:.2f}MB")

    return {
        'model': model_name,
        'video': video_path,
        'device': device,
        'parameters_millions': params_millions,
        'flops_giga': flops_giga,
        'total_frames': frame_count,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'avg_latency_ms': avg_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'memory_peak_mb': memory_peak,
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after
    }


def save_results_to_csv(results, filename='yolo_benchmark_results.csv'):
    """
    Save benchmark results to CSV file
    """
    if not results:
        print("No results to save")
        return

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\n✓ Results saved to: {filename}")
    return df


def create_visualizations(df, video_name):
    """
    Create comprehensive visualization from benchmark results
    """
    if df.empty:
        print("No data for visualization")
        return

    # Set up the plotting style
    plt.style.use('default')

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'YOLOv8 Benchmark Results - {video_name}', fontsize=16, fontweight='bold')

    # Extract model names (remove .pt extension)
    model_names = [name.replace('.pt', '') for name in df['model']]

    # 1. FPS vs Model Size
    axes[0, 0].bar(model_names, df['avg_fps'], color='skyblue', alpha=0.8)
    axes[0, 0].set_title('FPS vs Model Size')
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Average FPS')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Latency vs Model Size
    axes[0, 1].bar(model_names, df['avg_latency_ms'], color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('Average Latency vs Model Size')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Average Latency (ms)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Memory Usage vs Model Size
    axes[0, 2].bar(model_names, df['memory_peak_mb'], color='lightgreen', alpha=0.8)
    axes[0, 2].set_title('Memory Usage vs Model Size')
    axes[0, 2].set_xlabel('Model')
    axes[0, 2].set_ylabel('Peak Memory (MB)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].tick_params(axis='x', rotation=45)

    # 4. Parameters vs Model Size
    axes[1, 0].bar(model_names, df['parameters_millions'], color='gold', alpha=0.8)
    axes[1, 0].set_title('Parameters vs Model Size')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Parameters (Millions)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. FLOPs vs Model Size
    axes[1, 1].bar(model_names, df['flops_giga'], color='plum', alpha=0.8)
    axes[1, 1].set_title('FLOPs vs Model Size')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('FLOPs (Giga)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 6. FPS vs Parameters (scatter plot)
    axes[1, 2].scatter(df['parameters_millions'], df['avg_fps'], s=100, alpha=0.7, c='red')
    axes[1, 2].set_title('FPS vs Parameters')
    axes[1, 2].set_xlabel('Parameters (Millions)')
    axes[1, 2].set_ylabel('Average FPS')
    axes[1, 2].grid(True, alpha=0.3)

    # Add model labels to scatter plot
    for i, model in enumerate(model_names):
        axes[1, 2].annotate(model, (df['parameters_millions'].iloc[i], df['avg_fps'].iloc[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()

    # Save the comprehensive chart
    chart_filename = f'comprehensive_benchmark_{video_name}.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive benchmark chart saved as: {chart_filename}")
    plt.close()


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='YOLOv8 Comprehensive Benchmarking Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run main.py --video test_short.mp4                    # Benchmark all models\n"
            "  uv run main.py --video test_short.mp4 --model yolov8n.pt # Benchmark specific model\n"
            "  uv run main.py --video test_short.mp4 --plot             # Benchmark all models with plots\n"
            "  uv run main.py --video test_short.mp4 --model yolov8n.pt --plot  # Benchmark specific model with plots\n"
            "  uv run main.py --video test_short.mp4 --device gpu       # Use GPU for inference\n"
            "  uv run main.py --video test_short.mp4 --device cpu       # Use CPU for inference"
        )
    )

    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to the video file for benchmarking'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='Specific YOLOv8 model to benchmark (if not specified, all models will be tested)'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comprehensive plots and individual metric charts'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu'],
        default='cpu',
        help='Device to use for inference (default: cpu)'
    )

    return parser.parse_args()


def main():
    """
    Main benchmarking function
    """
    # Parse command line arguments
    args = parse_arguments()

    print("🚀 YOLOv8 Comprehensive Benchmarking Tool")
    print("=" * 50)

    # Configuration based on arguments
    video_path = args.video
    generate_plots = args.plot
    device = args.device

    # Set model list based on arguments
    if args.model:
        model_names = [args.model]
        print(f"🎯 Testing specific model: {args.model}")
    else:
        model_names = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        print(f"🎯 Testing all YOLOv8 models: {', '.join([m.replace('.pt', '') for m in model_names])}")

    # Check for video file
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        print("   Please ensure the video file exists in the current directory")
        return

    # Check device availability
    if device == 'gpu' and not torch.cuda.is_available():
        print(f"⚠️  Warning: GPU requested but CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"📹 Video file: {video_path}")
    print(f"🖥️  Device: {device.upper()}")
    print(f"📊 Generate plots: {'Yes' if generate_plots else 'No'}")
    print("-" * 50)

    results = []

    # Run benchmarks for the specified video and model(s)
    print(f"\n📹 Processing video: {video_path}")

    for model_name in model_names:
        result = benchmark_model(model_name, video_path, device)
        if result:
            results.append(result)

    # Save results to CSV
    if results:
        df = save_results_to_csv(results)

        # Create visualizations if --plot flag is used
        if generate_plots:
            print("\n📊 Generating visualizations...")
            video_name = os.path.basename(video_path).split('.')[0]
            create_visualizations(df, video_name)

        # Print summary
        print("\n" + "=" * 50)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 50)

        video_name = os.path.basename(video_path).split('.')[0]
        print(f"\n🎬 Video: {video_name}")
        print("-" * 30)

        # Find best performing models
        best_fps = max(results, key=lambda x: x['avg_fps'])
        fastest_latency = min(results, key=lambda x: x['avg_latency_ms'])
        lowest_memory = min(results, key=lambda x: x['memory_peak_mb'])

        print(f"🏆 Best FPS: {best_fps['model'].replace('.pt', '')} ({best_fps['avg_fps']:.2f} FPS)")
        print(f"⚡ Fastest Latency: {fastest_latency['model'].replace('.pt', '')} ({fastest_latency['avg_latency_ms']:.2f}ms)")
        print(f"💾 Lowest Memory: {lowest_memory['model'].replace('.pt', '')} ({lowest_memory['memory_peak_mb']:.2f}MB)")

        print(f"\n✅ Benchmarking completed successfully!")
        print(f"📁 Results saved to: yolo_benchmark_results.csv")

        if generate_plots:
            print(f"📊 Comprehensive chart saved to: comprehensive_benchmark_{video_name}.png")
        else:
            print(f"📊 No plots generated (use --plot flag to generate visualizations)")
    else:
        print("❌ No benchmark results generated")


if __name__ == "__main__":
    main()
