# NutriVision AI: Model Optimization & Latency Benchmark Report

Comparison of Original FP32 Baseline vs. Pruned (20%, 30%, 50%) and INT8 Quantized models.

| Model Variant | Type / Sparsity | Size (MB) | Size Delta | CPU Latency Mean (ms) | CPU P95 (ms) | CPU Speedup | GPU Latency (ms) | Top-1 Acc (%) | Top-5 Acc (%) | Calorie MAE (kcal) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FP32 Baseline** | FP32 Dense | 96.17 MB | Baseline | 44.81 ms | 49.62 ms | **1.0x** | 8.29 ms | 77.93% | 94.41% | 48.99 kcal |
| **Pruned (20%)** | 20% Conv L1 Sparsity | 96.19 MB | Baseline | 47.2 ms | 51.68 ms | **0.95x** | 8.74 ms | 76.2% | 93.02% | 49.8 kcal |
| **Pruned (30%)** | 30% Conv L1 Sparsity | 96.19 MB | Baseline | 44.1 ms | 47.04 ms | **1.02x** | 8.59 ms | 75.93% | 92.75% | 50.56 kcal |
| **Pruned (50%)** | 50% Conv L1 Sparsity | 96.19 MB | Baseline | 45.71 ms | 51.82 ms | **0.98x** | 9.04 ms | 75.07% | 92.49% | 51.61 kcal |
| **INT8 Quantized** | Dynamic INT8 Linear | 91.65 MB | -4.7% | 45.39 ms | 49.08 ms | **0.99x** | N/A | 79.52% | 94.48% | 47.94 kcal |

> **Benchmark Setup:** Latencies measured on 100 iterations after 15 warmup cycles using `torch.randn(1, 3, 224, 224)` inputs.
> **Note on Quantization:** Dynamic INT8 converts Linear layer weights to 8-bit integers, optimized specifically for fast CPU inference.
> **Accuracy Setup:** Top-1, Top-5 and calorie MAE are measured on 1504 held-out test-split images (raw kcal targets, model outputs de-normalized).
