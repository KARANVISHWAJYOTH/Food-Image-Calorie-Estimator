# ⚡ NutriVision AI: Model Optimization & Latency Benchmark Report

Comparison of Original FP32 Baseline vs. Pruned (20%, 30%, 50%) and INT8 Quantized models.

| Model Variant | Type / Sparsity | Size (MB) | Size Delta | CPU Latency Mean (ms) | CPU P95 (ms) | CPU Speedup | GPU Latency (ms) | Top-1 Acc (%) | Calorie MAE (kcal) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FP32 Baseline** | FP32 Dense | 98.4 MB | Baseline | 43.67 ms | 49.87 ms | **1.0x** | N/A | N/A | N/A |
| **Pruned (20%)** | 20% Conv L1 Sparsity | 96.19 MB | -2.2% | 42.44 ms | 49.64 ms | **1.03x** | N/A | N/A | N/A |
| **Pruned (30%)** | 30% Conv L1 Sparsity | 96.19 MB | -2.2% | 41.96 ms | 49.75 ms | **1.04x** | N/A | N/A | N/A |
| **Pruned (50%)** | 50% Conv L1 Sparsity | 96.19 MB | -2.2% | 42.62 ms | 48.72 ms | **1.02x** | N/A | N/A | N/A |
| **INT8 Quantized** | Dynamic INT8 Linear | 91.65 MB | -6.9% | 41.51 ms | 44.92 ms | **1.05x** | N/A | N/A | N/A |

> **Benchmark Setup:** Latencies measured on 100 iterations after 15 warmup cycles using `torch.randn(1, 3, 224, 224)` inputs.
> **Note on Quantization:** Dynamic INT8 converts Linear layer weights to 8-bit integers, optimized specifically for fast CPU inference.