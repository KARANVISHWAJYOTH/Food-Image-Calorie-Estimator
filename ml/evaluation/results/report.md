# 📊 NutriVision AI – Held-Out Test Split Evaluation Report

## 1. Classification Performance (Test Split)

| Metric | Value |
|---|---|
| **Top-1 Accuracy** | 78.86% |
| **Top-5 Accuracy** | 93.92% |
| **Macro Precision** | 0.7898 |
| **Macro Recall** | 0.7886 |
| **Macro F1 Score** | 0.7882 |
| **Total Test Samples** | 15150 |

## 2. Nutritional Regression Performance

| Nutritional Target | MAE | RMSE | Pearson r | R² Score | MAPE |
|---|---|---|---|---|---|
| Calories | 48.44 kcal | 67.29 kcal | 0.845 | 0.714 | 15.70% |
| Protein | 3.99 g | 5.48 g | 0.844 | 0.712 | — |
| Carbohydrates | 7.87 g | 11.00 g | 0.821 | 0.673 | — |
| Fat | 3.35 g | 4.92 g | 0.811 | 0.657 | — |
| **Overall Macro Avg** | 15.91 | 22.17 | 0.830 | 0.689 | — |

## 3. Per-Class Accuracy: 10 Most Challenging Classes

| Rank | Dish Category Name | Class Index | Test Samples | Correct | Accuracy |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | **tuna_tartare** | `99` | 150 | 76 | **50.7%** |
| 2 | **bread_pudding** | `8` | 150 | 78 | **52.0%** |
| 3 | **huevos_rancheros** | `56` | 150 | 82 | **54.7%** |
| 4 | **pork_chop** | `77` | 150 | 86 | **57.3%** |
| 5 | **ravioli** | `82` | 150 | 86 | **57.3%** |
| 6 | **foie_gras** | `39` | 150 | 91 | **60.7%** |
| 7 | **filet_mignon** | `37` | 150 | 92 | **61.3%** |
| 8 | **apple_pie** | `0` | 150 | 94 | **62.7%** |
| 9 | **chocolate_mousse** | `22` | 150 | 94 | **62.7%** |
| 10 | **steak** | `93` | 150 | 95 | **63.3%** |

## 4. Limitations & Modeling Nuances

> **Proxy-Label Calorie Noise:**  
> Food-101 images do not possess direct per-dish laboratory bomb calorimetry or chemical nutrient assays. > Instead, continuous nutritional regression targets were derived by aligning Food-101 culinary categories > to cafeteria dish ground-truth distributions from the Nutrition5k dataset using fuzzy text matching and > manual ingredient heuristics. Consequently, the reported Calorie MAE and MAPE partly reflect proxy-label > variance, recipe heterogeneity, and culinary preparation differences across real-world food captures, > rather than solely visual prediction error.