# Benchmarking Tree-Based Machine Learning Models for HOMO–LUMO Gap Prediction in tmQM Complex Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ACS JCIM Standard](https://img.shields.io/badge/ACS-JCIM%20Format-red.svg)](https://pubs.acs.org/journal/jcisd8)

This repository contains the dataset preprocessing scripts, model training pipelines, and figure generation workflows for evaluating tree-based machine learning surrogate models on the **tmQM (Transition Metal Quantum Mechanics)** dataset ($N = 195,206$).

---

## 📌 Project Overview

Density Functional Theory (DFT) calculations for transition metal complexes (TMCs) carry substantial computational overhead. Machine learning surrogate architectures offer a rapid alternative for structural and electronic property predictions. 

In this benchmark study, we systematically evaluate **Multiple Linear Regression**, **XGBoost**, and **Random Forest** models across global scalar quantum chemical descriptors to establish the empirical performance ceiling of non-spatial tabular features for predicting the HOMO–LUMO orbital gap:

$$\text{HOMO--LUMO Gap} = E_{\text{LUMO}} - E_{\text{HOMO}}$$

---

## 📊 Benchmark Results

| Model Architecture | Feature Representation | Test $R^2$ | Test RMSE (eV) |
| :--- | :--- | :---: | :---: |
| **Linear Regression Baseline** | Raw Tabular | 0.085 | 0.0321 |
| **XGBoost Regressor** | Raw Tabular | **0.294** | 0.0282 |
| **Random Forest Regressor** | Physical Descriptors | **0.738** | **0.0172** |

### Key Physical Insights
* **Primary Feature Drivers:** Partial charge on the central metal (`Metal_q`, weight $\approx 0.26$) and total electronic energy (`Electronic_E`, weight $\approx 0.21$) are the single most influential global descriptors for predicting orbital splitting.
* **Target Leakage Prevention:** Orbital energies (`HOMO_Energy` and `LUMO_Energy`) were rigorously excluded from all model training features.
* **Spatial Limitations:** Global scalar features lack local ligand-field symmetry and stereochemical coordination information, demonstrating the necessity of 3D spatial representations (e.g., GNNs or 3D graph representations).

---

## 📁 Repository Structure

```text
├── feature_importance.png       # Random Forest Descriptor Importance Rankings
├── real_parity_plot.png         # Baseline Tabular XGBoost Parity Plot (R² = 0.294)
├── tmqm_correlation_matrix.png  # Descriptor Pearson Correlation Heatmap
├── run_pipeline.py              # Main execution pipeline script
├── requirements.txt             # Python dependency requirements
├── LICENSE                      # License file
└── README.md                    # Project documentation
