# Benchmarking Tree-Based Machine Learning Models for HOMO–LUMO Gap Prediction in tmQM Complex Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ACS JCIM Standard](https://img.shields.io/badge/ACS-JCIM%20Format-red.svg)](https://pubs.acs.org/journal/jcisd8)

This repository contains the dataset preprocessing scripts, model training pipelines, and figure generation workflows for evaluating tree-based machine learning surrogate models on the **tmQM (Transition Metal Quantum Mechanics)** dataset ($N = 195,206$).

---

## 📌 Project Overview

Density Functional Theory (DFT) calculations for transition metal complexes (TMCs) carry substantial computational overhead. Machine learning surrogate architectures offer a rapid alternative for structural and electronic property predictions. 

In this benchmark study, we systematically evaluate **Multiple Linear Regression**, **XGBoost**, and **Random Forest** models across global scalar quantum chemical descriptors to establish the empirical performance ceiling of non-spatial tabular features for predicting the HOMO–LUMO orbital gap:

$$\mathrm{HOMO\text{--}LUMO\ Gap} = E_{\mathrm{LUMO}} - E_{\mathrm{HOMO}}$$

---

## 📊 Benchmark Results

| Model Architecture | Feature Representation | Test $R^2$ | Test RMSE (eV) |
| :--- | :--- | :---: | :---: |
| **Linear Regression Baseline** | Raw Tabular | 0.085 | 0.0321 |
| **XGBoost Regressor** | Raw Tabular | **0.294** | 0.0282 |
| **Random Forest Regressor** | Physical Descriptors | **0.738** | **0.0172** |

### Key Physical Insights
* **Primary Feature Drivers:** Partial charge on the central metal ($\text{Metal\_q}$, weight $\approx 0.26$) and total electronic energy ($\text{Electronic\_E}$, weight $\approx 0.21$) are the single most influential global descriptors for predicting orbital splitting.
* **Target Leakage Prevention:** Orbital energies ($\text{HOMO\_Energy}$ and $\text{LUMO\_Energy}$) were rigorously excluded from all model training features.
* **Spatial Limitations:** Global scalar features lack local ligand-field symmetry and stereochemical coordination information, demonstrating the necessity of 3D spatial representations (e.g., GNNs or 3D graph representations).

---

## 📁 Repository Structure

```text
├── data/
│   └── tmQM_dataset.csv                   # Raw parsed tmQM quantum descriptors (N=195,206)
├── figures/
│   ├── real_parity_plot (1).png           # Baseline Tabular XGBoost Parity Plot (R² = 0.294)
│   ├── feature_importance (1).png         # Random Forest Descriptor Importance Rankings
│   └── tmqm_correlation_matrix (1).png    # Descriptor Pearson Correlation Heatmap
├── scripts/
│   ├── 01_data_preprocessing.py          # Dataset parsing and target leakage prevention
│   ├── 02_model_benchmarking.py          # Model training and metric evaluations
│   └── 03_plot_generation.py             # Generates publication-ready figures
├── main.tex                               # Manuscript LaTeX source code (ACS Format)
├── tmQM_ML_Surrogate_Model_Manuscript.pdf # Compiled Manuscript PDF
├── README.md                              # Project documentation
└── LICENSE                                # MIT License
