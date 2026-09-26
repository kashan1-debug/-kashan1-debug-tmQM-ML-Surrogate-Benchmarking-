# Benchmarking Tree-Based Machine Learning Architectures on tmQM

This repository contains the official Python scripts for benchmarking Multiple Linear Regression, Random Forest, and XGBoost surrogate models against quantum chemical descriptors extracted from the **tmQM dataset** ($N = 86,665$).

## Overview
We evaluate standard scalar tabular parameters, transformed polynomial spaces, non-linear feature engineering, and 1D Coulomb spectrum representations to predict DFT-calculated HOMO-LUMO energy gaps in mononuclear transition metal complexes.

## Files in Repository
- `run_pipeline.py`: Main Python execution script for dataset ingestion, feature engineering, and model training.
- `requirements.txt`: Required Python dependencies.
- Generated figures: Correlation matrix, feature importance rankings, and ACS-formatted parity plots.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
