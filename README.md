# Project Kassandra – ML Codefest

This repository contains an end-to-end machine learning pipeline developed for Project Kassandra (ML Codefest – FinTech Track).

## Overview
The system predicts the next trading day’s closing price of a stock using historical market data and alternative sentiment data (Google Trends).

## Pipeline Structure
- Phase 1: Live data collection using yfinance and Google Trends
- Phase 2: Feature engineering, leakage-safe model training, and next-day price prediction

## How to Run
```bash
pip install -r requirements.txt
python data_check.py
python phase2_features.py
python train_model.py
