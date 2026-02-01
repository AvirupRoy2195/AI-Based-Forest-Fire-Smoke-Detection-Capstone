# AI-Based Forest Fire Smoke Detection Capstone

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This capstone project implements an advanced **AI pipeline for reducing forest fire disasters** by detecting smoke and fire anomalies from IoT sensor data.

Moving beyond simple binary classification, this system incorporates:

- **Robust Statistical Analysis** (Normality tests, Correlation).
- **Dual-Track Modeling**: Unsupervised Gaussian Anomaly Detection vs. Supervised Ensembles.
- **Operational Intelligence**: Spatial Risk Heatmaps & Drone Deployment Strategies.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Quick Start](#quick-start)
3. [Methodology](#methodology)
4. [System Architecture](#system-architecture)
5. [Key Features](#key-features)
6. [Future Improvements](#future-improvements)

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `Final_Forest_Fire_Smoke_Detection.ipynb` | **Main Notebook** - Complete pipeline from data to drone strategy |
| `Forest Fire Smoke Dataset.xlsx` | Source IoT sensor dataset |
| `ARCHITECTURE.md` | Detailed system design diagrams |
| `METHODOLOGY.md` | In-depth approach explanation |
| `requirements.txt` | Python dependencies |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/AvirupRoy2195/AI-Based-Forest-Fire-Smoke-Detection-Capstone.git
cd AI-Based-Forest-Fire-Smoke-Detection-Capstone
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Analysis

```bash
jupyter notebook Final_Forest_Fire_Smoke_Detection.ipynb
```

---

## Methodology

### Why Dual Modeling?

Forest fires are **rare events** (anomalies), but we often have labeled historical data for classification. This project implements **both approaches** to maximize robustness:

### Approach A: Gaussian Anomaly Detection (Unsupervised)

**Concept**: The vast majority of sensor readings represent "Normal" conditions. We model this distinct "Normal" state as a **Multivariate Gaussian distribution**.

**Mathematical Foundation**:

```
P(x; mu, Sigma) = (1 / ((2*pi)^(n/2) * |Sigma|^(1/2))) * exp(-0.5 * (x - mu)^T * Sigma^(-1) * (x - mu))
```

**Steps**:

1. Fit Gaussian (mu, Sigma) on **Normal class data only**
2. Calculate probability density P(x) for all samples
3. Optimize threshold (epsilon) to maximize F1-score
4. Decision: If P(x) < epsilon -> Classify as **Anomaly (Fire)**

**Advantage**: Detects *new, unseen* smoke signatures that deviate from normal patterns.

### Approach B: Supervised Ensemble (Labeled Data)

**Concept**: Leveraging labeled examples of "Fire" vs "No Fire" with robust classifiers.

**Algorithms Used**:

| Model | Purpose |
|-------|---------|
| Random Forest | Captures non-linear sensor interactions |
| SVM (RBF Kernel) | Optimal hyperplane separation |
| Gradient Boosting | Sequential error correction |
| Logistic Regression | Baseline linear model |

**Imbalance Handling**: We use **SMOTE (Synthetic Minority Over-sampling Technique)** to create synthetic fire examples, preventing the model from ignoring the minority class.

### Statistical Rigor

We don't just "fit" models - we validate assumptions:

| Test | Purpose |
|------|---------|
| **Shapiro-Wilk** | Determines if features need Log/Power transformation |
| **VIF (Variance Inflation Factor)** | Removes redundant features to improve stability |
| **Correlation Heatmap** | Identifies multicollinearity between sensors |

---

## System Architecture

### High-Level Data Flow

```
[IoT Sensors] --> [Data Ingestion] --> [Statistical Validation]
                                              |
                      +-----------------------+-----------------------+
                      |                                               |
              [Normal Data Only]                              [Labeled Data]
                      |                                               |
           [Gaussian Anomaly Model]                     [Supervised Classifiers]
                      |                                               |
                      +-----------------------+-----------------------+
                                              |
                                     [Is Anomaly?]
                                              |
                                    [Alarm Triggered]
                                              |
                                 [Spatial Risk Analysis]
                                              |
                                  [Drone Dispatch Logic]
```

### Component Design

#### A. Data Ingestion Layer

- **Input**: Raw sensor data (Temperature, Humidity, TVOC, eCO2, PM values)
- **Processing**: Deduplication, Missing Value Imputation, Outlier Detection

#### B. Statistical Analysis Engine

- **Normality Checks**: Shapiro-Wilk tests for feature distribution
- **Correlation Analysis**: Heatmaps to identify redundant features

#### C. Modeling Core (Dual-Track)

1. **Unsupervised Track**: Multivariate Gaussian with threshold optimization
2. **Supervised Track**: SMOTE + Ensemble Voting (RF, GBM, SVM)

#### D. Operational Output

- **Risk Heatmap**: Visualizes high-probability fire zones on 2D map
- **Drone Command**: K-Means Clustering to find optimal station coordinates

---

## Key Features

### 1. Robust Data Science

- Full **skewness and kurtosis** analysis
- **Shapiro-Wilk normality tests** to validate scaling needs
- **VIF Analysis** to remove multicollinearity

### 2. Advanced Modeling

- Automatic **"Winner" model selection** based on F1-score
- **Probability calibration** for reliable risk scores
- **Cross-validation** for robust performance estimation

### 3. Operational Intelligence

- **GPS Simulation**: Projects predictions onto spatial coordinates
- **Fire Risk Heatmap**: Visual intensity map of danger zones
- **Drone Deployment Strategy**: K-Means clustering for optimal station placement

---

## Future Improvements

| Enhancement | Description |
|-------------|-------------|
| **Edge AI** | Port models to TFLite for on-drone inference |
| **Sensor Fusion** | Integrate real-time weather API data |
| **Time-Series** | Implement LSTM for temporal trend analysis |
| **Real GPS** | Replace simulation with actual sensor coordinates |

---

## License

MIT License - See LICENSE file for details.

---
*Capstone Project Submission - AI-Based Forest Fire Smoke Detection*
