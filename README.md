# AI-Based Forest Fire Smoke Detection Capstone

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

## \ud83d\udd25 Project Overview

This capstone project implements an advanced **AI pipeline for reducing forest fire disasters** by detecting smoke and fire anomalies from IoT sensor data.

Moving beyond simple binary classification, this system incorporates:

- **Robust Statistical Analysis** (Normality tests, Correlation).
- **Dual-Track Modeling**: Unsupervised Gaussian Anomaly Detection vs. Supervised Ensembles (GBM/RF/SVM).
- **Operational Intelligence**: Spatial Risk Heatmaps & Drone Deployment Strategies.

## \ud83d\udcc2 Repository Structure

| File | Description |
| :--- | :--- |
| `Final_Forest_Fire_Smoke_Detection.ipynb` | **Main Notebook**. Complete pipeline from data to drone strategy. |
| `Project_Documentation.txt` | Exhaustive step-by-step methodology guide. |
| `ARCHITECTURE.md` | System design and data flow diagrams. |
| `requirements.txt` | Python dependencies. |
| `Dataset.xlsx` | (Example) Forest Fire Smoke Dataset. |

## \ud83d\ude80 Quick Start

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

Launch Jupyter Notebook and run the main file:

```bash
jupyter notebook Final_Forest_Fire_Smoke_Detection.ipynb
```

## \ud83d\udcca Key Features

### 1. Robust Data Science

- **Descriptive Statistics**: Full skewness and kurtosis check.
- **Normality Tests**: Shapiro-Wilk test to validate scaling needs.
- **VIF Analysis**: Variance Inflation Factor to remove multicollinearity.

### 2. Advanced Modeling

- **Approach A: Gaussian Anomaly Detection**:
  - Multivariate Gaussian modeling on 'Normal' data.
  - Optimized probability thresholding ($\epsilon$).
- **Approach B: Supervised Classification**:
  - SMOTE for class imbalance.
  - Ensemble of **Gradient Boosting, Random Forest, SVM**.
  - Automatic "Winner" model selection.

### 3. Operational Intelligence

- **Risk Heatmap**: Simulates GPS coordinates and projects predicted fire risk.
- **Drone Deployment**: Uses **K-Means Clustering** to find optimal station coordinates for drone fleets.

## \ud83d\udee0\ufe0f Future Improvements

- **Edge AI**: Porting models to TFLite for on-drone interference.
- **Sensor Fusion**: Integrating real-time weather API data.
- **Time-Series**: Implementing LSTM for temporal trend analysis.

---
*Capstone Project Submission*
