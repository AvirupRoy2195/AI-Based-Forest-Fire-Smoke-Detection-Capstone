# Methodology Explanation

## 1. Why Dual Modeling?

Forest fires are rare events ("Anomalies"), but we often have labeled historical data ("Classification"). This project implements **both** approaches to maximize robustness.

### Approach A: Gaussian Anomaly Detection (Unsupervised)

- **Concept**: The vast majority of sensor readings represent "Normal" conditions. We model this distinct "Normal" state as a Multivariate Gaussian distribution.
- **Math**:
  $$P(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$
- **Advantage**: Detects *new, unseen* types of smoke signatures that deviate from the norm, even if not present in the training set.

### Approach B: Supervised Ensemble (Supervised)

- **Concept**: leveraging labeled examples of "Fire" vs "No Fire".
- **Algorithms**:
  - **Random Forest**: Captures non-linear interactions between sensors (e.g., High Temp + Low Humidity).
  - **SVM**: Finds the optimal hyperplane separating fire signatures.
- **Imbalance Handling**: We use **SMOTE (Synthetic Minority Over-sampling Technique)** to create synthetic fire examples, preventing the model from ignoring the minority class.

## 2. Statistical Rigor

We don't just "fit" models. We validate assumptions:

- **Shapiro-Wilk Test**: Determines if features need Log/Power transformation (Box-Cox).
- **VIF (Variance Inflation Factor)**: Removes redundant features (like PM1.0 and PM2.5 often being identical) to improve model stability.

## 3. Operational Strategy

Prediction is useless without action.

- **Spatial Analysis**: We project predictions onto a 2D map to visualize the *spread* of risk.
- **Drone Logic**: Instead of sending drones randomly, we use **K-Means Clustering** to find the geographic *centers* of high-risk clusters, optimizing response time.
