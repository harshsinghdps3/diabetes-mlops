# Diabetes Prediction MLOps Project

This project implements a machine learning pipeline for predicting diabetes outcomes. It includes robust MLOps practices such as experiment tracking with MLflow, logging, data validation, and modular code structure.

## Project Structure

```
├── configs/            # Configuration files
│   └── config.yaml
├── data/               # Data directory
│   ├── raw/            # Original dataset
│   └── processed/      # Processed data for training/testing
├── models/             # Saved models
├── src/                # Source code
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── train.py        # Training pipeline
│   └── evaluate.py     # Model evaluation
├── tests/              # Unit tests
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup

1.  **Clone the repository** (if applicable).
2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Loading & Validation
Loads the raw data, validates columns, and preprocesses missing values (zeros).
```bash
python src/data_loader.py
```

### 2. Training
Trains a Logistic Regression model using a pipeline (Imputation -> Scaling -> Model). Logs metrics and parameters to MLflow.
```bash
python src/train.py
```
*   **Training Accuracy**: ~0.7720
*   **Artifacts**: Model saved to `models/model_v1.joblib`

### 3. Evaluation
Evaluates the trained model on the test set.
```bash
python src/evaluate.py
```

## Recent Results

**Confusion Matrix**:
```
[[82 17]
 [21 34]]
```

**Classification Report**:
```
              precision    recall  f1-score   support

           0       0.80      0.83      0.81        99
           1       0.67      0.62      0.64        55

    accuracy                           0.75       154
   macro avg       0.73      0.72      0.73       154
weighted avg       0.75      0.75      0.75       154
```

## Key Features

*   **MLflow Integration**: Tracks experiments, parameters, and metrics.
*   **Logging**: Replaced standard `print` statements with Python's `logging` module for better observability.
*   **Data Validation**: Ensures required columns exist before processing.
*   **Type Hinting**: All functions include type hints for better code quality and developer experience.
