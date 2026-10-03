# MLOps CIA1 — Predictive Maintenance Pipeline

## Project Overview

This project implements a basic end-to-end **MLOps pipeline** for predicting machine failures using the **AI4I 2020 Predictive Maintenance Dataset**.

The project demonstrates practical MLOps concepts including:

* MLOps lifecycle design
* Data ingestion and validation
* Data preprocessing and feature engineering
* Machine learning model training
* Model evaluation
* MLflow experiment tracking
* DVC-based dataset tracking and versioning
* Git/GitHub integration

Additional components such as Kubeflow workflow orchestration, Feast feature serving, and cloud deployment/comparison are planned for the next stages of the project.

---

## Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Problem Type:** Binary Classification

**Target Variable:** `Machine failure`

The dataset contains **10,000 records and 14 columns**.

### Features Used

The following features are used for model development:

* `Type`
* `Air temperature [K]`
* `Process temperature [K]`
* `Rotational speed [rpm]`
* `Torque [Nm]`
* `Tool wear [min]`

The columns `UDI` and `Product ID` are identifiers and are excluded from modelling.

The failure-mode indicators:

* `TWF`
* `HDF`
* `PWF`
* `OSF`
* `RNF`

are excluded because they provide information directly related to machine failure and can introduce target leakage.

---

# MLOps Lifecycle

The implemented lifecycle follows:

```text
Data Collection
       ↓
Data Ingestion
       ↓
Data Validation
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Deployment
       ↓
Monitoring
       ↓
Retraining
```

### Current Implementation

The following stages have been implemented:

1. **Data Ingestion**

   * Loads the AI4I dataset using Pandas.
   * Validates file availability.

2. **Data Validation**

   * Checks missing values.
   * Checks duplicate records.
   * Checks data types.
   * Examines target distribution and unique target values.

3. **Data Preprocessing**

   * Removes identifier columns.
   * Removes target-leakage columns.
   * Performs one-hot encoding on `Type`.
   * Performs an 80/20 stratified train-test split.
   * Applies `StandardScaler` using the training data.

4. **Model Training**

   * K-Nearest Neighbors
   * Random Forest

5. **Model Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Classification report

---

# Project Structure

```text
MLOps_CIA1/
│
├── data/
│   ├── ai4i2020.csv
│   ├── ai4i2020.csv.dvc
│   ├── ai4i2020_processed.csv
│   └── ai4i2020_processed.csv.dvc
│
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── mlflow_tracking.py
│   └── create_modified_dataset.py
│
├── docs/
│   └── task1_mlops_lifecycle.md
│
├── notebooks/
│
├── .dvc/
│
├── mlruns/
│
├── .gitignore
├── README.md
└── ...
```

---

# Model Development

Two classification models were implemented:

## 1. K-Nearest Neighbors

Configuration:

```text
n_neighbors = 5
```

## 2. Random Forest

Configuration:

```text
n_estimators = 100
random_state = 42
```

---

# Model Evaluation Results

The models were evaluated on the held-out test dataset.

| Model         | Training Accuracy | Testing Accuracy | Precision | Recall | F1-score |
| ------------- | ----------------: | ---------------: | --------: | -----: | -------: |
| KNN           |            0.9804 |           0.9740 |    0.8333 | 0.2941 |   0.4348 |
| Random Forest |            1.0000 |           0.9815 |    0.8780 | 0.5294 |   0.6606 |

The project tracks multiple classification metrics rather than relying only on accuracy, since recall and F1-score provide additional information about failure-class detection.

---

# MLflow Experiment Tracking

MLflow is used to track and compare machine learning experiments.

### Experiment

```text
AI4I_Predictive_Maintenance
```

### Parameters Tracked

For KNN:

```text
n_neighbors = 5
```

For Random Forest:

```text
n_estimators = 100
random_state = 42
```

### Metrics Tracked

* Training accuracy
* Testing accuracy
* Precision
* Recall
* F1-score

The MLflow UI has been successfully configured and verified with separate runs for both KNN and Random Forest.

### Run MLflow UI

From the project root:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

---

# DVC — Data Version Control

DVC is used to track the datasets separately from the source code.

The project was initialized using:

```bash
git init
dvc init
```

## Original Dataset

The original dataset is tracked using:

```bash
dvc add data/ai4i2020.csv
```

Original dataset:

```text
10,000 rows × 14 columns
```

DVC metadata:

```text
data/ai4i2020.csv.dvc
```

## Processed Dataset

A preprocessing script was used to create a processed version of the dataset.

The processed dataset:

```text
10,000 rows × 8 columns
```

DVC metadata:

```text
data/ai4i2020_processed.csv.dvc
```

The processed dataset includes:

* Removed identifier columns
* Removed target-leakage columns
* One-hot encoded `Type`

### DVC Verification

DVC successfully reports the datasets as up to date:

```bash
dvc status
```

The project therefore maintains separate DVC-tracked representations of the original and processed datasets.

---

# Git and GitHub

Git is used to version the project source code and DVC metadata.

GitHub is used as the remote repository for the project.

The repository contains:

* Source code
* Documentation
* DVC metadata
* Project configuration
* README documentation

DVC manages dataset tracking while Git manages the project code and DVC metadata.

---

# Technologies Used

| Category            | Technology                       |
| ------------------- | -------------------------------- |
| Programming         | Python                           |
| Data Processing     | Pandas                           |
| Machine Learning    | Scikit-learn                     |
| Experiment Tracking | MLflow                           |
| Data Versioning     | DVC                              |
| Source Control      | Git                              |
| Remote Repository   | GitHub                           |
| Dataset             | AI4I 2020 Predictive Maintenance |

---

# Completed Assignment Components

### Task 1 — MLOps Lifecycle

**Status: Completed**

Implemented and documented:

* Data ingestion
* Data validation
* Preprocessing
* Feature engineering
* Model training
* Model evaluation
* Lifecycle/tool mapping

### Task 2 — MLflow Experiment Tracking

**Status: Completed**

Implemented:

* Two ML models
* Parameter tracking
* Training accuracy
* Testing accuracy
* Precision
* Recall
* F1-score
* MLflow UI comparison

### Task 3 — DVC

**Status: Completed**

Implemented:

* Git repository initialization
* DVC initialization
* Original dataset tracking
* Processed dataset creation
* Processed dataset tracking
* Git/GitHub integration

---

# Remaining Components

The following components are planned for the next stages:

### Task 4 — Kubeflow

Planned workflow:

```text
Data Collection
       ↓
Data Validation
       ↓
Preprocessing
       ↓
Training
       ↓
Evaluation
       ↓
Deployment
```

If a full Kubeflow environment is not practical because of cloud/IAM limitations, a simulated Kubeflow workflow will be implemented and documented.

### Task 5 — Feast and Cloud

Planned components:

* Define three machine-related features
* Demonstrate feature-store concepts
* Explain offline and online feature serving
* Compare AWS SageMaker and Google Vertex AI
* Explore cloud deployment where feasible

---

# Current Project Status

```text
Task 1 — MLOps Lifecycle       ████████████████████  100%
Task 2 — MLflow                 ████████████████████  100%
Task 3 — DVC                    ████████████████████  100%
Task 4 — Kubeflow               ░░░░░░░░░░░░░░░░░░░░    0%
Task 5 — Feast + Cloud          ░░░░░░░░░░░░░░░░░░░░    0%
```

The project is currently ready to proceed with **Kubeflow workflow implementation** and subsequently the **Feast/cloud component**.

