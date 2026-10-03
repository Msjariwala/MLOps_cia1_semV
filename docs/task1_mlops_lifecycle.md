# Task 1 — MLOps Lifecycle Analysis

## Dataset

AI4I 2020 Predictive Maintenance Dataset

## Problem Type

Binary Classification

## Target Variable

Machine failure

## ML Models

1. K-Nearest Neighbors (KNN)
2. Random Forest

---

## 1. MLOps Lifecycle

The complete MLOps lifecycle for this project consists of:

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
Model Deployment
        ↓
Model Monitoring
        ↓
Retraining
        ↺

---

## 2. Data Ingestion

The AI4I 2020 Predictive Maintenance Dataset is provided as a CSV file.

The dataset is loaded into a Pandas DataFrame using Python.

The data ingestion script reads the original dataset from the project's data directory.

---

## 3. Data Validation

The following validation checks are performed:

- Number of rows and columns
- Missing values
- Duplicate records
- Data types
- Target variable values
- Target class distribution

These checks ensure that the dataset is suitable for further processing.

---

## 4. Data Preprocessing

The following preprocessing operations will be performed:

- Remove irrelevant identifier columns
- Encode the categorical Type feature
- Separate input features and target variable
- Split the data into training and testing datasets
- Apply feature scaling where required

Feature scaling is particularly important for the KNN model.

---

## 5. Model Training

Two classification models will be trained:

### K-Nearest Neighbors

KNN predicts the class of a new observation based on its nearest observations.

### Random Forest

Random Forest uses multiple decision trees to perform classification.

Both models will be trained using the same training dataset.

---

## 6. Model Evaluation

The trained models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

The experiments will later be tracked using MLflow.

---

## 7. Model Deployment

After evaluation, the selected trained model can be deployed to a cloud ML platform such as AWS SageMaker or Google Vertex AI.

The deployed model can receive machine sensor values and return a machine failure prediction.

---

## 8. Model Monitoring

After deployment, the model should be monitored for:

- Prediction performance
- Data drift
- Changes in input data distribution
- Model degradation

If significant degradation is detected, the model can be retrained using updated data.

---

## 9. End-to-End Lifecycle

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
Model Deployment
        ↓
Model Monitoring
        ↓
Model Retraining
        ↺