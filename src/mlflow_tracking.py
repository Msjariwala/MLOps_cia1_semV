import mlflow
import mlflow.sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import load_data, preprocess_data


# --------------------------------------------------
# 1. Load and preprocess data
# --------------------------------------------------

df = load_data()

X_train, X_test, y_train, y_test, scaler = preprocess_data(df)


# --------------------------------------------------
# 2. Create MLflow experiment
# --------------------------------------------------

mlflow.set_experiment("AI4I_Predictive_Maintenance")


# --------------------------------------------------
# 3. Function for training + MLflow tracking
# --------------------------------------------------

def train_and_track(model, model_name, parameters):

    with mlflow.start_run(run_name=model_name):

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Metrics
        train_accuracy = accuracy_score(
            y_train,
            train_predictions
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        precision = precision_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        # Log parameters
        mlflow.log_params(parameters)

        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log trained model
        # mlflow.sklearn.log_model(
        #     model,
        #     "model"
        # )

        # Display results
        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        print("Parameters:")
        print(parameters)

        print(f"Training Accuracy : {train_accuracy:.4f}")
        print(f"Testing Accuracy  : {test_accuracy:.4f}")
        print(f"Precision         : {precision:.4f}")
        print(f"Recall            : {recall:.4f}")
        print(f"F1-score          : {f1:.4f}")


# --------------------------------------------------
# 4. KNN Experiment
# --------------------------------------------------

knn_parameters = {
    "n_neighbors": 5
}

knn = KNeighborsClassifier(
    n_neighbors=5
)

train_and_track(
    knn,
    "KNN",
    knn_parameters
)


# --------------------------------------------------
# 5. Random Forest Experiment
# --------------------------------------------------

rf_parameters = {
    "n_estimators": 100,
    "random_state": 42
}

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

train_and_track(
    random_forest,
    "Random Forest",
    rf_parameters
)