from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from preprocessing import load_data, preprocess_data


# --------------------------------------------------
# Load and preprocess data
# --------------------------------------------------

df = load_data()

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler
) = preprocess_data(df)


# --------------------------------------------------
# Define models
# --------------------------------------------------

knn = KNeighborsClassifier(
    n_neighbors=5
)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# Train models
# --------------------------------------------------

knn.fit(X_train, y_train)

random_forest.fit(X_train, y_train)


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

knn_predictions = knn.predict(X_test)

rf_predictions = random_forest.predict(X_test)


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_model(model_name, y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# --------------------------------------------------
# Evaluate both models
# --------------------------------------------------

knn_results = evaluate_model(
    "K-Nearest Neighbors",
    y_test,
    knn_predictions
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)


# --------------------------------------------------
# Comparison
# --------------------------------------------------

print("\n\nMODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':<25}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1-score':<12}"
)

print("-" * 70)

for result in [knn_results, rf_results]:

    print(
        f"{result['model']:<25}"
        f"{result['accuracy']:<12.4f}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
        f"{result['f1_score']:<12.4f}"
    )