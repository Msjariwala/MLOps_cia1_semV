from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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
# Model 1: K-Nearest Neighbors
# --------------------------------------------------

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Model 2: Random Forest
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

knn_predictions = knn.predict(X_test)

rf_predictions = random_forest.predict(X_test)


# --------------------------------------------------
# Basic output
# --------------------------------------------------

print("Model training completed.")

print("\nKNN predictions:")
print(knn_predictions[:20])

print("\nRandom Forest predictions:")
print(rf_predictions[:20])

print("\nNumber of KNN predictions:", len(knn_predictions))
print("Number of Random Forest predictions:", len(rf_predictions))