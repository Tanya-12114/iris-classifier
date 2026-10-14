import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Classification_Experiment")

def main():
    # Set experiment name in MLflow
    mlflow.set_experiment("Iris_Classification_Experiment")

    # Load Iris dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # Hyperparameters
    n_estimators = 100
    max_depth = 3

    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Predict & calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters & metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        # Log model artifact
        mlflow.sklearn.log_model(model, "iris_rf_model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        print(f"Model trained successfully! Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()