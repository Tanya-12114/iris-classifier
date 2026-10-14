"""Train a Random Forest on the Iris dataset and log it to MLflow."""
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Iris_Classification_Experiment"
MODEL_ARTIFACT_NAME = "iris_rf_model"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    n_estimators = 100
    max_depth = 3

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            model,
            name=MODEL_ARTIFACT_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Also keep a plain joblib copy of the latest model.
        joblib.dump(model, "model.pkl")

        print(f"Model trained successfully! Accuracy: {accuracy:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
