"""Unit and integration tests for the Iris classifier."""
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
SPECIES = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(ROOT / "Iris.csv")


@pytest.fixture(scope="module")
def trained(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data["Species"], test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# ---- Unit tests -----------------------------------------------------------
def test_dataset_shape(data):
    assert len(data) == 150
    assert set(FEATURES + ["Species"]).issubset(data.columns)


def test_dataset_has_three_balanced_classes(data):
    counts = data["Species"].value_counts()
    assert set(counts.index) == SPECIES
    assert (counts == 50).all()


def test_dataset_has_no_missing_values(data):
    assert not data[FEATURES + ["Species"]].isnull().any().any()


def test_model_accuracy_above_threshold(trained):
    model, X_test, y_test = trained
    assert accuracy_score(y_test, model.predict(X_test)) >= 0.90


def test_predictions_are_valid_species(trained):
    model, X_test, _ = trained
    assert set(model.predict(X_test)).issubset(SPECIES)


@pytest.mark.parametrize(
    "sample, expected",
    [
        ([5.1, 3.5, 1.4, 0.2], "Iris-setosa"),
        ([6.0, 2.8, 4.5, 1.4], "Iris-versicolor"),
        ([6.9, 3.1, 5.8, 2.3], "Iris-virginica"),
    ],
)
def test_known_samples(trained, sample, expected):
    model, _, _ = trained
    X = pd.DataFrame([sample], columns=FEATURES)
    assert model.predict(X)[0] == expected


# ---- Integration tests ----------------------------------------------------
def test_train_script_logs_to_mlflow(tmp_path):
    """Run train.py in a scratch directory and check its outputs."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "train.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Model trained successfully" in result.stdout
    assert (tmp_path / "mlflow.db").exists()
    model = joblib.load(tmp_path / "model.pkl")
    assert len(model.predict([load_iris().data[0]])) == 1
