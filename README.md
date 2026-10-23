# 🌸 Iris Flower Classification

A small end-to-end machine learning project that classifies iris flowers into **Setosa**, **Versicolor** or **Virginica** from four measurements (sepal length/width, petal length/width).

It includes an exploratory notebook, an MLflow-tracked training script, and two interactive web apps (Gradio and Streamlit).

## Project structure

| File | Purpose |
|------|---------|
| `Iris Flower Classification.ipynb` | Exploratory notebook: data loading, pairplot, KNN classifier (k=3, ~96% test accuracy) |
| `Iris.csv` | Iris dataset (150 rows, 3 classes) |
| `train.py` | Trains a Random Forest, logs params/accuracy/model to MLflow and saves `model.pkl` |
| `app.py` | Gradio app that automatically loads the latest trained model from MLflow and predicts species |
| `streamlit_app.py` | Streamlit app that trains a Random Forest from `Iris.csv` on startup and predicts species |
| `model.pkl` | Latest Random Forest saved by `train.py` (joblib format) |
| `mlflow.db`, `mlruns/` | MLflow tracking database and model artifacts (created by `train.py`, git-ignored) |
| `requirements*.txt` | Dependencies: app (`requirements.txt`), CI (`-ci`), full local (`-dev`) |
| `tests/` | pytest unit, integration and Streamlit app tests |
| `Jenkinsfile` | Jenkins CI pipeline: checkout, install, test, train, archive, deploy check |
| `jenkins/Dockerfile`, `docker-compose.yml` | Local Jenkins with Python installed |
| `.gitignore` | Keeps MLflow data, virtual envs and caches out of git |

## Getting started

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

`requirements.txt` holds only what the deployed Streamlit app needs; `requirements-ci.txt` is used by Jenkins; `requirements-dev.txt` is the full local setup.

### 2. Train the model (MLflow)

```bash
python train.py
```

This trains a `RandomForestClassifier` (`n_estimators=100`, `max_depth=3`) on an 80/20 split and logs everything to the `Iris_Classification_Experiment` experiment in `mlflow.db`. Expected output:

```
Model trained successfully! Accuracy: 1.0000
```

To browse runs, parameters and metrics:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

### 3. Run the Gradio app (uses the MLflow model)

```bash
python app.py
```

`app.py` automatically loads the most recent finished run from the `Iris_Classification_Experiment` experiment, so there is no run ID to copy. If you haven't trained yet, it will tell you to run `python train.py` first. To get a temporary public link, use `python app.py --share`.

### 4. Run the Streamlit app (standalone)

This app does not need MLflow. It trains on `Iris.csv` when it starts.

```bash
streamlit run streamlit_app.py
```

Run it from the project folder so `Iris.csv` can be found.

## Model

- **Algorithm:** Random Forest (scikit-learn)
- **Features:** sepal length, sepal width, petal length, petal width (cm)
- **Classes:** Iris-setosa, Iris-versicolor, Iris-virginica
- **Test accuracy:** 100% on the 30-sample hold-out set (`random_state=42`)
- **Notebook baseline:** KNN (k=3) reaches about 95.6% on a stratified 30% test split

The Iris dataset is small and cleanly separable, so near-perfect accuracy is expected and does not say much about performance on harder problems.

## Tech stack

Python · scikit-learn · pandas · NumPy · MLflow · Gradio · Streamlit

## Dataset

Fisher's Iris dataset (UCI Machine Learning Repository).

## Tests and CI

```bash
pytest
```

The `Jenkinsfile` runs the same tests on every push, then runs `train.py` and archives `model.pkl`. See `JENKINS_SETUP.md` for setting up Jenkins.
