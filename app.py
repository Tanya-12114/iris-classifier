"""Gradio app that serves the latest Random Forest model logged to MLflow."""
import gradio as gr
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Iris_Classification_Experiment"
MODEL_ARTIFACT_NAME = "iris_rf_model"

mlflow.set_tracking_uri(TRACKING_URI)


def load_latest_model():
    """Find the most recent finished run of the experiment and load its model."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(
            f"No finished runs found in experiment '{EXPERIMENT_NAME}'.\n"
            "Run `python train.py` first to train and log a model."
        )
    run = runs.iloc[0]
    run_id = run["run_id"]
    accuracy = run.get("metrics.accuracy")
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
    print(f"Loading MLflow model from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri), run_id, accuracy


model, RUN_ID, ACCURACY = load_latest_model()
accuracy_text = f"{ACCURACY * 100:.2f}%" if ACCURACY is not None else "n/a"

iris = load_iris()
df = pd.DataFrame(
    iris.data,
    columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
)
df["Species"] = iris.target_names[iris.target]


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=iris.feature_names,
    )
    prediction = model.predict(features.values)
    species_name = iris.target_names[prediction[0]]
    return f"### The predicted iris species is: **{species_name.capitalize()}**"


with gr.Blocks() as demo:
    gr.Markdown("# Iris Flower Species Predictor (MLflow Integrated)")
    gr.Markdown(
        "This app serves the model tracked and logged in **MLflow** "
        f"(`{TRACKING_URI}`)."
    )

    with gr.Tab("Prediction & Demo"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Enter Flower Measurements")
                sepal_length = gr.Slider(4.0, 8.0, value=5.1, step=0.1, label="Sepal Length (cm)")
                sepal_width = gr.Slider(2.0, 4.5, value=3.5, step=0.1, label="Sepal Width (cm)")
                petal_length = gr.Slider(1.0, 7.0, value=1.4, step=0.1, label="Petal Length (cm)")
                petal_width = gr.Slider(0.1, 2.5, value=0.2, step=0.1, label="Petal Width (cm)")
                predict_btn = gr.Button("Predict Species", variant="primary")

            with gr.Column():
                gr.Markdown("### MLflow Model Metadata")
                gr.Markdown(
                    f"- **Run ID:** `{RUN_ID}`\n"
                    f"- **Logged accuracy:** {accuracy_text}"
                )
                output_result = gr.Markdown("### Prediction result will appear here")

        predict_btn.click(
            fn=predict_species,
            inputs=[sepal_length, sepal_width, petal_length, petal_width],
            outputs=output_result,
        )

    with gr.Tab("Dataset Preview"):
        gr.Markdown("### Iris Dataset Sample Preview")
        gr.DataFrame(df.head(10), interactive=False)

if __name__ == "__main__":
    # Set a public share link only if you explicitly want one: python app.py --share
    import sys

    demo.launch(share="--share" in sys.argv)
