import streamlit as st
import mlflow.pyfunc
import numpy as np

# Load your MLflow model (make sure the path points to your local MLflow artifact folder)
# Example: model = mlflow.pyfunc.load_model("runs:/<run_id>/model") or a local path
@st.cache_resource
def load_model():
    # Replace with your actual relative path to the MLflow model folder
    return mlflow.pyfunc.load_model("model") 

model = load_model()

st.title("Iris Flower Classification")
st.write("Predict the Iris flower species using an MLflow-backed model.")

# Input sliders for the 4 Iris features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    # Format input for MLflow model expectations
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    
    st.success(f"Predicted Iris Species: **{prediction[0]}**")