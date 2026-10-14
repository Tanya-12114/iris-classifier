import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Iris Flower Predictor", layout="centered")
st.title("Iris Flower Species Predictor")
st.markdown(
    """
    <style>
    /* Set app background to light gray */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;  
        color: #000000 !important;             
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: white !important;
        color: black !important;
        border: 2px dashed black !important;
        border-radius: 8px;
        padding: 20px;
    }
    [data-testid="stFileUploaderDropzone"] span, 
    [data-testid="stFileUploaderDropzone"] p {
        color: black !important;
        font-weight: 600;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload your Iris dataset CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Normalize column names (lowercase + remove spaces/underscores)
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "").str.replace("_", "")

    # Detect feature and target columns automatically
    feature_cols = [col for col in data.columns if 'sepal' in col or 'petal' in col]
    target_col = [col for col in data.columns if 'species' in col][0]

    X = data[feature_cols]
    y = data[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    st.success("✅ Model trained successfully!")

    # Model accuracy
    accuracy = model.score(X_test, y_test)
    st.write(f"Model Accuracy: **{accuracy*100:.2f}%**")

    # Step 2: Get min/max/mean values for sliders
    st.subheader("Enter Flower Measurements:")

    sepal_length = st.slider("Sepal Length (cm)", float(X.iloc[:,0].min()), float(X.iloc[:,0].max()), float(X.iloc[:,0].mean()))
    sepal_width  = st.slider("Sepal Width (cm)",  float(X.iloc[:,1].min()), float(X.iloc[:,1].max()), float(X.iloc[:,1].mean()))
    petal_length = st.slider("Petal Length (cm)", float(X.iloc[:,2].min()), float(X.iloc[:,2].max()), float(X.iloc[:,2].mean()))
    petal_width  = st.slider("Petal Width (cm)",  float(X.iloc[:,3].min()), float(X.iloc[:,3].max()), float(X.iloc[:,3].mean()))

    if st.button("Predict"):
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_features)[0]
        st.success(f"The predicted Iris species is: **{prediction}**")

        # Optional: Show flower image
        # Optional: Show flower image
        flower_images = {
            'Iris-setosa': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxHVVjSM5dIobxwIfyjAmjkS_tWc4ssD6yCQ&s',
            'Iris-versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
            'Iris-virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg',
            'setosa': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxHVVjSM5dIobxwIfyjAmjkS_tWc4ssD6yCQ&s',
            'versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
            'virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
        }

        if prediction in flower_images:
            st.image(flower_images[prediction], caption=f"{prediction.capitalize()}", use_container_width=True)

