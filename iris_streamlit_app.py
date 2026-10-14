import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris Flower Predictor", layout="centered")
st.title("Iris Flower Species Predictor")

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;  
        color: #000000 !important;            
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Automatically load local Iris dataset
@st.cache_resource
def load_and_train_model():
    try:
        data = pd.read_csv("Iris.csv")
    except FileNotFoundError:
        st.error("Iris.csv file not found in the project directory!")
        return None, None
    
    # Normalize column names
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "").str.replace("_", "")

    # Detect feature and target columns automatically
    feature_cols = [col for col in data.columns if 'sepal' in col or 'petal' in col]
    target_col = [col for col in data.columns if 'species' in col][0]

    X = data[feature_cols]
    y = data[target_col]

    # Train-test split & Model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, X, accuracy

model, X, accuracy = load_and_train_model()

if model is not None:
    st.success("✅ Model trained successfully on local dataset!")
    st.write(f"Model Accuracy: **{accuracy*100:.2f}%**")

    st.subheader("Enter Flower Measurements:")

    sepal_length = st.slider("Sepal Length (cm)", float(X.iloc[:,0].min()), float(X.iloc[:,0].max()), float(X.iloc[:,0].mean()))
    sepal_width  = st.slider("Sepal Width (cm)",  float(X.iloc[:,1].min()), float(X.iloc[:,1].max()), float(X.iloc[:,1].mean()))
    petal_length = st.slider("Petal Length (cm)", float(X.iloc[:,2].min()), float(X.iloc[:,2].max()), float(X.iloc[:,2].mean()))
    petal_width  = st.slider("Petal Width (cm)",  float(X.iloc[:,3].min()), float(X.iloc[:,3].max()), float(X.iloc[:,3].mean()))

    if st.button("Predict"):
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_features)[0]
        st.success(f"The predicted Iris species is: **{prediction}**")

        flower_images = {
            'Iris-setosa': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxHVVjSM5dIobxwIfyjAmjkS_tWc4ssD6yCQ&s',
            'Iris-versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
            'Iris-virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg',
            'setosa': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxHVVjSM5dIobxwIfyjAmjkS_tWc4ssD6yCQ&s',
            'versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
            'virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
        }

        if prediction in flower_images:
            st.image(flower_images[prediction], caption=f"{str(prediction).capitalize()}", use_container_width=True)