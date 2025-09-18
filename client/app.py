import streamlit as st
from api_client import api_client

st.title("Facial Attractiveness Prediction")

if file := st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"]):
    image = file.read()
    prediction = api_client.predict(image)
    st.success(f"Predicted attractiveness score: {prediction}")
