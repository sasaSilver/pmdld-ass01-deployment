import requests
import streamlit as st
from api_client import api_client

st.title("Facial Attractiveness Prediction")

if file := st.file_uploader(
    label="Upload your image", type=["png", "jpg", "jpeg", "webp"]
):
    if file.type not in ("image/png", "image/jpeg", "image/webp", "image/jpg"):
        st.error("Please upload an image file.")
    st.spinner("Predicting...")
    image_file = (file.name, file.getvalue(), file.type)
    try:
        prediction = api_client.predict(image_file)
        st.success(f"Predicted attractiveness score: {prediction}")
    except requests.exceptions.HTTPError as e:
        st.error(e)
