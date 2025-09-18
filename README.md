# Assignment 01 - Deployment for the PMLDL course

This repo contains a FastAPI api and a client with Streamlit containerized with Docker.

## Repo Structure

The structure of this repo is different from the recommended one in the assignment description.

```
├── app -- FastAPI app
│   ├── api/ -- FastAPI app code
│   └── models -- Model itself, its training code, and the saved model
│       └── saved/ -- Saved models
│   └── __main__.py -- FastAPI app entry point
├── client -- Streamlit app
│   └── __main__.py -- Streamlit app entry point
├── data -- Data-related files for dataset analysis and the dataset itself
│    ├── dataset/ -- Dataset files
│    └── analyze.ipynb -- Jupyter notebook for data analysis
└── deployment
    ├── Dockerfile.fastapi -- Dockerfile for the FastAPI app
    ├── Dockerfile.streamlit -- Dockerfile for the Streamlit app
    └── docker-compose.yml -- Docker Compose file for both apps
```

## Implemented model

The model is an "attractiveness classifier" that predicts the attractiveness score of a face based on its image.

