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
├── dataset -- Data-related files for dataset analysis and the dataset itself
│    ├── data/ -- Dataset files
│    └── analyze.ipynb -- Jupyter notebook for data analysis
└── deployment
    ├── Dockerfile.fastapi -- Dockerfile for the FastAPI app
    ├── Dockerfile.streamlit -- Dockerfile for the Streamlit app
    └── docker-compose.yml -- Docker Compose file for both apps
```

## Implemented model

The model is an "attractiveness classifier" that predicts the attractiveness score of a face based on its image.

## Data

The dataset description can be found in the `dataset/README.md` file.
This project uses `dvc` for data versioning with a local repository.

## Deployment

The deployment docker files and in `deployment/` folder, containing `Dockerfile.api`, `Dockerfile.client`, and `docker-compose.yml` files.

To run the containers, run

```
docker-compose -f deployment/docker-compose.yml up
```