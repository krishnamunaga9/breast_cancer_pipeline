# Breast Cancer Classification Pipeline

This repository contains code for building a machine learning pipeline using Kubeflow to classify breast cancer as malignant or benign. The pipeline includes steps for data preprocessing, feature engineering, model training, and evaluation. An optional deployment step using Docker and Flask is also included.

## Project Structure

- `data_preprocessing.py`: Script for data retrieval and preprocessing.
- `feature_engineering.py`: Script for feature engineering.
- `model_training.py`: Script for model training and evaluation.
- `kubeflow_pipeline.py`: Definition of the Kubeflow pipeline.
- `app.py`: Flask application for serving the model.
- `Dockerfile`: Dockerfile for containerizing the Flask app.
- `requirements.txt`: List of Python dependencies.

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
