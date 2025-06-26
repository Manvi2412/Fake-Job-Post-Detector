# Fake Job Post Detector

A machine learning-powered web application that detects whether a job description is **real or fake** using natural language processing and classification algorithms.

---

## Project Overview

Fake job postings are a growing concern in today's online job market. This tool helps users **quickly assess the authenticity** of a job description using a trained machine learning model.

Built using:
- **TF-IDF** for text vectorization
- **XGBoost** for classification
- **Streamlit** for interactive web interface

---

## Model Info

- **Model Type:** TF-IDF + XGBoost Classifier  
- **Metrics Achieved:**
  - Accuracy: ~92%
  - F1 Score: ~0.83
  - AUC Score: ~0.98

---

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Joblib (model saving/loading)
- Streamlit (web app framework)

---

 Features
-Clean and simple UI to paste job descriptions

-Real-time prediction using trained ML model

-Confidence score with every prediction

-Preprocessing with input cleaning and TF-IDF

 Dataset Info
Contains fields like:

Title, Location, Department

Description, Requirements

Employment Type, Industry, Function

Format: data/fake_job_postings.xlsx

Author
Manvi Taneja
B.Tech in Information Technology (2026)
Delhi, India

Acknowledgements
Streamlit for frontend

XGBoost + Scikit-learn for modeling

Dataset: Kaggle
