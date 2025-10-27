# Fake Job Post Detector

A machine learning-powered web application that detects whether a job description is **real or fake** using natural language processing and classification algorithms.

---

## Project Overview

Fake job postings are a growing concern in today's online job market. This tool helps users **quickly assess the authenticity** of a job description using a trained machine learning model.

Built using:
- **TF-IDF** for text vectorization
- **XGBoost** for classification
- **Streamlit** for interactive web interface

The app provides:
- Real-time predictions
- Confidence scores
- A clean, professional interface built with Streamlit

---

## Model Info

- **Model Type:** TF-IDF + XGBoost Classifier  
- **Metrics Achieved:**
  - Accuracy: ~92%
  - F1 Score: ~0.83
  - AUC Score: ~0.98

## Features

- Paste any job description into the app
- Instant prediction – Real or Fake
- Confidence score bar showing probabilities
- Modern dark UI with interactive design and icons
- Preprocessing pipeline with TF-IDF
- Deployed-ready Streamlit app

---

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Joblib (model saving/loading)
- Streamlit (web app framework)

---

 Dataset Info
Contains fields like:

Title, Location, Department

Description, Requirements

Employment Type, Industry, Function

Format: data/fake_job_postings.xlsx

🧪 **How It Works**
- The user pastes a job description into the input area.
- The app preprocesses the text and converts it into TF-IDF features.
- The XGBoost classifier predicts whether the posting is Real or Fake.
- The result and confidence percentage are displayed in a clean, styled UI.

**Run locally**
```bash
# Clone the repository
git clone https://github.com/yourusername/fake-job-post-detector.git
cd fake-job-post-detector

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run main.py
```


**Author**
Manvi Taneja
B.Tech in Information Technology (2026)
Delhi, India

Acknowledgements
Streamlit for frontend

XGBoost + Scikit-learn for modeling

Dataset: Kaggle
