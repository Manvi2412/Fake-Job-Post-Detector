{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f1d5d0e5-e062-44f7-9f7d-44dcbf1c9356",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import re\n",
    "\n",
    "model = joblib.load('fake_job_detector_xgb.pkl')\n",
    "vectorizer = joblib.load('tfidf_vectorizer.pkl')\n",
    "\n",
    "def clean_text_simple(text):\n",
    "    text = text.lower()\n",
    "    text = re.sub(r'[^a-z\\s]', '', text)\n",
    "    text = re.sub(r'\\s+', ' ', text)\n",
    "    return text.strip()\n",
    "\n",
    "st.title(\"🕵️‍♀️ Fake Job Post Detector\")\n",
    "st.markdown(\"Paste any job description below and check if it's likely to be **Real** or **Fake**.\")\n",
    "\n",
    "job_input = st.text_area(\"📄 Job Description\", height=300)\n",
    "\n",
    "if st.button(\"🔍 Predict\"):\n",
    "    if job_input.strip() == \"\":\n",
    "        st.warning(\"Please enter a job description.\")\n",
    "    else:\n",
    "        clean = clean_text_simple(job_input)\n",
    "        vector = vectorizer.transform([clean])\n",
    "        pred = model.predict(vector)[0]\n",
    "        proba = model.predict_proba(vector)[0][1]\n",
    "\n",
    "        if pred == 1:\n",
    "            st.error(f\"🚨 Prediction: FAKE (Confidence: {proba:.2f})\")\n",
    "        else:\n",
    "            st.success(f\"✅ Prediction: REAL (Confidence: {1 - proba:.2f})\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a961951-0aed-453c-ae1f-019b225b0f13",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import re\n",
    "\n",
    "model = joblib.load('fake_job_detector_xgb.pkl')\n",
    "vectorizer = joblib.load('tfidf_vectorizer.pkl')\n",
    "\n",
    "def clean_text_simple(text):\n",
    "    text = text.lower()\n",
    "    text = re.sub(r'[^a-z\\s]', '', text)\n",
    "    text = re.sub(r'\\s+', ' ', text)\n",
    "    return text.strip()\n",
    "\n",
    "st.title(\"🕵️‍♀️ Fake Job Post Detector\")\n",
    "st.markdown(\"Paste any job description below and check if it's likely to be **Real** or **Fake**.\")\n",
    "\n",
    "job_input = st.text_area(\"📄 Job Description\", height=300)\n",
    "\n",
    "if st.button(\"🔍 Predict\"):\n",
    "    if job_input.strip() == \"\":\n",
    "        st.warning(\"Please enter a job description.\")\n",
    "    else:\n",
    "        clean = clean_text_simple(job_input)\n",
    "        vector = vectorizer.transform([clean])\n",
    "        pred = model.predict(vector)[0]\n",
    "        proba = model.predict_proba(vector)[0][1]\n",
    "\n",
    "        if pred == 1:\n",
    "            st.error(f\"🚨 Prediction: FAKE (Confidence: {proba:.2f})\")\n",
    "        else:\n",
    "            st.success(f\"✅ Prediction: REAL (Confidence: {1 - proba:.2f})\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
