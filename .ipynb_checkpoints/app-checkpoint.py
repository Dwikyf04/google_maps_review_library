{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "fa2b040a-aba6-4de4-a48f-4e5f592df574",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: scikit-learn in c:\\users\\dwiky\\miniforge3\\lib\\site-packages (1.7.2)\n",
      "Requirement already satisfied: joblib in c:\\users\\dwiky\\miniforge3\\lib\\site-packages (1.5.2)\n",
      "Requirement already satisfied: numpy>=1.22.0 in c:\\users\\dwiky\\miniforge3\\lib\\site-packages (from scikit-learn) (2.3.4)\n",
      "Requirement already satisfied: scipy>=1.8.0 in c:\\users\\dwiky\\miniforge3\\lib\\site-packages (from scikit-learn) (1.16.2)\n",
      "Requirement already satisfied: threadpoolctl>=3.1.0 in c:\\users\\dwiky\\miniforge3\\lib\\site-packages (from scikit-learn) (3.6.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install scikit-learn joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e2cc44b-c60b-4bbe-a047-66fa5237c451",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import joblib\n",
    "import streamlit as st\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cedb801-6624-4adc-9b9c-5728711fe0e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import joblib\n",
    "\n",
    "# === Load Model & Vectorizer ===\n",
    "tfidf = joblib.load(r\"C:\\Users\\dwiky\\Documents\\googlemaps_projectt\\Models\\tfidf_vectorizer.pkl\")\n",
    "model = joblib.load(r\"C:\\Users\\dwiky\\Documents\\googlemaps_projectt\\Models\\svm_sentiment_model.pkl\")\n",
    "\n",
    "print(\"✅ Model dan TF-IDF berhasil dimuat!\")\n",
    "\n",
    "# === Setup Flask ===\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route(\"/\")\n",
    "def home():\n",
    "    return jsonify({\"message\": \"API Sentimen Google Maps aktif!\"})\n",
    "\n",
    "@app.route(\"/predict\", methods=[\"POST\"])\n",
    "def predict():\n",
    "    data = request.get_json()\n",
    "    text = data.get(\"text\", \"\")\n",
    "\n",
    "    # Gunakan tfidf (bukan vectorizer)\n",
    "    X = tfidf.transform([text])\n",
    "    pred = model.predict(X)[0]\n",
    "\n",
    "    return jsonify({\n",
    "        \"text\": text,\n",
    "        \"predicted_sentiment\": pred\n",
    "    })\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(host=\"0.0.0.0\", port=5000)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "017152be-5487-4dc1-b69f-169c50fb9b89",
   "metadata": {},
   "outputs": [],
   "source": [
    "streamlit run app_streamlit.py\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bf56648d-34d6-4d27-9ddb-d1f6a454b6ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "!streamlit run \"C:\\Users\\dwiky\\Documents\\googlemaps_projectt\\app_streamlit.py\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "099a1471-fdae-460d-84e5-fcf5abc6298d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
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
   "version": "3.12.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
