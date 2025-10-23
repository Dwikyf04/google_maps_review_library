import streamlit as st
import joblib
import numpy as np
import pandas as pd

# === Load model dan vectorizer ===
@st.cache_resource
def load_models():
    tfidf = joblib.load("Models/tfidf_vectorizer.pkl")
    svm_model = joblib.load("Models/svm_sentiment_model.pkl")
    kmeans_model = joblib.load("Models/kmeans.pkl")
    return tfidf, svm_model, kmeans_model

tfidf, svm_model, kmeans_model = load_models()

# === Judul Aplikasi ===
st.title("📊 Sistem Analisis & Rekomendasi Ulasan Perpustakaan Nasional")
st.markdown("Model: **SVM (Sentimen)** + **K-Means (Clustering)** + **TF-IDF (Feature Extraction)**")

# === Input pengguna ===
user_input = st.text_area("Masukkan ulasan pengguna di sini:", "")

if st.button("Analisis Ulasan"):
    if user_input.strip():
        X = tfidf.transform([user_input])
        sentiment_pred = svm_model.predict(X)[0]
        cluster_pred = kmeans_model.predict(X)[0]

        st.subheader("🔍 Hasil Analisis:")
        st.write(f"**Sentimen:** {sentiment_pred}")
        st.write(f"**Cluster:** {cluster_pred}")

        # Rekomendasi berdasarkan cluster
        if cluster_pred == 0:
            st.success("📚 Rekomendasi: Ulasan ini mirip dengan kelompok pembaca yang memberikan **ulasan positif** dan sering merekomendasikan layanan pustaka digital.")
        elif cluster_pred == 1:
            st.info("🧐 Rekomendasi: Ulasan ini termasuk kelompok dengan **penilaian netral**, mungkin perlu peningkatan pelayanan.")
        else:
            st.warning("😔 Rekomendasi: Termasuk cluster dengan **ulasan negatif**, fokuskan perbaikan pada pengalaman pengunjung.")

    else:
        st.warning("Harap masukkan teks terlebih dahulu!")

st.markdown("---")
st.caption("Dibuat oleh Nanda | Analisis Sentimen & Sistem Rekomendasi Perpustakaan 📚")