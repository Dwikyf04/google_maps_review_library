# [File: app_streamlit.py]

import streamlit as st
import pandas as pd
import joblib
import numpy as np  # Diperlukan oleh model scikit-learn

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Analisis & Rekomendasi Perpus", layout="wide")

# --- 1. Fungsi Pemuatan Data (Cache) ---

@st.cache_resource
def load_models():
    """
    Memuat model TF-IDF, SVM, dan K-Means dari file .pkl.
    Fungsi ini hanya dijalankan sekali.
    """
    try:
        tfidf = joblib.load("Models/tfidf_vectorizer.pkl")
        svm_model = joblib.load("Models/svm_sentiment_model.pkl")
        kmeans_model = joblib.load("Models/kmeans.pkl")
        return tfidf, svm_model, kmeans_model
    except FileNotFoundError:
        st.error("File model (.pkl) tidak ditemukan di folder /Models.")
        st.warning("Pastikan folder 'Models' berisi file .pkl sudah ada di repositori GitHub.")
        return None, None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

@st.cache_data
def load_library_data(file_path="data/data_perpustakaan_review.csv"):
    """
    Memuat data perpustakaan yang sudah diolah dari file CSV.
    File ini digunakan untuk Tab Rekomendasi.
    """
    try:
        df = pd.read_csv(file_path)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['skor_kualitas'] = pd.to_numeric(df['skor_kualitas'], errors='coerce')
        df['persen_positif'] = pd.to_numeric(df['persen_positif'], errors='coerce')
        df.dropna(subset=['skor_kualitas', 'rating', 'kota', 'nama_perpustakaan'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"File data '{file_path}' tidak ditemukan.")
        st.warning("Pastikan file 'data_perpustakaan.csv' sudah ada di repositori GitHub.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data CSV: {e}")
        return pd.DataFrame()

# --- 2. Memuat Semua Data & Model ---

# Memuat model untuk Tab 2
tfidf, svm_model, kmeans_model = load_models()
# Memuat data CSV untuk Tab 1
library_data = load_library_data()

# --- 3. Judul Utama Aplikasi ---
st.title("📊 Sistem Analisis & Rekomendasi Ulasan Perpustakaan")
st.markdown("Pilih tab di bawah untuk melihat rekomendasi perpustakaan atau menganalisis ulasan baru.")

# --- 4. Membuat Tab ---
tab1, tab2 = st.tabs(["🏆 Rekomendasi Perpustakaan", "🔍 Analisis Ulasan Individual"])

# --- 5. Isi Tab 1: Rekomendasi Perpustakaan (Kode Aplikasi #2) ---
with tab1:
    st.header("Temukan Perpustakaan Terbaik di Kota Anda")
    
    if not library_data.empty:
        available_cities = sorted(library_data['kota'].unique())
        if available_cities:
            selected_city = st.selectbox(
                "📍 Pilih Kota Anda:",
                options=available_cities,
                index=None,
                placeholder="Pilih kota..."
            )

            if selected_city:
                st.markdown("---")
                st.subheader(f"Rekomendasi Perpustakaan Terbaik di {selected_city}:")
                
                city_libraries = library_data[library_data['kota'] == selected_city].copy()
                recommended_libraries = city_libraries.sort_values(by='skor_kualitas', ascending=False)

                if not recommended_libraries.empty:
                    for i, (_, row) in enumerate(recommended_libraries.head(5).iterrows()):
                        st.markdown(f"#### {i + 1}. {row['nama_perpustakaan']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="⭐ Rating Google", value=f"{row['rating']:.1f} / 5")
                        with col2:
                            if 'persen_positif' in row and pd.notna(row['persen_positif']):
                                st.metric(label="👍 Sentimen Positif", value=f"{row['persen_positif']:.0%}")
                            elif 'skor_kualitas' in row and pd.notna(row['skor_kualitas']):
                                st.metric(label="💯 Skor Kualitas", value=f"{row['skor_kualitas']:.2f}")
                        
                        if 'url_google_maps' in row and pd.notna(row['url_google_maps']) and row['url_google_maps'].startswith('http'):
                            st.link_button("Lihat di Google Maps ↗️", row['url_google_maps'])
                        st.divider()
                else:
                    st.info(f"Belum ada data rekomendasi perpustakaan untuk {selected_city}.")
        else:
            st.warning("Tidak ada data kota yang tersedia di file CSV.")
    else:
        st.error("Data perpustakaan tidak dapat dimuat. Periksa file 'data/data_perpustakaan.csv'.")

# --- 6. Isi Tab 2: Analisis Ulasan Individual (Kode Aplikasi #1) ---
with tab2:
    st.header("Analisis Sentimen & Cluster Ulasan Baru")
    st.markdown("Model: **SVM (Sentimen)** + **K-Means (Clustering)** + **TF-IDF (Feature Extraction)**")

    # Memeriksa apakah model berhasil dimuat
    if tfidf is not None and svm_model is not None and kmeans_model is not None:
        user_input = st.text_area("Masukkan ulasan pengguna di sini:", "", key="input_ulasan")

        if st.button("Analisis Ulasan"):
            if user_input.strip():
                try:
                    # Proses prediksi
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
                
                except Exception as e:
                    st.error(f"Gagal melakukan prediksi: {e}")

            else:
                st.warning("Harap masukkan teks terlebih dahulu!")
    else:
        st.error("Model analisis (SVM/K-Means/TF-IDF) gagal dimuat. Tab ini tidak dapat berfungsi.")


# --- 7. Footer ---
st.markdown("---")
st.caption("Dibuat oleh Nanda | Analisis Sentimen & Sistem Rekomendasi Perpustakaan 📚")

