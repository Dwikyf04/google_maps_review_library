# [File: app_streamlit.py]

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
import folium
import gspread
import requests
import re
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from streamlit_folium import st_folium
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from collections import Counter

LABEL_POSITIF = "Positive"
LABEL_NEGATIF = "Negative"
LABEL_NETRAL = "Neutral"

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Analisis & Rekomendasi Perpus", layout="wide")

# --- 1. Fungsi Pemuatan Data (Cache) ---

@st.cache_resource
def load_models():
    """
    Memuat model TF-IDF, SVM, dan K-Means dari file .pkl.
    """
    try:
        tfidf = joblib.load("Models/tfidf_vectorizer.pkl")
        svm_model = joblib.load("Models/svm_sentiment_model.pkl")
        kmeans_model = joblib.load("Models/kmeans.pkl")
        profil_nama = joblib.load("Models/profil_nama.pkl")
        profil_vektor = joblib.load("Models/profil_vektor.pkl")
        return tfidf, svm_model, kmeans_model, profil_nama, profil_vektor
    except FileNotFoundError:
        st.error("File model (.pkl) tidak ditemukan di folder /Models.")
        return None, None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

@st.cache_data
def load_library_data(file_path="data_perpustakaan.csv"):
    """
    Memuat data perpustakaan yang SUDAH DIOLAH (RINGKASAN) dari file CSV.
    File ini digunakan untuk Tab Rekomendasi.
    """
    try:
        df = pd.read_csv(file_path)
        # (Pastikan nama kolom 'kota' dan 'nama_perpustakaan' sesuai dengan file Anda)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['skor_kualitas'] = pd.to_numeric(df['skor_kualitas'], errors='coerce')
        df['persen_positif'] = pd.to_numeric(df['persen_positif'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df.dropna(subset=['skor_kualitas', 'rating', 'city', 'Place_name', 'latitude', 'longitude'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"File data '{file_path}' (RINGKASAN) tidak ditemukan.")
        return pd.DataFrame()
    except KeyError as e:
        st.error(f"Kolom {e} tidak ditemukan di 'data_perpustakaan.csv'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data CSV (Ringkasan): {e}")
        return pd.DataFrame()

# --- BARU: Fungsi untuk memuat SEMUA ulasan individual ---
@st.cache_data
def load_review_data(file_path="data_perpustakaan_review.csv"):
    """
    Memuat data ulasan MENTAH/INDIVIDUAL dari file CSV.
    File ini digunakan untuk menampilkan ulasan.
    """
    try:
        df_reviews = pd.read_csv(
            file_path,
            usecols=['Place_name', 'sentiment', 'Komentar'] 
        )
        df_reviews.dropna(inplace=True)
        return df_reviews
    except FileNotFoundError:
        st.warning(f"File '{file_path}' (Ulasan Individual) tidak ditemukan. Contoh ulasan tidak akan ditampilkan.")
        return pd.DataFrame()
    except ValueError:
        # Error jika usecols tidak ditemukan
        st.warning(f"Kolom di '{file_path}' tidak lengkap. Setidaknya butuh 'Place_name', 'sentiment', 'Komentar'.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Gagal memuat file ulasan individual: {e}")
        return pd.DataFrame()
# ---------------------------------------------------------


# --- 2. Memuat Semua Data & Model ---
tfidf, svm_model, kmeans_model, profil_nama, profil_vektor = load_models()
library_data = load_library_data()
# --- BARU: Memuat semua ulasan ---
all_reviews = load_review_data()
# ---------------------------------

with st.sidebar:
    st.image("Logo.png", width=100) # Opsional: Ganti dengan URL logo Anda
    selected_page = option_menu(
        menu_title="Menu Utama",
        options=["Beranda", "Rekomendasi", "Analisis Ulasan", "About","Feedback"],
        icons=["house-door-fill", "star-fill", "search", "info-circle-fill"],
        menu_icon="compass-fill",
        default_index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Dibuat oleh Nanda | 2025")

# [GANTI BAGIAN INI DI app.py ANDA]

elif selected_page == "Beranda":
    
    # 1. Kotak Info Biru (Mirip target)
    st.info("ℹ️ **Selamat Datang di Sistem Rekomendasi Buku!** Temukan buku favorit Anda berikutnya di sini.")

    # 2. Search Bar (Mirip target)
    st.text_input(
        "Search, what are you looking for?", 
        placeholder="Cari berdasarkan judul, penulis, atau topik...",
        key="home_search"
    )
    
    st.write("") # Memberi spasi
    
    # 3. Grid Ikon (Menggunakan HTML/CSS kustom untuk meniru tampilan)
    
    # Definisikan CSS untuk tombol-tombol ikon
    # Ini adalah 'sihir' untuk membuat tampilannya mirip
    st.markdown("""
    <style>
    .icon-button {
        background-color: #16a085; /* Warna hijau mirip target */
        border-radius: 15px;      /* Sudut membulat */
        padding: 20px;
        text-align: center;
        color: white !important;  /* Paksa warna teks jadi putih */
        height: 140px;            /* Tinggi konsisten */
        text-decoration: none;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: background-color 0.3s;
    }
    .icon-button:hover {
        background-color: #1abc9c; /* Warna hover lebih cerah */
        color: white !important;   /* Paksa warna teks jadi putih */
        text-decoration: none;
    }
    .icon-button-icon {
        font-size: 48px;          /* Ukuran ikon emoji */
        line-height: 1;
    }
    .icon-button-text {
        margin-top: 10px;
        font-weight: bold;
        font-size: 14px;
    }
    /* Sembunyikan dekorasi link default Streamlit */
    a:link, a:visited {
        text-decoration: none !important;
        color: inherit !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Tombol-tombol ini menggunakan HTML kustom agar bisa di-style.
    # Karena itu, mereka tidak bisa diklik untuk mengubah halaman Streamlit
    # secara langsung. Mereka saat ini HANYA VISUAL.
    
    with col1:
        st.markdown("""
            <div class="icon-button">
                <div class="icon-button-icon">📅</div>
                <div class="icon-button-text">Latest Additions</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="icon-button">
                <div class="icon-button-icon">🔎</div>
                <div class="icon-button-text">Advanced Search</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="icon-button">
                <div class="icon-button-icon">🗂️</div>
                <div class="icon-button-text">Browse Repository</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
            <div class="icon-button">
                <div class="icon-button-icon">ℹ️</div>
                <div class="icon-button-text">About us</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
            <div class="icon-button">
                <div class="icon-button-icon">📜</div>
                <div class="icon-button-text">Policies</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- SISA DARI HALAMAN BERANDA ANDA ---
    # (Letakkan kode metrik dan chart Anda sebelumnya di sini)
    
    st.subheader("Data Overview")
    if not df_books.empty:
        m1, m2, m3 = st.columns(3)
        
        m1.metric("Total Judul Buku", f"{df_books['title'].nunique()} Judul")
        
        try:
            if 'authors' in df_books.columns:
                all_authors = df_books['authors'].dropna().astype(str).unique()
                m2.metric("Total Penulis", f"{len(all_authors)} Penulis")
            else:
                m2.metric("Total Penulis", "N/A")
        except Exception:
            m2.metric("Total Penulis", "N/A")

        if 'categories' in df_books.columns:
            m3.metric("Jumlah Kategori", f"{df_books['categories'].nunique()} Kategori")
        else:
            m3.metric("Jumlah Kategori", "N/A")
            
        st.dataframe(df_books[['title', 'authors', 'categories']].head(10), use_container_width=True)
        
    else:
        st.info("Data buku belum dimuat...")


# ===============================================
# Halaman 2: REKOMENDASI (Kode Lengkap & Diperbaiki)
# ===============================================
elif selected_page == "Rekomendasi":
    st.markdown("""
        <div style='text-align:center; padding: 10px;'>
            <h1>Rekomendasi Perpustakaan di Kota Anda</h1>
            <p style='font-size:18px;'>Temukan perpustakaan terbaik berbasis analisis ribuan ulasan Google Maps</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    # --- Pastikan library_data dimuat ---
    if not library_data.empty:
        # Ganti 'city' jika nama kolom kota Anda berbeda
        available_cities = sorted(library_data['city'].unique()) 
        
        if available_cities:
            
            # --- 1. KUMPULKAN SEMUA INPUT PENGGUNA ---
            selected_city = st.selectbox(
                "📍 Pilih Kota Anda:", options=available_cities, index=None, placeholder="Pilih kota..."
            )
            sort_options = {
                "Skor Terbaik (Rekomendasi)": "skor_kualitas",
                "Rating Google Tertinggi": "rating",
                "Sentimen Paling Positif": "persen_positif"
            }
            sort_by_label = st.selectbox("📊 Urutkan berdasarkan:", options=sort_options.keys())
            sort_by_column = sort_options[sort_by_label]
            min_rating = st.slider("Minimal rating:", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
            st.markdown("---")
            st.subheader("Filter Tambahan Berdasarkan Topik Ulasan")
            filter_options = {
                "Tampilkan Semua": None, "Koleksi Lengkap": "lengkap", "Ramah Disabilitas": "disabilitas",
                "Tempat Nyaman": "nyaman", "Pelayanan Staf": "ramah" # Sesuaikan keyword
            }
            selected_filter_label = st.selectbox("Tampilkan perpustakaan yang sering disebut:", options=filter_options.keys())
            selected_keyword = filter_options[selected_filter_label]

            # --- 2. PROSES & FILTER DATA ---
            if selected_city:
                st.markdown("---")
                st.subheader(f"Rekomendasi di {selected_city} (Diurutkan: {sort_by_label}, Min Rating: {min_rating}⭐)")
                
                # Filter Awal
                # Ganti 'city' jika perlu
                city_libraries = library_data[
                    (library_data['city'] == selected_city) & 
                    (library_data['rating'] >= min_rating)
                ].copy()

                # Filter Keyword
                if selected_keyword and not all_reviews.empty:
                    # Ganti 'Komentar' dan 'Place_name' jika perlu
                    matching_reviews = all_reviews[
                        all_reviews['Komentar'].str.contains(selected_keyword, case=False, na=False)
                    ]
                    matching_libraries_names = matching_reviews['Place_name'].unique() 
                    city_libraries = city_libraries[
                        city_libraries['Place_name'].isin(matching_libraries_names) 
                    ]
                
                # Urutkan
                recommended_libraries = city_libraries.sort_values(
                    by=sort_by_column, ascending=False
                ).head(5)

                # --- 3. TAMPILKAN HASIL ---
                
                if not recommended_libraries.empty:
                    st.subheader("Peta Lokasi Teratas")
                    try:
                        st.map(recommended_libraries[['latitude', 'longitude']])
                    except KeyError as e:
                         st.warning(f"Kolom {e} tidak ada untuk peta.")

                    st.subheader("Detail Peringkat")
                    
                    # Definisikan URL dasar gambar dan fungsi normalisasi DI LUAR LOOP
                    GITHUB_IMAGE_URL = "https://raw.githubusercontent.com/Dwikyf04/google_maps_review_library/main/images/"
                    
                    def normalize_filename(name):
                        name = str(name).lower().strip() # Pastikan string
                        name = name.replace(" ", "-")
                        name = re.sub(r"[^a-z0-9\-]", "", name)
                        return name

                    # --- SATU LOOP UTAMA UNTUK MENAMPILKAN SEMUA ---
                    for i, (_, row) in enumerate(recommended_libraries.iterrows()):
                        # Membuat kontainer (kartu)
                        with st.container(border=True):
                            
                            # --- Logika Menampilkan Gambar ---
                            gambar_url = None
                            if "Image_filename" in row and pd.notna(row["Image_filename"]):
                                file_base = str(row["Image_filename"]).split('.')[0] # Ambil nama tanpa ekstensi
                            else:
                                file_base = normalize_filename(row["Place_name"]) # Buat dari nama tempat

                            image_formats = ["jpg", "jpeg", "png", "webp"]
                            
                            for ext in image_formats:
                                img_url = f"{GITHUB_IMAGE_URL}{file_base}.{ext}"
                                try:
                                    response = requests.head(img_url, timeout=3) # Cek header saja
                                    if response.status_code == 200:
                                        gambar_url = img_url
                                        break # Hentikan jika ketemu
                                except requests.exceptions.RequestException:
                                    pass # Abaikan error koneksi

                            if gambar_url:
                                st.image(gambar_url, caption=row['Place_name'], use_container_width=True)
                            else:
                                st.caption("🖼️ Gambar tidak tersedia")
                            # --- Akhir Logika Gambar ---

                            # Tampilkan Nama Perpustakaan
                            st.markdown(f"### {i + 1}. {row['Place_name']}") 
                            
                            # Tampilkan Metrik & Bagan
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.metric(label="⭐ Rating Google", value=f"{row['rating']:.1f} / 5")
                                st.metric(label="👍 Sentimen Positif", value=f"{row['persen_positif']:.0%}")
                            with col2:
                                st.write("**Distribusi Sentimen:**")
                                try:
                                    chart_data = pd.DataFrame({
                                        "Tipe Sentimen": ["positif", "negative", "neutral"], # Perbaiki typo jika perlu
                                        "Jumlah Ulasan": [
                                            row['jumlah_positif'], 
                                            row['jumlah_negative'], 
                                            row['jumlah_neutral']  
                                        ]
                                    })
                                    st.bar_chart(chart_data, x="Tipe Sentimen", y="Jumlah Ulasan", color="Tipe Sentimen")
                                except KeyError:
                                    st.caption("Kolom jumlah sentimen tidak ada.")
                                except Exception as e:
                                     st.caption(f"Gagal membuat bagan: {e}")
                            
                            # Tampilkan Link Google Maps
                            if 'url_google_maps' in row and pd.notna(row['url_google_maps']) and row['url_google_maps'].startswith('http'):
                                st.link_button("Lihat di Google Maps ↗️", row['url_google_maps'])

                            # Tampilkan Expander Ulasan
                            with st.expander(f"Lihat contoh ulasan untuk {row['Place_name']}"): 
                                if not all_reviews.empty:
                                    try:
                                        library_reviews = all_reviews[all_reviews['Place_name'] == row['Place_name']] 
                                        
                                        if selected_keyword:
                                            st.write(f"**Contoh Ulasan yang Menyebut '{selected_keyword}':**")
                                            matching_keyword_reviews = library_reviews[
                                                library_reviews['Komentar'].str.contains(selected_keyword, case=False, na=False)
                                            ]
                                            if not matching_keyword_reviews.empty:
                                                for _, review_row in matching_keyword_reviews.head(3).iterrows():
                                                    if review_row['sentiment'] == 'Positif': # Gunakan LABEL_POSITIF jika perlu
                                                        st.success(f"• {review_row['Komentar']}")
                                                    elif review_row['sentiment'] == 'Negatif': # Gunakan LABEL_NEGATIF jika perlu
                                                        st.warning(f"• {review_row['Komentar']}")
                                                    else: 
                                                        st.info(f"• {review_row['Komentar']}")
                                            else:
                                                st.caption(f"Tidak ada contoh ulasan yang menyebut '{selected_keyword}'.")
                                        else: # Tampilkan default Positif/Negatif
                                            st.write("**Contoh Ulasan Positif:**")
                                            pos_reviews = library_reviews[library_reviews['sentiment'] == LABEL_POSITIF]['Komentar'].head(3)
                                            if not pos_reviews.empty:
                                                for review_text in pos_reviews: st.success(f"• {review_text}")
                                            else: st.caption("Tidak ada contoh ulasan positif.")
                                            
                                            st.write("**Contoh Ulasan Negatif:**")
                                            neg_reviews = library_reviews[library_reviews['sentiment'] == LABEL_NEGATIF]['Komentar'].head(3)
                                            if not neg_reviews.empty:
                                                for review_text in neg_reviews: st.warning(f"• {review_text}")
                                            else: st.caption("Tidak ada contoh ulasan negatif.")
                                    except KeyError as e:
                                         st.caption(f"Kolom {e} tidak ditemukan.")
                                    except Exception as e:
                                         st.caption(f"Gagal menampilkan ulasan: {e}")
                                else:
                                    st.caption("File ulasan individual tidak dapat dimuat.")
                        st.write("") # Spasi antar kartu
                else:
                    st.info(f"Tidak ada perpustakaan di {selected_city} yang memenuhi kriteria filter Anda.")
        else:
            st.warning("Tidak ada data kota yang tersedia di file CSV.")
    else:
        st.error("Data perpustakaan (Ringkasan) tidak dapat dimuat.")


elif selected_page == "Analisis Ulasan":
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1> Analisis Sentimen & Topik Ulasan Baru</h1>
            <p style='font-size:18px;'>Masukkan ulasan dan sistem akan memprediksi sentimen + topik ulasan berdasarkan clustering.</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    
    nama_cluster = {
        0: "Fasilitas & Kenyamanan",
        1: "Kelengkapan Koleksi",
        2: "Kualitas Pelayanan Staf"
    }

    if tfidf is not None and svm_model is not None and kmeans_model is not None:

        user_input = st.text_area("Masukkan ulasan pengguna di sini:", "", key="input_ulasan")

        if st.button("Analisis Ulasan"):
            if user_input.strip():
                try:
                    # TF-IDF transform
                    X = tfidf.transform([user_input])

                    # Prediksi
                    sentiment_pred = svm_model.predict(X)[0]
                    cluster_pred = kmeans_model.predict(X)[0]
                    cluster_name = nama_cluster.get(cluster_pred, f"Cluster {cluster_pred}")

                    st.subheader(" Hasil Analisis")
                    st.write(f"**Sentimen Terdeteksi:** {sentiment_pred}")
                    st.write(f"**Topik Utama Ulasan:** {cluster_name}")

                    rekomendasi_dic = {
                        0: "Fasilitas nyaman seperti ruangan AC, WiFi, dan tempat duduk nyaman",
                        1: "Ketersediaan koleksi buku dan akses digital sangat baik",
                        2: "Pelayanan staf ramah dan cepat"
                    }
                    st.info(f"Insight otomatis: {rekomendasi_dic.get(cluster_pred)}")

                    # === Kata Kunci TF-IDF ===
                    st.markdown("---")
                    st.subheader("Kata Penting dalam Ulasan")

                    feature_names = tfidf.get_feature_names_out()
                    scores = X.toarray().flatten()

                    df_scores = pd.DataFrame({
                        "Kata": feature_names,
                        "Skor TF-IDF": scores
                    })

                    top_words = df_scores[df_scores['Skor TF-IDF'] > 0].nlargest(7, "Skor TF-IDF")

                    if not top_words.empty:
                        st.table(top_words)
                    else:
                        st.caption("Tidak ada kata penting (ulasan terlalu pendek).")

                    # === REKOMENDASI BERDASARKAN SIMILARITY ===
                    if 'profil_vektor' in globals() and profil_vektor is not None:
                        st.markdown("---")
                        st.subheader("Rekomendasi Perpustakaan yang Relevan")
                        st.caption("Berdasarkan kemiripan teks ulasan Anda dengan ulasan perpustakaan lain")

                        try:
                            similarity_scores = cosine_similarity(X, profil_vektor).flatten()
                            top_indices = similarity_scores.argsort()[::-1][:5]

                            rekomendasi_df = pd.DataFrame({
                                "Perpustakaan": [profil_nama[i] for i in top_indices],
                                "Similarity (%)": [round(similarity_scores[i] * 100, 2) for i in top_indices]
                            })

                            st.dataframe(rekomendasi_df, use_container_width=True)

                            if not library_data.empty:
                                st.subheader("Lokasi Perpustakaan Rekomendasi")
                                display_map = library_data[
                                    library_data['Place_name'].isin(rekomendasi_df['Perpustakaan'])
                                ]
                                st.map(display_map[['latitude', 'longitude']])
                                st.markdown("### 🔗 Akses Google Maps:")
                                for _, row in display_map.iterrows():
                                    if pd.notna(row['url_google_maps']) and row['url_google_maps'].startswith("http"):
                                        st.link_button(f"📍 {row['Place_name']}", row['url_google_maps'])
                                else:
                                    st.caption(f"{row['Place_name']} tidak memiliki tautan Google Maps.")
                        except Exception as e:
                            st.warning(f"Gagal menghitung similarity: {e}")

                    else:
                        st.info("⚠️ Model rekomendasi belum tersedia.")

                except Exception as e:
                    st.error(f"❌ Gagal memprediksi: {e}")

            else:
                st.warning("Masukkan teks ulasan terlebih dahulu!")

    else:
        st.error("⚠️ Model gagal dimuat. Pastikan semua file .pkl sudah tersedia dalam folder Models.")



elif selected_page == "About":
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1> About </h1>
            <p style='font-size:18px;'>Tentang portofolio ini</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    
    st.markdown("""
    <div style='text-align: justify; font-size: 17px; line-height: 1.6;'>
    
    web ini dibbuat oleh saya sendiri sebagai proyek untuk portofolio saya dan pengimpelentasian ilmu-ilmu yang saya pelajari baik melalui perkuliahan maupun melalui bootchamp. 
    <br><br>
    saya membuat website ini dilatarbelakangi oleh keresahan saya setiap ingin pergi ke perpustakaan tetapi masih perlu bertanya ke teman terkait review perpustakaan tersebut.
    Oleh Karena itu, saya ingin menyediakan platform analisis berbasis data untuk membantu pengguna menemukan perpustakaan terbaik di Indonesia.
    <br> <br>
    Hal ini juga Sejalan dengan temuan Solomon et.al  (2012), media sosial memiliki peran penting dalam memengaruhi perilaku konsumen. 
    Mereka menjelaskan bahwa media sosial menyediakan ruang interaksi bebas bagi para penggunanya untuk saling bertukar informasi berupa ulasan, penilaian, foto, serta pengalaman pribadi. 
    Budaya yang terbentuk di dalam platform tersebut dapat memberikan dampak besar terhadap keputusan seseorang dalam membeli atau menggunakan suatu layanan, termasuk dalam menentukan pilihan destinasi yang ingin dikunjungi (Rutbah & Prama, 2025).
    <br><br>
    </div>
    """, unsafe_allow_html=True)
                
    st.divider()

    st.markdown("""
    ### Metodologi
    Proyek ini menggabungkan dua pendekatan *machine learning*:
    
    1.  **Analisis Sentimen (Supervised Learning)**
        * **Model:** `LinearSVC` (Support Vector Machine).
        * **Tujuan:** Mengklasifikasikan setiap ulasan ke dalam sentimen **Positif**, **Negatif**, atau **Netral**.
        * **Fitur:** `TfidfVectorizer` (n-grams 1-2, max 3000 fitur).
    
    2.  **Clustering Topik (Unsupervised Learning)**
        * **Model:** `K-Means Clustering` (K=3).
        * **Tujuan:** Mengelompokkan ulasan ke dalam 3 topik utama secara otomatis. Berdasarkan analisis, topik tersebut adalah:
            * **Cluster 0: Fasilitas & Kenyamanan** (membahas wifi, ac, tempat duduk, gedung).
            * **Cluster 1: Kelengkapan Koleksi** (membahas buku, jurnal, digital, lengkap).
            * **Cluster 2: Kualitas Pelayanan** (membahas staf, antrian, pelayanan, ramah).
    
    3.  **Skor Rekomendasi (Tab Rekomendasi)**
        * Skor dihitung secara offline menggunakan formula:
        * `skor_kualitas = (0.6 * Rating_Google_Normalized) + (0.4 * Persentase_Sentimen_Positif)`
    
    4.  **Rekomendasi Sesuai Selera (Tab Analisis ulasan)**
        * Menggunakan **Content-Based Filtering**.
        * Profil TF-IDF dari setiap perpustakaan dicocokkan dengan vektor TF-IDF dari input pengguna menggunakan **Cosine Similarity**.

    5. **Referensi**
       * Fajri Koto, and Gemala Y. Rahmaningtyas "InSet Lexicon: Evaluation of a Word List for Indonesian Sentiment Analysis in Microblogs". IEEE in the 21st International Conference on Asian Language Processing (IALP), Singapore, December 2017.*
       * Rutba, S. A., & Pramana, S. (2025). Aspect-based Sentiment Analysis and Topic Modelling of International Media on Indonesia Tourism Sector Recovery. Indonesian Journal of Tourism and Leisure, 6(1), 76-94.* 
       * https://github.com/adeariniputri/text-preprocesing*
       * Solomon, M., Russell-Bennett, R., & Previte, J. (2012). Consumer behaviour: Buying, having, being (3rd ed.). Pearson Australia.
    6. **Dataset**
        * Seluruh data ulasan dan rating diambil dari **Google Maps**.
    """)

elif selected_page == "Feedback":

    # === HEADER ===
    st.markdown("""
        <div style='text-align:center; padding: 15px;'>
            <h2>Formulir Feedback Pengguna</h2>
            <p style='font-size:17px;'>Masukan Anda sangat berharga bagi pengembangan aplikasi ini 🙌</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # === KONEKSI KE GOOGLE SHEETS — MENGGUNAKAN STREAMLIT SECRETS ===
    SCOPE = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
        client = gspread.authorize(creds)
        sheet = client.open("feedback_portofolio").sheet1
    except Exception as e:
        st.error(f"⚠️ Tidak dapat terhubung ke Google Sheets: {e}")
        sheet = None

    # === FORM INPUT FEEDBACK ===
    with st.form("feedback_form"):
        user_name = st.text_input("Nama (opsional)")
        user_city = st.text_input("Kota Asal (opsional)")
        user_rating = st.slider("Seberapa puas Anda dengan aplikasi ini?", 1, 5, 5)
        user_feedback = st.text_area("Kritik / Saran Anda ✍️")

        submitted = st.form_submit_button("Kirim Feedback ✅")

    # === SIMPAN FEEDBACK ===
    if submitted:
        if not user_feedback.strip():
            st.warning("Mohon isi feedback terlebih dahulu ✅")
        elif sheet is None:
            st.error("❌ Feedback gagal dikirim karena koneksi database belum siap.")
        else:
            try:
                sheet.append_row([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user_name,
                    user_city,
                    user_rating,
                    user_feedback
                ])
                st.success("✨ Terima kasih! Feedback Anda berhasil dikirim.")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Gagal menyimpan feedback: {e}")





    










































































































































































