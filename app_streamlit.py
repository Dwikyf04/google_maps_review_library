# [File: app_streamlit.py]

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu

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
        # Kita hanya butuh kolom-kolom ini untuk menghemat memori
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
    st.image("https://i.pinimg.com/736x/12/f9/ed/12f9ed73b852fd466830c23ab8fb575e.jpg", width=100) # Opsional: Ganti dengan URL logo Anda
    selected_page = option_menu(
        menu_title="Menu Utama",
        options=["Beranda", "Rekomendasi", "Analisis Ulasan", "About"],
        icons=["house-door-fill", "star-fill", "search", "info-circle-fill"],
        menu_icon="compass-fill",
        default_index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Dibuat oleh Nanda | 2025")

if selected_page == "Beranda":
    st.header("Analisis & Rekomendasi Perpustakaan")
    st.markdown("Aplikasi ini membantu Anda menemukan perpustakaan terbaik berdasarkan ulasan nyata pengguna Google Maps.")
    st.divider()

    st.subheader("Ringkasan Data")
    if not library_data.empty and not all_reviews.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Perpustakaan", f"{library_data['Place_name'].nunique()} Perpus")
        col2.metric("Total Ulasan Dianalisis", f"{len(all_reviews)} Ulasan")
        col3.metric("Jumlah Kota", f"{library_data['city'].nunique()} Kota")
    else:
        st.info("Data sedang dimuat atau tidak ditemukan.")

    st.subheader("Peta Sebaran Perpustakaan")
    if not library_data.empty:
        st.map(library_data[['latitude', 'longitude', 'Place_name']])
    
    st.subheader("Kota dengan Skor Kualitas Rata-rata Tertinggi")
    if not library_data.empty:
        top_cities = library_data.groupby('city')['skor_kualitas'].mean().nlargest(5)
        st.bar_chart(top_cities)
    col1, col2 = st.columns([1, 2]) # Kolom 1 lebih kecil
            
# ===============================================
# Halaman 2: REKOMENDASI (Kode Tab 1 Lama Anda)
# ===============================================
# Di app_streamlit.py
# ===============================================
# Halaman 2: REKOMENDASI 
# ===============================================
elif selected_page == "Rekomendasi":
    st.header("🏆 Temukan Perpustakaan Terbaik di Kota Anda")
    
    # Periksa apakah data ringkasan perpustakaan berhasil dimuat
    if not library_data.empty:
        # Ambil daftar kota unik dari data ringkasan
        # Ganti 'kota' jika nama kolom di data_perpustakaan.csv berbeda
        available_cities = sorted(library_data['city'].unique()) 
        
        if available_cities:
            
            # --- 1. KUMPULKAN SEMUA INPUT PENGGUNA ---
            
            # INPUT 1: Pilih Kota
            selected_city = st.selectbox(
                "📍 Pilih Kota Anda:",
                options=available_cities,
                index=None, # Default tidak ada yang terpilih
                placeholder="Pilih kota..."
            )
            
            # INPUT 2: Opsi Urut
            sort_options = {
                "Skor Terbaik (Rekomendasi)": "skor_kualitas",
                "Rating Google Tertinggi": "rating",
                "Sentimen Paling Positif": "persen_positif"
            }
            sort_by_label = st.selectbox(
                "📊 Urutkan berdasarkan:",
                options=sort_options.keys() # Tampilkan label
            )
            sort_by_column = sort_options[sort_by_label] # Dapatkan nama kolom teknis

            # INPUT 3: Filter Rating
            min_rating = st.slider(
                "Tampilkan perpustakaan dengan minimal rating:",
                min_value=1.0, 
                max_value=5.0, 
                value=3.5, # Nilai default
                step=0.1
            )

            # INPUT 4: Filter Keyword
            st.markdown("---")
            st.subheader("Filter Tambahan Berdasarkan Topik Ulasan")
            filter_options = {
                "Tampilkan Semua": None,
                "Koleksi Lengkap": "lengkap",
                "Ramah Disabilitas": "disabilitas",
                "Tempat Nyaman": "nyaman",
                "Pelayanan Staf": "ramah" # Sesuaikan keyword jika perlu
            }
            selected_filter_label = st.selectbox(
                "Tampilkan perpustakaan yang sering disebut:",
                options=filter_options.keys()
            )
            selected_keyword = filter_options[selected_filter_label] # Dapatkan keyword (atau None)

            
            # --- 2. PROSES & FILTER DATA ---
            
            # Hanya jalankan jika pengguna sudah memilih kota
            if selected_city:
                st.markdown("---")
                st.subheader(f"Rekomendasi di {selected_city} (Diurutkan: {sort_by_label}, Min Rating: {min_rating}⭐)")
                
                # Filter Awal: Berdasarkan Kota dan Rating
                # Ganti 'kota' jika nama kolom Anda berbeda
                city_libraries = library_data[
                    (library_data['city'] == selected_city) & 
                    (library_data['rating'] >= min_rating)
                ].copy()

                # Filter Kedua: Berdasarkan Keyword (jika dipilih dan data ulasan ada)
                if selected_keyword and not all_reviews.empty:
                    # Cari di ulasan mentah ('all_reviews')
                    # Pastikan nama kolom 'Komentar' dan 'nama_perpustakaan' sesuai
                    matching_reviews = all_reviews[
                        all_reviews['Komentar'].str.contains(selected_keyword, case=False, na=False)
                    ]
                    matching_libraries_names = matching_reviews['Place_name'].unique() 
                    
                    # Filter data perpustakaan agar hanya menampilkan yang lolos keyword
                    # Ganti 'nama_perpustakaan' jika nama kolom Anda berbeda
                    city_libraries = city_libraries[
                        city_libraries['Place_name'].isin(matching_libraries_names) 
                    ]
                
                # Terakhir: Urutkan hasil akhir berdasarkan pilihan pengguna
                recommended_libraries = city_libraries.sort_values(
                    by=sort_by_column, 
                    ascending=False
                ).head(5) # Ambil 5 teratas

                
                # --- 3. TAMPILKAN HASIL ---
                
                if not recommended_libraries.empty:
                    # Tampilkan Peta
                    st.subheader("Peta Lokasi Teratas")
                    try:
                        # Pastikan kolom 'latitude' dan 'longitude' ada
                        st.map(recommended_libraries[['latitude', 'longitude']])
                    except KeyError as e:
                         st.warning(f"Kolom {e} tidak ada untuk peta.")

                    # Tampilkan Detail Peringkat dalam bentuk Kartu
                    st.subheader("Detail Peringkat")
                    for i, (_, row) in enumerate(recommended_libraries.iterrows()):
                        # Membuat kontainer (kartu) dengan border
                        with st.container(border=True):
                            # Ganti 'nama_perpustakaan' jika perlu
                            st.markdown(f"### {i + 1}. {row['Place_name']}") 
                            
                            # Kolom untuk metrik dan bagan
                            col1, col2 = st.columns([1, 2]) # Kolom 1 lebih kecil
                            with col1:
                                st.metric(label="⭐ Rating Google", value=f"{row['rating']:.1f} / 5")
                                st.metric(label="👍 Sentimen Positif", value=f"{row['persen_positif']:.0%}")
                            with col2:
                                st.write("**Distribusi Sentimen:**")
                                try:
                                    # Buat DataFrame mini untuk bagan
                                    chart_data = pd.DataFrame({
                                        "Tipe Sentimen": ["Positif", "Negatif", "Netral"],
                                        "Jumlah Ulasan": [
                                            row['jumlah_positif'], 
                                            row['jumlah_negatif'], # Pastikan kolom ini ada
                                            row['jumlah_netral']   # Pastikan kolom ini ada
                                        ]
                                    })
                                    st.bar_chart(chart_data, x="Tipe Sentimen", y="Jumlah Ulasan", color="Tipe Sentimen")
                                except KeyError:
                                    st.caption("Kolom jumlah sentimen (negatif/netral) tidak ditemukan di data_perpustakaan.csv.")
                                except Exception as e:
                                     st.caption(f"Gagal membuat bagan: {e}")
                            
                            # Tombol Link Google Maps
                            if 'url_google_maps' in row and pd.notna(row['url_google_maps']) and row['url_google_maps'].startswith('http'):
                                st.link_button("Lihat di Google Maps ↗️", row['url_google_maps'])

                            # Expander untuk Word Cloud
                            # Ganti 'nama_perpustakaan' jika perlu
                            with st.expander(f"Lihat Analisis Word Cloud untuk {row['Place_name']}"): 
                                if not all_reviews.empty:
                                    try:
                                        # Filter ulasan untuk perpustakaan ini
                                        # Ganti 'nama_perpustakaan' jika perlu
                                        library_reviews = all_reviews[all_reviews['Place_name'] == row['Place_name']] 
                                        
                                        # Gabungkan teks (Gunakan LABEL_POSITIF/NEGATIF yang didefinisikan di atas)
                                        # Ganti 'sentiment' dan 'Komentar' jika perlu
                                        text_positif = " ".join(review for review in library_reviews[library_reviews['sentiment'] == LABEL_POSITIF]['Komentar'])
                                        text_negatif = " ".join(review for review in library_reviews[library_reviews['sentiment'] == LABEL_NEGATIF]['Komentar'])
                                        
                                        wc_col1, wc_col2 = st.columns(2)
                                        with wc_col1:
                                            st.write("**Kata Kunci Positif:**")
                                            if text_positif:
                                                wc_pos = WordCloud(background_color="white", colormap="Greens", max_words=30, width=400, height=200).generate(text_positif)
                                                fig_pos, ax_pos = plt.subplots()
                                                ax_pos.imshow(wc_pos, interpolation='bilinear')
                                                ax_pos.axis('off')
                                                st.pyplot(fig_pos, use_container_width=True)
                                            else:
                                                st.caption("Tidak ada data ulasan positif.")
                                        with wc_col2:
                                            st.write("**Kata Kunci Negatif:**")
                                            if text_negatif:
                                                wc_neg = WordCloud(background_color="black", colormap="Reds", max_words=30, width=400, height=200).generate(text_negatif)
                                                fig_neg, ax_neg = plt.subplots()
                                                ax_neg.imshow(wc_neg, interpolation='bilinear')
                                                ax_neg.axis('off')
                                                st.pyplot(fig_neg, use_container_width=True)
                                            else:
                                                st.caption("Tidak ada data ulasan negatif.")
                                    except KeyError as e:
                                         st.caption(f"Kolom {e} tidak ditemukan di data ulasan individual.")
                                    except Exception as e:
                                         st.caption(f"Gagal membuat Word Cloud: {e}")
                                else:
                                    st.caption("File ulasan individual tidak dapat dimuat.")
                        st.write("") # Memberi spasi antar kartu
                else:
                    # Pesan jika tidak ada perpustakaan yang lolos semua filter
                    st.info(f"Tidak ada perpustakaan di {selected_city} yang memenuhi kriteria filter Anda.")
        else:
            st.warning("Tidak ada data kota yang tersedia di file CSV.")
    else:
        # Pesan jika data_perpustakaan.csv gagal dimuat
        st.error("Data perpustakaan (Ringkasan) tidak dapat dimuat.")


# --- 6. Isi Tab 2: Analisis Ulasan Individual ---
elif selected_page == "Analisis Ulasan":
    st.header("Analisis Sentimen & Topik Ulasan Baru")
    # (Pastikan nama cluster ini sesuai dengan analisis Anda)
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
                    X = tfidf.transform([user_input])
                    sentiment_pred = svm_model.predict(X)[0]
                    cluster_pred = kmeans_model.predict(X)[0]
                    cluster_name = nama_cluster.get(cluster_pred, f"Cluster {cluster_pred}")
                    
                    st.subheader("🔍 Hasil Analisis:")
                    st.write(f"**Sentimen:** {sentiment_pred}")
                    st.write(f"**Topik Utama Ulasan:** {cluster_name}")
                    
                    if cluster_pred == 0:
                        st.success(f"📚 Rekomendasi: Ulasan ini berfokus pada **{cluster_name}**.")
                    elif cluster_pred == 1:
                        st.info(f"🧐 Rekomendasi: Ulasan ini berfokus pada **{cluster_name}**.")
                    else:
                        st.warning(f"😔 Rekomendasi: Ulasan ini berfokus pada **{cluster_name}**.")

                    st.markdown("---")
                    st.subheader("Kata Kunci Paling Berpengaruh:")
                
                    feature_names = tfidf.get_feature_names_out()
                    
                    scores = X.toarray().flatten() 
                    
                    df_scores = pd.DataFrame({'kata': feature_names, 'skor_tfidf': scores})
                    
                    top_words = df_scores[df_scores['skor_tfidf'] > 0].sort_values(
                        by='skor_tfidf', 
                        ascending=False
                    ).head(5)
                    
                    if not top_words.empty:
                        st.dataframe(top_words, use_container_width=True)
                    else:
                        st.caption("Tidak ada kata kunci yang dikenali (mungkin semua stopwords).")
              
                except Exception as e:
                    st.error(f"Gagal melakukan prediksi: {e}")

            else:
                st.warning("Harap masukkan teks terlebih dahulu!")
    else:
        st.error("Model analisis (SVM/K-Means/TF-IDF) gagal dimuat. Tab ini tidak dapat berfungsi.")

elif selected_page == "About":
    st.header("About")
    st.markdown("""
    web ini dibbuat oleh saya sendiri sebagai proyek untuk portofolio saya dan pengimpelentasian ilmu-ilmu yang saya pelajari baik melalui perkuliahan maupun melalui bootchamp. 
    Tujuannya adalah untuk membangun sistem rekomendasi perpustakaan berdasarkan ulasan otentik dari Google Maps. saya membuat website ini dilatarbelakangi oleh keresahan saya setiap ingin pergi ke perpustakaan tetapi 
    masih perlu bertanya ke teman terkait review perpustakaan tersebut.

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
    
    4.  **Rekomendasi Sesuai Selera (Tab Analisis)**
        * Menggunakan **Content-Based Filtering**.
        * Profil TF-IDF dari setiap perpustakaan dicocokkan dengan vektor TF-IDF dari input pengguna menggunakan **Cosine Similarity**.
    
    ### Dataset
    * Seluruh data ulasan dan rating diambil dari **Google Maps**.
    """)

















































