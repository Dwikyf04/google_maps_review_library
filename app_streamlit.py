# [File: app_streamlit.py]

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
# === Konstanta Label Sentimen ===
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

if selected_page == "Beranda":
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1>Sistem Rekomendasi Perpustakaan Indonesia</h1>
            <p style='font-size:18px;'>Cari perpustakaan terbaik berbasis analisis ribuan ulasan Google Maps dengan NLP & Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.write("""
    Aplikasi ini menganalisis ribuan ulasan **Google Maps** untuk membantu pengguna menemukan 
    perpustakaan terbaik di Indonesia. Sistem menggabungkan teknologi **NLP (Natural Language Processing)** 
    dan **Machine Learning** untuk memberikan hasil yang akurat dan informatif.
    """)
    st.divider()
    if not library_data.empty and not all_reviews.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Perpustakaan", f"{library_data['Place_name'].nunique()}+ Perpustkaaan")
        col2.metric("Total Komentar", f"{len(all_reviews)}+ Komentar")
        col3.metric("Jumlah Kota", f"{library_data['city'].nunique()} Kota")
    else:
        st.info("Data sedang dimuat...")

    st.divider()

       # ✅ Fitur Utama
    st.markdown("### 🚀 Fitur Utama Aplikasi")
    fitur_cols = st.columns(2)
    fitur_cols[1].success("🎯 Rekomendasi Perpustakaan Terbaik")
    fitur_cols[0].info("🔎 Analisis Sentimen Ulasan Baru")
    #fitur_cols[2].warning("📊 Visualisasi Data Interaktif")

    st.divider()
    

    # 🗺️ Peta Persebaran
    st.subheader("Peta Sebaran Perpustakaan")
    if not library_data.empty:
        st.map(library_data[['latitude', 'longitude', 'Place_name']])

    # 📊 Lima Kota Skor Terbaik
    st.subheader("Kota dengan Skor Kualitas Rata-rata Tertinggi")
    if not library_data.empty:
        top_cities = library_data.groupby('city')['skor_kualitas'].mean().nlargest(5)
        st.bar_chart(top_cities)
            # Insight Kota Terbaik
    if not library_data.empty:
        best_city = top_cities.index[0]
        best_score = top_cities.iloc[0]

        st.markdown("🔍 **Insight Kota:**")
        st.write(
            f"• **{best_city}** memiliki skor kualitas tertinggi: **{best_score:.2f}** ✅\n"
            f"• Menunjukkan kualitas layanan dan fasilitas yang sangat baik."
        )
    st.divider()
    # ⭐ Distribusi Rating
    st.subheader("Distribusi Rating Perpustakaan")
    if not library_data.empty:
        rating_counts = library_data['rating'].value_counts().sort_index()
        st.bar_chart(rating_counts)
        # Insight Distribusi Rating
    if not library_data.empty:
        avg_rating = library_data['rating'].mean()
        high_rating_pct = (library_data['rating'] >= 4.0).mean() * 100

        st.markdown(f"🔍 **Insight Rating:**")
        st.write(
            f"• Rata-rata rating perpustakaan: **{avg_rating:.2f} / 5**\n"
            f"• {high_rating_pct:.1f}% perpustakaan memiliki rating **≥ 4.0** ⭐\n"
        )

    st.divider()
    # 👍 Sentimen Positif vs Negatif
    st.subheader("Distribusi Sentimen Positif vs Negatif")
    if 'persen_positif' in library_data.columns:
        sentiment_summary = pd.DataFrame({
            "Positif (%)": library_data['persen_positif'] * 100,
            "Negatif (%)": (1 - library_data['persen_positif']) * 100
        })
        st.line_chart(sentiment_summary)
    else:
        st.warning("Data sentimen positif belum tersedia!")
    if 'persen_positif' in library_data.columns:
        avg_positive = library_data['persen_positif'].mean() * 100

        st.markdown("🔍 **Insight Sentimen:**")
        st.write(
            f"• Sentimen positif rata-rata: **{avg_positive:.1f}%** 👍\n"
            f"• Pengunjung perpustakaan di Indonesia **dominan puas**."
        )
        
    st.divider()

    # 🏆 Top 10 Sentimen Positif
    st.subheader("Top 10 Perpustakaan dengan Sentimen Positif Tertinggi")
    if not library_data.empty:
        top_positive = library_data.sort_values(by="persen_positif", ascending=False).head(10)
        st.dataframe(
            top_positive[['Place_name', 'city', 'rating', 'persen_positif']],
            use_container_width=True
        )
    st.divider()
    
    #if not all_reviews.empty:
        #st.subheader("☁️ Word Cloud Ulasan Perpustakaan (Semua Kota)")
        #text_reviews = " ".join(all_reviews['Komentar'].astype(str))
        #if text_reviews.strip():
            #wc = WordCloud(width=800, height=500, background_color="white").generate(text_reviews)
            
            #fig_wc, ax_wc = plt.subplots()
            #ax_wc.imshow(wc, interpolation='bilinear')
            #ax_wc.axis('off')
            #st.pyplot(fig_wc)
        #else:
            #st.caption("Tidak ada ulasan.")


# ===============================================
# Halaman 2: REKOMENDASI (Kode Tab 1 Lama Anda)
# ===============================================
# Di app_streamlit.py
# ===============================================
# Halaman 2: REKOMENDASI 
# ===============================================
# ===============================================
# Halaman 2: REKOMENDASI (Kode Tab 1 Lama Anda)
# ===============================================
elif selected_page == "Rekomendasi":
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1> merekommendasikan perpustakaan di kota anda</h1>
            <p style='font-size:18px;'>Cari perpustakaan terbaik berbasis analisis ribuan ulasan Google Maps dengan NLP & Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    
    if not library_data.empty:
        # Asumsi kolom Anda bernama 'kota'
        available_cities = sorted(library_data['city'].unique()) 
        if available_cities:
            
            # --- 1. KUMPULKAN SEMUA INPUT PENGGUNA ---
            
            # INPUT 1: Pilih Kota
            selected_city = st.selectbox(
                "📍 Pilih Kota Anda:",
                options=available_cities,
                index=None,
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
                options=sort_options.keys()
            )
            sort_by_column = sort_options[sort_by_label] # Dapatkan nama kolom

            # INPUT 3: Filter Rating
            min_rating = st.slider(
                "Tampilkan perpustakaan dengan minimal rating:",
                min_value=1.0, max_value=5.0, value=3.5, step=0.1
            )

            # INPUT 4: Filter Keyword (URUTAN SUDAH BENAR)
            st.markdown("---")
            st.subheader("Filter Tambahan Berdasarkan Topik Ulasan")
            filter_options = {
                "Tampilkan Semua": None,
                "Koleksi Lengkap": "lengkap",
                "Ramah Disabilitas": "disabilitas",
                "Tempat Nyaman": "nyaman",
                "Pelayanan Staf": "ramah" # Contoh
            }
            selected_filter_label = st.selectbox(
                "Tampilkan perpustakaan yang sering disebut:",
                options=filter_options.keys()
            )
            selected_keyword = filter_options[selected_filter_label]

            
            # --- 2. PROSES & FILTER DATA ---
            
            if selected_city:
                st.markdown("---")
                # (Sisa kode filter Anda)
                city_libraries = library_data[
                    (library_data['city'] == selected_city) &
                    (library_data['rating'] >= min_rating)
                ].copy()

                if selected_keyword and not all_reviews.empty:
                    matching_reviews = all_reviews[
                        all_reviews['Komentar'].str.contains(selected_keyword, case=False, na=False)
                    ]
                    matching_libraries_names = matching_reviews['Place_name'].unique()
                    city_libraries = city_libraries[
                        city_libraries['Place_name'].isin(matching_libraries_names)
                    ]
                
                recommended_libraries = city_libraries.sort_values(
                    by=sort_by_column, 
                    ascending=False
                ).head(5)

                
                # --- 3. TAMPILKAN HASIL ---
                
                if not recommended_libraries.empty:
                    st.subheader("Peta Lokasi Teratas")
                    st.map(recommended_libraries[['latitude', 'longitude']])
                    
                    st.subheader("Detail Peringkat")
                   
                    for i, (_, row) in enumerate(recommended_libraries.iterrows()):
                        st.markdown(f"#### {i + 1}. {row['Place_name']}") 
                        # ... (Kode st.metric Anda) ...
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="⭐ Rating Google", value=f"{row['rating']:.1f} / 5")
                        with col2:
                            if 'persen_positif' in row and pd.notna(row['persen_positif']):
                                st.metric(label="👍 Sentimen Positif", value=f"{row['persen_positif']:.0%}")
                            elif 'skor_kualitas' in row and pd.notna(row['skor_kualitas']):
                                st.metric(label="💯 Skor Kualitas", value=f"{row['skor_kualitas']:.2f}")

                        # ... (Kode st.link_button Anda) ...
                        if 'url_google_maps' in row and pd.notna(row['url_google_maps']) and row['url_google_maps'].startswith('http'):
                            st.link_button("Lihat di Google Maps ↗️", row['url_google_maps'])

                        # --- PERUBAHAN DI SINI: Logika Expander ---
                        with st.expander(f"Lihat ulasan untuk {row['Place_name']}"):
                            if not all_reviews.empty:
                                # Filter ulasan hanya untuk perpustakaan ini
                                library_reviews = all_reviews[all_reviews['Place_name'] == row['Place_name']]
                                
                                # --- LOGIKA BARU ---
                                # JIKA PENGGUNA MEMILIH KEYWORD FILTER
                                if selected_keyword:
                                    st.write(f"**Ulasan yang Menyebut '{selected_keyword}':**")
                                    # Filter ulasan yang mengandung keyword
                                    matching_keyword_reviews = library_reviews[
                                        library_reviews['Komentar'].str.contains(selected_keyword, case=False, na=False)
                                    ]
                                    
                                    if not matching_keyword_reviews.empty:
                                        # Tampilkan 3 contoh, warnai berdasarkan sentimen
                                        for _, review_row in matching_keyword_reviews.head(3).iterrows():
                                            if review_row['sentiment'] == 'Positif':
                                                st.success(f"• {review_row['Komentar']}")
                                            elif review_row['sentiment'] == 'Negatif':
                                                st.warning(f"• {review_row['Komentar']}")
                                            else:
                                                st.info(f"• {review_row['Komentar']}")
                                    else:
                                        st.caption(f"Tidak ada ulasan yang menyebut '{selected_keyword}'.")
                                
                                # JIKA PENGGUNA TIDAK MEMILIH FILTER (Tampilkan Semua)
                                else:
                                    st.write("**Ulasan Positif:**")
                                    pos_reviews = library_reviews[library_reviews['sentiment'] == 'Positif']['Komentar'].head(3)
                                    if not pos_reviews.empty:
                                        for review_text in pos_reviews:
                                            st.success(f"• {review_text}")
                                    else:
                                        st.caption("Tidak ada ulasan positif.")
                                    
                                    st.write("**Ulasan Negatif:**")
                                    neg_reviews = library_reviews[library_reviews['sentiment'] == 'Negatif']['Komentar'].head(3)
                                    if not neg_reviews.empty:
                                        for review_text in neg_reviews:
                                            st.warning(f"• {review_text}")
                                    else:
                                        st.caption("Tidak ada ulasan negatif.")
                            else:
                                st.caption("File ulasan individual tidak dapat dimuat.")
                        st.divider()
                else:
                    st.info(f"Tidak ada perpustakaan di {selected_city} yang memenuhi kriteria filter Anda.")
        else:
            st.warning("Tidak ada data kota yang tersedia di file CSV.")
    else:
        st.error("Data perpustakaan (Ringkasan) tidak dapat dimuat.")


# --- 6. Isi Tab 2: Analisis Ulasan Individual ---
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

elif selected_page == "Feedback":
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1> Formulir Feedback Pengguna </h1>
            <p style='font-size:18px;'>Feedback untuk kami</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    FEEDBACK_FILE = "feedback_pengguna.csv"
    st.write("Bantu kami meningkatkan kualitas aplikasi ini dengan memberikan feedback!")

# Input form
    with st.form("feedback_form"):
        name = st.text_input("Nama (opsional)", "")
        city = st.text_input("Kota asal pengguna (opsional)", "")
        rating = st.slider("Seberapa puas Anda dengan aplikasi ini?", 1, 5, 4)
        feedback = st.text_area("Masukan / kritik / saran Anda", "")

        submitted = st.form_submit_button("Kirim")

    if submitted:
        if feedback.strip() == "":
            st.warning("Mohon isi kritik atau saran terlebih dahulu ✅")
        else:
            new_data = pd.DataFrame([{
                "timestamp": datetime.datetime.now(),
                "name": name,
                "city": city,
                "rating": rating,
                "feedback": feedback
            }])

        try:
            # Jika file sudah ada → append
            existing_data = pd.read_csv(FEEDBACK_FILE)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        except FileNotFoundError:
            # Jika file belum ada → buat file baru
            updated_data = new_data

        updated_data.to_csv(FEEDBACK_FILE, index=False)

        st.success("✅ Terima kasih! Feedback Anda sudah dikirim.")
        st.balloons()

    


























































































