import pandas as pd
import joblib
import numpy as np
import sys

# --- Pengaturan ---
FILE_ULASAN_MENTAH = "data_perpustakaan_review.csv"
FILE_TFIDF = "Models/tfidf_vectorizer.pkl"
FILE_NAMA_OUTPUT = "profil_nama.pkl"
FILE_VEKTOR_OUTPUT = "profil_vektor.pkl"

print("Memulai pembuatan profil perpustakaan...")

# --- 1. Muat Data Ulasan Mentah ---
try:
    df_reviews = pd.read_csv(FILE_ULASAN_MENTAH, usecols=['nama_perpustakaan', 'Komentar'])
    df_reviews = df_reviews.dropna()
    print("Berhasil memuat data ulasan mentah.")
except FileNotFoundError:
    print(f"Error: File '{FILE_ULASAN_MENTAH}' tidak ditemukan.")
    sys.exit()
except ValueError:
    print(f"Error: Kolom 'nama_perpustakaan' atau 'Komentar' tidak ada di {FILE_ULASAN_MENTAH}.")
    sys.exit()

# --- 2. Buat Profil Teks (Gabungkan Ulasan) ---
print("Menggabungkan ulasan untuk membuat profil teks...")
# Gabungkan semua ulasan untuk setiap perpustakaan menjadi satu string teks besar
profil_teks = df_reviews.groupby('nama_perpustakaan')['Komentar'].apply(' '.join)

# Simpan daftar nama perpustakaan (penting untuk urutan)
profil_nama = profil_teks.index.tolist()
joblib.dump(profil_nama, FILE_NAMA_OUTPUT)
print(f"Profil nama disimpan ke '{FILE_NAMA_OUTPUT}'.")

# --- 3. Muat TF-IDF Vectorizer ---
try:
    tfidf = joblib.load(FILE_TFIDF)
    print("Berhasil memuat TF-IDF Vectorizer.")
except FileNotFoundError:
    print(f"Error: File '{FILE_TFIDF}' tidak ditemukan di folder 'Models/'.")
    sys.exit()

# --- 4. Ubah Profil Teks menjadi Profil Vektor ---
print("Mengubah profil teks menjadi profil vektor TF-IDF...")
# Gunakan .transform() (JANGAN .fit_transform()) agar tetap di 'ruang' yang sama
profil_vektor = tfidf.transform(profil_teks)

# Simpan matriks vektor
joblib.dump(profil_vektor, FILE_VEKTOR_OUTPUT)
print(f"Profil vektor disimpan ke '{FILE_VEKTOR_OUTPUT}'.")
print("\n✅ SUKSES! File profil telah dibuat.")