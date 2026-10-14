# 🎬 Movie Recommendation System (MovieLens 100K)

Sistem rekomendasi film berbasis **Content-Based Filtering** menggunakan algoritma **Cosine Similarity** pada fitur genre film dari dataset MovieLens 100K. Proyek ini dibangun menggunakan Python dan didistribusikan dalam bentuk dashboard interaktif menggunakan **Streamlit**.

---

## 📌 Fitur Utama

- **🏠 Beranda**: Ringkasan proyek, statistik dataset utama, dan quick test rekomendasi.
- **📦 Data Collecting**: Tinjauan dataset (u.data dan u.item) serta unduh data otomatis.
- **🔍 Data Exploring (EDA)**: Analisis distribusi rating, sebaran genre, film terpopuler, dan korelasi antar fitur.
- **🧹 Data Preparation**: Penanganan missing value, duplikasi, dan pembangunan matriks fitur genre (*item profile*).
- **🤖 Modeling & Evaluasi**: Rekomendasi film interaktif berbasis *Cosine Similarity* disertai matriks similarity dan evaluasi genre (*precision*).
- **📖 Dokumentasi**: Panduan metodologi dan formula matematika yang digunakan.

---

## 🛠️ Tech Stack & Dependensi

- **Python** (>= 3.9)
- **Streamlit** (UI Dashboard)
- **Pandas & NumPy** (Manipulasi data)
- **Scikit-Learn** (Perhitungan Cosine Similarity)
- **Matplotlib & Seaborn** (Visualisasi data)

---

## 🚀 Menjalankan Secara Lokal

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/Lahen194/MovieRecommendation.git
   cd MovieRecommendation
   ```

2. **Buat & aktifkan virtual environment (opsional namun disarankan):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Panduan Deploy ke Streamlit Community Cloud

1. Upload seluruh file proyek ke repository **GitHub** (publik atau privat).
2. Kunjungi [Streamlit Community Cloud](https://share.streamlit.io/).
3. Login menggunakan akun GitHub Anda.
4. Klik tombol **New app**.
5. Pilih:
   - **Repository**: Repositori proyek ini
   - **Branch**: `main` (atau `master`)
   - **Main file path**: `app.py`
6. Klik **Deploy!**

---

## 👤 Author
- **Nama**: Fakhira Lahen Siregar
- **Dicoding ID**: MC009D5X2078
