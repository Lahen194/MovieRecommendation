# -*- coding: utf-8 -*-
"""
Sistem Rekomendasi Film Berbasis Genre dengan Content-Based Filtering

Author  : Fakhira Lahen Siregar
ID      : MC009D5X2078
Dataset : MovieLens 100K - https://grouplens.org/datasets/movielens/
"""

import io
import os
import random
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "ml-100k"

GENRE_COLS = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

MOVIE_COLUMNS = (
    ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL']
    + ['unknown'] + GENRE_COLS
)


# --- 1. Data Collecting ---

def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Membaca file u.data dan mengembalikan DataFrame ratings."""
    path = os.path.join(data_dir, "u.data")
    ratings = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    return ratings


def load_movies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Membaca file u.item dan mengembalikan DataFrame metadata film."""
    path = os.path.join(data_dir, "u.item")
    movies = pd.read_csv(
        path,
        sep='|',
        encoding='latin-1',
        names=MOVIE_COLUMNS,
        usecols=range(24)
    )
    movies = movies.drop(
        columns=['release_date', 'video_release_date', 'IMDb_URL', 'unknown']
    )
    movies.rename(columns={'movie_id': 'item_id'}, inplace=True)
    return movies


def download_and_extract_data(data_dir: str = DATA_DIR):
    """Download dan extract dataset MovieLens 100K jika belum ada."""
    if not os.path.exists(data_dir):
        print(f"Dataset tidak ditemukan di {data_dir}. Mengunduh...")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        zip_path = "ml-100k.zip"
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)
            print("Dataset berhasil diunduh dan diekstrak.")
        except Exception as e:
            print(f"Gagal mengunduh dataset: {e}")
            raise


def load_data(data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Memuat semua data dan menggabungkan ratings + movies berdasarkan item_id."""
    download_and_extract_data(data_dir)
    ratings = load_ratings(data_dir)
    movies  = load_movies(data_dir)
    df      = pd.merge(ratings, movies, on="item_id")
    return ratings, movies, df


# --- 2. Data Exploring ---

def get_data_info(df: pd.DataFrame) -> str:
    """Mengembalikan output df.info() sebagai string."""
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def get_data_shape(df: pd.DataFrame) -> dict:
    """Mengembalikan dict ringkasan dimensi baris dan kolom."""
    numeric_cols    = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return {
        "baris"           : df.shape[0],
        "kolom"           : df.shape[1],
        "kolom_numerik"   : len(numeric_cols),
        "kolom_kategorikal": len(categorical_cols),
        "nama_numerik"    : numeric_cols,
        "nama_kategorikal": categorical_cols,
    }


def get_nunique_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mengembalikan DataFrame berisi jumlah nilai unik tiap kolom."""
    return (
        pd.DataFrame({
            "Kolom"         : df.columns,
            "Tipe Data"     : df.dtypes.values,
            "Nilai Unik"    : [df[c].nunique() for c in df.columns],
        })
        .reset_index(drop=True)
    )


def get_duplicate_titles(movies: pd.DataFrame) -> pd.DataFrame:
    """Mengembalikan data film yang memiliki judul sama dengan item_id berbeda."""
    dupes = (
        movies[movies.duplicated(subset=['title'], keep=False)]
        .sort_values('title')[['item_id', 'title']]
        .reset_index(drop=True)
    )
    return dupes


def get_rating_value_counts(ratings: pd.DataFrame) -> pd.DataFrame:
    """Mengembalikan distribusi frekuensi nilai rating."""
    vc = ratings['rating'].value_counts().sort_index()
    return pd.DataFrame({"Rating": vc.index, "Jumlah": vc.values})


def get_top_rated_movies(ratings: pd.DataFrame, movies: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Mengembalikan top-N film yang paling banyak dirating beserta rata-ratanya."""
    result = (
        ratings.merge(movies[['item_id', 'title']], on='item_id')
        .groupby('title')
        .agg(
            jumlah_rating =('rating', 'count'),
            rata_rata_rating=('rating', 'mean')
        )
        .sort_values('jumlah_rating', ascending=False)
        .head(top_n)
        .reset_index()
    )
    result['rata_rata_rating'] = result['rata_rata_rating'].round(2)
    result.columns = ['Judul Film', 'Jumlah Rating', 'Rata-rata Rating']
    return result


def get_genre_counts(movies: pd.DataFrame) -> pd.Series:
    """Menghitung total film per genre, diurutkan secara menurun."""
    return movies[GENRE_COLS].sum().sort_values(ascending=False)


# --- 3. Data Preparation ---

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Memeriksa jumlah dan persentase missing value pada setiap kolom."""
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    result  = pd.DataFrame({
        "Kolom"        : missing.index,
        "Missing Value": missing.values,
        "Persentase (%)" : pct.values,
    })
    return result[result["Missing Value"] >= 0].reset_index(drop=True)


def check_duplicates(df: pd.DataFrame) -> int:
    """Mengembalikan jumlah baris duplikat dalam DataFrame."""
    return int(df.duplicated().sum())


def build_movie_features(df: pd.DataFrame) -> pd.DataFrame:
    """Membangun item profile biner (title x genre) setelah menghapus duplikasi judul."""
    movie_features = (
        df[['title'] + GENRE_COLS]
        .drop_duplicates(subset=['title'])
        .set_index('title')
    )
    return movie_features


# --- 4. Model Development ---

def build_cosine_similarity(movie_features: pd.DataFrame) -> pd.DataFrame:
    """Menghitung matriks cosine similarity antar film berdasarkan profil genre."""
    cosine_sim = cosine_similarity(movie_features)
    return pd.DataFrame(
        cosine_sim,
        index=movie_features.index,
        columns=movie_features.index
    )


def build_model(data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pipeline pemrosesan data hingga pembentukan matriks similarity."""
    ratings, movies, df = load_data(data_dir)
    movie_features = build_movie_features(df)
    cosine_sim_df  = build_cosine_similarity(movie_features)
    return ratings, movies, df, movie_features, cosine_sim_df


def recommend_movies(
    title: str,
    cosine_sim_df: pd.DataFrame,
    top_n: int = 5
) -> list[str] | str:
    """Merekomendasikan top-N film yang mirip berdasarkan cosine similarity genre."""
    if title not in cosine_sim_df.columns:
        return f"Film '{title}' tidak ditemukan dalam database."

    sim_scores = (
        cosine_sim_df[title]
        .drop(title)
        .sort_values(ascending=False)
    )
    return sim_scores.head(top_n).index.tolist()


def get_genres(title: str, movie_features: pd.DataFrame) -> str:
    """Mengembalikan daftar genre film yang dipisahkan koma."""
    if title not in movie_features.index:
        return "-"
    row    = movie_features.loc[title]
    genres = [g for g in GENRE_COLS if row[g] == 1]
    return ', '.join(genres) if genres else "Unknown"


# --- 5. Evaluasi ---

def evaluate_recommendation(
    title: str,
    cosine_sim_df: pd.DataFrame,
    movie_features: pd.DataFrame,
    top_n: int = 5
) -> pd.DataFrame | str:
    """Mengevaluasi rekomendasi secara kualitatif berdasarkan perbandingan genre."""
    results = recommend_movies(title, cosine_sim_df, top_n)
    if isinstance(results, str):
        return results

    input_genres = get_genres(title, movie_features)
    rows = [{"Judul Film": title, "Genre": input_genres, "Similarity": 1.0, "Keterangan": "Film Referensi"}]

    for rec in results:
        sim  = round(float(cosine_sim_df.loc[title, rec]), 4)
        rows.append({
            "Judul Film"  : rec,
            "Genre"       : get_genres(rec, movie_features),
            "Similarity"  : sim,
            "Keterangan"  : "Rekomendasi"
        })

    return pd.DataFrame(rows)


def get_random_sample_titles(movie_features: pd.DataFrame, n: int = 5) -> list[str]:
    """Mengembalikan n judul film acak dari dataset."""
    return random.sample(movie_features.index.tolist(), n)


# --- 6. Visualisasi ---

def plot_rating_distribution(ratings: pd.DataFrame) -> plt.Figure:
    """Bar chart distribusi rating (1–5)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='rating', data=ratings, palette='mako', ax=ax)
    ax.set_title('Distribusi Rating', fontsize=14)
    ax.set_xlabel('Rating', fontsize=11)
    ax.set_ylabel('Jumlah', fontsize=11)
    fig.tight_layout()
    return fig


def plot_genre_distribution(movies: pd.DataFrame) -> plt.Figure:
    """Bar chart distribusi jumlah film per genre."""
    genre_counts = get_genre_counts(movies)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis', ax=ax)
    ax.set_title('Distribusi Genre Film', fontsize=14)
    ax.set_xlabel('Genre', fontsize=11)
    ax.set_ylabel('Jumlah Film', fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig


def plot_top_movie_rating_dist(
    title: str,
    df: pd.DataFrame
) -> plt.Figure | None:
    """Bar chart distribusi rating untuk satu film tertentu."""
    subset = df[df['title'] == title]
    if subset.empty:
        return None
    mean_rating = round(subset['rating'].mean(), 2)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='rating', data=subset, palette='mako', ax=ax)
    ax.set_title(f'{title}', fontsize=12)
    ax.set_xlabel(f'Rating  (Rata-rata: {mean_rating})', fontsize=10)
    ax.set_ylabel('Jumlah', fontsize=10)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap korelasi antar kolom genre dan rating."""
    corr = df[GENRE_COLS + ['rating']].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr, annot=True, cmap='coolwarm', fmt=".2f",
        ax=ax, annot_kws={"size": 7}, linewidths=0.3
    )
    ax.set_title('Correlation Matrix — Genre & Rating', fontsize=14)
    fig.tight_layout()
    return fig


def plot_similarity_heatmap(
    title: str,
    recommendations: list[str],
    cosine_sim_df: pd.DataFrame
) -> plt.Figure:
    """Heatmap cosine similarity antara film referensi dan rekomendasinya."""
    films       = [title] + recommendations
    sim_subset  = cosine_sim_df.loc[films, films]
    fig, ax     = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        sim_subset, annot=True, fmt=".2f",
        cmap='YlOrRd', linewidths=0.5,
        ax=ax, annot_kws={"size": 9}
    )
    ax.set_title('Cosine Similarity — Film Referensi & Rekomendasi', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


# --- Main Entry Point ---

if __name__ == "__main__":
    print("=" * 55)
    print("  Sistem Rekomendasi Film — MovieLens 100K")
    print("=" * 55)

    print("\n[1/3] Memuat dataset...")
    ratings, movies, df, movie_features, cosine_sim_df = build_model()
    print("  ✔ Dataset berhasil dimuat.")
    print(f"  • Total rating  : {len(ratings):,}")
    print(f"  • Total film    : {len(movies):,}")
    print(f"  • Total user    : {ratings['user_id'].nunique():,}")
    print(f"  • Film unik     : {len(movie_features):,}")

    print("\n[2/3] Contoh judul film (acak):")
    for i, t in enumerate(get_random_sample_titles(movie_features), 1):
        print(f"  {i}. {t}")

    print("\n[3/3] Rekomendasi Film")
    title  = input("  Masukkan judul film: ")
    result = recommend_movies(title, cosine_sim_df, top_n=5)

    if isinstance(result, str):
        print(f"\n  ⚠ {result}")
    else:
        print(f"\n  Rekomendasi untuk '{title}':")
        for i, rec in enumerate(result, 1):
            print(f"    {i}. {rec}  [{get_genres(rec, movie_features)}]")