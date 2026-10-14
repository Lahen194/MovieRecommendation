# ──────────────────────────────────────────────────────────────────────────────
# app.py  —  Streamlit App: Sistem Rekomendasi Film (MovieLens 100K)
# Author : Fakhira Lahen Siregar | ID: MC009D5X2078
# ──────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from movierecommendation import (
    GENRE_COLS,
    build_model,
    check_duplicates,
    check_missing_values,
    evaluate_recommendation,
    get_data_info,
    get_data_shape,
    get_duplicate_titles,
    get_genre_counts,
    get_genres,
    get_nunique_table,
    get_random_sample_titles,
    get_rating_value_counts,
    get_top_rated_movies,
    plot_correlation_heatmap,
    plot_genre_distribution,
    plot_rating_distribution,
    plot_similarity_heatmap,
    plot_top_movie_rating_dist,
    recommend_movies,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e0e0e0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }

/* ── Hero ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}
.hero-subtitle { font-size: 1rem; color: #94a3b8; margin-bottom: 1.5rem; }

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.055);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1.4rem 1.75rem;
    margin-bottom: 1.2rem;
}

/* ── Section header ── */
.sec-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 4px solid #a78bfa;
    padding-left: 0.7rem;
    margin-bottom: 0.9rem;
}

/* ── Step badge (numbered) ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 700;
    color: white;
    margin-right: 0.5rem;
    flex-shrink: 0;
}

/* ── Insight box ── */
.insight-box {
    background: rgba(96, 165, 250, 0.08);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    color: #bfdbfe;
    font-size: 0.92rem;
    line-height: 1.7;
}
.insight-box strong { color: #93c5fd; }

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value { font-size: 1.9rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem; }

/* ── Recommendation card ── */
.rec-card {
    background: rgba(167,139,250,0.09);
    border: 1px solid rgba(167,139,250,0.22);
    border-radius: 11px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.65rem;
    transition: all 0.2s ease;
}
.rec-card:hover { background: rgba(167,139,250,0.18); transform: translateX(4px); }
.rec-number { font-size: 1.3rem; font-weight: 700; color: #a78bfa; margin-right: 0.65rem; }
.rec-title  { font-size: 0.97rem; font-weight: 600; color: #f1f5f9; }
.rec-genre  { font-size: 0.78rem; color: #94a3b8; margin-top: 0.3rem; }

/* ── Genre badge ── */
.badge {
    display: inline-block;
    background: rgba(96,165,250,0.13);
    border: 1px solid rgba(96,165,250,0.28);
    border-radius: 20px;
    padding: 2px 9px;
    font-size: 0.74rem;
    color: #60a5fa;
    margin: 2px;
}

/* ── Code block override ── */
code { background: rgba(255,255,255,0.08) !important; color: #86efac !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white; border: none; border-radius: 9px;
    font-weight: 600; padding: 0.5rem 1.5rem;
    transition: all 0.2s ease; width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"]  { gap: 6px; }
.stTabs [data-baseweb="tab"]       { border-radius: 8px 8px 0 0; color: #94a3b8; font-weight: 500; }
.stTabs [aria-selected="true"]     { background: rgba(167,139,250,0.15); color: #a78bfa; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

hr { border-color: rgba(255,255,255,0.07); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: transparent matplotlib figure
# ──────────────────────────────────────────────────────────────────────────────
def dark_fig(fig: plt.Figure) -> plt.Figure:
    """Membuat latar belakang figure transparan dan label berwarna putih."""
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.set_facecolor('none')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor((1, 1, 1, 0.15))
    return fig


def badges(genre_str: str) -> str:
    """Mengubah string genre menjadi HTML badge."""
    return "".join(
        f'<span class="badge">{g.strip()}</span>'
        for g in genre_str.split(',') if g.strip()
    )


# ──────────────────────────────────────────────────────────────────────────────
# Load model (cached — hanya dijalankan sekali)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🚀 Memuat dataset & model...")
def get_model():
    ratings, movies, df, movie_features, cosine_sim_df = build_model()
    return ratings, movies, df, movie_features, cosine_sim_df


ratings, movies, df, movie_features, cosine_sim_df = get_model()
all_titles = sorted(movie_features.index.tolist())

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — navigasi
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Movie Recommender")
    st.markdown(
        "Sistem rekomendasi film berbasis **Content-Based Filtering** "
        "menggunakan **Cosine Similarity** pada fitur genre."
    )
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        options=[
            "🏠 Beranda",
            "📦 Data Collecting",
            "🔍 Data Exploring",
            "🧹 Data Preparation",
            "🤖 Modeling & Evaluasi",
            "📖 Dokumentasi",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    **Dataset:** MovieLens 100K  
    **Author:** Fakhira Lahen Siregar  
    **ID:** MC009D5X2078
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.markdown('<div class="hero-title">🎬 Sistem Rekomendasi Film</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Berbasis Genre dengan Content-Based Filtering — MovieLens 100K</div>',
        unsafe_allow_html=True,
    )

    # Metrik ringkasan
    info = get_data_shape(df)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [f"{len(ratings):,}", f"{len(movies):,}", f"{ratings['user_id'].nunique():,}", f"{ratings['rating'].mean():.2f}"],
        ["Total Rating", "Total Film", "Total User", "Rata-rata Rating"],
    ):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">📌 Tentang Proyek</div>
            <p style="color:#cbd5e1;line-height:1.85">
                Pada era digital ini, konsumen memiliki akses ke ribuan film dan serial TV melalui
                berbagai platform streaming. Proyek ini membangun <strong>sistem rekomendasi film</strong>
                yang memberikan saran film kepada pengguna berdasarkan kemiripan <strong>genre</strong>
                menggunakan pendekatan <em>Content-Based Filtering</em>.
            </p>
            <p style="color:#cbd5e1;line-height:1.85;margin-top:0.75rem">
                Dataset yang digunakan adalah <strong>MovieLens 100K</strong> dari GroupLens Research,
                berisi 100.000 rating dari 943 pengguna terhadap 1.682 film.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">🗺️ Alur Proyek</div>
            <table style="width:100%;border-collapse:collapse;color:#cbd5e1">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.07)">
                    <td style="padding:8px 0"><span class="step-badge">1</span> <strong>Data Collecting</strong></td>
                    <td style="padding:8px;color:#94a3b8">Download & load dataset dari GroupLens</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.07)">
                    <td style="padding:8px 0"><span class="step-badge">2</span> <strong>Data Exploring</strong></td>
                    <td style="padding:8px;color:#94a3b8">Memahami struktur, distribusi, dan pola data</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.07)">
                    <td style="padding:8px 0"><span class="step-badge">3</span> <strong>Data Preparation</strong></td>
                    <td style="padding:8px;color:#94a3b8">Membersihkan data & membangun item profile</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.07)">
                    <td style="padding:8px 0"><span class="step-badge">4</span> <strong>Modeling</strong></td>
                    <td style="padding:8px;color:#94a3b8">Hitung cosine similarity antar film</td>
                </tr>
                <tr>
                    <td style="padding:8px 0"><span class="step-badge">5</span> <strong>Evaluasi</strong></td>
                    <td style="padding:8px;color:#94a3b8">Verifikasi rekomendasi secara kualitatif</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">📂 File Dataset</div>
            <ul style="color:#cbd5e1;line-height:2.1;padding-left:1.1rem;font-size:0.9rem">
                <li><code>u.data</code> — 100.000 rating</li>
                <li><code>u.item</code> — 1.682 film</li>
                <li><code>u.user</code> — 943 user</li>
                <li><code>u.genre</code> — 19 genre</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">📐 Model</div>
            <p style="color:#cbd5e1;font-size:0.9rem;line-height:1.8">
                <strong>Content-Based Filtering</strong><br>
                Menggunakan <strong>Cosine Similarity</strong> pada
                vektor biner genre (18 dimensi) untuk mengukur
                kemiripan antar film.
            </p>
            <blockquote style="border-left:3px solid #a78bfa;padding-left:0.8rem;color:#94a3b8;font-style:italic;font-size:0.85rem">
                cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
            </blockquote>
        </div>
        """, unsafe_allow_html=True)

        # Top genre pie-style count
        genre_counts = get_genre_counts(movies)
        st.markdown('<div class="glass-card"><div class="sec-header">🏆 Top 5 Genre</div>', unsafe_allow_html=True)
        for g, cnt in genre_counts.head(5).items():
            pct = int(cnt / genre_counts.sum() * 100)
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;color:#cbd5e1;font-size:0.88rem;margin-bottom:6px">
                <span>{g}</span><span style="color:#a78bfa;font-weight:600">{int(cnt)} film</span>
            </div>
            <div style="background:rgba(255,255,255,0.08);border-radius:99px;height:6px;margin-bottom:10px">
                <div style="background:linear-gradient(90deg,#7c3aed,#60a5fa);width:{pct}%;height:6px;border-radius:99px"></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA COLLECTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Data Collecting":
    st.markdown('<div class="hero-title">📦 Data Collecting</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Pengumpulan dan pemuatan dataset MovieLens 100K.</div>', unsafe_allow_html=True)

    # ── Sumber Dataset ──
    st.markdown("""
    <div class="glass-card">
        <div class="sec-header">🌐 Sumber Dataset</div>
        <p style="color:#cbd5e1;line-height:1.8">
            Dataset diambil dari <strong>GroupLens Research</strong>:
            <a href="https://grouplens.org/datasets/movielens/" style="color:#60a5fa">
            https://grouplens.org/datasets/movielens/</a>
        </p>
        <p style="color:#94a3b8;font-size:0.9rem;margin-top:0.5rem">
            MovieLens 100K adalah kumpulan data film berukuran sedang yang banyak digunakan untuk penelitian
            sistem rekomendasi. Dataset ini terdiri dari 100.000 rating yang diberikan oleh pengguna
            terhadap film, dan sangat cocok untuk eksperimen dalam collaborative filtering maupun
            content-based filtering.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Kode download ──
    st.markdown('<div class="glass-card"><div class="sec-header">⬇️ Download & Load Dataset</div>', unsafe_allow_html=True)
    st.code("""# Download dataset dari GroupLens
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip""", language="bash")

    st.code("""# Membaca file ratings (u.data)
ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\\t",
    names=["user_id", "item_id", "rating", "timestamp"]
)
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Membaca file movies (u.item)
movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|", encoding="latin-1",
    names=MOVIE_COLUMNS, usecols=range(24)
)

# Gabungkan ratings + movies
df = pd.merge(ratings, movies, on="item_id")""", language="python")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Deskripsi file ──
    st.markdown('<div class="glass-card"><div class="sec-header">📄 Deskripsi File Dataset</div>', unsafe_allow_html=True)
    file_data = {
        "File"   : ["u.data", "u.item", "u.user", "u.genre"],
        "Deskripsi": [
            "Rating data — user_id, item_id, rating, timestamp",
            "Movie metadata — movie_id, title, tanggal, URL, genre (biner)",
            "User metadata — user_id, age, gender, occupation, zip code",
            "Daftar genre dan ID-nya",
        ],
        "Contoh Baris": [
            "196 | 242 | 3 | 881250949",
            "1 | Toy Story (1995) | 01-Jan-1995 | ... | 0|0|0|1|1|1|...",
            "1 | 24 | M | technician | 85711",
            "unknown|0",
        ],
    }
    st.dataframe(pd.DataFrame(file_data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Preview dataframe ──
    tab_r, tab_m, tab_df = st.tabs(["DataFrame: ratings", "DataFrame: movies", "DataFrame: df (gabungan)"])

    with tab_r:
        st.markdown(f"**{len(ratings):,} baris × {ratings.shape[1]} kolom**")
        st.dataframe(ratings.head(10), use_container_width=True, hide_index=True)

    with tab_m:
        st.markdown(f"**{len(movies):,} baris × {movies.shape[1]} kolom**")
        st.dataframe(movies.head(10), use_container_width=True, hide_index=True)

    with tab_df:
        st.markdown(f"**{len(df):,} baris × {df.shape[1]} kolom**")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box">
        <strong>💡 Data Insight:</strong><br>
        • Dataset berhasil dibaca menggunakan <code>pd.read_csv</code> dan disimpan ke variabel
          <code>ratings</code> (file u.data) dan <code>movies</code> (file u.item).<br>
        • Penggabungan dua dataframe disesuaikan dengan <code>item_id</code> sehingga setiap rating
          bisa dikaitkan langsung dengan judul film yang diberi rating tersebut.<br>
        • Hasil penggabungan menghasilkan <strong>100.000 baris dan 23 kolom</strong>.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Data Exploring":
    st.markdown('<div class="hero-title">🔍 Data Exploring</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Menilai struktur data, distribusi, dan pola menarik dalam dataset.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Assessing Data",
        "🔗 item_id vs title",
        "⭐ Distribusi Rating",
        "🎭 Distribusi Genre",
        "🌡️ Korelasi",
    ])

    # ── Tab 1: Assessing ──
    with tab1:
        st.markdown('<div class="sec-header">📊 Assessing Data</div>', unsafe_allow_html=True)

        info = get_data_shape(df)
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip(
            [c1, c2, c3, c4],
            [info["baris"], info["kolom"], info["kolom_numerik"], info["kolom_kategorikal"]],
            ["Baris", "Kolom", "Kolom Numerik", "Kolom Kategorikal"],
        ):
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val:,}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="sec-header">ℹ️ df.info()</div>', unsafe_allow_html=True)
            st.code(get_data_info(df), language="text")

        with col_r:
            st.markdown('<div class="sec-header">📈 df.describe()</div>', unsafe_allow_html=True)
            st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">🔢 Nilai Unik per Kolom (df.nunique())</div>', unsafe_allow_html=True)
        st.dataframe(get_nunique_table(df), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Data Insight:</strong><br>
            • Dataset berisi <strong>100.000 baris dan 23 kolom</strong>.<br>
            • Terdapat <strong>21 kolom numerik</strong> (user_id, item_id, rating, genre),
              <strong>1 kolom kategorikal</strong> (title), dan <strong>1 kolom datetime</strong> (timestamp).<br>
            • Rata-rata nilai kolom rating adalah <strong>3.5</strong> — menunjukkan kecenderungan pengguna
              memberikan rating di atas tengah.<br>
            • Diidentifikasi perbedaan nilai unik antara <code>item_id</code> (1682) dan <code>title</code> (1664)
              — mengindikasikan adanya duplikasi judul.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: item_id vs title ──
    with tab2:
        st.markdown('<div class="sec-header">🔗 Hubungan item_id dan title</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8">Mengidentifikasi duplikasi: judul sama dengan item_id berbeda.</p>', unsafe_allow_html=True)

        st.code("""dupe_titles = movies[movies.duplicated(subset=['title'], keep=False)].sort_values('title')
print(dupe_titles[['item_id', 'title']])""", language="python")

        dupes = get_duplicate_titles(movies)
        if dupes.empty:
            st.success("Tidak ditemukan duplikasi judul film.")
        else:
            st.warning(f"Ditemukan **{len(dupes)} baris** dengan judul film yang sama (namun item_id berbeda).")
            st.dataframe(dupes, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Data Insight:</strong><br>
            • Setelah dicek dengan <code>movies.duplicated(subset=['title'])</code>, ditemukan beberapa
              duplikasi judul film dengan <code>item_id</code> berbeda.<br>
            • Duplikasi ini <strong>tidak dihapus</strong> dari data utama karena tidak mengganggu proses
              content-based filtering — hanya saja ada kemungkinan satu film muncul dua kali dalam hasil rekomendasi.<br>
            • Pada tahap pembangunan item profile, duplikasi judul <strong>dihapus</strong> menggunakan
              <code>drop_duplicates(subset=['title'])</code>.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Distribusi Rating ──
    with tab3:
        st.markdown('<div class="sec-header">⭐ Distribusi Rating</div>', unsafe_allow_html=True)

        col_l, col_r = st.columns([1.5, 1])
        with col_l:
            fig = plot_rating_distribution(ratings)
            st.pyplot(dark_fig(fig), use_container_width=True)
        with col_r:
            st.markdown("<br>", unsafe_allow_html=True)
            vc = get_rating_value_counts(ratings)
            vc['Persentase (%)'] = (vc['Jumlah'] / vc['Jumlah'].sum() * 100).round(1)
            st.dataframe(vc, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Data Insight:</strong><br>
            • Rating <strong>4</strong> adalah rating yang paling banyak diberikan pengguna untuk keseluruhan film.<br>
            • Distribusi rating condong ke kanan (positively skewed) — pengguna cenderung memberikan rating
              yang baik, dengan median dan modus di angka 4.<br>
            • Rating 1 adalah rating yang paling jarang diberikan.
        </div>
        """, unsafe_allow_html=True)

        # Top 5 film yang paling banyak dirating
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">🏆 Distribusi Rating per Film (Top 5)</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:0.9rem">Tidak semua film dengan rating terbanyak adalah film dengan rata-rata rating tertinggi.</p>', unsafe_allow_html=True)

        top5 = get_top_rated_movies(ratings, movies, top_n=5)
        top5_titles = top5['Judul Film'].tolist()

        cols_top = st.columns(5)
        for col, title_top in zip(cols_top, top5_titles):
            with col:
                fig_t = plot_top_movie_rating_dist(title_top, df)
                if fig_t:
                    st.pyplot(dark_fig(fig_t), use_container_width=True)

    # ── Tab 4: Distribusi Genre ──
    with tab4:
        st.markdown('<div class="sec-header">🎭 Distribusi Genre Film</div>', unsafe_allow_html=True)
        fig_genre = plot_genre_distribution(movies)
        st.pyplot(dark_fig(fig_genre), use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            genre_counts = get_genre_counts(movies)
            gdf = pd.DataFrame({"Genre": genre_counts.index, "Jumlah Film": genre_counts.values.astype(int)})
            st.dataframe(gdf, use_container_width=True, hide_index=True)
        with col_r:
            st.markdown("""
            <div class="insight-box">
                <strong>💡 Data Insight:</strong><br>
                • Genre <strong>Drama</strong> memiliki jumlah film terbanyak (>700 film),
                  diikuti oleh <strong>Comedy</strong> dan <strong>Action</strong>.<br>
                • Genre <strong>Fantasy</strong>, <strong>Western</strong>, dan <strong>Film-Noir</strong>
                  memiliki jumlah film yang relatif sedikit (di bawah 50 film).<br>
                • Satu film bisa memiliki lebih dari satu genre (multi-label), sehingga total jumlah
                  genre > jumlah film.
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 5: Korelasi ──
    with tab5:
        st.markdown('<div class="sec-header">🌡️ Heatmap Korelasi antar Genre dan Rating</div>', unsafe_allow_html=True)
        st.code("correlation_matrix = df.corr(numeric_only=True)", language="python")

        fig_corr = plot_correlation_heatmap(df)
        st.pyplot(dark_fig(fig_corr), use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Data Insight:</strong><br>
            • Korelasi antar genre mencerminkan pola yang logis. Misalnya:<br>
              &nbsp;&nbsp;— <strong>Action</strong> dan <strong>Adventure</strong> sering muncul bersamaan.<br>
              &nbsp;&nbsp;— <strong>Children's</strong> berkorelasi positif dengan <strong>Animation</strong>.<br>
              &nbsp;&nbsp;— <strong>Animation</strong> juga berkorelasi dengan <strong>Musical</strong>.<br>
            • Kolom <strong>rating</strong> memiliki korelasi yang sangat rendah terhadap semua genre —
              artinya genre bukan penentu utama tinggi-rendahnya rating yang diberikan pengguna.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧹 Data Preparation":
    st.markdown('<div class="hero-title">🧹 Data Preparation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Membersihkan data dan membangun item profile untuk modeling.</div>', unsafe_allow_html=True)

    step_tabs = st.tabs([
        "1️⃣ Missing Value & Duplikasi",
        "2️⃣ Fitur Genre",
        "3️⃣ Item Profile",
    ])

    # ── Step 1: Missing & Duplikasi ──
    with step_tabs[0]:
        st.markdown('<div class="sec-header">🔎 Pemeriksaan Missing Value</div>', unsafe_allow_html=True)
        st.code("""isnull = df.isnull().sum()
print(f"Banyak missing value :\\n{isnull}")""", language="python")

        missing_df = check_missing_values(df)
        all_zero   = (missing_df["Missing Value"] == 0).all()

        if all_zero:
            st.success("✅ Tidak ada missing value pada seluruh kolom!")
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">🔎 Pemeriksaan Duplikasi Baris</div>', unsafe_allow_html=True)
        st.code("duplikat = df.duplicated().sum()\nprint(f'Banyak duplikat : {duplikat}')", language="python")

        dup_count = check_duplicates(df)
        if dup_count == 0:
            st.success(f"✅ Tidak ada baris duplikat. ({dup_count} duplikat ditemukan)")
        else:
            st.warning(f"⚠️ Ditemukan {dup_count} baris duplikat.")

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Data Insight:</strong><br>
            • Dataset <strong>tidak memiliki missing value</strong> pada kolom manapun.<br>
            • Dataset <strong>tidak memiliki baris duplikat</strong> pada dataframe gabungan <code>df</code>.<br>
            • Data siap digunakan untuk proses modeling tanpa perlu imputasi atau penghapusan.
        </div>
        """, unsafe_allow_html=True)

    # ── Step 2: Fitur Genre ──
    with step_tabs[1]:
        st.markdown('<div class="sec-header">🏷️ Mendefinisikan Kolom Genre</div>', unsafe_allow_html=True)
        st.code("""genre_cols = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]""", language="python")

        st.markdown(f"Total **{len(GENRE_COLS)} genre** yang digunakan sebagai fitur:")
        badge_html = "".join([f'<span class="badge">{g}</span>' for g in GENRE_COLS])
        st.markdown(badge_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">📋 Contoh Data Genre (One-Hot Encoding)</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:0.9rem">Dataset sudah dalam format biner — tidak perlu one-hot encoding tambahan.</p>', unsafe_allow_html=True)
        sample_genre = df[['title'] + GENRE_COLS].drop_duplicates(subset=['title']).head(5)
        st.dataframe(sample_genre, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Data Insight:</strong><br>
            • Data genre sudah dalam format biner (0/1) dari dataset aslinya.<br>
            • Tidak diperlukan proses <strong>TF-IDF</strong> karena tidak ada fitur teks.<br>
            • Tidak diperlukan proses <strong>one-hot encoding</strong> karena data sudah dalam format yang tepat.
        </div>
        """, unsafe_allow_html=True)

    # ── Step 3: Item Profile ──
    with step_tabs[2]:
        st.markdown('<div class="sec-header">🎛️ Membangun Item Profile</div>', unsafe_allow_html=True)
        st.code("""# Buat item profile: setiap film unik sebagai baris, genre sebagai kolom
movie_features = (
    df[['title'] + genre_cols]
    .drop_duplicates(subset=['title'])   # hapus duplikasi judul
    .set_index('title')                  # jadikan title sebagai index
)""", language="python")

        st.markdown(f"**Dimensi item profile:** {movie_features.shape[0]} film × {movie_features.shape[1]} genre")
        st.dataframe(movie_features.head(10), use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Data Insight:</strong><br>
            • Item profile berhasil dibuat dengan setiap baris mewakili satu film unik.<br>
            • Judul film dijadikan <strong>index</strong> untuk memudahkan akses saat inferensi.<br>
            • Duplikasi judul berhasil dihapus: dari total rating data ke <strong>film unik</strong>.<br>
            • Data siap digunakan untuk menghitung cosine similarity pada tahap modeling.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODELING & EVALUASI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Modeling & Evaluasi":
    st.markdown('<div class="hero-title">🤖 Modeling & Evaluasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Hitung cosine similarity dan gunakan model untuk mendapatkan rekomendasi film.</div>', unsafe_allow_html=True)

    tab_model, tab_rec, tab_eval = st.tabs([
        "📐 Model Development",
        "✨ Rekomendasi Film",
        "📋 Evaluasi",
    ])

    # ── Model Development ──
    with tab_model:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("""
            <div class="glass-card">
                <div class="sec-header">📐 Cosine Similarity</div>
                <p style="color:#cbd5e1;line-height:1.8;font-size:0.9rem">
                    Untuk mengukur kemiripan antar film berdasarkan vektor genre biner,
                    digunakan <strong>Cosine Similarity</strong>:
                </p>
                <blockquote style="border-left:3px solid #a78bfa;padding-left:0.8rem;color:#94a3b8;font-style:italic">
                    cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
                </blockquote>
                <ul style="color:#94a3b8;font-size:0.88rem;line-height:1.9;margin-top:0.5rem">
                    <li>Nilai <strong>0</strong> = tidak ada genre yang sama</li>
                    <li>Nilai <strong>1</strong> = genre identik sempurna</li>
                    <li>Cocok untuk data biner (0/1)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            st.markdown("""
            <div class="glass-card">
                <div class="sec-header">✅ Kelebihan & ❌ Kekurangan</div>
                <table style="color:#cbd5e1;width:100%;border-collapse:collapse;font-size:0.88rem">
                    <tr><td style="padding:5px;color:#34d399">✅ Tidak perlu riwayat rating user</td></tr>
                    <tr><td style="padding:5px;color:#34d399">✅ Cocok untuk cold-start problem</td></tr>
                    <tr><td style="padding:5px;color:#34d399">✅ Transparan & mudah dijelaskan</td></tr>
                    <tr><td style="padding:5px;color:#f87171">❌ Tidak personalisasi per pengguna</td></tr>
                    <tr><td style="padding:5px;color:#f87171">❌ Hanya berdasarkan genre (terbatas)</td></tr>
                    <tr><td style="padding:5px;color:#f87171">❌ Tidak menangkap kualitas film</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header">💻 Kode: Membangun Model</div>', unsafe_allow_html=True)
        st.code("""from sklearn.metrics.pairwise import cosine_similarity

# Hitung cosine similarity antar semua pasangan film
cosine_sim = cosine_similarity(movie_features)

# Ubah ke DataFrame agar lebih mudah diakses
cosine_sim_df = pd.DataFrame(
    cosine_sim,
    index=movie_features.index,    # index = judul film
    columns=movie_features.index   # kolom = judul film
)""", language="python")

        st.markdown('<div class="sec-header">💻 Kode: Fungsi Rekomendasi</div>', unsafe_allow_html=True)
        st.code("""def recommend_movies(title, cosine_sim_df, top_n=5):
    if title not in cosine_sim_df.columns:
        return f"Film '{title}' tidak ditemukan dalam database."

    sim_scores = (
        cosine_sim_df[title]
        .drop(title)               # hilangkan film itu sendiri
        .sort_values(ascending=False)
    )
    return sim_scores.head(top_n).index.tolist()""", language="python")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">👁️ Preview Similarity Matrix (5×5)</div>', unsafe_allow_html=True)
        st.dataframe(cosine_sim_df.iloc[:5, :5].round(4), use_container_width=True)

    # ── Rekomendasi ──
    with tab_rec:
        st.markdown('<div class="glass-card"><div class="sec-header">🎯 Cari Rekomendasi Film</div>', unsafe_allow_html=True)

        col_inp, col_n = st.columns([3, 1])
        with col_inp:
            selected = st.selectbox(
                "Pilih atau cari judul film:",
                options=all_titles,
                index=all_titles.index("Toy Story (1995)") if "Toy Story (1995)" in all_titles else 0,
            )
        with col_n:
            top_n = st.number_input("Top N", min_value=3, max_value=20, value=5)

        btn_rec = st.button("✨ Rekomendasikan", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Genre film yang dipilih
        if selected:
            input_genre = get_genres(selected, movie_features)
            st.markdown(f"""
            <div class="glass-card">
                <div class="sec-header">🏷️ Genre: {selected}</div>
                {badges(input_genre)}
            </div>
            """, unsafe_allow_html=True)

        if btn_rec and selected:
            results = recommend_movies(selected, cosine_sim_df, top_n=int(top_n))

            if isinstance(results, str):
                st.error(results)
            else:
                st.markdown(f'<div class="sec-header">🌟 Top {top_n} Rekomendasi untuk "{selected}"</div>', unsafe_allow_html=True)

                for i, rec in enumerate(results, 1):
                    rec_genre = get_genres(rec, movie_features)
                    sim_score = cosine_sim_df.loc[selected, rec]
                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="display:flex;align-items:center;justify-content:space-between">
                            <div>
                                <span class="rec-number">#{i}</span>
                                <span class="rec-title">{rec}</span>
                                <div class="rec-genre" style="margin-top:0.35rem">{badges(rec_genre)}</div>
                            </div>
                            <div style="text-align:right;flex-shrink:0;margin-left:1rem">
                                <div style="font-size:1rem;font-weight:700;color:#34d399">{sim_score*100:.1f}%</div>
                                <div style="font-size:0.7rem;color:#64748b">similarity</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Heatmap similarity
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="sec-header">🔥 Heatmap Cosine Similarity</div>', unsafe_allow_html=True)
                fig_sim = plot_similarity_heatmap(selected, results, cosine_sim_df)
                st.pyplot(dark_fig(fig_sim), use_container_width=True)

    # ── Evaluasi ──
    with tab_eval:
        st.markdown('<div class="sec-header">📋 Evaluasi Kualitatif</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <p style="color:#cbd5e1;line-height:1.85;font-size:0.9rem">
                Model ini menggunakan pendekatan <strong>Content-Based Filtering</strong>, sehingga
                evaluasi dilakukan secara <strong>kualitatif</strong> — yaitu dengan memverifikasi
                bahwa genre dari film yang direkomendasikan sesuai dengan genre film referensi.
            </p>
            <p style="color:#94a3b8;font-size:0.88rem;margin-top:0.75rem">
                Evaluasi kuantitatif seperti <em>Precision@K</em>, <em>Recall@K</em>, atau
                <em>MRR</em> memerlukan <strong>ground truth</strong> (data film mana yang
                benar-benar relevan per pengguna), yang tidak tersedia pada pendekatan item-item
                berbasis konten ini.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header">🎲 Evaluasi dengan Film Acak</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:0.9rem">Pilih film acak kemudian verifikasi hasilnya:</p>', unsafe_allow_html=True)

        if st.button("🎲 Ambil Film Acak", use_container_width=False):
            st.session_state['eval_title'] = get_random_sample_titles(movie_features, n=1)[0]

        eval_title = st.session_state.get('eval_title', "Toy Story (1995)")
        eval_selected = st.selectbox(
            "Film untuk dievaluasi:",
            options=all_titles,
            index=all_titles.index(eval_title) if eval_title in all_titles else 0,
            key="eval_select",
        )

        if eval_selected:
            eval_df = evaluate_recommendation(eval_selected, cosine_sim_df, movie_features, top_n=5)
            if isinstance(eval_df, str):
                st.error(eval_df)
            else:
                st.dataframe(eval_df, use_container_width=True, hide_index=True)
                st.markdown("""
                <div class="insight-box">
                    <strong>💡 Cara Baca Tabel:</strong><br>
                    • Baris pertama (Similarity = 1.0) adalah <strong>film referensi</strong>.<br>
                    • Baris selanjutnya adalah <strong>film yang direkomendasikan</strong> beserta genre dan skor similarity-nya.<br>
                    • Verifikasi: pastikan genre film rekomendasi <strong>sesuai atau mirip</strong> dengan genre referensi.
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DOKUMENTASI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 Dokumentasi":
    st.markdown('<div class="hero-title">📖 Dokumentasi Proyek</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Laporan lengkap proyek machine learning terapan.</div>', unsafe_allow_html=True)

    doc_tabs = st.tabs([
        "📌 Project Overview",
        "🏢 Business Understanding",
        "📊 Data Understanding",
        "🧹 Data Preparation",
        "🤖 Modeling",
        "📋 Evaluasi",
        "✅ Kesimpulan",
    ])

    with doc_tabs[0]:
        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">📌 Project Overview</div>
            <p style="color:#cbd5e1;line-height:1.85">
                Pada era digital ini, konsumen memiliki akses ke ribuan film dan serial TV melalui berbagai
                platform streaming. Tantangan yang muncul adalah bagaimana membantu pengguna menemukan konten
                yang sesuai dengan preferensi mereka di antara banyaknya pilihan yang tersedia. Oleh karena
                itu, dibutuhkan sebuah <strong>sistem rekomendasi film</strong> yang dapat memberikan saran
                kepada pengguna berdasarkan karakteristik film (Content-Based Filtering).
            </p>
            <p style="color:#cbd5e1;line-height:1.85;margin-top:0.75rem">
                <strong>MovieLens 100K</strong> adalah dataset sistem rekomendasi yang paling banyak digunakan
                untuk penelitian dan pembelajaran. Dataset ini berisi data 100.000 rating dari 943 pengguna
                terhadap 1.682 film. Dengan membangun sistem rekomendasi menggunakan pendekatan content-based
                filtering, proyek ini bertujuan untuk menyarankan film yang serupa dengan film yang pernah
                disukai pengguna berdasarkan informasi genre.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">📚 Referensi</div>
            <ol style="color:#cbd5e1;line-height:2;font-size:0.9rem">
                <li>
                    R. Vyshnavi, <em>"Genre Based Movie Recommendation System Using Collaborative Filtering,"</em>
                    International Journal of Research Publication and Reviews, vol. 4, no. 12, pp. 4029–4032, Dec. 2023.
                    <br><span style="color:#94a3b8">Penelitian ini menggunakan data MovieLens dengan fitur genre dan K-Means clustering
                    untuk meningkatkan kualitas rekomendasi.</span>
                </li>
                <li style="margin-top:0.5rem">
                    S.-M. Choi et al., <em>"A movie recommendation algorithm based on genre correlations,"</em>
                    Expert Systems with Applications, vol. 39, no. 9, pp. 8079–8085, 2012.
                    <br><span style="color:#94a3b8">Menggunakan korelasi antar genre untuk mengatasi masalah cold-start dan sparsity
                    pada collaborative filtering.</span>
                </li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with doc_tabs[1]:
        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">❓ Problem Statements</div>
            <ul style="color:#cbd5e1;line-height:2">
                <li>Bagaimana membangun sistem rekomendasi film yang dapat menyarankan film berdasarkan genre?</li>
                <li>Bagaimana memanfaatkan data genre untuk menentukan kesamaan antar film?</li>
            </ul>
        </div>
        <div class="glass-card">
            <div class="sec-header">🎯 Goals</div>
            <ul style="color:#cbd5e1;line-height:2">
                <li>Mengembangkan sistem rekomendasi berdasarkan genre film.</li>
                <li>Memberikan rekomendasi film serupa berdasarkan input film yang disukai pengguna.</li>
            </ul>
        </div>
        <div class="glass-card">
            <div class="sec-header">💡 Solution Statements</div>
            <ul style="color:#cbd5e1;line-height:2">
                <li><strong>Content-Based Filtering</strong>: merekomendasikan film berdasarkan kemiripan fitur konten (genre) menggunakan cosine similarity.</li>
                <li><strong>Alternatif (tidak digunakan)</strong>: Collaborative Filtering — memanfaatkan kesamaan pola rating antar pengguna, namun gagal pada cold-start.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with doc_tabs[2]:
        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">📊 Deskripsi Dataset</div>
            <p style="color:#cbd5e1;line-height:1.8">
                Dataset: <a href="https://grouplens.org/datasets/movielens/100k/" style="color:#60a5fa">MovieLens 100K — GroupLens Research</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

        desc_data = {
            "File"        : ["u.data", "u.item", "u.user", "u.genre"],
            "Konten"      : ["100.000 rating", "1.682 film + metadata", "943 user + profil", "Daftar 19 genre"],
            "Kolom Utama" : [
                "user_id, item_id, rating, timestamp",
                "movie_id, title, release_date, genre (biner)",
                "user_id, age, gender, occupation, zip_code",
                "nama genre, id genre",
            ],
        }
        st.dataframe(pd.DataFrame(desc_data), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="insight-box">
            <strong>Insight Dataset:</strong><br>
            • Berisi 100.000 baris, 23 kolom setelah penggabungan.<br>
            • 21 kolom numerik, 1 kolom kategorikal (title), 1 kolom datetime (timestamp).<br>
            • Rata-rata rating: 3.5 | Rating terpopuler: 4.<br>
            • Film terbanyak dirating: Star Wars (1977), Contact (1997), Fargo (1996).<br>
            • Genre terbanyak: Drama, Comedy, Action. Genre tersedikit: Western, Film-Noir, Fantasy.
        </div>
        """, unsafe_allow_html=True)

    with doc_tabs[3]:
        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">🧹 Langkah-langkah Data Preparation</div>
            <ol style="color:#cbd5e1;line-height:2.2;font-size:0.9rem">
                <li><strong>Membaca file data</strong> — u.data dan u.item ke dalam dataframe ratings dan movies.</li>
                <li><strong>Penghapusan kolom</strong> — release_date, video_release_date, IMDb_URL, dan unknown dihapus dari movies.</li>
                <li><strong>Penyesuaian nama kolom</strong> — movie_id diubah menjadi item_id agar konsisten.</li>
                <li><strong>Penggabungan dataframe</strong> — ratings dan movies digabung berdasarkan item_id.</li>
                <li><strong>Konversi timestamp</strong> — dikonversi ke format datetime menggunakan pd.to_datetime().</li>
                <li><strong>Pemeriksaan missing value</strong> — tidak ada missing value ditemukan.</li>
                <li><strong>Pemeriksaan duplikasi</strong> — tidak ada baris duplikat.</li>
                <li><strong>Mendefinisikan genre_cols</strong> — 18 genre sebagai fitur.</li>
                <li><strong>Membangun item profile</strong> — movie_features dengan title sebagai index, drop_duplicates pada title.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with doc_tabs[4]:
        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">🤖 Pendekatan: Content-Based Filtering</div>
            <p style="color:#cbd5e1;line-height:1.85;font-size:0.9rem">
                Model dibangun dengan pendekatan Content-Based Filtering — merekomendasikan film berdasarkan
                kemiripan konten (genre) antar film. Kelebihannya adalah bisa digunakan bahkan jika pengguna
                belum memberikan banyak rating (cold-start). Kekurangannya adalah model tidak mempelajari
                preferensi unik tiap pengguna.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-header">📝 Contoh Output Rekomendasi</div>', unsafe_allow_html=True)
        st.code("""Rekomendasi film mirip 'Toy Story (1995)':
1. Aladdin and the King of Thieves (1996)    [Animation, Children's, Comedy]
2. Goofy Movie, A (1995)                     [Animation, Children's, Comedy]
3. Aladdin (1992)                            [Animation, Adventure, Children's, Comedy, Musical, Romance]
4. Heavyweights (1994)                       [Children's, Comedy]
5. Beavis and Butt-head Do America (1996)    [Animation, Comedy]""", language="text")

    with doc_tabs[5]:
        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">📋 Evaluasi Kualitatif</div>
            <p style="color:#cbd5e1;line-height:1.85;font-size:0.9rem">
                Evaluasi dilakukan secara kualitatif dengan langkah berikut:
            </p>
            <ol style="color:#cbd5e1;line-height:2;font-size:0.9rem">
                <li>Cetak 5 film acak dari dataset sebagai opsi.</li>
                <li>Input film referensi dari daftar tersebut.</li>
                <li>Tampilkan 5 film yang direkomendasikan.</li>
                <li>Buat dataframe berisi title dan genre_list (gabungan genre sebagai string).</li>
                <li>Verifikasi genre: bandingkan genre film input dengan genre film rekomendasi.</li>
            </ol>
        </div>
        <div class="glass-card">
            <div class="sec-header">❓ Mengapa Tidak Ada Evaluasi Kuantitatif?</div>
            <p style="color:#cbd5e1;line-height:1.85;font-size:0.9rem">
                Evaluasi kuantitatif seperti <strong>Precision@K</strong>, <strong>Recall@K</strong>,
                atau <strong>Mean Reciprocal Rank (MRR)</strong> memerlukan <em>ground truth</em> —
                yaitu data yang menunjukkan film mana yang benar-benar relevan untuk setiap pengguna.
                Pada sistem Content-Based Filtering dengan input berupa satu judul film, tidak tersedia
                ground truth eksplisit, sehingga metrik tersebut tidak dapat dihitung secara langsung.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with doc_tabs[6]:
        st.markdown("""
        <div class="glass-card">
            <div class="sec-header">✅ Kesimpulan</div>
            <p style="color:#cbd5e1;line-height:1.85">
                Sistem rekomendasi berhasil dibangun dan diuji. Rekomendasi diberikan berdasarkan
                kemiripan genre menggunakan <strong>Cosine Similarity</strong>, dan hasil evaluasi
                kualitatif menunjukkan sistem mampu memberikan saran film yang relevan — film yang
                direkomendasikan memiliki genre yang sama atau mirip dengan film referensi.
                Pendekatan ini cocok untuk kasus <strong>cold-start</strong> karena tidak bergantung
                pada interaksi pengguna.
            </p>
        </div>
        <div class="glass-card">
            <div class="sec-header">📚 Referensi</div>
            <ul style="color:#cbd5e1;line-height:2;font-size:0.9rem">
                <li>MovieLens Dataset: <a href="https://grouplens.org/datasets/movielens/" style="color:#60a5fa">grouplens.org/datasets/movielens</a></li>
                <li>Scikit-learn Documentation: <a href="https://scikit-learn.org" style="color:#60a5fa">scikit-learn.org</a></li>
                <li>Vyshnavi (2023), Choi et al. (2012)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
