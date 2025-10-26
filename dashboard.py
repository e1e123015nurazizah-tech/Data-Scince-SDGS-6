# ================================================
# 🌊 Streamlit Dashboard SDG 6: Clean Water & Sanitation
# (Perbaikan: relative paths, safe loading, info capture)
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import io
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import altair as alt


st.set_page_config(
    page_title="Dashboard SDG 6",
    layout="wide",
    page_icon="💧"
)

# ==== GLOBAL CSS UNTUK SEMUA HALAMAN ====
page_bg = """
<style>

[data-testid="stAppViewContainer"] {
    background-color: #E8F8FF; /* Warna lembut tema air */
}

[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #bde9f7, #88dce8); /* Sidebar tetap elegan */
}

/* Optional: memperhalus container dan kartu */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


# =====================================================
# 🧭 SIDEBAR NAVIGASI (Versi Estetik & Bertema Air)
# =====================================================
st.markdown(
    """
    <style>
    /* ===== Sidebar Styling ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #caf0f8 0%, #ade8f4 50%, #90e0ef 100%);
        color: #023e8a;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p {
        color: #023e8a;
    }

    /* Radio button styling */
    div[role="radiogroup"] > label {
        background-color: #ffffff55;
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 5px;
        color: #023e8a !important;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        border: 1px solid transparent;
    }

    div[role="radiogroup"] > label:hover {
        background-color: #48cae4;
        color: white !important;
        border: 1px solid #0077b6;
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 24px !important;
        font-weight: 700;
        color: #03045e;
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<h1 class='sidebar-title'>🌍 Navigasi Dashboard</h1>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard", "💧 Data Wrangling", "📊 EDA", "📈 Visualisasi Lanjutan", "🤖 Pemodelan"]
)


# =====================================================
# 📂 PATH DATASET (ditempatkan di luar supaya global)
# =====================================================
path_indexes = os.path.join("dataset", "indexes.csv")
path_drinking_water = os.path.join("dataset", "share-of-the-population-using-safely-managed-drinking-water-sources.csv")
path_pollution = os.path.join("dataset", "water_pollution_disease.csv")

def safe_read_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path), None
        except Exception as e:
            return None, f"Error membaca file: {e}"
    else:
        return None, "File tidak ditemukan"


# =====================================================
# 🏠 DASHBOARD UTAMA (Versi Interaktif & Elegan)
# =====================================================
if menu == "🏠 Dashboard":
    from streamlit_lottie import st_lottie
    import requests

    # Animasi Lottie
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_water = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_yr6zz3wv.json")

    # Judul utama & penelitian
    st.markdown(
        """
        <h1 style='text-align: center; color: #0077b6;'>
            💧 Dashboard SDG 6: Clean Water and Sanitation
        </h1>
        <h3 style='text-align: center; color: #0096c7;'>
            Analisis dan Prediksi Akses Air Minum Terkelola Menggunakan Algoritma 
            <b>XGBoost, LightGBM,</b> dan <b>SVR</b> Berdasarkan Indikator Sanitasi dan Layanan Publik
        </h3>
        """,
        unsafe_allow_html=True
    )

    # Dua kolom untuk layout rapi
    col1, col2 = st.columns([1, 2])
    with col1:
        st_lottie(lottie_water, height=230, key="water_anim")
    with col2:
        st.markdown(
            """
            ### 🌊 Tujuan SDG 6:
            Menjamin ketersediaan dan pengelolaan air bersih serta sanitasi yang berkelanjutan untuk semua.  

            Dashboard ini membantu menganalisis dan memvisualisasikan data terkait:
            - Akses air minum terkelola  
            - Sanitasi dan kebersihan  
            - Polusi air di berbagai negara  

            ### 💡 Manfaat:
            - Mendukung pemahaman kondisi air bersih global  
            - Membantu pengambilan kebijakan terkait sanitasi  
            - Menganalisis hubungan antara indeks SDG dan akses air bersih  
            """
        )

    st.markdown("---")
    st.info("📊 Gunakan menu di sebelah kiri untuk menjelajahi data, visualisasi, dan model prediksi air bersih.")


# =====================================================
# 💧 DATA WRANGLING (Gathering, Assessing, Cleaning)
# =====================================================
elif menu == "💧 Data Wrangling":
    import io, os, re
    st.title("💧 Data Wrangling")
    st.markdown("""
    Tahapan *data wrangling* mencakup tiga proses utama:
    1. **Gathering Data** – pengumpulan dataset  
    2. **Assessing Data** – pemeriksaan struktur, missing values, dan duplikasi  
    3. **Cleaning Data** – pembersihan dan standarisasi data  

    ---
    """)

    tab_gather, tab_assess, tab_clean = st.tabs(
        ["📥 Gathering Data", "🔍 Assessing Data", "🧽 Cleaning Data"]
    )

    # =====================================================
    # Helper: baca CSV dengan aman
    # =====================================================
    def safe_read_csv(path):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df, None
            except Exception as e:
                return None, str(e)
        else:
            return None, "file not found"

    # =====================================================
    # 🟦 TAB 1: GATHERING DATA
    # =====================================================
    with tab_gather:
        st.subheader("📥 Gathering Data")
        st.markdown("Dataset dimuat dari folder `dataset/` yang berada satu level dengan file `dashboard.py`.")

        col1, col2 = st.columns(2)
        df_indexes, err_idx = safe_read_csv(path_indexes)
        df_drinking_water, err_drink = safe_read_csv(path_drinking_water)
        df_pollution, err_poll = safe_read_csv(path_pollution)

        with col1:
            if df_indexes is not None:
                st.success("✅ **indexes.csv** berhasil dimuat.")
                st.dataframe(df_indexes.head(), use_container_width=True)
            else:
                st.error(f"❌ Gagal memuat indexes.csv: {err_idx}")

        with col2:
            if df_drinking_water is not None:
                st.success("✅ **drinking_water.csv** berhasil dimuat.")
                st.dataframe(df_drinking_water.head(), use_container_width=True)
            else:
                st.error(f"❌ Gagal memuat drinking_water.csv: {err_drink}")

        st.markdown("---")
        if df_pollution is not None:
            st.success("✅ **water_pollution_disease.csv** berhasil dimuat.")
            st.dataframe(df_pollution.head(), use_container_width=True)
        else:
            st.warning(f"⚠️ Dataset pollution tidak tersedia: {err_poll} (opsional)")

    # =====================================================
    # 🟦 TAB 2: ASSESSING DATA
    # =====================================================
    with tab_assess:
        st.subheader("🔍 Assessing Data")
        st.markdown("""
        Pemeriksaan mencakup:
        - Struktur data (jumlah baris, kolom, dan tipe data)  
        - Missing values & duplikasi  
        - Statistik deskriptif numerik  

        """)

        def dataset_overview(df, name):
            """Tampilkan ringkasan seperti df.info(), tapi versi visual."""
            if df is None:
                st.warning(f"⚠️ Dataset {name} tidak tersedia.")
                return
            st.markdown(f"### 🧾 {name}")

            # Info dasar
            st.info(f"Ukuran DataFrame: **{df.shape[0]} baris × {df.shape[1]} kolom**")

            # Tipe data
            type_df = pd.DataFrame({
                'Kolom': df.columns,
                'Tipe Data': df.dtypes.astype(str),
                'Jumlah Null': df.isnull().sum(),
                'Persentase Null (%)': (df.isnull().mean() * 100).round(2)
            })
            st.dataframe(type_df, use_container_width=True)

            # Statistik numerik
            num_cols = df.select_dtypes(include=[np.number])
            if not num_cols.empty:
                st.markdown("**Statistik Deskriptif (kolom numerik):**")
                st.dataframe(num_cols.describe().T, use_container_width=True)

            # Duplikasi
            dup = df.duplicated().sum()
            if dup > 0:
                st.warning(f"Ada {dup} baris duplikat di dataset.")
            else:
                st.success("Tidak ada baris duplikat ✅")

            st.markdown("---")

        dataset_overview(df_indexes, "SDG Indexes")
        dataset_overview(df_drinking_water, "Safely Managed Drinking Water")
        dataset_overview(df_pollution, "Water Pollution & Disease")

        # =====================================================
    # 🟦 TAB 3: CLEANING DATA  (GANTI BAGIAN INI DENGAN KODE INI)
    # =====================================================
    with tab_clean:
        st.subheader("🧽 Cleaning Data (lengkap — sesuai Colab)")
        st.markdown("""
        Langkah cleaning (sama persis dengan notebook Colab):
        - Rename kolom spesifik untuk df_indexes (lengkap)  
        - Standardisasi nama kolom (lower, strip, ganti spasi/dash)  
        - Standardisasi isi kategori (country, region, dsb.)  
        - Normalisasi tahun: hanya simpan 2000–2024  
        - Interpolasi nilai hilang (linear) per negara + ffill/bfill  
        - Hapus duplikat  
        """)

        # === RENAME MAP LENGKAP (persis seperti di Colab) ===
        rename_map = {
            'population_using_at_least_basic_drinking_water_services_(%)___rural': 'basic_drinking_water_rural',
            'population_using_at_least_basic_drinking_water_services_(%)___total': 'basic_drinking_water_total',
            'population_using_at_least_basic_drinking_water_services_(%)___urban': 'basic_drinking_water_urban',
            'population_using_safely_managed_drinking_water_services_(%)___rural': 'managed_drinking_water_rural',
            'population_using_safely_managed_drinking_water_services_(%)___total': 'managed_drinking_water_total',
            'population_using_safely_managed_drinking_water_services_(%)___urban': 'managed_drinking_water_urban',
            'population_using_at_least_basic_sanitation_services_(%)___rural': 'basic_sanitation_rural',
            'population_using_at_least_basic_sanitation_services_(%)___total': 'basic_sanitation_total',
            'population_using_at_least_basic_sanitation_services_(%)___urban': 'basic_sanitation_urban',
            'population_using_safely_managed_sanitation_services_(%)___rural': 'managed_sanitation_rural',
            'population_using_safely_managed_sanitation_services_(%)___total': 'managed_sanitation_total',
            'population_using_safely_managed_sanitation_services_(%)___urban': 'managed_sanitation_urban',
            'population_with_basic_handwashing_facilities_at_home_(%)___rural': 'basic_handwashing_rural',
            'population_with_basic_handwashing_facilities_at_home_(%)___total': 'basic_handwashing_total',
            'population_with_basic_handwashing_facilities_at_home_(%)___urban': 'basic_handwashing_urban',
            'population_practising_open_defecation_(%)___rural': 'open_defecation_rural',
            'population_practising_open_defecation_(%)___total': 'open_defecation_total',
            'population_practising_open_defecation_(%)___urban': 'open_defecation_urban'
        }

        # -------------------------
        # Cleaning df_indexes
        # -------------------------
        if df_indexes is not None:
            df = df_indexes.copy()
            st.markdown("#### 🔹 df_indexes — Proses cleaning")

            # 1) Normalisasi nama kolom (lower, strip, replace)
            df.columns = (
                df.columns.astype(str)
                .str.lower()
                .str.strip()
                .str.replace(' ', '_')
                .str.replace('-', '_')
                .str.replace('country_code', 'code')
            )

            # 2) Apply rename_map (lengkap)
            df.rename(columns=rename_map, inplace=True)

            # 3) Standarisasi isi kategori: country & region
            if 'country' in df.columns:
                df['country'] = df['country'].astype(str).str.replace(r'\s*\([^)]*\)', '', regex=True).str.lower().str.strip()
            if 'region' in df.columns:
                df['region'] = df['region'].astype(str).str.replace(r'\s*\([^)]*\)', '', regex=True).str.lower().str.strip()

            # 4) Normalisasi tahun 2000-2024
            removed_rows_year = 0
            if 'year' in df.columns:
                orig = len(df)
                df = df[df['year'].between(2000, 2024)].copy()
                removed_rows_year = orig - len(df)

            # 5) Interpolasi NaN untuk semua kolom hasil rename_map (jika ada)
            cols_to_fill = [c for c in rename_map.values() if c in df.columns]
            st.write("Kolom yang akan di-interpolasi (jika tersedia):", cols_to_fill)
            for col in cols_to_fill:
                df[col] = df.groupby('country')[col].transform(lambda x: x.interpolate(method='linear').ffill().bfill())

            # 6) Cek NaN tersisa & duplikat — lalu hapus duplikat
            nan_after = int(df.isnull().sum().sum())
            dup_before = int(df_indexes.duplicated().sum())
            df.drop_duplicates(inplace=True)
            dup_after = int(df.duplicated().sum())
            dup_removed = dup_before - dup_after if dup_before > 0 else 0

            # 7) Tampilkan ringkasan hasil cleaning df_indexes
            st.success("Hasil cleaning — df_indexes")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write("- Baris dihapus (tahun diluar 2000–2024):", removed_rows_year)
                st.write("- Total NaN setelah interpolasi:", nan_after)
                st.write("- Duplikat dihapus:", dup_removed)
            with col_b:
                st.write("**Kolom (contoh up to 40 kolom):**")
                st.write(df.columns.tolist()[:40])
            st.write("Preview hasil akhir (5 baris):")
            st.dataframe(df.head(), use_container_width=True)

            # replace original
            df_indexes = df.copy()
        else:
            st.info("df_indexes tidak tersedia — skip cleaning df_indexes.")

        # -------------------------
        # Cleaning df_drinking_water
        # -------------------------
        if df_drinking_water is not None:
            df = df_drinking_water.copy()
            st.markdown("#### 🔹 df_drinking_water — Proses cleaning")

            # 1) Rename spesifik (sama persis)
            # Note: kolom di file bisa berbeda case; gunakan match case-insensitive jika perlu
            if 'Usage of safely managed drinking water services' in df.columns:
                df.rename(columns={'Usage of safely managed drinking water services': 'managed_drinking_water'}, inplace=True)
            # 2) semua nama kolom ke lowercase
            df.columns = [col.lower() for col in df.columns]

            # 3) Standarisasi isi kolom 'country'
            if 'country' in df.columns:
                df['country'] = df['country'].astype(str).str.replace(r'\s*\([^)]*\)', '', regex=True).str.lower().str.strip()

            # 4) Normalisasi tahun
            removed_dw = 0
            if 'year' in df.columns:
                orig = len(df)
                df = df[df['year'].between(2000, 2024)].copy()
                removed_dw = orig - len(df)

            # 5) Interpolasi managed_drinking_water
            col_to_fill = 'managed_drinking_water'
            before_nan = after_nan = None
            if col_to_fill in df.columns:
                before_nan = int(df[col_to_fill].isnull().sum())
                if 'country' in df.columns:
                    df[col_to_fill] = df.groupby('country')[col_to_fill].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
                after_nan = int(df[col_to_fill].isnull().sum())

            # 6) Hapus duplikat
            dup_before_dw = int(df_drinking_water.duplicated().sum())
            df.drop_duplicates(inplace=True)
            dup_after_dw = int(df.duplicated().sum())
            dup_removed_dw = dup_before_dw - dup_after_dw if dup_before_dw > 0 else 0

            # 7) Ringkasan
            st.success("Hasil cleaning — df_drinking_water")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write("- Baris dihapus (tahun diluar 2000–2024):", removed_dw)
                if before_nan is not None:
                    st.write(f"- NaN '{col_to_fill}' sebelum→sesudah: {before_nan} → {after_nan}")
                else:
                    st.write(f"- Kolom '{col_to_fill}' tidak ditemukan atau tidak bisa diinterpolasi.")
                st.write("- Duplikat dihapus:", dup_removed_dw)
            with col_b:
                st.write("**Kolom (contoh up to 30):**")
                st.write(df.columns.tolist()[:30])
            st.write("Preview hasil akhir (5 baris):")
            st.dataframe(df.head(), use_container_width=True)

            df_drinking_water = df.copy()
        else:
            st.info("df_drinking_water tidak tersedia — skip cleaning df_drinking_water.")

        # -------------------------
        # Cleaning df_pollution
        # -------------------------
        if df_pollution is not None:
            df = df_pollution.copy()
            st.markdown("#### 🔹 df_pollution — Proses cleaning")

            # 1) Standarisasi nama kolom (lower, strip, hapus unit dlm kurung, hilangkan tanda kutip, ganti spasi/dash/slash)
            new_cols = df.columns.to_list()
            new_cols = [str(col).lower().strip() for col in new_cols]
            new_cols = [re.sub(r'\(.*\)', '', col).strip() for col in new_cols]
            new_cols = [col.replace('"', '') for col in new_cols]
            new_cols = [col.replace(' ', '_').replace('/', '_').replace('-', '_') for col in new_cols]
            new_cols = [re.sub(r'[^a-z0-9_]', '', col) for col in new_cols]
            df.columns = new_cols

            # 2) Standarisasi isi kategori
            if 'country' in df.columns:
                df['country'] = df['country'].astype(str).str.replace(r'\s*\([^)]*\)', '', regex=True).str.lower().str.strip()
            if 'region' in df.columns:
                df['region'] = df['region'].astype(str).str.replace(r'\s*\([^)]*\)', '', regex=True).str.lower().str.strip()
            if 'water_source_type' in df.columns:
                df['water_source_type'] = df['water_source_type'].astype(str).str.lower().str.strip()
            if 'water_treatment_method' in df.columns:
                df['water_treatment_method'] = df['water_treatment_method'].astype(str).str.lower().str.strip()

            # 3) Normalisasi tahun 2000-2024 (jika ada)
            removed_pol = 0
            if 'year' in df.columns:
                orig = len(df)
                df = df[df['year'].between(2000, 2024)].copy()
                removed_pol = orig - len(df)

            # 4) Cek NaN total & duplikat lalu hapus duplikat
            nan_total = int(df.isnull().sum().sum())
            dup_before_pol = int(df_pollution.duplicated().sum())
            df.drop_duplicates(inplace=True)
            dup_after_pol = int(df.duplicated().sum())
            dup_removed_pol = dup_before_pol - dup_after_pol if dup_before_pol > 0 else 0

            # 5) Ringkasan
            st.success("Hasil cleaning — df_pollution")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write("- Baris dihapus (tahun diluar 2000–2024):", removed_pol)
                st.write("- Jumlah NaN (total):", nan_total)
                st.write("- Duplikat dihapus:", dup_removed_pol)
            with col_b:
                st.write("**Contoh Kolom (max 20):**")
                st.write(df.columns.tolist()[:20])
            st.write("Preview hasil akhir (5 baris):")
            st.dataframe(df.head(), use_container_width=True)

            df_pollution = df.copy()
        else:
            st.info("df_pollution tidak tersedia — skip cleaning df_pollution.")

        st.success("✅ Semua langkah cleaning selesai — data siap untuk EDA & visualisasi.")

        # =======================================================
        # 🌍 MERGE & PENYATUAN DATA (Langkah Akhir Cleaning)
        # =======================================================
        st.markdown("### 🌍 Menggabungkan dan Menyatukan Dataset")

        try:
            # --- Merge df_indexes dengan df_pollution ---
            st.write("🔹 Menggabungkan df_indexes dengan df_pollution ...")
            df_master_global = pd.merge(df_indexes, df_pollution, on=['country', 'year'], how='outer')
            st.write(f"Hasil sementara (Merge 1): {df_master_global.shape}")

            # --- Merge hasilnya dengan df_drinking_water ---
            st.write("🔹 Menggabungkan hasil sebelumnya dengan df_drinking_water ...")
            df_master_global = pd.merge(df_master_global, df_drinking_water, on=['country', 'year'], how='outer')
            st.success(f"✅ Merge selesai! Ukuran dataset akhir: {df_master_global.shape}")

            # --- Membersihkan kolom redundan hasil merge ---
            st.markdown("### 🧩 Membersihkan Kolom Redundan")

            # 1️⃣ Kolom Air Minum
            if 'managed_drinking_water_total' in df_master_global.columns and 'managed_drinking_water' in df_master_global.columns:
                df_master_global['managed_drinking_water'] = df_master_global['managed_drinking_water_total'].fillna(
                    df_master_global['managed_drinking_water']
                )
                df_master_global.drop(columns=['managed_drinking_water_total'], inplace=True, errors='ignore')
                st.write("✔ Kolom 'managed_drinking_water' telah dikonsolidasi.")

            # 2️⃣ Kolom Sanitasi
            if 'managed_sanitation_total' in df_master_global.columns and 'sanitation_coverage' in df_master_global.columns:
                df_master_global['sanitation_coverage'] = df_master_global['managed_sanitation_total'].fillna(
                    df_master_global['sanitation_coverage']
                )
                df_master_global.drop(columns=['managed_sanitation_total'], inplace=True, errors='ignore')
                st.write("✔ Kolom 'sanitation_coverage' telah dikonsolidasi.")

            # 3️⃣ Kolom Region
            if 'region_x' in df_master_global.columns and 'region_y' in df_master_global.columns:
                df_master_global['region'] = df_master_global['region_x'].fillna(df_master_global['region_y'])
                df_master_global.drop(columns=['region_x', 'region_y'], inplace=True)
                st.write("✔ Kolom 'region' telah dikonsolidasi.")
            elif 'region_x' in df_master_global.columns:
                df_master_global.rename(columns={'region_x': 'region'}, inplace=True)
            elif 'region_y' in df_master_global.columns:
                df_master_global.rename(columns={'region_y': 'region'}, inplace=True)

            # --- Simpan hasil cleaning untuk tahapan berikut ---
            st.session_state["df_master_global"] = df_master_global
            st.success("💾 Dataset hasil merge dan cleaning telah disimpan untuk digunakan di tahap EDA & Modeling.")
            st.dataframe(df_master_global.head(10))

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat proses merge: {e}")



# =====================================================
# 📊 EDA
# =====================================================

elif menu == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    if "df_master_global" not in st.session_state:
        st.warning("⚠️ Data belum dimuat. Silakan lakukan proses Data Wrangling dan Cleaning terlebih dahulu.")
    else:
        df_master_global = st.session_state["df_master_global"]

        # Buat tab untuk setiap pertanyaan
        tabs = st.tabs([
            "1️⃣ PDB vs Akses Sanitasi Dasar",
            "2️⃣ BABS vs Kasus Diare",
            "3️⃣ Air Minum vs Kematian Bayi",
            "4️⃣ Peningkatan Akses Air Aman",
            "5️⃣ Penurunan Kasus Kolera"
        ])

        # ======================================================
        # 🧩 Pertanyaan 1
        # ======================================================
        with tabs[0]:
            st.subheader("1️⃣ Apakah ada korelasi antara PDB per Kapita dengan Akses Sanitasi Dasar?")

            cols_q1 = ['gdp_per_capita', 'basic_sanitation_total', 'country', 'year', 'region']
            df_q1 = df_master_global[cols_q1].dropna()

            # st.markdown(f"Jumlah data: **{len(df_q1)} baris** setelah pembersihan (drop missing).")

            base = alt.Chart(df_q1).encode(
                x=alt.X('gdp_per_capita:Q', scale=alt.Scale(type="log"), title='PDB per Kapita (USD - Skala Log)'),
                y=alt.Y('basic_sanitation_total:Q', title='Akses Sanitasi Dasar (% Populasi)'),
                tooltip=[
                    alt.Tooltip('country', title='Negara'),
                    alt.Tooltip('year', title='Tahun'),
                    alt.Tooltip('region', title='Wilayah'),
                    alt.Tooltip('gdp_per_capita', title='PDB per Kapita', format='$,.0f'),
                    alt.Tooltip('basic_sanitation_total', title='Sanitasi Dasar', format='.1f')
                ]
            ).properties(width=900, height=500, title="PDB vs Akses Sanitasi Dasar")

            points = base.mark_circle(opacity=0.5).encode(color='region:N')
            regression = base.transform_regression('gdp_per_capita', 'basic_sanitation_total').mark_line(color='red', strokeWidth=3)
            st.altair_chart((points + regression).interactive(), use_container_width=True)
            st.markdown("📈 **Interpretasi:** Meskipun PDB per kapita meningkat, akses sanitasi dasar TIDAK selalu meningkat proporsional. Garis regresi yang hampir mendatar menunjukkan korelasi yang lemah.")


        # ======================================================
        # 🧩 Pertanyaan 2
        # ======================================================
        with tabs[1]:
            st.subheader("2️⃣ Bagaimana hubungan antara angka BABS dan kasus Diare?")
            cols_q2 = ['open_defecation_total', 'diarrheal_cases_per_100000_people', 'country', 'year', 'region']
            df_q2 = df_master_global[cols_q2].dropna()

            # st.markdown(f"Data bersih: **{len(df_q2)} baris** (setelah dropna).")

            scatter = alt.Chart(df_q2).mark_circle(size=80, opacity=0.6).encode(
                x=alt.X('open_defecation_total:Q', title='Populasi BABS (%)'),
                y=alt.Y('diarrheal_cases_per_100000_people:Q', title='Kasus Diare per 100.000 Penduduk'),
                color=alt.Color('region:N', title='Wilayah'),
                tooltip=[
                    'country:N', 'year:Q', 'region:N',
                    alt.Tooltip('open_defecation_total:Q', title='BABS (%)', format='.2f'),
                    alt.Tooltip('diarrheal_cases_per_100000_people:Q', title='Kasus Diare', format='.1f')
                ]
            ).properties(
                title="Hubungan BABS dan Kasus Diare",
                width=900, height=500
            )

            regression = alt.Chart(df_q2).mark_line(color='red', strokeWidth=3, opacity=0.8).transform_regression(
                'open_defecation_total', 'diarrheal_cases_per_100000_people'
            ).encode(x='open_defecation_total:Q', y='diarrheal_cases_per_100000_people:Q')

            st.altair_chart((scatter + regression).interactive(), use_container_width=True)
            st.markdown("📈 **Interpretasi:** Hubungan antara BABS dan kasus diare adalah LEMAH atau HAMPIR TIDAK ADA. Terlihat dari garis regresi hampir datar tidak ada tren linear juga titik tersebar acak tidak ada pola yang konsisten")


        # ======================================================
        # 🧩 Pertanyaan 3
        # ======================================================
        with tabs[2]:
            st.subheader("3️⃣ Seberapa besar pengaruh akses air minum dasar terhadap angka kematian bayi?")
            cols_q3 = ['basic_drinking_water_total', 'infant_mortality_rate', 'country', 'year', 'region']
            df_q3 = df_master_global[cols_q3].dropna()

            # st.markdown(f"Jumlah data: **{len(df_q3)} baris** (setelah dropna).")

            base = alt.Chart(df_q3).encode(
                x=alt.X('basic_drinking_water_total:Q', title='Akses Air Minum Dasar (%)'),
                y=alt.Y('infant_mortality_rate:Q', title='Angka Kematian Bayi (per 1.000 kelahiran)'),
                tooltip=['country:N', 'year:Q', 'region:N']
            ).properties(width=900, height=500, title="Akses Air Minum Dasar vs Angka Kematian Bayi")

            points = base.mark_circle(opacity=0.5).encode(color='region:N')
            regression = base.transform_regression('basic_drinking_water_total', 'infant_mortality_rate').mark_line(color='red', strokeWidth=3)
            st.altair_chart((points + regression).interactive(), use_container_width=True)
            st.markdown("📈 **Interpretasi:** ")


        # ======================================================
        # 🧩 Pertanyaan 4 — Peningkatan Akses Air Minum Aman
        # ======================================================
        with tabs[3]:
            st.subheader("4️⃣ Wilayah dengan peningkatan tercepat dalam akses air minum aman (10 tahun terakhir)")

            cols_q4 = ['year', 'region', 'managed_drinking_water']
            df_q4 = df_master_global[cols_q4].dropna()
            df_agg = df_q4.groupby(['year', 'region']).mean().reset_index()

            max_year = df_agg['year'].max()
            min_year = max_year - 10
            df_last10 = df_agg[df_agg['year'] >= min_year]

            changes = []
            for region in df_last10['region'].unique():
                df_r = df_last10[df_last10['region'] == region].sort_values('year')
                if len(df_r) >= 2:
                    first, last = df_r.iloc[0], df_r.iloc[-1]
                    change = last['managed_drinking_water'] - first['managed_drinking_water']
                    pct = (change / first['managed_drinking_water'] * 100) if first['managed_drinking_water'] > 0 else 0
                    changes.append({
                        'region': region,
                        'start': first['managed_drinking_water'],
                        'end': last['managed_drinking_water'],
                        'change': change,
                        'pct_change': pct
                    })

            df_change = pd.DataFrame(changes).sort_values('change', ascending=False)

            # --- Bar Chart ---
            bars = alt.Chart(df_change).mark_bar(
                cornerRadiusTopRight=5,
                cornerRadiusBottomRight=5
            ).encode(
                x=alt.X('change:Q', title='Perubahan Akses Air Minum Aman (% poin)'),
                y=alt.Y('region:N', sort='-x', title='Wilayah'),
                color=alt.Color('change:Q', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=[
                    alt.Tooltip('region:N', title='Wilayah'),
                    alt.Tooltip('start:Q', title='Tahun Awal', format='.1f'),
                    alt.Tooltip('end:Q', title='Tahun Akhir', format='.1f'),
                    alt.Tooltip('change:Q', title='Perubahan (% poin)', format='+.2f')
                ]
            )

            # --- Label Angka di Ujung Bar ---
            text = alt.Chart(df_change).mark_text(
                align='left',
                baseline='middle',
                dx=5,  # jarak dari ujung bar
                fontSize=12,
                fontWeight='bold',
                color='black'
            ).encode(
                x=alt.X('change:Q'),
                y=alt.Y('region:N', sort='-x'),
                text=alt.Text('change:Q', format='+.1f')
            )

            chart = (bars + text).properties(
                width=900,
                height=400,
                title="Peningkatan Akses Air Minum Aman (10 Tahun Terakhir)"
            )

            st.altair_chart(chart, use_container_width=True)
            st.markdown("📈 **Interpretasi:** Wilayah dengan batang terpanjang menunjukkan peningkatan akses air minum aman terbesar dalam 10 tahun terakhir.")



        # ======================================================
        # 🧩 Pertanyaan 5 — Penurunan Kasus Kolera
        # ======================================================
        with tabs[4]:
            st.subheader("5️⃣ Wilayah dengan penurunan tercepat dalam kasus Kolera (10 tahun terakhir)")

            # --- Persiapan Data ---
            cholera_col = 'cholera_cases_per_100000_people'
            cols_q5 = ['year', 'region', cholera_col]
            df_q5 = df_master_global[cols_q5].dropna()
            df_agg = df_q5.groupby(['year', 'region']).mean().reset_index()

            max_year = df_agg['year'].max()
            min_year = max_year - 10
            df_last10 = df_agg[df_agg['year'] >= min_year]

            changes = []
            for region in df_last10['region'].unique():
                df_r = df_last10[df_last10['region'] == region].sort_values('year')
                if len(df_r) >= 2:
                    first, last = df_r.iloc[0], df_r.iloc[-1]
                    change = last[cholera_col] - first[cholera_col]
                    pct = (change / first[cholera_col] * 100) if first[cholera_col] > 0 else 0
                    changes.append({
                        'region': region,
                        'start': first[cholera_col],
                        'end': last[cholera_col],
                        'change': change,
                        'pct_change': pct
                    })

            df_change_b = pd.DataFrame(changes).sort_values('change', ascending=True)

            # --- Bar Chart Utama ---
            bars = alt.Chart(df_change_b).mark_bar(
                cornerRadiusTopRight=5,
                cornerRadiusBottomRight=5
            ).encode(
                x=alt.X('change:Q', title='Perubahan Kasus Kolera (per 100k penduduk)'),
                y=alt.Y('region:N', sort='x', title='Wilayah'),
                color=alt.Color('change:Q', scale=alt.Scale(scheme='redblue', reverse=True), legend=None),
                tooltip=[
                    alt.Tooltip('region:N', title='Wilayah'),
                    alt.Tooltip('start:Q', title='Tahun Awal', format='.1f'),
                    alt.Tooltip('end:Q', title='Tahun Akhir', format='.1f'),
                    alt.Tooltip('change:Q', title='Perubahan (per 100k)', format='+.2f')
                ]
            )

            # --- Label Positif (kenaikan kasus, kanan bar) ---
            text_pos = alt.Chart(df_change_b).transform_filter(
                "datum.change >= 0"
            ).mark_text(
                align='left',
                baseline='middle',
                dx=5,
                fontSize=12,
                fontWeight='bold',
                color='black'
            ).encode(
                x=alt.X('change:Q'),
                y=alt.Y('region:N', sort='x'),
                text=alt.Text('change:Q', format='+.2f')
            )

            # --- Label Negatif (penurunan kasus, kiri bar) ---
            text_neg = alt.Chart(df_change_b).transform_filter(
                "datum.change < 0"
            ).mark_text(
                align='right',
                baseline='middle',
                dx=-5,
                fontSize=12,
                fontWeight='bold',
                color='black'
            ).encode(
                x=alt.X('change:Q'),
                y=alt.Y('region:N', sort='x'),
                text=alt.Text('change:Q', format='+.2f')
            )

            # --- Gabungkan ---
            chart = (bars + text_pos + text_neg).properties(
                width=900,
                height=400,
                title="Penurunan Kasus Kolera (10 Tahun Terakhir)"
            )

            # --- Tampilkan ---
            st.altair_chart(chart, use_container_width=True)

            st.markdown("""
            📉 **Interpretasi:**
            Wilayah dengan batang paling ke kiri menunjukkan penurunan tercepat dalam kasus kolera selama 10 tahun terakhir.
            Nilai negatif berarti jumlah kasus per 100.000 penduduk telah menurun, sedangkan nilai positif menunjukkan peningkatan.
            """)


# =====================================================
# 📈 Visualisasi Lanjutan
# =====================================================
elif menu == "📈 Visualisasi Lanjutan":
    st.title("📈 Visualisasi Lanjutan")
    st.info("Berisi visualisasi mendalam terkait outlier, tren perubahan, dan korelasi antar variabel.")

    if "df_master_global" not in st.session_state:
        st.warning("⚠️ Data belum dimuat. Silakan lakukan proses Data Wrangling dan Cleaning terlebih dahulu.")
    else:
        df_master_global = st.session_state["df_master_global"]

        # Buat tab untuk 5 bagian visualisasi lanjutan
        tabs = st.tabs([
            "1️⃣ Analisis Outlier (IQR)",
            "2️⃣ Round-Based Gapping (10 Tahun Terakhir)",
            "3️⃣ Heatmap Korelasi Variabel Numerik",
            "4️⃣ Kesenjangan Akses Air & Sanitasi (Kota vs Desa)",
            "5️⃣ Proporsi Jenis Sumber Air per Wilayah"
        ])

        # ======================================================
        # 🧩 TAB 1: Analisis Outlier (IQR)
        # ======================================================
        with tabs[0]:
            st.subheader("1️⃣ Deteksi & Visualisasi Outlier (Metode IQR)")

            df_to_process = df_master_global.copy()
            numerical_cols = df_to_process.select_dtypes(include=np.number).columns.tolist()
            if "year" in numerical_cols:
                numerical_cols.remove("year")

            outlier_indices = set()
            for col in numerical_cols:
                if df_to_process[col].notna().any():
                    Q1 = df_to_process[col].quantile(0.25)
                    Q3 = df_to_process[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        col_outliers = df_to_process[(df_to_process[col] < lower) | (df_to_process[col] > upper)].index
                        outlier_indices.update(col_outliers)

            num_outlier_rows = len(outlier_indices)
            st.write(f"📊 Ditemukan **{num_outlier_rows} baris** yang mengandung outlier dari {len(numerical_cols)} kolom numerik.")

            cols_for_boxplot_viz = [
                'diarrheal_cases_per_100000_people','cholera_cases_per_100000_people',
                'infant_mortality_rate','basic_sanitation_total',
                'basic_drinking_water_total','managed_drinking_water',
                'open_defecation_total','gdp_per_capita',
                'healthcare_access_index','bacteria_count','ph_level'
            ]
            existing_cols = [c for c in cols_for_boxplot_viz if c in df_to_process.columns]

            st.markdown("### 📦 Visualisasi Boxplot per Variabel")
            for col_name in existing_cols:
                use_log = any(k in col_name for k in ['cases', 'gdp', 'bacteria'])
                chart = alt.Chart(df_to_process).mark_boxplot(extent='min-max', size=30).encode(
                    y=alt.Y(f'{col_name}:Q',
                            scale=alt.Scale(type='log', zero=False) if use_log else alt.Scale(),
                            title=f'{col_name} {"(Log)" if use_log else ""}')
                ).properties(width=300, height=300, title=f'Distribusi & Outlier - {col_name}')
                st.altair_chart(chart, use_container_width=True)

            # Simpan data tanpa outlier ke session_state
            df_no_outliers = df_to_process.drop(index=list(outlier_indices))
            st.session_state["df_no_outliers"] = df_no_outliers
            st.success(f"✅ {len(outlier_indices)} baris outlier dihapus. Dataset bersih disimpan ke `df_no_outliers`.")

        # ======================================================
        # 🧩 TAB 2: Round-Based Gapping — versi urut per warna tren
        # ======================================================
        with tabs[1]:
            st.subheader("2️⃣ Analisis Perubahan Akses Air Minum Layak (Round-Based Gapping)")

            df_gap = df_master_global.copy()
            required_cols = ['region', 'year', 'managed_drinking_water']

            if all(col in df_gap.columns for col in required_cols):
                latest_year = int(df_gap['year'].max())
                earliest_year = latest_year - 10
                df_10yr = df_gap[df_gap['year'].between(earliest_year, latest_year)]

                # Hitung perubahan (gap) per region
                df_gap_region = (
                    df_10yr.groupby('region')['managed_drinking_water']
                    .agg(['first', 'last'])
                    .reset_index()
                )
                df_gap_region['gap'] = df_gap_region['last'] - df_gap_region['first']

                # Tambahkan label tren warna
                df_gap_region['trend_color'] = df_gap_region['gap'].apply(
                    lambda x: 'Meningkat' if x > 0 else ('Menurun' if x < 0 else 'Stabil')
                )

                # Tentukan urutan kelompok warna
                color_order = ['Meningkat', 'Stabil', 'Menurun']
                color_scale = alt.Scale(
                    domain=color_order,
                    range=['#2ECC71', '#F1C40F', '#E74C3C']
                )

                # Buat kolom pembantu untuk sorting agar kelompok warna berdekatan
                order_map = {'Meningkat': 1, 'Stabil': 2, 'Menurun': 3}
                df_gap_region['trend_order'] = df_gap_region['trend_color'].map(order_map)

                # Urutkan dulu berdasarkan tren (warna), lalu besar nilai gap (desc)
                df_gap_region = df_gap_region.sort_values(
                    by=['trend_order', 'gap'],
                    ascending=[True, False]
                )

                # Buat urutan label sumbu Y sesuai hasil pengelompokan
                region_order = df_gap_region['region'].tolist()

                # Chart batang horizontal
                chart_gap_region = (
                    alt.Chart(df_gap_region)
                    .mark_bar(size=25, cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                    .encode(
                        y=alt.Y('region:N', sort=region_order, title='Wilayah'),
                        x=alt.X('gap:Q', title=f'Perubahan Akses Air Minum Layak ({earliest_year}–{latest_year})'),
                        color=alt.Color('trend_color:N', scale=color_scale, title='Arah Perubahan'),
                        tooltip=[
                            alt.Tooltip('region:N', title='Wilayah'),
                            alt.Tooltip('first:Q', title=f'Akses Tahun {earliest_year}', format=',.2f'),
                            alt.Tooltip('last:Q', title=f'Akses Tahun {latest_year}', format=',.2f'),
                            alt.Tooltip('gap:Q', title='Perubahan (Gap)', format='+.2f'),
                            alt.Tooltip('trend_color:N', title='Tren')
                        ]
                    )
                    .properties(
                        title=f"Round-Based Gapping: Perubahan Akses Air Minum Layak per Wilayah ({earliest_year}–{latest_year})",
                        width=900,
                        height=450
                    )
                )

                # Tambahkan garis nol
                zero_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='gray', strokeDash=[4, 4]).encode(x='x:Q')

                # Tambahkan label nilai perubahan di sisi batang
                text_labels = alt.Chart(df_gap_region).mark_text(
                    align='left',
                    dx=6,
                    fontWeight='bold'
                ).encode(
                    y=alt.Y('region:N', sort=region_order),
                    x=alt.X('gap:Q'),
                    text=alt.Text('gap:Q', format='+.2f'),
                    color=alt.value('black')
                )

                st.altair_chart(chart_gap_region + text_labels + zero_line, use_container_width=True)

                st.markdown("""
                📊 **Keterangan:**
                - 🟩 Meningkat 
                - 🟨 Stabil 
                - 🟥 Menurun  
                """)

            else:
                st.error("❌ Kolom 'region', 'year', atau 'managed_drinking_water' tidak ditemukan dalam data.")

                st.subheader("2️⃣ Analisis Perubahan Akses Air Minum Layak (Round-Based Gapping)")

                df_gap = df_master_global.copy()
                required_cols = ['region', 'year', 'managed_drinking_water']

                if all(col in df_gap.columns for col in required_cols):
                        latest_year = int(df_gap['year'].max())
                        earliest_year = latest_year - 10
                        df_10yr = df_gap[df_gap['year'].between(earliest_year, latest_year)]

                        # Hitung gap per region
                        df_gap_region = (
                            df_10yr.groupby('region')['managed_drinking_water']
                            .agg(['first', 'last'])
                            .reset_index()
                        )
                        df_gap_region['gap'] = df_gap_region['last'] - df_gap_region['first']

                        # Tambahkan kategori tren warna
                        df_gap_region['trend_color'] = df_gap_region['gap'].apply(
                            lambda x: 'Meningkat' if x > 0 else ('Menurun' if x < 0 else 'Stabil')
                        )

                        # Urutkan sesuai kelompok warna dan besar perubahan
                        trend_order = {'Meningkat': 1, 'Stabil': 2, 'Menurun': 3}
                        df_gap_region['sort_order'] = df_gap_region['trend_color'].map(trend_order)

                        # Sort by group (warna) lalu besar gap (desc untuk meningkat, asc untuk menurun)
                        df_gap_region = df_gap_region.sort_values(
                            by=['sort_order', 'gap'], ascending=[True, False]
                        )

                        # Buat daftar urutan region untuk sumbu Y
                        region_order = df_gap_region['region'].tolist()

                        # Skala warna
                        color_scale = alt.Scale(domain=['Meningkat', 'Stabil', 'Menurun'],
                                                range=['#2ECC71', '#F1C40F', '#E74C3C'])

                        threshold = 5.0  # ambang jarak label

                        # Bar utama
                        bars = (
                            alt.Chart(df_gap_region)
                            .mark_bar(size=30, cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                            .encode(
                                y=alt.Y('region:N', sort=region_order, title='Wilayah'),
                                x=alt.X('gap:Q', title=f'Perubahan Akses Air Minum Layak ({earliest_year}–{latest_year})'),
                                color=alt.Color('trend_color:N', scale=color_scale, title='Arah Perubahan'),
                                tooltip=[
                                    alt.Tooltip('region:N', title='Wilayah'),
                                    alt.Tooltip('first:Q', title=f'Akses Tahun {earliest_year}', format=',.2f'),
                                    alt.Tooltip('last:Q', title=f'Akses Tahun {latest_year}', format=',.2f'),
                                    alt.Tooltip('gap:Q', title='Perubahan (Gap)', format='+.2f'),
                                    alt.Tooltip('trend_color:N', title='Tren')
                                ]
                            )
                        )

                        # Label (empat lapisan seperti sebelumnya)
                        text_pos_inside = (
                            alt.Chart(df_gap_region[df_gap_region['gap'] >= threshold])
                            .mark_text(align='right', dx=-6, color='white', fontWeight='bold')
                            .encode(y='region:N', x='gap:Q', text=alt.Text('gap:Q', format='+.2f'))
                        )

                        text_pos_outside = (
                            alt.Chart(df_gap_region[(df_gap_region['gap'] > 0) & (df_gap_region['gap'] < threshold)])
                            .mark_text(align='left', dx=6, color='black', fontWeight='bold')
                            .encode(y='region:N', x='gap:Q', text=alt.Text('gap:Q', format='+.2f'))
                        )

                        text_neg_inside = (
                            alt.Chart(df_gap_region[df_gap_region['gap'] <= -threshold])
                            .mark_text(align='left', dx=6, color='white', fontWeight='bold')
                            .encode(y='region:N', x='gap:Q', text=alt.Text('gap:Q', format='+.2f'))
                        )

                        text_neg_outside = (
                            alt.Chart(df_gap_region[(df_gap_region['gap'] < 0) & (df_gap_region['gap'] > -threshold)])
                            .mark_text(align='right', dx=-6, color='black', fontWeight='bold')
                            .encode(y='region:N', x='gap:Q', text=alt.Text('gap:Q', format='+.2f'))
                        )

                        zero_line = alt.Chart(pd.DataFrame({'x':[0]})).mark_rule(color='gray', strokeDash=[4,4]).encode(x='x:Q')

                        chart_gap_region = (
                            bars + text_pos_inside + text_pos_outside + text_neg_inside + text_neg_outside + zero_line
                        ).properties(
                            title=f"Round-Based Gapping: Perubahan Akses Air Minum Layak per Wilayah ({earliest_year}–{latest_year})",
                            width=900,
                            height=450
                        )

                        st.altair_chart(chart_gap_region, use_container_width=True)

                        st.markdown("""
                        📊 **Interpretasi:**
                        - 🟩 **Meningkat** di bagian atas  
                        - 🟨 **Stabil** di tengah  
                        - 🟥 **Menurun** di bawah  
                        - Nilai positif menunjukkan peningkatan akses air minum layak selama 10 tahun terakhir.
                        """)
                else:
                        st.error("❌ Kolom 'region', 'year', atau 'managed_drinking_water' tidak ditemukan dalam data.")

        # ======================================================
        # 🧩 TAB 3: Heatmap Korelasi Variabel Numerik
        # ======================================================
        with tabs[2]:
            st.subheader("3️⃣ Heatmap Korelasi Variabel Numerik")

            if "df_no_outliers" not in st.session_state:
                st.warning("⚠️ Jalankan tab 'Analisis Outlier' terlebih dahulu untuk membuat df_no_outliers.")
            else:
                df_no_outliers = st.session_state["df_no_outliers"]

                numerical_cols = df_no_outliers.select_dtypes(include=np.number).columns.tolist()
                if "year" in numerical_cols:
                    numerical_cols.remove("year")

                df_corr = df_no_outliers[numerical_cols].corr().reset_index().rename(columns={'index': 'variable_1'})
                df_corr_long = df_corr.melt(id_vars='variable_1', var_name='variable_2', value_name='correlation')

                base = alt.Chart(df_corr_long).encode(
                    x=alt.X('variable_1:N', sort=numerical_cols, title=None, axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('variable_2:N', sort=numerical_cols, title=None),
                    color=alt.Color('correlation:Q', scale=alt.Scale(domain=[-1, 1], scheme='redblue')),
                    tooltip=['variable_1', 'variable_2', alt.Tooltip('correlation:Q', format='.2f')]
                ).properties(width=800, height=800, title="Heatmap Korelasi Variabel Numerik (Data Tanpa Outlier)")

                heatmap = base.mark_rect()
                text = base.mark_text(fontSize=8).encode(
                    text=alt.Text('correlation:Q', format='.2f'),
                    color=alt.condition('abs(datum.correlation) > 0.6', alt.value('white'), alt.value('black'))
                )

                st.altair_chart(heatmap + text, use_container_width=True)

        # ======================================================
        # 🧩 TAB 4: Kesenjangan Akses Air & Sanitasi
        # ======================================================
        with tabs[3]:
            st.subheader("🌆 Kesenjangan Akses Air & Sanitasi antara Kota dan Desa")
            st.markdown("""
            Visualisasi ini menunjukkan **rata-rata akses air minum layak dan sanitasi dasar**
            berdasarkan wilayah dan membandingkan antara **Kota (Urban)** dan **Desa (Rural)**.
            Tujuannya untuk melihat kesenjangan layanan antar area permukiman di setiap region.
            """)

            # --- Langkah 1: Cek apakah DataFrame sudah tersedia ---
            if 'df_master_global' not in locals() and 'df_master_global' not in st.session_state:
                st.error("❌ DataFrame `df_master_global` tidak ditemukan. Jalankan proses merge data terlebih dahulu.")
            else:
                # Ambil DataFrame dari session_state kalau ada
                df_master_global = st.session_state.get('df_master_global', df_master_global)

                # --- Langkah 2: Tentukan kolom yang relevan ---
                cols_disparity = [
                    'region',
                    'basic_drinking_water_urban',
                    'basic_drinking_water_rural',
                    'basic_sanitation_urban',
                    'basic_sanitation_rural'
                ]

                existing_cols_disp = [col for col in cols_disparity if col in df_master_global.columns]
                missing_cols_disp = set(cols_disparity) - set(existing_cols_disp)
                if missing_cols_disp:
                    st.warning(f"⚠️ Beberapa kolom berikut tidak ditemukan: {missing_cols_disp}")

                # --- Langkah 3: Hitung rata-rata per wilayah ---
                df_avg_disp = df_master_global[existing_cols_disp].groupby('region').mean().reset_index()

                # --- Langkah 4: Ubah format (melt) agar mudah divisualisasikan ---
                df_melted_disp = df_avg_disp.melt(
                    id_vars=['region'],
                    var_name='indicator',
                    value_name='average_access_percent'
                )

                # Tambahkan kolom keterangan tipe dan lokasi
                df_melted_disp['type'] = df_melted_disp['indicator'].apply(
                    lambda x: 'Air Minum Dasar' if 'water' in x else 'Sanitasi Dasar'
                )
                df_melted_disp['location'] = df_melted_disp['indicator'].apply(
                    lambda x: 'Kota (Urban)' if 'urban' in x else 'Desa (Rural)'
                )

                # --- Langkah 5: Buat visualisasi Altair ---
                chart_disp = alt.Chart(df_melted_disp).mark_bar().encode(
                    x=alt.X('region:N', title='Wilayah', axis=alt.Axis(labelAngle=-40)),
                    y=alt.Y('average_access_percent:Q', title='Rata-rata Akses (%)', axis=alt.Axis(format='.0f')),
                    color=alt.Color('location:N', title='Lokasi', scale=alt.Scale(scheme='tableau10')),
                    column=alt.Column('type:N', title='Indikator SDG 6'),
                    tooltip=[
                        alt.Tooltip('region:N', title='Wilayah'),
                        alt.Tooltip('type:N', title='Indikator'),
                        alt.Tooltip('location:N', title='Lokasi'),
                        alt.Tooltip('average_access_percent:Q', title='Rata-rata Akses (%)', format='.1f')
                    ]
                ).properties(
                    title='Kesenjangan Rata-rata Akses Air & Sanitasi (Kota vs Desa)',
                    width=300
                ).configure_axis(
                    labelFontSize=11,
                    titleFontSize=12
                ).configure_header(
                    titleFontSize=13,
                    labelFontSize=11
                ).interactive()

                st.altair_chart(chart_disp, use_container_width=True)

                # # --- Langkah 6: Interpretasi Otomatis ---
                # st.markdown("### 📊 Interpretasi Otomatis")
                # st.write("""
                # Dari grafik di atas dapat diamati bahwa:
                # - **Akses air minum dan sanitasi di wilayah perkotaan (Urban)** umumnya lebih tinggi dibandingkan pedesaan (Rural).
                # - Kesenjangan yang besar mengindikasikan perlunya **intervensi kebijakan untuk wilayah pedesaan**, terutama terkait infrastruktur dasar.
                # - Wilayah dengan nilai rata-rata mendekati atau sama antara Urban dan Rural menunjukkan **pemerataan layanan yang lebih baik**.
                # """)


        # ======================================================
        # 🧩 TAB 4: Proporsi Jenis Sumber Air per Wilayah
        # ======================================================
        with tabs[4]:
            st.subheader("5️⃣ Proporsi Jenis Sumber Air per Wilayah")

            cols_source = ['region', 'water_source_type']
            if all(col in df_master_global.columns for col in cols_source):
                df_source = df_master_global[cols_source].dropna(subset=['water_source_type'])

                df_counts = df_source.groupby(['region', 'water_source_type']).size().reset_index(name='count')
                df_totals = df_source.groupby(['region']).size().reset_index(name='total_count')
                df_perc = pd.merge(df_counts, df_totals, on='region')
                df_perc['percentage'] = (df_perc['count'] / df_perc['total_count']) * 100

                chart2 = alt.Chart(df_perc).mark_bar().encode(
                    x=alt.X('region:N', title='Wilayah', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('percentage:Q', title='Proporsi Penggunaan Sumber Air (%)'),
                    color=alt.Color('water_source_type:N', title='Jenis Sumber Air'),
                    order=alt.Order('percentage:Q', sort='descending'),
                    tooltip=[
                        alt.Tooltip('region', title='Wilayah'),
                        alt.Tooltip('water_source_type', title='Jenis Sumber'),
                        alt.Tooltip('percentage', title='Persentase', format='.1f'),
                        alt.Tooltip('count', title='Jumlah Data')
                    ]
                ).properties(
                    title='Distribusi Sumber Air Berdasarkan Wilayah',
                    width=alt.Step(80)
                ).interactive()

                st.altair_chart(chart2, use_container_width=True)
            else:
                st.warning("Kolom 'region' dan 'water_source_type' tidak ditemukan di dataset.")

# =====================================================
# 🤖 PEMODELAN
# =====================================================
elif menu == "🤖 Pemodelan":
    st.title("🤖 Pemodelan Machine Learning")
    st.info("Halaman ini berisi hasil pelatihan model dan uji coba prediksi interaktif.")

    if "df_master_global" not in st.session_state:
        st.warning("⚠️ Data belum dimuat. Silakan lakukan proses Data Wrangling dan Cleaning terlebih dahulu.")
    else:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import RobustScaler, PowerTransformer
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor
        import numpy as np
        import pandas as pd
        import altair as alt
        import pickle
        import os
        import joblib
        import pickle

        df_air = st.session_state["df_master_global"].copy()

        # =======================================
        # Tabs
        # =======================================
        tab1, tab2 = st.tabs(["📈 Hasil Pemodelan", "🧮 Uji Coba Prediksi"])

        # =====================================================
        # TAB 1️⃣ : HASIL PEMODELAN
        # =====================================================
        with tab1:
            fitur = [
                'basic_sanitation_total',
                'basic_drinking_water_total',
                'open_defecation_total',
                'sanitation_coverage'
            ]
            target = 'managed_drinking_water'

            df_air = df_air[fitur + [target]].dropna()

            st.subheader("🧩 Fitur & Target yang Digunakan")
            st.markdown("""
            - **Fitur (X):**
              - `basic_sanitation_total`
              - `basic_drinking_water_total`
              - `open_defecation_total`
              - `sanitation_coverage`
            - **Target (y):**
              - `managed_drinking_water`
            """)

            # Feature Engineering
            df_air["ratio_sani_open"] = df_air["basic_sanitation_total"] / (df_air["open_defecation_total"] + 1)
            df_air["ratio_water_sani"] = df_air["basic_drinking_water_total"] / (df_air["basic_sanitation_total"] + 1)
            df_air["service_index"] = (
                df_air["basic_sanitation_total"] + df_air["basic_drinking_water_total"] + df_air["sanitation_coverage"]
            ) / 3

            st.subheader("⚙️ Feature Engineering")
            st.markdown("""
            - **`ratio_sani_open`** = basic_sanitation_total / (open_defecation_total + 1)  
            - **`ratio_water_sani`** = basic_drinking_water_total / (basic_sanitation_total + 1)  
            - **`service_index`** = Rata-rata dari basic_sanitation_total, basic_drinking_water_total, dan sanitation_coverage  
            """)

            all_features = fitur + ["ratio_sani_open", "ratio_water_sani", "service_index"]

            # Heatmap
            st.subheader("🔥 Korelasi Variabel Numerik")
            corr = df_air[all_features + [target]].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

            # Split & Scaling
            X = df_air[all_features]
            y = df_air[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = RobustScaler()
            pt = PowerTransformer()
            X_train_scaled = pt.fit_transform(scaler.fit_transform(X_train))
            X_test_scaled = pt.transform(scaler.transform(X_test))

            # Model Training
            models = {
                "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, random_state=42),
                "CatBoost": CatBoostRegressor(iterations=400, learning_rate=0.05, depth=6, verbose=0, random_state=42),
                "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42),
            }

            results = []
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                results.append({
                    "Model": name,
                    "R2 Train": r2_score(y_train, y_pred_train),
                    "R2 Test": r2_score(y_test, y_pred_test),
                    "MAE Test": mean_absolute_error(y_test, y_pred_test),
                    "RMSE Test": np.sqrt(mean_squared_error(y_test, y_pred_test))
                })

            results_df = pd.DataFrame(results).sort_values(by="R2 Test", ascending=False)
            st.subheader("📊 Hasil Evaluasi Model")
            st.dataframe(results_df.style.highlight_max(axis=0, color='#A5D6A7'))

            best_model_name = results_df.iloc[0]["Model"]
            best_model = models[best_model_name]
            st.session_state["fit_features"] = all_features

            st.success(f"✅ Model terbaik: **{best_model_name}**")
            st.info("Untuk memastikan hasil prediksi identik dengan Colab, model dari file `model.pkl`, `scaler.pkl`, dan `transformer.pkl` akan digunakan di tab berikutnya.")

        # =====================================================
        # TAB 2️⃣ : UJI COBA PREDIKSI (PAKAI MODEL COLAB)
        # =====================================================
        with tab2:
            st.subheader("🧮 Uji Coba Prediksi (Menggunakan Model Colab)")

            model_file = "model.pkl"
            scaler_file = "scaler.pkl"
            transformer_file = "transformer.pkl"

            if not (os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(transformer_file)):
                st.error("❌ File model tidak lengkap di direktori kerja.")
            else:
                try:
                    model = joblib.load(model_file)
                    scaler = joblib.load(scaler_file)
                    transformer = joblib.load(transformer_file)
                except:
                    model = pickle.load(open(model_file, "rb"))
                    scaler = pickle.load(open(scaler_file, "rb"))
                    transformer = pickle.load(open(transformer_file, "rb"))

                st.success("✅ Model dan transformer berhasil dimuat dari file .pkl")

                st.markdown("Masukkan nilai fitur untuk memprediksi **Managed Drinking Water (%):**")
                user_input = {}
                col1, col2 = st.columns(2)
                with col1:
                    user_input["basic_sanitation_total"] = st.number_input("Sanitasi Dasar (%)", 0.0, 100.0, 90.0)
                    user_input["open_defecation_total"] = st.number_input("BABS (%)", 0.0, 100.0, 2.5)
                with col2:
                    user_input["basic_drinking_water_total"] = st.number_input("Air Minum Dasar (%)", 0.0, 100.0, 90.0)
                    user_input["sanitation_coverage"] = st.number_input("Cakupan Sanitasi (%)", 0.0, 100.0, 88.0)

                # Tombol prediksi
                if st.button("🔮 Prediksi Sekarang"):
                    input_df = pd.DataFrame([user_input])
                    input_df["ratio_sani_open"] = input_df["basic_sanitation_total"] / (input_df["open_defecation_total"] + 1)
                    input_df["ratio_water_sani"] = input_df["basic_drinking_water_total"] / (input_df["basic_sanitation_total"] + 1)
                    input_df["service_index"] = (
                        input_df["basic_sanitation_total"] + input_df["basic_drinking_water_total"] + input_df["sanitation_coverage"]
                    ) / 3

                    # Pastikan fitur sesuai dengan training
                    X_input = input_df[st.session_state["fit_features"]]
                    X_scaled = transformer.transform(scaler.transform(X_input))
                    y_pred = model.predict(X_scaled)[0]
                    y_pred = max(0, min(100, y_pred))

                    st.success(f"💧 Prediksi Akses Air Minum Terkelola: **{y_pred:.2f}%**")

                    # Visualisasi hasil
                    if y_pred >= 80:
                        color, kategori = "#4CAF50", "Tinggi"
                    elif y_pred >= 50:
                        color, kategori = "#FFC107", "Sedang"
                    else:
                        color, kategori = "#E74C3C", "Rendah"

                    chart_data = pd.DataFrame({
                        "Kategori": ["Prediksi Managed Drinking Water"],
                        "Persentase": [y_pred]
                    })

                    bar_chart = alt.Chart(chart_data).mark_bar(size=80, color=color).encode(
                        x=alt.X("Kategori:N", title=""),
                        y=alt.Y("Persentase:Q", title="Persentase (%)", scale=alt.Scale(domain=[0, 100])),
                        tooltip=["Persentase"]
                    )

                    st.altair_chart(bar_chart.properties(width=400, height=300))
                    st.info(f"Kategori hasil prediksi: **{kategori}** ({y_pred:.2f}%)")
