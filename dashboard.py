import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ==================================
# Konfigurasi Halaman Streamlit
# ==================================
st.set_page_config(
    page_title="Dashboard SDG 6 - Air Bersih",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Dashboard Analisis Data SDG 6: Air Bersih & Sanitasi")
st.markdown("---")

# ==================================
# Session State untuk menyimpan data
# ==================================
if 'datasets' not in st.session_state:
    st.session_state.datasets = {
        "Indeks SDG": None,
        "Air Minum Aman": None,
        "Kelayakan Air": None,
        "Polusi & Penyakit": None
    }

# ==================================
# Fungsi Loading Data dari Upload
# ==================================
def load_uploaded_file(uploaded_file):
    """Memuat dataset dari file yang diupload."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"⚠️ Error membaca file: {e}")
            return None
    return None

# ==================================
# Fungsi Assessing Data
# ==================================
def display_assessment(df, df_name):
    """Menampilkan hasil assessment data (Info, Describe, NaN, Duplikat)."""
    st.header(f"🔍 Assessing Data: {df_name}")

    st.subheader("1. Info Dataset")
    buffer = StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("2. Statistik Deskriptif")
    st.dataframe(df.describe())

    st.subheader("3. Missing Values (Nilai Kosong)")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.dataframe(missing_values[missing_values > 0].sort_values(ascending=False).to_frame(name='Jumlah NaN'))
    else:
        st.success("✅ Tidak ada missing values!")
    st.write(f"Total nilai NaN: {missing_values.sum()}")

    st.subheader("4. Duplikat Data")
    duplicates_count = df.duplicated().sum()
    st.write(f"Jumlah baris data duplikat: {duplicates_count}")

# ==================================
# Fungsi Cleaning Data
# ==================================
def clean_indexes(df_orig):
    """Membersihkan DataFrame Indeks SDG."""
    df = df_orig.copy()
    
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

    original_cols = df.columns.tolist()
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace('(', '', regex=False).str.replace(')', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace('___', '_', regex=False)
        .str.replace('__', '_', regex=False)
    )
    standardized_cols = df.columns.tolist()

    rename_map_std = {}
    original_map_keys_lower = {k.lower().strip() : v for k, v in rename_map.items()}

    for orig_col, std_col in zip(original_cols, standardized_cols):
        orig_col_lower_strip = orig_col.lower().strip()
        if orig_col_lower_strip in original_map_keys_lower:
            rename_map_std[std_col] = original_map_keys_lower[orig_col_lower_strip]

    df.rename(columns=rename_map_std, inplace=True)
    df.rename(columns={'country_code': 'code'}, inplace=True, errors='ignore')

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )
    df = df.loc[:,~df.columns.duplicated()]

    if 'country' in df.columns:
        df['country'] = df['country'].astype(str).str.title().str.strip()
    if 'region' in df.columns:
        df['region'] = df['region'].astype(str).str.lower().str.strip()

    if 'year' in df.columns:
        df = df[df['year'].between(2000, 2024)].copy()

    cols_to_fill = [v.lower() for v in rename_map.values()]
    cols_to_fill = list(set(cols_to_fill))

    if 'country' in df.columns:
        for col in cols_to_fill:
             if col in df.columns and df[col].isnull().any():
                df[col] = df.groupby('country')[col].transform(
                    lambda x: x.interpolate(method='linear').ffill().bfill()
                )

    df.drop_duplicates(inplace=True)
    return df

def clean_drinking_water(df_orig):
    """Membersihkan DataFrame Akses Air Minum Aman."""
    df = df_orig.copy()
    df.rename(columns={'Usage of safely managed drinking water services': 'managed_drinking_water'}, inplace=True)
    df.columns = (df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_'))
    if 'country' in df.columns:
        df['country'] = df['country'].astype(str).str.title().str.strip()
    if 'year' in df.columns:
        df = df[df['year'].between(2000, 2023)].copy()
    col_to_fill = 'managed_drinking_water'
    if col_to_fill in df.columns and 'country' in df.columns:
        df[col_to_fill] = df.groupby('country')[col_to_fill].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
    elif col_to_fill in df.columns:
         df[col_to_fill] = df[col_to_fill].interpolate(method='linear').ffill().bfill()
    df.drop_duplicates(inplace=True)
    return df

def clean_potability(df_orig):
    """Membersihkan DataFrame Kelayakan Air."""
    df = df_orig.copy()
    df.columns = (df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_'))
    cols_to_fill = ['ph', 'sulfate', 'trihalomethanes']
    for col in cols_to_fill:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def clean_pollution(df_orig):
    """Membersihkan DataFrame Polusi & Penyakit."""
    df = df_orig.copy()
    new_cols = df.columns.to_list()
    new_cols = [col.lower().strip() for col in new_cols]
    new_cols = [re.sub(r'\(.*\)', '', col).strip() for col in new_cols]
    new_cols = [col.replace('"', '') for col in new_cols]
    new_cols = [col.replace(' ', '_').replace('/', '_').replace('-', '_') for col in new_cols]
    new_cols = [re.sub(r'[^a-z0-9_]', '', col) for col in new_cols]
    
    seen = {}
    final_cols = []
    for i, col in enumerate(new_cols):
        if col in seen:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_cols.append(col)
    df.columns = final_cols

    if 'country' in df.columns:
        df['country'] = df['country'].astype(str).str.title().str.strip()
    if 'region' in df.columns:
        df['region'] = df['region'].astype(str).str.lower().str.strip()
    if 'water_source_type' in df.columns:
        df['water_source_type'] = df['water_source_type'].astype(str).str.lower().str.strip()
    if 'water_treatment_method' in df.columns:
        df['water_treatment_method'] = df['water_treatment_method'].astype(str).str.lower().str.strip()
    if 'year' in df.columns:
        df = df[df['year'].between(2000, 2024)].copy()
    df.drop_duplicates(inplace=True)
    return df

cleaning_functions = {
    "Indeks SDG": clean_indexes,
    "Air Minum Aman": clean_drinking_water,
    "Kelayakan Air": clean_potability,
    "Polusi & Penyakit": clean_pollution
}

# ==================================
# Sidebar Navigasi
# ==================================
st.sidebar.header("🧭 Dashboard Navigasi")

# Upload Files Section
st.sidebar.markdown("### 📤 Upload Dataset CSV")
st.sidebar.markdown("Upload file CSV Anda di bawah ini:")

dataset_mappings = {
    "Indeks SDG": "indexes.csv",
    "Air Minum Aman": "drinking_water.csv",
    "Kelayakan Air": "water_potability.csv",
    "Polusi & Penyakit": "water_pollution_disease.csv"
}

for dataset_name, expected_filename in dataset_mappings.items():
    uploaded_file = st.sidebar.file_uploader(
        f"📁 {dataset_name}",
        type=['csv'],
        key=f'upload_{dataset_name}',
        help=f"Expected: {expected_filename}"
    )
    
    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)
        if df is not None:
            st.session_state.datasets[dataset_name] = df
            st.sidebar.success(f"✅ {dataset_name} loaded!")

st.sidebar.markdown("---")

# Stage Selection
st.sidebar.markdown("### 🔄 Pilih Tahapan")
stage = st.sidebar.radio(
    "Tahapan Data Wrangling:",
    ("Gathering Data", "Assessing Data", "Cleaning Data"),
    key="stage_selection"
)

st.sidebar.markdown("---")

# Dataset Selection (for Assessing and Cleaning stages)
selected_df_name = None
if stage in ["Assessing Data", "Cleaning Data"]:
    # Filter hanya dataset yang sudah diupload
    available_datasets = [name for name, df in st.session_state.datasets.items() if df is not None]
    
    if available_datasets:
        selected_df_name = st.sidebar.selectbox(
            "📊 Pilih Dataset:",
            available_datasets,
            key="df_select"
        )
    else:
        st.sidebar.warning("⚠️ Upload dataset terlebih dahulu!")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** Upload semua dataset CSV di atas, lalu pilih tahapan analisis yang diinginkan.")

# ==================================
# Main Area Display Logic
# ==================================

# -----------------
# TAHAP GATHERING
# -----------------
if stage == "Gathering Data":
    st.header("📂 Gathering Data")
    st.markdown("### Status Upload Dataset")
    
    # Check upload status
    uploaded_count = sum(1 for df in st.session_state.datasets.values() if df is not None)
    total_datasets = len(st.session_state.datasets)
    
    if uploaded_count == 0:
        st.info("👈 Silakan upload file CSV di sidebar untuk memulai!")
    else:
        st.success(f"✅ {uploaded_count}/{total_datasets} dataset telah diupload")
    
    st.markdown("---")
    
    # Display uploaded datasets
    for name, df in st.session_state.datasets.items():
        st.subheader(f"Dataset: {name}")
        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jumlah Baris", df.shape[0])
            with col2:
                st.metric("Jumlah Kolom", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            with st.expander("👁️ Lihat Preview Data"):
                st.dataframe(df.head(10))
            
            with st.expander("📋 Lihat Nama Kolom"):
                st.write(df.columns.tolist())
        else:
            st.warning("⏳ Belum diupload")
        st.markdown("---")

# -----------------
# TAHAP ASSESSING
# -----------------
elif stage == "Assessing Data":
    st.header("📊 Assessing Data")
    
    if selected_df_name:
        df_to_assess = st.session_state.datasets[selected_df_name]
        if df_to_assess is not None:
            display_assessment(df_to_assess, selected_df_name)
        else:
            st.warning("❌ Dataset tidak ditemukan. Silakan upload terlebih dahulu.")
    else:
        st.info("👈 Upload dataset dan pilih di sidebar untuk melihat assessment.")

# -----------------
# TAHAP CLEANING
# -----------------
elif stage == "Cleaning Data":
    st.header("🧹 Cleaning Data")
    
    if selected_df_name:
        df_original = st.session_state.datasets[selected_df_name]
        
        if df_original is not None:
            cleaning_func = cleaning_functions.get(selected_df_name)
            
            if cleaning_func:
                st.subheader(f"Proses Cleaning untuk: {selected_df_name}")
                
                with st.spinner("🔄 Sedang membersihkan data..."):
                    try:
                        df_cleaned = cleaning_func(df_original)
                        st.success("✅ Cleaning berhasil!")
                    except Exception as e:
                        st.error(f"❌ Error saat cleaning: {e}")
                        df_cleaned = None
                
                if df_cleaned is not None:
                    st.markdown("---")
                    st.subheader("📊 Perbandingan Data")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📥 Data SEBELUM Cleaning")
                        st.write(f"**Ukuran:** {df_original.shape[0]} baris × {df_original.shape[1]} kolom")
                        
                        # Info
                        with st.expander("📄 Info Dataset"):
                            buffer_before = StringIO()
                            df_original.info(buf=buffer_before)
                            st.text(buffer_before.getvalue())
                        
                        # Missing Values
                        with st.expander("🔍 Missing Values"):
                            missing_before = df_original.isnull().sum()
                            if missing_before.sum() > 0:
                                st.dataframe(
                                    missing_before[missing_before > 0]
                                    .sort_values(ascending=False)
                                    .to_frame(name='Jumlah NaN')
                                )
                            else:
                                st.success("Tidak ada missing values")
                            st.write(f"**Total NaN:** {missing_before.sum()}")
                        
                        # Preview
                        with st.expander("👁️ Preview Data"):
                            st.dataframe(df_original.head(10))
                    
                    with col2:
                        st.markdown("### 📤 Data SETELAH Cleaning")
                        st.write(f"**Ukuran:** {df_cleaned.shape[0]} baris × {df_cleaned.shape[1]} kolom")
                        
                        # Info
                        with st.expander("📄 Info Dataset"):
                            buffer_after = StringIO()
                            df_cleaned.info(buf=buffer_after)
                            st.text(buffer_after.getvalue())
                        
                        # Missing Values
                        with st.expander("🔍 Missing Values"):
                            missing_after = df_cleaned.isnull().sum()
                            if missing_after.sum() > 0:
                                st.dataframe(
                                    missing_after[missing_after > 0]
                                    .sort_values(ascending=False)
                                    .to_frame(name='Jumlah NaN')
                                )
                            else:
                                st.success("✅ Tidak ada missing values!")
                            st.write(f"**Total NaN:** {missing_after.sum()}")
                        
                        # Preview
                        with st.expander("👁️ Preview Data"):
                            st.dataframe(df_cleaned.head(10))
                    
                    # Summary of Changes
                    st.markdown("---")
                    st.subheader("📈 Ringkasan Perubahan")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        rows_removed = df_original.shape[0] - df_cleaned.shape[0]
                        st.metric("Baris Dihapus", rows_removed, delta=f"{rows_removed} rows")
                    with col_b:
                        nan_before = df_original.isnull().sum().sum()
                        nan_after = df_cleaned.isnull().sum().sum()
                        st.metric("Missing Values", nan_after, delta=f"{nan_after - nan_before}")
                    with col_c:
                        dupes_removed = df_original.duplicated().sum() - df_cleaned.duplicated().sum()
                        st.metric("Duplikat Dihapus", dupes_removed)
                    
                    # Download Button
                    st.markdown("---")
                    csv = df_cleaned.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Data Cleaned (CSV)",
                        data=csv,
                        file_name=f"{selected_df_name.lower().replace(' ', '_')}_cleaned.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.warning(f"⚠️ Fungsi cleaning untuk '{selected_df_name}' belum tersedia.")
        else:
            st.warning("❌ Dataset tidak ditemukan. Silakan upload terlebih dahulu.")
    else:
        st.info("👈 Upload dataset dan pilih di sidebar untuk melakukan cleaning.")