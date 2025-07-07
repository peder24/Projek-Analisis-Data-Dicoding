import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Peminjaman Sepeda", layout="wide")

# Judul aplikasi
st.title("Analisis Peminjaman Sepeda")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    hour_df = pd.read_csv("hour.csv")  # Ganti dengan path file Anda
    day_df = pd.read_csv("day.csv")  # Ganti dengan path file Anda
    
    # Tambahkan label-label kategori
    hour_df['season_label'] = hour_df['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    day_df['season_label'] = day_df['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    
    hour_df['weathersit_label'] = hour_df['weathersit'].map({
        1: 'Clear', 
        2: 'Mist/Cloudy', 
        3: 'Light Rain/Snow', 
        4: 'Heavy Rain/Snow'
    })
    day_df['weathersit_label'] = day_df['weathersit'].map({
        1: 'Clear', 
        2: 'Mist/Cloudy', 
        3: 'Light Rain/Snow', 
        4: 'Heavy Rain/Snow'
    })
    
    hour_df['yr_label'] = hour_df['yr'].map({0: '2011', 1: '2012'})
    day_df['yr_label'] = day_df['yr'].map({0: '2011', 1: '2012'})
    
    hour_df['workingday_label'] = hour_df['workingday'].map({0: 'Weekend/Holiday', 1: 'Working Day'})
    day_df['workingday_label'] = day_df['workingday'].map({0: 'Weekend/Holiday', 1: 'Working Day'})
    
    hour_df['weekday_label'] = hour_df['weekday'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    day_df['weekday_label'] = day_df['weekday'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    # Tambahkan kolom untuk bulan
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    hour_df['month_name'] = hour_df['mnth'].map(month_names)
    day_df['month_name'] = day_df['mnth'].map(month_names)
    
    # Tambahkan kategori untuk variabel cuaca
    hour_df['temp_actual'] = hour_df['temp'] * 41  # Normalized to actual temperature
    hour_df['atemp_actual'] = hour_df['atemp'] * 50  # Normalized to actual temperature
    hour_df['hum_actual'] = hour_df['hum'] * 100  # Normalized to actual percentage
    hour_df['windspeed_actual'] = hour_df['windspeed'] * 67  # Normalized to actual wind speed
    
    day_df['temp_actual'] = day_df['temp'] * 41
    day_df['atemp_actual'] = day_df['atemp'] * 50
    day_df['hum_actual'] = day_df['hum'] * 100
    day_df['windspeed_actual'] = day_df['windspeed'] * 67
    
    # Kategori suhu - tambahkan ke kedua dataframe
    hour_df['temp_category'] = pd.cut(
        hour_df['temp_actual'], 
        bins=[0, 10, 20, 30, 41], 
        labels=['Cold (0-10°C)', 'Cool (10-20°C)', 'Warm (20-30°C)', 'Hot (30-41°C)']
    )
    
    day_df['temp_category'] = pd.cut(
        day_df['temp_actual'], 
        bins=[0, 10, 20, 30, 41], 
        labels=['Cold (0-10°C)', 'Cool (10-20°C)', 'Warm (20-30°C)', 'Hot (30-41°C)']
    )
    
    # Kategori kelembaban - tambahkan ke kedua dataframe
    hour_df['hum_category'] = pd.cut(
        hour_df['hum_actual'], 
        bins=[0, 25, 50, 75, 100], 
        labels=['Very Dry (0-25%)', 'Dry (25-50%)', 'Humid (50-75%)', 'Very Humid (75-100%)']
    )
    
    day_df['hum_category'] = pd.cut(
        day_df['hum_actual'], 
        bins=[0, 25, 50, 75, 100], 
        labels=['Very Dry (0-25%)', 'Dry (25-50%)', 'Humid (50-75%)', 'Very Humid (75-100%)']
    )
    
    # Kategori kecepatan angin - tambahkan ke kedua dataframe
    hour_df['windspeed_category'] = pd.cut(
        hour_df['windspeed_actual'], 
        bins=[0, 15, 30, 45, 67], 
        labels=['Calm (0-15)', 'Light Breeze (15-30)', 'Moderate Wind (30-45)', 'Strong Wind (45+)']
    )
    
    day_df['windspeed_category'] = pd.cut(
        day_df['windspeed_actual'], 
        bins=[0, 15, 30, 45, 67], 
        labels=['Calm (0-15)', 'Light Breeze (15-30)', 'Moderate Wind (30-45)', 'Strong Wind (45+)']
    )
    
    # Indeks kenyamanan - tambahkan ke kedua dataframe
    hour_df['comfort_index'] = (
        (1 - abs(hour_df['temp_actual'] - 25) / 35) * 50 +  # Suhu optimal sekitar 25°C
        (1 - hour_df['hum_actual'] / 100) * 30 +  # Kelembaban rendah lebih nyaman
        (1 - hour_df['windspeed_actual'] / 67) * 20  # Angin rendah lebih nyaman
    )
    hour_df['comfort_index'] = hour_df['comfort_index'].clip(0, 100)
    hour_df['comfort_category'] = pd.cut(
        hour_df['comfort_index'], 
        bins=[0, 25, 50, 75, 100], 
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )
    
    day_df['comfort_index'] = (
        (1 - abs(day_df['temp_actual'] - 25) / 35) * 50 +  # Suhu optimal sekitar 25°C
        (1 - day_df['hum_actual'] / 100) * 30 +  # Kelembaban rendah lebih nyaman
        (1 - day_df['windspeed_actual'] / 67) * 20  # Angin rendah lebih nyaman
    )
    day_df['comfort_index'] = day_df['comfort_index'].clip(0, 100)
    day_df['comfort_category'] = pd.cut(
        day_df['comfort_index'], 
        bins=[0, 25, 50, 75, 100], 
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )
    
    # Tambahkan kolom tanggal untuk filtering - cara sederhana yang lebih robust
    try:
        # Jika kolom dteday sudah dalam format datetime string
        day_df['date'] = pd.to_datetime(day_df['dteday'])
        hour_df['date'] = pd.to_datetime(hour_df['dteday'])
    except:
        # Alternatif jika tidak bisa langsung dikonversi
        st.warning("Kolom tanggal perlu dibuat manual. Menggunakan metode alternatif.")
        
        # Fungsi untuk membuat tanggal dari tahun, bulan, dan hari
        def create_date(row):
            year = 2011 if row['yr'] == 0 else 2012
            month = row['mnth']
            day = 1  # Default ke hari pertama bulan jika tidak ada info hari
            
            if 'dteday' in row and pd.notna(row['dteday']):
                try:
                    parts = row['dteday'].split('-')
                    if len(parts) >= 3:
                        day = int(parts[2])
                except:
                    pass
            
            return datetime(year, month, day)
        
        # Terapkan fungsi untuk membuat tanggal
        day_df['date'] = day_df.apply(create_date, axis=1)
        hour_df['date'] = hour_df.apply(create_date, axis=1)
    
    return hour_df, day_df

# Muat data
try:
    hour_df, day_df = load_data()
    
    # Sidebar untuk navigasi dan filter
    st.sidebar.title("Navigasi")
    
    # FITUR INTERAKTIF 1: Filter data berdasarkan rentang tanggal
    st.sidebar.header("Filter Data")
    min_date = day_df['date'].min().date()
    max_date = day_df['date'].max().date()
    
    # Filter tanggal dengan date_input
    date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Pastikan pengguna memilih rentang tanggal yang valid
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask_day = (day_df['date'].dt.date >= start_date) & (day_df['date'].dt.date <= end_date)
        filtered_day_df = day_df[mask_day]
        
        mask_hour = (hour_df['date'].dt.date >= start_date) & (hour_df['date'].dt.date <= end_date)
        filtered_hour_df = hour_df[mask_hour]
    else:
        filtered_day_df = day_df
        filtered_hour_df = hour_df
        st.sidebar.warning("Pilih rentang tanggal yang valid")
    
    # FITUR INTERAKTIF 2: Filter berdasarkan musim
    selected_seasons = st.sidebar.multiselect(
        "Pilih Musim",
        options=day_df['season_label'].unique(),
        default=day_df['season_label'].unique()
    )
    
    if selected_seasons:
        filtered_day_df = filtered_day_df[filtered_day_df['season_label'].isin(selected_seasons)]
        filtered_hour_df = filtered_hour_df[filtered_hour_df['season_label'].isin(selected_seasons)]
    
    # FITUR INTERAKTIF 3: Filter berdasarkan kondisi cuaca
    selected_weather = st.sidebar.multiselect(
        "Pilih Kondisi Cuaca",
        options=day_df['weathersit_label'].unique(),
        default=day_df['weathersit_label'].unique()
    )
    
    if selected_weather:
        filtered_day_df = filtered_day_df[filtered_day_df['weathersit_label'].isin(selected_weather)]
        filtered_hour_df = filtered_hour_df[filtered_hour_df['weathersit_label'].isin(selected_weather)]
    
    # FITUR INTERAKTIF 4: Filter berdasarkan tipe hari
    workingday_options = {
        "Semua": None,
        "Hari Kerja": 1,
        "Akhir Pekan/Libur": 0
    }
    selected_workingday_label = st.sidebar.radio(
        "Pilih Tipe Hari",
        options=list(workingday_options.keys()),
        index=0
    )
    
    selected_workingday = workingday_options[selected_workingday_label]
    if selected_workingday is not None:
        filtered_day_df = filtered_day_df[filtered_day_df['workingday'] == selected_workingday]
        filtered_hour_df = filtered_hour_df[filtered_hour_df['workingday'] == selected_workingday]
    
    # Informasi tentang filter yang diterapkan
    st.sidebar.markdown("---")
    st.sidebar.subheader("Informasi Filter")
    st.sidebar.info(f"""
    Data yang ditampilkan:
    - Periode: {start_date} hingga {end_date}
    - Musim: {', '.join(selected_seasons)}
    - Cuaca: {', '.join(selected_weather)}
    - Tipe Hari: {selected_workingday_label}
    
    Total data: {len(filtered_day_df)} hari
    """)
    
    # Menu navigasi utama
    analysis_type = st.sidebar.radio(
        "Pilih Jenis Analisis:",
        ["Pola Temporal", "Pengaruh Cuaca"]
    )
    
    if analysis_type == "Pola Temporal":
        st.header("Analisis Pola Temporal Peminjaman Sepeda")
        
        # Tampilkan subtab untuk berbagai visualisasi pola temporal
        temporal_tabs = st.tabs(["Pola Musiman", "Pola Mingguan", "Pola Tahunan"])
        
        with temporal_tabs[0]:  # Pola Musiman
            st.subheader("Pola Peminjaman Berdasarkan Musim dan Waktu")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Persiapkan data untuk visualisasi
                seasonal_hourly_pattern = filtered_hour_df.groupby(['season_label', 'hr']).agg({
                    'cnt': 'mean',
                    'casual': 'mean',
                    'registered': 'mean'
                }).reset_index()
                
                seasonal_hourly_pivot = seasonal_hourly_pattern.pivot(index='hr', columns='season_label', values='cnt')
                
                # Plot 1: Pola peminjaman per jam berdasarkan musim
                fig, ax = plt.subplots(figsize=(10, 6))
                seasonal_hourly_pivot.plot(ax=ax)
                plt.title('Pola Peminjaman Sepeda per Jam Berdasarkan Musim', fontsize=12)
                plt.xlabel('Jam', fontsize=10)
                plt.ylabel('Rata-rata Jumlah Peminjaman', fontsize=10)
                plt.xticks(range(0, 24))
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Plot 2: Distribusi peminjaman berdasarkan musim
                fig, ax = plt.subplots(figsize=(10, 6))
                season_totals = filtered_day_df.groupby('season_label').agg({
                    'cnt': 'sum',
                    'casual': 'sum',
                    'registered': 'sum'
                })
                season_totals.plot(kind='bar', ax=ax)
                plt.title('Total Peminjaman Berdasarkan Musim', fontsize=12)
                plt.xlabel('Musim', fontsize=10)
                plt.ylabel('Total Peminjaman', fontsize=10)
                plt.xticks(rotation=45)
                plt.legend(title='Tipe Pengguna')
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Plot 3: Perbandingan pengguna kasual vs terdaftar per musim
                fig, ax = plt.subplots(figsize=(10, 6))
                seasonal_user_ratio = filtered_day_df.groupby('season_label').agg({
                    'casual': 'sum',
                    'registered': 'sum'
                })
                seasonal_user_ratio['total'] = seasonal_user_ratio['casual'] + seasonal_user_ratio['registered']
                seasonal_user_ratio['casual_pct'] = seasonal_user_ratio['casual'] / seasonal_user_ratio['total'] * 100
                seasonal_user_ratio['registered_pct'] = seasonal_user_ratio['registered'] / seasonal_user_ratio['total'] * 100

                user_type_data = pd.DataFrame({
                    'Casual': seasonal_user_ratio['casual_pct'],
                    'Registered': seasonal_user_ratio['registered_pct']
                })
                user_type_data.plot(kind='bar', stacked=True, ax=ax)
                plt.title('Persentase Tipe Pengguna Berdasarkan Musim', fontsize=12)
                plt.xlabel('Musim', fontsize=10)
                plt.ylabel('Persentase (%)', fontsize=10)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col4:
                # Plot 4: Heatmap pola peminjaman per jam dan musim
                fig, ax = plt.subplots(figsize=(10, 6))
                seasonal_hour_pivot = seasonal_hourly_pattern.pivot(index='hr', columns='season_label', values='cnt')
                sns.heatmap(seasonal_hour_pivot, cmap='YlOrRd', annot=False, fmt='.0f', 
                           cbar_kws={'label': 'Rata-rata Peminjaman'}, ax=ax)
                plt.title('Heatmap Peminjaman Sepeda: Jam vs Musim', fontsize=12)
                plt.xlabel('Musim', fontsize=10)
                plt.ylabel('Jam', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
        
        with temporal_tabs[1]:  # Pola Mingguan
            st.subheader("Pola Peminjaman Berdasarkan Hari dalam Seminggu")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pola mingguan berdasarkan musim
                fig, ax = plt.subplots(figsize=(10, 6))
                weekly_seasonal_pattern = filtered_day_df.groupby(['season_label', 'weekday_label']).agg({
                    'cnt': 'mean',
                }).reset_index()
                
                weekly_seasonal_pivot = weekly_seasonal_pattern.pivot(index='weekday_label', columns='season_label', values='cnt')
                
                # Mengurutkan hari dalam seminggu dengan benar
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_seasonal_pivot = weekly_seasonal_pivot.reindex(day_order)
                
                weekly_seasonal_pivot.plot(kind='bar', ax=ax)
                plt.title('Rata-rata Peminjaman Sepeda per Hari Berdasarkan Musim', fontsize=12)
                plt.xlabel('Hari dalam Seminggu', fontsize=10)
                plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                plt.legend(title='Musim')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Perbandingan hari kerja vs akhir pekan
                fig, ax = plt.subplots(figsize=(10, 6))
                workday_seasonal = filtered_day_df.groupby(['season_label', 'workingday_label']).agg({
                    'cnt': 'mean'
                }).reset_index()
                workday_seasonal_pivot = workday_seasonal.pivot(index='workingday_label', columns='season_label', values='cnt')
                workday_seasonal_pivot.plot(kind='bar', ax=ax)
                plt.title('Rata-rata Peminjaman: Hari Kerja vs Akhir Pekan per Musim', fontsize=12)
                plt.xlabel('Tipe Hari', fontsize=10)
                plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                plt.xticks(rotation=0)
                plt.grid(True, alpha=0.3, axis='y')
                plt.legend(title='Musim')
                plt.tight_layout()
                st.pyplot(fig)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Pola jam berdasarkan tipe hari (hari kerja vs akhir pekan)
                fig, ax = plt.subplots(figsize=(10, 6))
                workday_hourly = filtered_hour_df.groupby(['workingday', 'hr']).agg({
                    'cnt': 'mean'
                }).reset_index()
                
                # Cek apakah ada data untuk kedua tipe hari
                if 0 in workday_hourly['workingday'].values and 1 in workday_hourly['workingday'].values:
                    workday_hourly_pivot = workday_hourly.pivot(index='hr', columns='workingday', values='cnt')
                    workday_hourly_pivot.columns = ['Weekend/Holiday', 'Working Day']
                    workday_hourly_pivot.plot(kind='line', ax=ax)
                    plt.title('Pola Peminjaman per Jam: Hari Kerja vs Akhir Pekan', fontsize=12)
                    plt.xlabel('Jam', fontsize=10)
                    plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                    plt.xticks(range(0, 24))
                    plt.grid(True, alpha=0.3)
                    plt.legend(title='Tipe Hari')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Data tidak cukup untuk membandingkan pola per jam berdasarkan tipe hari.")
            
            with col4:
                # Komposisi pengguna berdasarkan tipe hari
                fig, ax = plt.subplots(figsize=(10, 6))
                user_day_type = filtered_hour_df.groupby('workingday').agg({
                    'casual': 'sum',
                    'registered': 'sum'
                })
                
                # Pastikan ada data untuk kedua tipe hari
                if len(user_day_type) == 2:
                    user_day_type['total'] = user_day_type['casual'] + user_day_type['registered']
                    user_day_type['casual_pct'] = user_day_type['casual'] / user_day_type['total'] * 100
                    user_day_type['registered_pct'] = user_day_type['registered'] / user_day_type['total'] * 100

                    day_type_labels = ['Weekend/Holiday', 'Working Day']
                    user_comp_data = pd.DataFrame({
                        'Casual': user_day_type['casual_pct'].values,
                        'Registered': user_day_type['registered_pct'].values
                    }, index=day_type_labels)

                    user_comp_data.plot(kind='bar', stacked=True, ax=ax)
                    plt.title('Komposisi Pengguna Berdasarkan Tipe Hari', fontsize=12)
                    plt.xlabel('Tipe Hari', fontsize=10)
                    plt.ylabel('Persentase (%)', fontsize=10)
                    plt.xticks(rotation=0)
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.legend(title='Tipe Pengguna')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Data tidak cukup untuk membandingkan komposisi pengguna berdasarkan tipe hari.")
        
        with temporal_tabs[2]:  # Pola Tahunan
            st.subheader("Analisis Pertumbuhan dan Tren Tahunan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Perbandingan peminjaman antara 2011 dan 2012
                fig, ax = plt.subplots(figsize=(10, 6))
                yearly_comparison = filtered_day_df.groupby('yr_label').agg({
                    'cnt': 'mean',
                    'casual': 'mean',
                    'registered': 'mean'
                })
                
                if len(yearly_comparison) > 0:
                    yearly_comparison.plot(kind='bar', ax=ax)
                    plt.title('Rata-rata Peminjaman Harian: 2011 vs 2012', fontsize=12)
                    plt.xlabel('Tahun', fontsize=10)
                    plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.legend(title='Tipe Pengguna')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Tidak ada data untuk perbandingan tahunan.")
            
            with col2:
                # Tren bulanan per tahun
                fig, ax = plt.subplots(figsize=(10, 6))
                monthly_yearly_trend = filtered_day_df.groupby(['yr_label', 'month_name']).agg({
                    'cnt': 'mean'
                }).reset_index()
                
                if not monthly_yearly_trend.empty:
                    monthly_yearly_pivot = monthly_yearly_trend.pivot(index='month_name', columns='yr_label', values='cnt')
                    
                    # Mengurutkan bulan secara kronologis
                    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                                'July', 'August', 'September', 'October', 'November', 'December']
                    monthly_yearly_pivot = monthly_yearly_pivot.reindex(month_order)
                    monthly_yearly_pivot.plot(kind='line', marker='o', ax=ax)
                    plt.title('Tren Peminjaman Bulanan: 2011 vs 2012', fontsize=12)
                    plt.xlabel('Bulan', fontsize=10)
                    plt.ylabel('Rata-rata Peminjaman Harian', fontsize=10)
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    plt.legend(title='Tahun')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Tidak ada data yang cukup untuk menampilkan tren bulanan per tahun.")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Pertumbuhan berdasarkan musim
                fig, ax = plt.subplots(figsize=(10, 6))
                season_yearly_trend = filtered_day_df.groupby(['yr_label', 'season_label']).agg({
                    'cnt': 'mean'
                }).reset_index()
                
                if not season_yearly_trend.empty:
                    season_yearly_pivot = season_yearly_trend.pivot(index='season_label', columns='yr_label', values='cnt')
                    
                    # Hanya lakukan perhitungan growth jika ada data untuk kedua tahun
                    if '2011' in season_yearly_pivot.columns and '2012' in season_yearly_pivot.columns:
                        season_yearly_pivot['growth_pct'] = (season_yearly_pivot['2012'] / season_yearly_pivot['2011'] - 1) * 100

                        growth_data = pd.DataFrame({
                            'Growth (%)': season_yearly_pivot['growth_pct']
                        })
                        growth_data.plot(kind='bar', color='green', ax=ax)
                        plt.title('Pertumbuhan Peminjaman per Musim (2011 ke 2012)', fontsize=12)
                        plt.xlabel('Musim', fontsize=10)
                        plt.ylabel('Pertumbuhan (%)', fontsize=10)
                        plt.xticks(rotation=45)
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Data tidak cukup untuk menghitung pertumbuhan tahunan. Pilih rentang tanggal yang mencakup kedua tahun.")
                else:
                    st.warning("Tidak ada data yang cukup untuk menampilkan pertumbuhan berdasarkan musim.")
            
            with col4:
                # Pertumbuhan berdasarkan tipe hari
                fig, ax = plt.subplots(figsize=(10, 6))
                daytype_yearly_trend = filtered_day_df.groupby(['yr_label', 'workingday_label']).agg({
                    'cnt': 'mean'
                }).reset_index()
                
                if not daytype_yearly_trend.empty:
                    daytype_yearly_pivot = daytype_yearly_trend.pivot(index='workingday_label', columns='yr_label', values='cnt')
                    
                    # Hanya lakukan perhitungan growth jika ada data untuk kedua tahun
                    if '2011' in daytype_yearly_pivot.columns and '2012' in daytype_yearly_pivot.columns:
                        daytype_yearly_pivot['growth_pct'] = (daytype_yearly_pivot['2012'] / daytype_yearly_pivot['2011'] - 1) * 100

                        daytype_growth_data = pd.DataFrame({
                            'Growth (%)': daytype_yearly_pivot['growth_pct']
                        })
                        daytype_growth_data.plot(kind='bar', color='purple', ax=ax)
                        plt.title('Pertumbuhan Peminjaman per Tipe Hari (2011 ke 2012)', fontsize=12)
                        plt.xlabel('Tipe Hari', fontsize=10)
                        plt.ylabel('Pertumbuhan (%)', fontsize=10)
                        plt.xticks(rotation=0)
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Data tidak cukup untuk menghitung pertumbuhan berdasarkan tipe hari. Pilih rentang tanggal yang mencakup kedua tahun.")
                else:
                    st.warning("Tidak ada data yang cukup untuk menampilkan pertumbuhan berdasarkan tipe hari.")
    
    else:  # Pengaruh Cuaca
        st.header("Analisis Pengaruh Cuaca Terhadap Peminjaman Sepeda")
        
        # FITUR INTERAKTIF 5: Filter rentang suhu untuk analisis
        st.sidebar.header("Filter Parameter Cuaca")
        
        # Slider untuk rentang suhu
        temp_min = float(filtered_day_df['temp_actual'].min())
        temp_max = float(filtered_day_df['temp_actual'].max())
        
        if temp_min != temp_max:  # Pastikan ada rentang nilai yang valid
            temp_range = st.sidebar.slider(
                "Rentang Suhu (°C)",
                min_value=temp_min,
                max_value=temp_max,
                value=(temp_min, temp_max)
            )
            
            # Terapkan filter suhu
            filtered_day_df = filtered_day_df[(filtered_day_df['temp_actual'] >= temp_range[0]) & 
                                            (filtered_day_df['temp_actual'] <= temp_range[1])]
            filtered_hour_df = filtered_hour_df[(filtered_hour_df['temp_actual'] >= temp_range[0]) & 
                                              (filtered_hour_df['temp_actual'] <= temp_range[1])]
        
        # Slider untuk rentang kelembaban
        hum_min = float(filtered_day_df['hum_actual'].min())
        hum_max = float(filtered_day_df['hum_actual'].max())
        
        if hum_min != hum_max:  # Pastikan ada rentang nilai yang valid
            humidity_range = st.sidebar.slider(
                "Rentang Kelembaban (%)",
                min_value=hum_min,
                max_value=hum_max,
                value=(hum_min, hum_max)
            )
            
            # Terapkan filter kelembaban
            filtered_day_df = filtered_day_df[(filtered_day_df['hum_actual'] >= humidity_range[0]) & 
                                             (filtered_day_df['hum_actual'] <= humidity_range[1])]
            filtered_hour_df = filtered_hour_df[(filtered_hour_df['hum_actual'] >= humidity_range[0]) & 
                                               (filtered_hour_df['hum_actual'] <= humidity_range[1])]
        
        # Tampilkan subtab untuk berbagai visualisasi pengaruh cuaca
        weather_tabs = st.tabs(["Kondisi Cuaca", "Faktor Cuaca", "Interaksi Cuaca"])
        
        with weather_tabs[0]:  # Kondisi Cuaca
            st.subheader("Peminjaman Berdasarkan Kondisi Cuaca")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Peminjaman berdasarkan kondisi cuaca
                if not filtered_day_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    weather_totals = filtered_day_df.groupby('weathersit_label').agg({
                        'cnt': 'mean',
                        'casual': 'mean',
                        'registered': 'mean'
                    })
                    
                    if not weather_totals.empty:
                        weather_totals.plot(kind='bar', ax=ax)
                        plt.title('Rata-rata Peminjaman Berdasarkan Kondisi Cuaca', fontsize=12)
                        plt.xlabel('Kondisi Cuaca', fontsize=10)
                        plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                        plt.xticks(rotation=45)
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.legend(title='Tipe Pengguna')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Tidak ada data yang cukup untuk visualisasi peminjaman berdasarkan kondisi cuaca.")
                else:
                    st.warning("Tidak ada data yang memenuhi kriteria filter.")
            
            with col2:
                # Pola peminjaman per jam berdasarkan kondisi cuaca
                if not filtered_hour_df.empty:
                    weather_impact = filtered_hour_df.groupby(['weathersit_label', 'hr']).agg({
                        'cnt': 'mean',
                    }).reset_index()
                    
                    if not weather_impact.empty and len(weather_impact['weathersit_label'].unique()) > 0:
                        weather_hour_pivot = weather_impact.pivot(index='hr', columns='weathersit_label', values='cnt')
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.lineplot(data=weather_hour_pivot, ax=ax)
                        plt.title('Pola Peminjaman per Jam Berdasarkan Kondisi Cuaca', fontsize=12)
                        plt.xlabel('Jam', fontsize=10)
                        plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                        plt.xticks(range(0, 24))
                        plt.grid(True, alpha=0.3)
                        plt.legend(title='Kondisi Cuaca')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Tidak ada data yang cukup untuk visualisasi pola per jam berdasarkan kondisi cuaca.")
                else:
                    st.warning("Tidak ada data yang memenuhi kriteria filter.")
            
            # Sensitivitas segmen pengguna terhadap cuaca
            if not filtered_day_df.empty and 'Clear' in filtered_day_df['weathersit_label'].values:
                fig, ax = plt.subplots(figsize=(10, 6))
                weather_sensitivity = filtered_day_df.groupby('weathersit_label').agg({
                    'casual': 'mean',
                    'registered': 'mean'
                })
                
                # Normalisasi berdasarkan nilai 'Clear'
                clear_casual = weather_sensitivity.loc['Clear', 'casual']
                clear_registered = weather_sensitivity.loc['Clear', 'registered']
                
                # Hanya lakukan perhitungan jika nilai referensi tidak nol
                if clear_casual > 0 and clear_registered > 0:
                    weather_sensitivity['casual_norm'] = (weather_sensitivity['casual'] / clear_casual) * 100
                    weather_sensitivity['registered_norm'] = (weather_sensitivity['registered'] / clear_registered) * 100
                    
                    weather_sensitivity[['casual_norm', 'registered_norm']].rename(
                        columns={'casual_norm': 'Casual', 'registered_norm': 'Registered'}
                    ).plot(kind='bar', ax=ax)
                    
                    plt.title('Sensitivitas Segmen Pengguna Terhadap Cuaca', fontsize=12)
                    plt.xlabel('Kondisi Cuaca', fontsize=10)
                    plt.ylabel('Persentase dari Cuaca Cerah (%)', fontsize=10)
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.legend(title='Tipe Pengguna')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Nilai referensi untuk cuaca cerah terlalu kecil untuk perhitungan yang bermakna.")
            else:
                st.warning("Data tidak cukup untuk menghitung sensitivitas cuaca. Pastikan filter mencakup hari dengan cuaca cerah.")
        
        with weather_tabs[1]:  # Faktor Cuaca
            st.subheader("Pengaruh Faktor Cuaca (Suhu, Kelembaban, Angin)")
            
            # FITUR INTERAKTIF 6: Pilih tipe pengguna untuk analisis
            user_type = st.radio(
                "Pilih Tipe Pengguna untuk Analisis",
                ["Semua Pengguna", "Pengguna Kasual", "Pengguna Terdaftar"],
                horizontal=True
            )
            
            # Tentukan kolom yang akan dianalisis berdasarkan pilihan pengguna
            if user_type == "Pengguna Kasual":
                count_column = 'casual'
                title_suffix = "Pengguna Kasual"
            elif user_type == "Pengguna Terdaftar":
                count_column = 'registered'
                title_suffix = "Pengguna Terdaftar"
            else:
                count_column = 'cnt'
                title_suffix = "Semua Pengguna"
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Peminjaman berdasarkan kategori suhu
                if not filtered_hour_df.empty:
                    temp_impact = filtered_hour_df.groupby('temp_category').agg({
                        count_column: 'mean'
                    })
                    
                    if not temp_impact.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        temp_impact.plot(kind='bar', ax=ax, color='tab:red')
                        plt.title(f'Rata-rata Peminjaman Berdasarkan Kategori Suhu ({title_suffix})', fontsize=12)
                        plt.xlabel('Kategori Suhu', fontsize=10)
                        plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                        plt.xticks(rotation=45)
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Tidak ada data yang cukup untuk visualisasi pengaruh suhu.")
                else:
                    st.warning("Tidak ada data yang memenuhi kriteria filter.")
                
                # Peminjaman berdasarkan kategori kelembaban
                if not filtered_hour_df.empty:
                    humidity_impact = filtered_hour_df.groupby('hum_category').agg({
                        count_column: 'mean'
                    })
                    
                    if not humidity_impact.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        humidity_impact.plot(kind='bar', ax=ax, color='tab:blue')
                        plt.title(f'Rata-rata Peminjaman Berdasarkan Kategori Kelembaban ({title_suffix})', fontsize=12)
                        plt.xlabel('Kategori Kelembaban', fontsize=10)
                        plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                        plt.xticks(rotation=45)
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Tidak ada data yang cukup untuk visualisasi pengaruh kelembaban.")
                else:
                    st.warning("Tidak ada data yang memenuhi kriteria filter.")
            
            with col2:
                # Peminjaman berdasarkan kategori kecepatan angin
                if not filtered_hour_df.empty:
                    wind_impact = filtered_hour_df.groupby('windspeed_category').agg({
                        count_column: 'mean'
                    })
                    
                    if not wind_impact.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        wind_impact.plot(kind='bar', ax=ax, color='tab:green')
                        plt.title(f'Rata-rata Peminjaman Berdasarkan Kategori Kecepatan Angin ({title_suffix})', fontsize=12)
                        plt.xlabel('Kategori Kecepatan Angin', fontsize=10)
                        plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                        plt.xticks(rotation=45)
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Tidak ada data yang cukup untuk visualisasi pengaruh kecepatan angin.")
                else:
                    st.warning("Tidak ada data yang memenuhi kriteria filter.")
                
                # Peminjaman berdasarkan indeks kenyamanan
                if not filtered_hour_df.empty:
                    comfort_analysis = filtered_hour_df.groupby('comfort_category').agg({
                        count_column: 'mean'
                    })
                    
                    if not comfort_analysis.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        comfort_analysis.plot(kind='bar', ax=ax, color='tab:purple')
                        plt.title(f'Peminjaman Berdasarkan Indeks Kenyamanan ({title_suffix})', fontsize=12)
                        plt.xlabel('Kategori Kenyamanan', fontsize=10)
                        plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                        plt.xticks(rotation=45)
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Tidak ada data yang cukup untuk visualisasi berdasarkan indeks kenyamanan.")
                else:
                    st.warning("Tidak ada data yang memenuhi kriteria filter.")
            
            # Interaksi antara suhu dan kelembaban
            if not filtered_hour_df.empty:
                temp_hum_interaction = filtered_hour_df.groupby(['temp_category', 'hum_category']).agg({
                    count_column: 'mean'
                }).reset_index()
                
                if not temp_hum_interaction.empty:
                    temp_hum_pivot = temp_hum_interaction.pivot(index='temp_category', columns='hum_category', values=count_column)
                    
                    if not temp_hum_pivot.empty:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.heatmap(temp_hum_pivot, cmap='YlOrRd', annot=True, fmt='.0f', 
                                  cbar_kws={'label': f'Rata-rata Peminjaman ({title_suffix})'}, ax=ax)
                        plt.title(f'Interaksi Suhu dan Kelembaban Terhadap Peminjaman ({title_suffix})', fontsize=12)
                        plt.xlabel('Kategori Kelembaban', fontsize=10)
                        plt.ylabel('Kategori Suhu', fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Tidak ada data yang cukup untuk visualisasi interaksi suhu dan kelembaban.")
                else:
                    st.warning("Tidak ada data yang cukup untuk pengelompokan berdasarkan suhu dan kelembaban.")
            else:
                st.warning("Tidak ada data yang memenuhi kriteria filter.")
        
        with weather_tabs[2]:  # Interaksi Cuaca
            st.subheader("Interaksi antara Cuaca, Musim, dan Tipe Hari")
            
            # FITUR INTERAKTIF 7: Pilih parameter untuk perbandingan
            comparison_param = st.selectbox(
                "Pilih Parameter untuk Analisis Interaksi",
                ["Kondisi Cuaca", "Kategori Suhu", "Kategori Kelembaban", "Kategori Kecepatan Angin"]
            )
            
            # Tentukan kolom yang akan digunakan berdasarkan parameter yang dipilih
            if comparison_param == "Kondisi Cuaca":
                param_column = 'weathersit_label'
                title_part = "Cuaca"
            elif comparison_param == "Kategori Suhu":
                param_column = 'temp_category'
                title_part = "Suhu"
            elif comparison_param == "Kategori Kelembaban":
                param_column = 'hum_category'
                title_part = "Kelembaban"
            else:
                param_column = 'windspeed_category'
                title_part = "Kecepatan Angin"
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pengaruh parameter pada berbagai musim
                if not filtered_day_df.empty:
                    season_param_interaction = filtered_day_df.groupby(['season_label', param_column]).agg({
                        'cnt': 'mean'
                    }).reset_index()
                    
                    if not season_param_interaction.empty:
                        season_param_pivot = season_param_interaction.pivot(index='season_label', columns=param_column, values='cnt')
                        
                        if not season_param_pivot.empty:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            season_param_pivot.plot(kind='bar', ax=ax)
                            plt.title(f'Pengaruh {title_part} pada Berbagai Musim', fontsize=12)
                            plt.xlabel('Musim', fontsize=10)
                            plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                            plt.xticks(rotation=45)
                            plt.grid(True, alpha=0.3, axis='y')
                            plt.legend(title=comparison_param)
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning(f"Tidak ada data yang cukup untuk visualisasi pengaruh {title_part} pada musim.")
                    else:
                        st.warning(f"Tidak ada data yang cukup untuk pengelompokan berdasarkan musim dan {title_part}.")
                else:
                    st.warning("Tidak ada data yang memenuhi kriteria filter.")
            
            with col2:
                # Pengaruh parameter pada hari kerja vs akhir pekan
                if not filtered_day_df.empty:
                    workday_param_interaction = filtered_day_df.groupby(['workingday_label', param_column]).agg({
                        'cnt': 'mean'
                    }).reset_index()
                    
                    if not workday_param_interaction.empty:
                        workday_param_pivot = workday_param_interaction.pivot(index='workingday_label', columns=param_column, values='cnt')
                        
                        if not workday_param_pivot.empty:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            workday_param_pivot.plot(kind='bar', ax=ax)
                            plt.title(f'Pengaruh {title_part} pada Hari Kerja vs Akhir Pekan', fontsize=12)
                            plt.xlabel('Tipe Hari', fontsize=10)
                            plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                            plt.xticks(rotation=0)
                            plt.grid(True, alpha=0.3, axis='y')
                            plt.legend(title=comparison_param)
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning(f"Tidak ada data yang cukup untuk visualisasi pengaruh {title_part} pada tipe hari.")
                    else:
                        st.warning(f"Tidak ada data yang cukup untuk pengelompokan berdasarkan tipe hari dan {title_part}.")
                else:
                    st.warning("Tidak ada data yang memenuhi kriteria filter.")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Pengaruh parameter pada berbagai bulan
                if not filtered_day_df.empty:
                    month_param_interaction = filtered_day_df.groupby(['month_name', param_column]).agg({
                        'cnt': 'mean'
                    }).reset_index()
                    
                    if not month_param_interaction.empty:
                        # Mengurutkan bulan secara kronologis
                        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                                    'July', 'August', 'September', 'October', 'November', 'December']
                        
                        # Pivot dan plot
                        month_param_pivot = month_param_interaction.pivot(index='month_name', columns=param_column, values='cnt')
                        
                        if not month_param_pivot.empty:
                            # Reindex hanya jika semua bulan ada dalam data
                            month_param_pivot = month_param_pivot.reindex([m for m in month_order if m in month_param_pivot.index])
                            
                            # Plot sebagai heatmap untuk visualisasi yang lebih baik
                            fig, ax = plt.subplots(figsize=(12, 8))
                            sns.heatmap(month_param_pivot, cmap='YlOrRd', annot=True, fmt='.0f', 
                                      cbar_kws={'label': 'Rata-rata Peminjaman'}, ax=ax)
                            plt.title(f'Interaksi Bulan dan {title_part} Terhadap Peminjaman', fontsize=12)
                            plt.xlabel(comparison_param, fontsize=10)
                            plt.ylabel('Bulan', fontsize=10)
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning(f"Tidak ada data yang cukup untuk visualisasi interaksi bulan dan {title_part}.")
                    else:
                        st.warning(f"Tidak ada data yang cukup untuk pengelompokan berdasarkan bulan dan {title_part}.")
                else:
                    st.warning("Tidak ada data yang memenuhi kriteria filter.")
            
            with col4:
                # Korelasi antara variabel cuaca dan peminjaman
                if not filtered_hour_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    weather_corr = filtered_hour_df[['temp_actual', 'atemp_actual', 'hum_actual', 'windspeed_actual', 'cnt', 'casual', 'registered']].corr()
                    mask = np.triu(np.ones_like(weather_corr, dtype=bool))
                    sns.heatmap(weather_corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
                              cbar_kws={'label': 'Korelasi'}, ax=ax)
                    plt.title('Korelasi antara Variabel Cuaca dan Peminjaman', fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Tidak ada data yang memenuhi kriteria filter untuk analisis korelasi.")
            
            # Tambahan: Analisis cuaca berdasarkan segmen pengguna
            st.subheader("Analisis Cuaca Berdasarkan Segmen Pengguna")
            
            # Perbandingan sensitivitas pengguna kasual vs terdaftar terhadap cuaca
            if not filtered_hour_df.empty:
                casual_weather = filtered_hour_df.groupby(param_column)['casual'].mean()
                registered_weather = filtered_hour_df.groupby(param_column)['registered'].mean()
                
                # Hanya lakukan perhitungan jika ada data
                if not casual_weather.empty and not registered_weather.empty:
                    # Periksa apakah nilai maksimum lebih besar dari nol untuk menghindari pembagian dengan nol
                    casual_max = casual_weather.max()
                    registered_max = registered_weather.max()
                    
                    if casual_max > 0 and registered_max > 0:
                        casual_normalized = casual_weather / casual_max * 100
                        registered_normalized = registered_weather / registered_max * 100
                        
                        weather_comparison = pd.DataFrame({
                            'Casual': casual_normalized,
                            'Registered': registered_normalized
                        })
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        weather_comparison.plot(kind='bar', ax=ax)
                        plt.title(f'Sensitivitas Relatif Pengguna Kasual vs Terdaftar Terhadap {title_part}', fontsize=12)
                        plt.xlabel(comparison_param, fontsize=10)
                        plt.ylabel('Persentase dari Nilai Maksimum (%)', fontsize=10)
                        plt.xticks(rotation=45)
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.legend(title='Tipe Pengguna')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Nilai maksimum terlalu kecil untuk perhitungan yang bermakna.")
                else:
                    st.warning("Tidak cukup data untuk membuat perbandingan sensitivitas pengguna.")
            else:
                st.warning("Tidak ada data yang memenuhi kriteria filter.")

    # Tampilkan ringkasan data
    st.sidebar.header("Informasi Dataset")
    if st.sidebar.checkbox("Tampilkan Statistik Dasar"):
        if not filtered_day_df.empty:
            st.sidebar.subheader("Statistik Data Harian")
            st.sidebar.dataframe(filtered_day_df[['cnt', 'casual', 'registered', 'temp_actual', 'hum_actual', 'windspeed_actual']].describe())
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("Jumlah Data per Kategori")
            st.sidebar.write(f"Jumlah hari dalam dataset (setelah filter): {len(filtered_day_df)}")
            st.sidebar.write(f"Musim: {filtered_day_df['season_label'].nunique()} kategori")
            st.sidebar.write(f"Kondisi cuaca: {filtered_day_df['weathersit_label'].nunique()} kategori")
        else:
            st.sidebar.warning("Tidak ada data yang tersisa setelah menerapkan filter.")

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
    st.exception(e)  # Tampilkan detail exception untuk membantu debugging