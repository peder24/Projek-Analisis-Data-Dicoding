import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Peminjaman Sepeda", layout="wide")

# Judul aplikasi
st.title("Analisis Peminjaman Sepeda")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    # Asumsikan data sudah diproses sebelumnya
    # Untuk keperluan contoh, kita akan membuat fungsi dummy untuk membuat dataframe
    
    # Buat data dummy untuk hour_df
    hour_df = pd.read_csv("./submission/dashboard/hour.csv")  # Ganti dengan path file Anda
    
    # Buat data dummy untuk day_df
    day_df = pd.read_csv("./submission/dashboard/day.csv")  # Ganti dengan path file Anda
    
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
    
    # Kategori suhu
    hour_df['temp_category'] = pd.cut(
        hour_df['temp_actual'], 
        bins=[0, 10, 20, 30, 41], 
        labels=['Cold (0-10°C)', 'Cool (10-20°C)', 'Warm (20-30°C)', 'Hot (30-41°C)']
    )
    
    # Kategori kelembaban
    hour_df['hum_category'] = pd.cut(
        hour_df['hum_actual'], 
        bins=[0, 25, 50, 75, 100], 
        labels=['Very Dry (0-25%)', 'Dry (25-50%)', 'Humid (50-75%)', 'Very Humid (75-100%)']
    )
    
    # Kategori kecepatan angin
    hour_df['windspeed_category'] = pd.cut(
        hour_df['windspeed_actual'], 
        bins=[0, 15, 30, 45, 67], 
        labels=['Calm (0-15)', 'Light Breeze (15-30)', 'Moderate Wind (30-45)', 'Strong Wind (45+)']
    )
    
    # Indeks kenyamanan
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
    
    return hour_df, day_df

# Muat data
try:
    hour_df, day_df = load_data()
    
    # Sidebar untuk navigasi
    st.sidebar.title("Navigasi")
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
            
            # Persiapkan data untuk visualisasi
            seasonal_hourly_pattern = hour_df.groupby(['season_label', 'hr']).agg({
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
            
            # Plot 2: Distribusi peminjaman berdasarkan musim
            fig, ax = plt.subplots(figsize=(10, 6))
            season_totals = day_df.groupby('season_label').agg({
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
        
        with temporal_tabs[1]:  # Pola Mingguan
            st.subheader("Pola Peminjaman Berdasarkan Hari dalam Seminggu")
            
            # Pola mingguan berdasarkan musim
            fig, ax = plt.subplots(figsize=(10, 6))
            weekly_seasonal_pattern = day_df.groupby(['season_label', 'weekday_label']).agg({
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
            
            # Pola jam berdasarkan tipe hari (hari kerja vs akhir pekan)
            fig, ax = plt.subplots(figsize=(10, 6))
            workday_hourly = hour_df.groupby(['workingday', 'hr']).agg({
                'cnt': 'mean'
            }).reset_index()
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
        
        with temporal_tabs[2]:  # Pola Tahunan
            st.subheader("Analisis Pertumbuhan dan Tren Tahunan")
            
            # Perbandingan peminjaman antara 2011 dan 2012
            fig, ax = plt.subplots(figsize=(10, 6))
            yearly_comparison = day_df.groupby('yr_label').agg({
                'cnt': 'mean',
                'casual': 'mean',
                'registered': 'mean'
            })
            yearly_comparison.plot(kind='bar', ax=ax)
            plt.title('Rata-rata Peminjaman Harian: 2011 vs 2012', fontsize=12)
            plt.xlabel('Tahun', fontsize=10)
            plt.ylabel('Rata-rata Peminjaman', fontsize=10)
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend(title='Tipe Pengguna')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tren bulanan per tahun
            fig, ax = plt.subplots(figsize=(10, 6))
            monthly_yearly_trend = day_df.groupby(['yr_label', 'month_name']).agg({
                'cnt': 'mean'
            }).reset_index()
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
    
    else:  # Pengaruh Cuaca
        st.header("Analisis Pengaruh Cuaca Terhadap Peminjaman Sepeda")
        
        # Tampilkan subtab untuk berbagai visualisasi pengaruh cuaca
        weather_tabs = st.tabs(["Kondisi Cuaca", "Faktor Cuaca", "Interaksi Cuaca"])
        
        with weather_tabs[0]:  # Kondisi Cuaca
            st.subheader("Peminjaman Berdasarkan Kondisi Cuaca")
            
            # Peminjaman berdasarkan kondisi cuaca
            fig, ax = plt.subplots(figsize=(10, 6))
            weather_totals = day_df.groupby('weathersit_label').agg({
                'cnt': 'mean',
                'casual': 'mean',
                'registered': 'mean'
            })
            weather_totals.plot(kind='bar', ax=ax)
            plt.title('Rata-rata Peminjaman Berdasarkan Kondisi Cuaca', fontsize=12)
            plt.xlabel('Kondisi Cuaca', fontsize=10)
            plt.ylabel('Rata-rata Peminjaman', fontsize=10)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend(title='Tipe Pengguna')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Pola peminjaman per jam berdasarkan kondisi cuaca
            weather_impact = hour_df.groupby(['weathersit_label', 'hr']).agg({
                'cnt': 'mean',
            }).reset_index()
            
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
        
        with weather_tabs[1]:  # Faktor Cuaca
            st.subheader("Pengaruh Faktor Cuaca (Suhu, Kelembaban, Angin)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Peminjaman berdasarkan kategori suhu
                temp_impact = hour_df.groupby('temp_category').agg({
                    'cnt': 'mean',
                    'casual': 'mean',
                    'registered': 'mean'
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                temp_impact.plot(kind='bar', ax=ax)
                plt.title('Rata-rata Peminjaman Berdasarkan Kategori Suhu', fontsize=12)
                plt.xlabel('Kategori Suhu', fontsize=10)
                plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                plt.legend(title='Tipe Pengguna')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Peminjaman berdasarkan kategori kelembaban
                humidity_impact = hour_df.groupby('hum_category').agg({
                    'cnt': 'mean',
                    'casual': 'mean',
                    'registered': 'mean'
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                humidity_impact.plot(kind='bar', ax=ax)
                plt.title('Rata-rata Peminjaman Berdasarkan Kategori Kelembaban', fontsize=12)
                plt.xlabel('Kategori Kelembaban', fontsize=10)
                plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                plt.legend(title='Tipe Pengguna')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Peminjaman berdasarkan kategori kecepatan angin
                wind_impact = hour_df.groupby('windspeed_category').agg({
                    'cnt': 'mean',
                    'casual': 'mean',
                    'registered': 'mean'
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                wind_impact.plot(kind='bar', ax=ax)
                plt.title('Rata-rata Peminjaman Berdasarkan Kategori Kecepatan Angin', fontsize=12)
                plt.xlabel('Kategori Kecepatan Angin', fontsize=10)
                plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                plt.legend(title='Tipe Pengguna')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Peminjaman berdasarkan indeks kenyamanan
                comfort_analysis = hour_df.groupby('comfort_category').agg({
                    'cnt': 'mean',
                    'casual': 'mean',
                    'registered': 'mean'
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                comfort_analysis.plot(kind='bar', ax=ax)
                plt.title('Peminjaman Berdasarkan Indeks Kenyamanan', fontsize=12)
                plt.xlabel('Kategori Kenyamanan', fontsize=10)
                plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                plt.legend(title='Tipe Pengguna')
                plt.tight_layout()
                st.pyplot(fig)
        
        with weather_tabs[2]:  # Interaksi Cuaca
            st.subheader("Interaksi antara Cuaca, Musim, dan Tipe Hari")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pengaruh cuaca pada berbagai musim
                season_weather_interaction = day_df.groupby(['season_label', 'weathersit_label']).agg({
                    'cnt': 'mean'
                }).reset_index()
                season_weather_pivot = season_weather_interaction.pivot(index='season_label', columns='weathersit_label', values='cnt')
                
                fig, ax = plt.subplots(figsize=(10, 6))
                season_weather_pivot.plot(kind='bar', ax=ax)
                plt.title('Pengaruh Cuaca pada Berbagai Musim', fontsize=12)
                plt.xlabel('Musim', fontsize=10)
                plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                plt.legend(title='Kondisi Cuaca')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interaksi antara suhu dan kelembaban
                temp_hum_interaction = hour_df.groupby(['temp_category', 'hum_category']).agg({
                    'cnt': 'mean'
                }).reset_index()
                temp_hum_pivot = temp_hum_interaction.pivot(index='temp_category', columns='hum_category', values='cnt')
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(temp_hum_pivot, cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Rata-rata Peminjaman'}, ax=ax)
                plt.title('Interaksi Suhu dan Kelembaban Terhadap Peminjaman', fontsize=12)
                plt.xlabel('Kategori Kelembaban', fontsize=10)
                plt.ylabel('Kategori Suhu', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Pengaruh cuaca pada hari kerja vs akhir pekan
                workday_weather_interaction = day_df.groupby(['workingday_label', 'weathersit_label']).agg({
                    'cnt': 'mean'
                }).reset_index()
                workday_weather_pivot = workday_weather_interaction.pivot(index='workingday_label', columns='weathersit_label', values='cnt')
                
                fig, ax = plt.subplots(figsize=(10, 6))
                workday_weather_pivot.plot(kind='bar', ax=ax)
                plt.title('Pengaruh Cuaca pada Hari Kerja vs Akhir Pekan', fontsize=12)
                plt.xlabel('Tipe Hari', fontsize=10)
                plt.ylabel('Rata-rata Peminjaman', fontsize=10)
                plt.xticks(rotation=0)
                plt.grid(True, alpha=0.3, axis='y')
                plt.legend(title='Kondisi Cuaca')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Korelasi antara variabel cuaca dan peminjaman
                weather_corr = hour_df[['temp_actual', 'atemp_actual', 'hum_actual', 'windspeed_actual', 'cnt', 'casual', 'registered']].corr()
                mask = np.triu(np.ones_like(weather_corr, dtype=bool))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(weather_corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Korelasi'}, ax=ax)
                plt.title('Korelasi antara Variabel Cuaca dan Peminjaman', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
    
    # Tampilkan pesan panduan jika terjadi kesalahan
    st.info("""
    Untuk menjalankan aplikasi ini, Anda memerlukan file data 'hour.csv' dan 'day.csv' 
    dari dataset Bike Sharing. Pastikan file-file tersebut tersedia di direktori yang sama 
    dengan aplikasi ini atau sesuaikan path pada fungsi `load_data()`.
    
    Dataset dapat diunduh dari: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    """)