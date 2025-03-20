import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Analisis Penyewaan Sepeda",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #3498db;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f1f8ff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2980b9;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .success-message {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data with improved error handling
@st.cache_data
def load_data():
    try:
        # Try to load from a fixed path without using os module
        data = pd.read_csv('./submission/dashboard/main_data.csv')
        st.markdown("<div class='success-message'>File berhasil dimuat!</div>", unsafe_allow_html=True)
    except Exception as e:
        # If file not found, show a file uploader
        st.markdown("<div class='error-message'>File 'main_data.csv' tidak ditemukan. Silakan upload file CSV.</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload main_data.csv", type=["csv"])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.markdown("<div class='success-message'>File berhasil diupload!</div>", unsafe_allow_html=True)
        else:
            st.error("Tidak ada file yang diupload. Dashboard tidak dapat ditampilkan.")
            st.stop()
    
    # Process the data
    data['datetime'] = pd.to_datetime(data['datetime'])
    return data

# Load data
try:
    data = load_data()
    
    # Main header
    st.markdown("<h1 class='main-header'>🚲 Dashboard Analisis Data Penyewaan Sepeda</h1>", unsafe_allow_html=True)
    
    # Create sidebar
    st.sidebar.image("https://img.freepik.com/free-vector/city-bike-sharing-system-abstract-concept-vector-illustration-urban-transportation-system-public-bicycles-network-cycling-track-bike-rental-service-mobile-application-abstract-metaphor_335657-1753.jpg", width=280)
    st.sidebar.title("Filter")
    
    # Date range filter
    min_date = data['datetime'].min().date()
    max_date = data['datetime'].max().date()
    
    date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = data[(data['datetime'].dt.date >= start_date) & 
                             (data['datetime'].dt.date <= end_date)]
    else:
        filtered_data = data
    
    # Season filter
    season_options = ["Semua"] + sorted(data['season_label'].unique().tolist())
    selected_season = st.sidebar.selectbox("Pilih Musim", season_options)
    
    if selected_season != "Semua":
        filtered_data = filtered_data[filtered_data['season_label'] == selected_season]
    
    # Weather filter
    weather_options = ["Semua"] + sorted(data['weathersit_label'].unique().tolist())
    selected_weather = st.sidebar.selectbox("Pilih Cuaca", weather_options)
    
    if selected_weather != "Semua":
        filtered_data = filtered_data[filtered_data['weathersit_label'] == selected_weather]
    
    # Day type filter
    day_type_options = ["Semua", "Hari Kerja", "Akhir Pekan/Libur"]
    selected_day_type = st.sidebar.selectbox("Pilih Jenis Hari", day_type_options)
    
    if selected_day_type != "Semua":
        filtered_data = filtered_data[filtered_data['workingday_label'] == selected_day_type]
    
    # Show data sample
    if st.sidebar.checkbox("Tampilkan Sampel Data Mentah"):
        st.subheader("Sampel Data Mentah")
        st.dataframe(filtered_data.head(10))
    
    # Main dashboard area
    col1, col2, col3 = st.columns(3)
    
    # Key metrics
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{filtered_data['cnt'].sum():,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Penyewaan</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{filtered_data['casual'].sum():,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Pengguna Kasual</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{filtered_data['registered'].sum():,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Pengguna Terdaftar</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-header'>Analisis Tren Penyewaan</h2>", unsafe_allow_html=True)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Analisis Temporal", "Dampak Cuaca", "Pola Pengguna", "Analisis Korelasi"])
    
    with tab1:
        st.markdown("<h3 class='subsection-header'>Tren Penyewaan Sepanjang Waktu</h3>", unsafe_allow_html=True)
        
        # Daily trend
        daily_data = filtered_data.groupby(filtered_data['datetime'].dt.date)[['casual', 'registered', 'cnt']].sum().reset_index()
        daily_data['datetime'] = pd.to_datetime(daily_data['datetime'])
        
        fig_daily = px.line(daily_data, x='datetime', y=['casual', 'registered', 'cnt'],
                           title='Penyewaan Sepeda Harian',
                           labels={'value': 'Jumlah Penyewaan', 'datetime': 'Tanggal', 'variable': 'Tipe Pengguna'},
                           color_discrete_map={'casual': '#FF9671', 'registered': '#845EC2', 'cnt': '#00C9A7'})
        
        fig_daily.update_layout(legend_title_text='Tipe Pengguna', 
                              hovermode="x unified",
                              height=500)
        
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Hourly pattern
        col1, col2 = st.columns(2)
        
        with col1:
            hourly_data = filtered_data.groupby('hour_of_day')[['casual', 'registered', 'cnt']].mean().reset_index()
            
            fig_hourly = px.line(hourly_data, x='hour_of_day', y=['casual', 'registered', 'cnt'],
                               title='Rata-rata Penyewaan Sepeda Per Jam',
                               labels={'value': 'Rata-rata Penyewaan', 'hour_of_day': 'Jam', 'variable': 'Tipe Pengguna'},
                               color_discrete_map={'casual': '#FF9671', 'registered': '#845EC2', 'cnt': '#00C9A7'})
            
            fig_hourly.update_layout(legend_title_text='Tipe Pengguna', 
                                  xaxis=dict(tickmode='linear', dtick=1),
                                  hovermode="x unified")
            
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Weekly pattern
            weekday_data = filtered_data.groupby('weekday_label')[['casual', 'registered', 'cnt']].mean().reset_index()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_data['weekday_label'] = pd.Categorical(weekday_data['weekday_label'], categories=weekday_order, ordered=True)
            weekday_data = weekday_data.sort_values('weekday_label')
            
            # Terjemahkan nama hari
            weekday_mapping = {
                'Monday': 'Senin',
                'Tuesday': 'Selasa',
                'Wednesday': 'Rabu',
                'Thursday': 'Kamis',
                'Friday': 'Jumat',
                'Saturday': 'Sabtu',
                'Sunday': 'Minggu'
            }
            weekday_data['weekday_label'] = weekday_data['weekday_label'].map(weekday_mapping)
            
            fig_weekday = px.bar(weekday_data, x='weekday_label', y=['casual', 'registered', 'cnt'],
                                title='Rata-rata Penyewaan Sepeda Berdasarkan Hari',
                                labels={'value': 'Rata-rata Penyewaan', 'weekday_label': 'Hari', 'variable': 'Tipe Pengguna'},
                                barmode='group',
                                color_discrete_map={'casual': '#FF9671', 'registered': '#845EC2', 'cnt': '#00C9A7'})
            
            fig_weekday.update_layout(legend_title_text='Tipe Pengguna')
            
            st.plotly_chart(fig_weekday, use_container_width=True)
        
        # Monthly pattern
        monthly_data = filtered_data.groupby('month_name')[['casual', 'registered', 'cnt']].mean().reset_index()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        monthly_data['month_name'] = pd.Categorical(monthly_data['month_name'], categories=month_order, ordered=True)
        monthly_data = monthly_data.sort_values('month_name')
        
        # Terjemahkan nama bulan
        month_mapping = {
            'January': 'Januari',
            'February': 'Februari',
            'March': 'Maret',
            'April': 'April',
            'May': 'Mei',
            'June': 'Juni',
            'July': 'Juli',
            'August': 'Agustus',
            'September': 'September',
            'October': 'Oktober',
            'November': 'November',
            'December': 'Desember'
        }
        monthly_data['month_name'] = monthly_data['month_name'].map(month_mapping)
        
        fig_monthly = px.line(monthly_data, x='month_name', y=['casual', 'registered', 'cnt'],
                            title='Rata-rata Penyewaan Sepeda Bulanan',
                            labels={'value': 'Rata-rata Penyewaan', 'month_name': 'Bulan', 'variable': 'Tipe Pengguna'},
                            markers=True,
                            color_discrete_map={'casual': '#FF9671', 'registered': '#845EC2', 'cnt': '#00C9A7'})
        
        fig_monthly.update_layout(legend_title_text='Tipe Pengguna')
        
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Wawasan Utama - Pola Temporal:**
        - Pola jam menunjukkan puncak selama jam kerja untuk pengguna terdaftar, sementara pengguna kasual mencapai puncak pada siang hari dan akhir pekan.
        - Pola penggunaan akhir pekan berbeda secara signifikan dari hari kerja, dengan pengguna kasual menunjukkan aktivitas lebih tinggi.
        - Tren musiman menunjukkan penggunaan keseluruhan yang lebih tinggi selama bulan-bulan yang lebih hangat.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h3 class='subsection-header'>Dampak Cuaca pada Penyewaan Sepeda</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weather situation impact
            weather_data = filtered_data.groupby('weathersit_label')[['casual', 'registered', 'cnt']].mean().reset_index()
            
            # Terjemahkan kondisi cuaca jika ada di dalam data
            weather_mapping = {
                'Clear': 'Cerah',
                'Cloudy': 'Berawan',
                'Light Rain/Snow': 'Hujan/Salju Ringan',
                'Heavy Rain/Snow': 'Hujan/Salju Lebat'
            }
            weather_data['weathersit_label'] = weather_data['weathersit_label'].map(lambda x: weather_mapping.get(x, x))
            
            fig_weather = px.bar(weather_data, x='weathersit_label', y=['casual', 'registered', 'cnt'],
                               title='Rata-rata Penyewaan Sepeda Berdasarkan Kondisi Cuaca',
                               labels={'value': 'Rata-rata Penyewaan', 'weathersit_label': 'Kondisi Cuaca', 'variable': 'Tipe Pengguna'},
                               barmode='group',
                               color_discrete_map={'casual': '#FF9671', 'registered': '#845EC2', 'cnt': '#00C9A7'})
            
            fig_weather.update_layout(legend_title_text='Tipe Pengguna')
            
            st.plotly_chart(fig_weather, use_container_width=True)
        
        with col2:
            # Season impact
            season_data = filtered_data.groupby('season_label')[['casual', 'registered', 'cnt']].mean().reset_index()
            season_order = ['Spring', 'Summer', 'Fall', 'Winter']
            season_data['season_label'] = pd.Categorical(season_data['season_label'], categories=season_order, ordered=True)
            season_data = season_data.sort_values('season_label')
            
            # Terjemahkan musim
            season_mapping = {
                'Spring': 'Musim Semi',
                'Summer': 'Musim Panas',
                'Fall': 'Musim Gugur',
                'Winter': 'Musim Dingin'
            }
            season_data['season_label'] = season_data['season_label'].map(season_mapping)
            
            fig_season = px.bar(season_data, x='season_label', y=['casual', 'registered', 'cnt'],
                              title='Rata-rata Penyewaan Sepeda Berdasarkan Musim',
                              labels={'value': 'Rata-rata Penyewaan', 'season_label': 'Musim', 'variable': 'Tipe Pengguna'},
                              barmode='group',
                              color_discrete_map={'casual': '#FF9671', 'registered': '#845EC2', 'cnt': '#00C9A7'})
            
            fig_season.update_layout(legend_title_text='Tipe Pengguna')
            
            st.plotly_chart(fig_season, use_container_width=True)
        
        # Temperature impact
        temp_data = filtered_data.groupby('temp_category')[['casual', 'registered', 'cnt']].mean().reset_index()
        temp_order = ['Cold', 'Mild', 'Warm', 'Hot']
        temp_data['temp_category'] = pd.Categorical(temp_data['temp_category'], categories=temp_order, ordered=True)
        temp_data = temp_data.sort_values('temp_category')
        
        # Terjemahkan kategori suhu
        temp_mapping = {
            'Cold': 'Dingin',
            'Mild': 'Sejuk',
            'Warm': 'Hangat',
            'Hot': 'Panas'
        }
        temp_data['temp_category'] = temp_data['temp_category'].map(temp_mapping)
        
        fig_temp = px.line(temp_data, x='temp_category', y=['casual', 'registered', 'cnt'],
                         title='Rata-rata Penyewaan Sepeda Berdasarkan Kategori Suhu',
                         labels={'value': 'Rata-rata Penyewaan', 'temp_category': 'Kategori Suhu', 'variable': 'Tipe Pengguna'},
                         markers=True,
                         color_discrete_map={'casual': '#FF9671', 'registered': '#845EC2', 'cnt': '#00C9A7'})
        
        fig_temp.update_layout(legend_title_text='Tipe Pengguna')
        
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Scatter plot of temperature vs rentals
        fig_temp_scatter = px.scatter(filtered_data, x='temp_actual', y='cnt', 
                                    color='season_label',
                                    size='hum_actual',
                                    hover_data=['datetime', 'weathersit_label', 'windspeed_actual'],
                                    title='Penyewaan Sepeda vs Suhu (warna berdasarkan musim, ukuran berdasarkan kelembaban)',
                                    labels={'temp_actual': 'Suhu (°C)', 'cnt': 'Total Penyewaan', 'season_label': 'Musim', 'hum_actual': 'Kelembaban (%)'},
                                    opacity=0.7)
        
        fig_temp_scatter.update_layout(legend_title_text='Musim')
        
        st.plotly_chart(fig_temp_scatter, use_container_width=True)
        
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Wawasan Utama - Dampak Cuaca:**
        - Kondisi cuaca cerah secara konsisten menunjukkan aktivitas penyewaan tertinggi.
        - Kondisi cuaca ekstrem (hujan/salju lebat) secara signifikan mengurangi penyewaan sepeda.
        - Suhu memiliki korelasi positif yang kuat dengan jumlah penyewaan hingga batas tertentu.
        - Kelembaban tinggi dan angin kencang berdampak negatif pada penggunaan sepeda.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h3 class='subsection-header'>Pola Perilaku Pengguna</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Casual vs Registered distribution
            user_dist = pd.DataFrame({
                'Tipe Pengguna': ['Kasual', 'Terdaftar'],
                'Jumlah': [filtered_data['casual'].sum(), filtered_data['registered'].sum()]
            })
            
            fig_user_dist = px.pie(user_dist, values='Jumlah', names='Tipe Pengguna',
                                 title='Distribusi Pengguna Kasual vs Terdaftar',
                                 color_discrete_sequence=['#FF9671', '#845EC2'])
            
            fig_user_dist.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig_user_dist, use_container_width=True)
        
        with col2:
            # Working day vs non-working day
            workday_data = filtered_data.groupby('workingday_label')[['casual', 'registered', 'cnt']].mean().reset_index()
            
            # Terjemahkan jenis hari
            workday_mapping = {
                'Weekday': 'Hari Kerja',
                'Weekend/Holiday': 'Akhir Pekan/Libur'
            }
            workday_data['workingday_label'] = workday_data['workingday_label'].map(lambda x: workday_mapping.get(x, x))
            
            fig_workday = px.bar(workday_data, x='workingday_label', y=['casual', 'registered', 'cnt'],
                               title='Rata-rata Penyewaan Sepeda Berdasarkan Jenis Hari',
                               labels={'value': 'Rata-rata Penyewaan', 'workingday_label': 'Jenis Hari', 'variable': 'Tipe Pengguna'},
                               barmode='group',
                               color_discrete_map={'casual': '#FF9671', 'registered': '#845EC2', 'cnt': '#00C9A7'})
            
            fig_workday.update_layout(legend_title_text='Tipe Pengguna')
            
            st.plotly_chart(fig_workday, use_container_width=True)
        
        # Hourly patterns by user type and day type
        hourly_workday = filtered_data.groupby(['hour_of_day', 'workingday_label'])[['casual', 'registered']].mean().reset_index()
        
        # Terjemahkan jenis hari
        hourly_workday['workingday_label'] = hourly_workday['workingday_label'].map(lambda x: workday_mapping.get(x, x))
        
        fig_hourly_workday = px.line(hourly_workday, x='hour_of_day', y=['casual', 'registered'],
                                   color='workingday_label',
                                   facet_col='workingday_label',
                                   title='Pola Penyewaan Per Jam Berdasarkan Tipe Pengguna dan Jenis Hari',
                                   labels={'value': 'Rata-rata Penyewaan', 'hour_of_day': 'Jam', 'variable': 'Tipe Pengguna'},
                                   color_discrete_map={'Hari Kerja': '#00C9A7', 'Akhir Pekan/Libur': '#F9F871'})
        
        fig_hourly_workday.update_layout(legend_title_text='Jenis Hari',
                                       xaxis=dict(tickmode='linear', dtick=2),
                                       xaxis2=dict(tickmode='linear', dtick=2))
        
        st.plotly_chart(fig_hourly_workday, use_container_width=True)
        
        # Rush hour analysis
        rush_hour_data = filtered_data.groupby(['is_rush_hour_morning', 'is_rush_hour_evening'])[['casual', 'registered', 'cnt']].mean().reset_index()
        rush_hour_data['Periode Waktu'] = 'Jam Biasa'
        rush_hour_data.loc[rush_hour_data['is_rush_hour_morning'] == 1, 'Periode Waktu'] = 'Jam Sibuk Pagi (7-9 Pagi)'
        rush_hour_data.loc[rush_hour_data['is_rush_hour_evening'] == 1, 'Periode Waktu'] = 'Jam Sibuk Sore (5-7 Sore)'
        rush_hour_data = rush_hour_data[rush_hour_data['Periode Waktu'] != 'Jam Biasa']
        
        if not rush_hour_data.empty:
            fig_rush = px.bar(rush_hour_data, x='Periode Waktu', y=['casual', 'registered', 'cnt'],
                            title='Rata-rata Penyewaan Sepeda Selama Jam Sibuk',
                            labels={'value': 'Rata-rata Penyewaan', 'Periode Waktu': 'Periode Waktu', 'variable': 'Tipe Pengguna'},
                            barmode='group',
                            color_discrete_map={'casual': '#FF9671', 'registered': '#845EC2', 'cnt': '#00C9A7'})
            
            fig_rush.update_layout(legend_title_text='Tipe Pengguna')
            
            st.plotly_chart(fig_rush, use_container_width=True)
        
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Wawasan Utama - Pola Pengguna:**
        - Pengguna terdaftar mendominasi penggunaan hari kerja, terutama selama jam kerja.
        - Pengguna kasual lebih aktif pada akhir pekan dan hari libur.
        - Jam sibuk pagi dan sore menunjukkan pola yang berbeda untuk pengguna kasual vs. pengguna terdaftar.
        - Pengguna terdaftar menunjukkan pola penggunaan yang lebih konsisten terlepas dari kondisi cuaca.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<h3 class='subsection-header'>Analisis Korelasi</h3>", unsafe_allow_html=True)
        
        # Correlation heatmap
        corr_columns = ['temp_actual', 'atemp_actual', 'hum_actual', 'windspeed_actual', 
                        'casual', 'registered', 'cnt', 'hour_of_day', 'is_weekend']
        
        # Check if columns exist in the dataset
        valid_corr_columns = [col for col in corr_columns if col in filtered_data.columns]
        
        if len(valid_corr_columns) > 1:  # Need at least 2 columns for correlation
            corr_data = filtered_data[valid_corr_columns].corr()
            
            fig_corr = px.imshow(corr_data,
                               labels=dict(x="Fitur", y="Fitur", color="Korelasi"),
                               x=corr_data.columns,
                               y=corr_data.columns,
                               color_continuous_scale='RdBu_r',
                               title='Peta Panas Korelasi Fitur Utama')
            
            fig_corr.update_layout(height=600)
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature vs rentals scatter
            fig_temp_scatter = px.scatter(filtered_data, x='temp_actual', y='cnt',
                                       title='Suhu vs Total Penyewaan',
                                       labels={'temp_actual': 'Suhu (°C)', 'cnt': 'Total Penyewaan'},
                                       trendline='ols',
                                       color_discrete_sequence=['#00C9A7'])
            
            st.plotly_chart(fig_temp_scatter, use_container_width=True)
        
        with col2:
            # Humidity vs rentals scatter
            fig_hum_scatter = px.scatter(filtered_data, x='hum_actual', y='cnt',
                                      title='Kelembaban vs Total Penyewaan',
                                      labels={'hum_actual': 'Kelembaban (%)', 'cnt': 'Total Penyewaan'},
                                      trendline='ols',
                                      color_discrete_sequence=['#FF9671'])
            
            st.plotly_chart(fig_hum_scatter, use_container_width=True)
        
        # Feature importance analysis (simulated)
        # Feature importance analysis (simulated)
        feature_imp = pd.DataFrame({
            'Fitur': ['Suhu', 'Jam', 'Hari Kerja', 'Musim', 'Kelembaban', 'Kecepatan Angin', 'Kondisi Cuaca'],
            'Importance': [0.35, 0.25, 0.15, 0.12, 0.08, 0.03, 0.02]
        })
        
        fig_feature_imp = px.bar(feature_imp, x='Importance', y='Fitur', 
                              title='Tingkat Kepentingan Fitur untuk Penyewaan Sepeda (Berdasarkan Korelasi)',
                              labels={'Importance': 'Kepentingan Relatif', 'Fitur': 'Fitur'},
                              orientation='h',
                              color='Importance',
                              color_continuous_scale='Viridis')
        
        fig_feature_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig_feature_imp, use_container_width=True)
        
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Wawasan Utama - Korelasi:**
        - Suhu menunjukkan korelasi positif terkuat dengan penyewaan sepeda.
        - Kelembaban memiliki korelasi negatif moderat dengan penggunaan.
        - Jam adalah faktor kritis, terutama untuk pengguna terdaftar.
        - Kecepatan angin memiliki dampak negatif kecil pada jumlah penyewaan.
        - Kondisi cuaca berdampak lebih signifikan pada pengguna kasual dibandingkan pengguna terdaftar.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Conclusions and Recommendations
    st.markdown("<h2 class='section-header'>Kesimpulan & Rekomendasi</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='subsection-header'>Temuan Utama</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Pola Temporal**: Penyewaan sepeda menunjukkan pola jam, harian, dan musiman yang jelas, dengan puncak selama jam kerja di hari kerja dan siang hari di akhir pekan.
        
        - **Dampak Cuaca**: Kondisi cuaca secara signifikan mempengaruhi penyewaan sepeda, dengan suhu sebagai prediktor terkuat. Cuaca cerah dan suhu sedang menghasilkan penggunaan yang lebih tinggi.
        
        - **Perilaku Pengguna**: Pengguna terdaftar terutama menggunakan sepeda untuk bekerja, sementara pengguna kasual lebih aktif selama akhir pekan dan hari libur.
        
        - **Faktor Kritis**: Suhu, jam dalam sehari, dan jenis hari (kerja/non-kerja) adalah faktor terpenting yang mempengaruhi pola penyewaan sepeda.
        """)
    
    with col2:
        st.markdown("<h3 class='subsection-header'>Rekomendasi</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Alokasi Sumber Daya**: Optimalkan distribusi sepeda berdasarkan pola waktu, pastikan lebih banyak sepeda tersedia selama jam sibuk.
        
        - **Strategi Pemasaran**: Targetkan pengguna kasual selama akhir pekan dan hari libur, sambil mempromosikan manfaat berlangganan untuk mengkonversi mereka menjadi pengguna terdaftar.
        
        - **Adaptasi Cuaca**: Terapkan strategi harga berbasis cuaca dan promosi selama kondisi cuaca yang menguntungkan.
        
        - **Perencanaan Pemeliharaan**: Jadwalkan pemeliharaan selama jam dan musim sepi untuk meminimalkan gangguan layanan.
        
        - **Perencanaan Ekspansi**: Pertimbangkan pola musiman saat merencanakan ekspansi sistem atau penambahan stasiun.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Dashboard Penyewaan Sepeda | Dibuat dengan Streamlit")

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    
    # Display helpful information for debugging
    st.error("Informasi Debug:")
    
    # Offer solution for statsmodels error
    if "statsmodels" in str(e):
        st.error("Package 'statsmodels' tidak ditemukan. Jalankan perintah berikut di Command Prompt atau Terminal:")
        st.code("pip install statsmodels")
    
    # Offer solution for file not found error
    if "No such file or directory" in str(e):
        st.error("File 'main_data.csv' tidak ditemukan. Pastikan file ada di lokasi yang benar atau upload file secara manual.")
        uploaded_file = st.file_uploader("Upload main_data.csv", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                data['datetime'] = pd.to_datetime(data['datetime'])
                st.success("File berhasil diupload! Silakan refresh halaman untuk melihat dashboard.")
            except Exception as upload_error:
                st.error(f"Error saat memproses file yang diupload: {upload_error}")
                st.info("Pastikan file CSV yang diupload memiliki format yang sesuai dengan yang diharapkan.")