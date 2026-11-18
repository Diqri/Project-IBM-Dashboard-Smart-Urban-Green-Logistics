import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Konfigurasi Halaman Dashboard
st.set_page_config(
    page_title="Green Logistics & Carbon-Cost Dashboard",
    page_icon="🌱",
    layout="wide"
)

# Judul Dashboard
st.title("🌱 Smart Urban Green Logistics & Carbon-Cost Optimization")
st.markdown("Dashboard analitik untuk memantau efisiensi biaya operasional dan jejak emisi karbon (*carbon footprint*) armada pengiriman perkotaan.")

# 1. Load Data
@st.cache_data
def load_data():
    # Membaca dataset
    df = pd.read_csv("logistics_data.csv")
    
    # Faktor Emisi Karbon (kg CO2 per km berdasarkan jenis bahan bakar)
    # Diesel: ~2.6 kg/km, Gasoline: ~2.3 kg/km, Electric: ~0.5 kg/km (emisi tidak langsung)
    emission_factors = {
        "Diesel": 2.6,
        "Gasoline": 2.3,
        "Electric": 0.5
    }
    
    # Biaya Bahan Bakar per km (dalam Rupiah / IDR estimasi)
    fuel_cost_per_km = {
        "Diesel": 4500,
        "Gasoline": 5000,
        "Electric": 1500
    }
    
    df['carbon_emission_kg'] = df.apply(lambda row: row['distance_km'] * emission_factors.get(row['fuel_type'], 2.0), axis=1)
    df['operational_cost_idr'] = df.apply(lambda row: row['distance_km'] * fuel_cost_per_km.get(row['fuel_type'], 4000), axis=1)
    
    return df

df = load_data()

# Sidebar untuk Filter Interaktif
st.sidebar.header("🔍 Filter Data Armada")
selected_fuel = st.sidebar.multiselect(
    "Pilih Jenis Bahan Bakar:",
    options=df['fuel_type'].unique(),
    default=df['fuel_type'].unique()
)

filtered_df = df[df['fuel_type'].isin(selected_fuel)]

# 2. Key Metrics (KPIs)
st.subheader("📊 Ringkasan Metrik Utama")
col1, col2, col3, col4 = st.columns(4)

total_distance = filtered_df['distance_km'].sum()
total_carbon = filtered_df['carbon_emission_kg'].sum()
total_cost = filtered_df['operational_cost_idr'].sum()
avg_load = filtered_df['load_weight_ton'].mean()

col1.metric("Total Jarak Tempuh", f"{total_distance:,.1f} km")
col2.metric("Total Emisi Karbon", f"{total_carbon:,.1f} kg CO2")
col3.metric("Total Biaya Operasional", f"Rp {total_cost:,.0f}")
col4.metric("Rata-rata Muatan", f"{avg_load:.2f} Ton")

st.markdown("---")

# 3. Visualisasi Grafik (Plotly)
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🔥 Emisi Karbon Berdasarkan Jenis Bahan Bakar")
    fig_carbon = px.bar(
        filtered_df, 
        x='route_name', 
        y='carbon_emission_kg', 
        color='fuel_type',
        title="Perbandingan Emisi Karbon tiap Rute",
        labels={'carbon_emission_kg': 'Emisi (kg CO2)', 'route_name': 'Rute Pengiriman'}
    )
    st.plotly_chart(fig_carbon, use_container_width=True)

with col_b:
    st.subheader("💰 Analisis Biaya vs Jarak Tempuh")
    fig_cost = px.scatter(
        filtered_df, 
        x='distance_km', 
        y='operational_cost_idr', 
        color='fuel_type',
        size='load_weight_ton',
        hover_data=['delivery_id', 'traffic_condition'],
        title="Korelasi Jarak terhadap Biaya Operasional",
        labels={'distance_km': 'Jarak (km)', 'operational_cost_idr': 'Biaya (IDR)'}
    )
    st.plotly_chart(fig_cost, use_container_width=True)

# 4. Tabel Data Mentah & Insight
st.subheader("📋 Detail Data Logistika")
st.dataframe(filtered_df, use_container_width=True)

st.markdown("---")

# 5. Machine Learning Prediction
st.subheader("🤖 Prediksi Emisi Karbon (Machine Learning)")
st.markdown("Model **Random Forest** dilatih dari data historis untuk memprediksi estimasi emisi karbon berdasarkan jarak, muatan, jenis bahan bakar, dan kondisi lalu lintas.")

@st.cache_resource
def train_model(df):
    le_fuel = LabelEncoder()
    le_traffic = LabelEncoder()

    df_ml = df.copy()
    df_ml['fuel_encoded'] = le_fuel.fit_transform(df_ml['fuel_type'])
    df_ml['traffic_encoded'] = le_traffic.fit_transform(df_ml['traffic_condition'])

    features = ['distance_km', 'load_weight_ton', 'fuel_encoded', 'traffic_encoded']
    target = 'carbon_emission_kg'

    X = df_ml[features]
    y = df_ml[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, le_fuel, le_traffic, mae, r2, features

model, le_fuel, le_traffic, mae, r2, features = train_model(df)

# Tampilkan performa model
m1, m2 = st.columns(2)
m1.metric("📉 Mean Absolute Error (MAE)", f"{mae:.2f} kg CO2")
m2.metric("📈 R² Score", f"{r2:.4f}")

st.markdown("### 🔮 Coba Prediksi Sendiri")
st.markdown("Masukkan parameter pengiriman baru untuk mendapatkan estimasi emisi karbon:")

pc1, pc2, pc3, pc4 = st.columns(4)

with pc1:
    pred_distance = st.number_input("Jarak Tempuh (km)", min_value=1.0, max_value=500.0, value=50.0, step=0.5)

with pc2:
    pred_load = st.number_input("Berat Muatan (Ton)", min_value=0.1, max_value=30.0, value=5.0, step=0.1)

with pc3:
    pred_fuel = st.selectbox("Jenis Bahan Bakar", options=list(le_fuel.classes_))

with pc4:
    pred_traffic = st.selectbox("Kondisi Lalu Lintas", options=list(le_traffic.classes_))

if st.button("🚀 Prediksi Emisi Karbon"):
    fuel_enc = le_fuel.transform([pred_fuel])[0]
    traffic_enc = le_traffic.transform([pred_traffic])[0]

    input_data = np.array([[pred_distance, pred_load, fuel_enc, traffic_enc]])
    prediction = model.predict(input_data)[0]

    st.success(f"✅ **Estimasi Emisi Karbon: {prediction:.2f} kg CO2**")

    # Visualisasi feature importance
    st.markdown("#### 📊 Tingkat Kepentingan Fitur (Feature Importance)")
    importance_df = pd.DataFrame({
        'Fitur': ['Jarak (km)', 'Muatan (Ton)', 'Jenis BBM', 'Kondisi Lalu Lintas'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig_imp = px.bar(
        importance_df,
        x='Importance',
        y='Fitur',
        orientation='h',
        title="Kontribusi Setiap Fitur terhadap Prediksi Emisi",
        labels={'Importance': 'Tingkat Kepentingan', 'Fitur': 'Fitur Input'},
        color='Importance',
        color_continuous_scale='greens'
    )
    st.plotly_chart(fig_imp, use_container_width=True)