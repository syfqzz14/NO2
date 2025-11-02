import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from datetime import timedelta

# --- Konfigurasi tampilan halaman ---
st.set_page_config(
    page_title="Prediksi NO₂ Sumenep",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Judul dan Deskripsi ---
st.markdown(
    """
    <h1 style='text-align: center; color: #00BFFF;'>Aplikasi Prediksi Konsentrasi NO₂ Sumenep</h1>
    <p style='text-align: center;'>Gunakan aplikasi ini untuk melakukan prediksi otomatis atau manual terhadap kadar NO₂ harian berdasarkan model KNN multi-output.</p>
    """,
    unsafe_allow_html=True
)

# --- Upload Dataset ---
uploaded_file = st.file_uploader("Unggah dataset NO₂ (format CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.set_index('time')
    df['NO2'] = df['NO2'].interpolate(method='time')

    # --- Bentuk supervised dataset ---
    n_lags = 4
    n_future = 7
    supervised = pd.DataFrame()
    for i in range(n_lags, 0, -1):
        supervised[f'NO2(t-{i})'] = df['NO2'].shift(i)
    for i in range(0, n_future):
        supervised[f'NO2(t+{i})'] = df['NO2'].shift(-i)
    supervised = supervised.dropna()

    # Pisahkan fitur dan target
    X = supervised[[f'NO2(t-{i})' for i in range(n_lags, 0, -1)]].values
    y = supervised[[f'NO2(t+{i})' for i in range(0, n_future)]].values

    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Normalisasi
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --- Model KNN Multi-Output ---
    knn = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5, weights='distance'))
    knn.fit(X_train, y_train)

    # --- Prediksi otomatis hari berikutnya ---
    last_values = df['NO2'].values[-n_lags:]
    X_last = scaler.transform([last_values])
    y_pred = knn.predict(X_last)[0]

    future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(n_future)]
    pred_df = pd.DataFrame({'Tanggal': future_dates, 'Prediksi_NO2': y_pred})

    # --- Pilihan Mode ---
    mode = st.radio(
        "Pilih Mode Prediksi:",
        ("Prediksi Otomatis", "Prediksi Manual Interaktif")
    )

    if mode == "Prediksi Otomatis":
        st.subheader("Prediksi Otomatis 7 Hari ke Depan")

        # Hasil hari pertama
        next_day = pred_df.iloc[0]
        st.success(
            f"Prediksi konsentrasi NO₂ untuk tanggal **{next_day['Tanggal'].date()}** "
            f"adalah **{next_day['Prediksi_NO2']:.6f} mol/m²**"
        )

        # --- Kategori Udara WHO ---
        val = next_day['Prediksi_NO2']
        if val < 0.00005:
            kategori = "🟢 Baik"
        elif val < 0.0001:
            kategori = "🟡 Sedang ⚠️"
        else:
            kategori = "🔴 Tidak Sehat ❌"

        st.info(f"Kategori Udara (WHO): {kategori}")

        # --- Grafik historis + prediksi ---
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.index, df['NO2'], 'o-', color='skyblue', label='Data Historis')
        ax.plot(pred_df['Tanggal'], pred_df['Prediksi_NO2'], 'ro', label='Prediksi')
        ax.set_title("Prediksi Konsentrasi NO₂ Sumenep")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Konsentrasi NO₂ (mol/m²)")
        ax.legend()
        st.pyplot(fig)

    else:
        st.subheader("Prediksi Manual Interaktif")
        manual_values = []
        st.markdown("Masukkan 4 nilai NO₂ terakhir:")
        for i in range(4, 0, -1):
            val = st.number_input(f"NO₂(t-{i})", value=float(df['NO2'].iloc[-i]))
            manual_values.append(val)
        X_manual = scaler.transform([manual_values])
        y_manual = knn.predict(X_manual)[0]
        st.write("### Hasil Prediksi 7 Hari ke Depan:")
        result_df = pd.DataFrame({
            'Tanggal': [df.index[-1] + timedelta(days=i+1) for i in range(7)],
            'Prediksi_NO₂': y_manual
        })
        st.dataframe(result_df)

        # Grafik manual
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(df.index, df['NO2'], 'o-', color='skyblue', label='Data Historis')
        ax2.plot(result_df['Tanggal'], result_df['Prediksi_NO₂'], 'ro', label='Prediksi Manual')
        ax2.legend()
        ax2.set_title("Prediksi Manual Konsentrasi NO₂ Sumenep")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Konsentrasi NO₂ (mol/m²)")
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; font-size: 13px;'>Dibuat oleh <b>Syafiq Azizi</b> menggunakan Streamlit</p>",
        unsafe_allow_html=True
    )

else:
    st.warning("Silakan unggah file dataset terlebih dahulu untuk memulai prediksi.")
