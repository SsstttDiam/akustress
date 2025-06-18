import streamlit as st
import numpy as np
import joblib

model = joblib.load("mentalhealth_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Prediksi Depresi Mahasiswa", layout="centered")
st.title("🧠 Prediksi Risiko Depresi Mahasiswa")
st.markdown(
    """
    Aplikasi ini digunakan untuk memprediksi apakah seorang mahasiswa **berpotensi mengalami depresi**
    berdasarkan lima indikator utama:
    
    - Tingkat stres  
    - Kualitas tidur  
    - Tekanan akademik  
    - Dukungan sosial  
    - Durasi penggunaan HP per hari
    """
)

with st.form("mental_health_form"):
    st.subheader("📋 Masukkan Data Mahasiswa:")
    
    stress = st.slider("1️⃣ Tingkat Stres", 1, 5, 3)
    sleep = st.slider("2️⃣ Kualitas Tidur", 1, 5, 3)
    academic = st.slider("3️⃣ Tekanan Akademik", 1, 5, 3)
    social = st.slider("4️⃣ Dukungan Sosial", 1, 5, 3)
    phone = st.number_input("5️⃣ Durasi Penggunaan HP per Hari (jam)", min_value=0.0, max_value=24.0, value=5.0, step=0.1)

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
    input_data = np.array([[stress, sleep, academic, social, phone]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.subheader("📢 Hasil Prediksi:")
    if prediction == 1:
        st.error("⚠️ Mahasiswa **berpotensi mengalami depresi**. Disarankan untuk mendapatkan perhatian dan dukungan yang tepat.")
    else:
        st.success("✅ Mahasiswa **tidak berpotensi mengalami depresi**. Tetap jaga kesehatan mental!")

    st.markdown("### 🔎 Data yang Dimasukkan:")
    st.json({
        "Tingkat Stres": stress,
        "Kualitas Tidur": sleep,
        "Tekanan Akademik": academic,
        "Dukungan Sosial": social,
        "Durasi HP (jam/hari)": phone
    })
