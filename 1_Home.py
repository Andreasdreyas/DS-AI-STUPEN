import streamlit as st

st.set_page_config(page_title="depan", layout="wide")

st.title("Prediksi Churn Pelanggan & Strategi Retensi Otomatis")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("Logo1.jpg")

st.markdown('<hr style="border: 2px solid orange;">', unsafe_allow_html=True)

st.markdown("""
Dalam era digital saat ini, industri berbasis langganan seperti telekomunikasi, layanan 
perangkat lunak (SaaS), dan platform streaming menghadapi tantangan besar dalam 
mempertahankan pelanggan mereka. Salah satu indikator performa yang paling krusial bagi 
perusahaan berbasis layanan berlangganan adalah customer churn yakni kondisi di mana 
pelanggan berhenti menggunakan layanan dalam jangka waktu tertentu. Tingkat churn yang 
tinggi bukan hanya menandakan ketidakpuasan pelanggan, tetapi juga berdampak langsung 
terhadap pendapatan dan kelangsungan bisnis. 
Oleh karena itu, penting bagi perusahaan untuk dapat memprediksi churn secara akurat serta 
mengambil tindakan proaktif melalui strategi retensi yang tepat sasaran. Proyek ini berfokus 
untuk mengembangkan sistem prediksi churn dan rekomendasi strategi retensi otomatis 
menggunakan data nyata dari Telco Customer Churn Dataset. Model akan menganalisis fitur 
seperti jenis kontrak, metode pembayaran, dan biaya bulanan untuk memprediksi risiko churn, 
lalu generative AI akan menghasilkan saran retensi yang personal dan relevan. Permasalahan 
yang diangkat adalah minimnya sistem adaptif yang bisa memberikan solusi retensi khusus 
berdasarkan profil pelanggan. """)