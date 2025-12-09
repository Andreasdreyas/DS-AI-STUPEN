import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import numpy as np
import shap
import os
import tensorflow as tf

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


# 0. FUNGSI BANTUAN UNTUK SHAP
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# 1. SETUP HALAMAN & SIDEBAR
st.set_page_config(page_title="Churn Prediction App", layout="wide")
with st.sidebar:
    st.subheader("🔑 Google AI (Gemini)")
    user_api_key = st.text_input("Masukkan Google API Key", type="password", help="Dapatkan di aistudio.google.com")
    
    if user_api_key:
        os.environ["GOOGLE_API_KEY"] = user_api_key
        st.success("API Key terdeteksi!")
    else:
        st.warning("Masukkan API Key untuk fitur Analisis AI.")

st.title("🧾 Customer Churn Prediction & Explainability")


# 2. LOAD ASSETS 
@st.cache_resource
def load_specific_asset(model_name, folder="Projek_Final_Stupen"):
    paths = [
        os.path.join(folder, model_name),
        model_name
    ]
    
    for path in paths:
        if os.path.exists(path):
            try:
                # CEK EKSTENSI FILE
                if path.endswith(".h5") or path.endswith(".keras"):
                    return tf.keras.models.load_model(path)
                else:
                    return joblib.load(path)
            except Exception as e:
                # Tampilkan error spesifik agar tahu kenapa gagal
                st.error(f"Gagal memuat {model_name}. Error: {e}")
                continue
    return None    

def load_all_assets():
    encoders = load_specific_asset('encoders_churn.joblib')
    scaler = load_specific_asset('scaler_churn.joblib')

    model_rf = load_specific_asset('model_rf_smote.joblib')
    model_ann = load_specific_asset('model_ann_smote.h5')
    model_svm = load_specific_asset('model_svm_smote.joblib')
    model_xgb = load_specific_asset('model_xgb_smote.joblib')
    
    return model_rf, model_ann, model_svm, model_xgb, encoders, scaler

model_rf, model_ann, model_svm, model_xgb, encoders_dict, scaler_obj = load_all_assets()

if model_rf is None:
    st.error("🚨 **File Aset Hilang!** Pastikan file model, encoders, dan scaler ada di folder yang sama.")
    st.stop()


# 3. FORM INPUT USER
options = {
    "Gender": ["","Male", "Female"],
    "Senior Citizen": ["","No", "Yes"], 
    "Partner": ["","No", "Yes"],
    "Dependents": ["","No", "Yes"],
    "Tenure Months": None, 
    "Phone Service": ["","No", "Yes"],
    "Multiple Lines": ["","No", "Yes", "No phone service"],
    "Internet Service": ["","No", "DSL", "Fiber optic"],
    "Online Security": ["","No", "Yes", "No internet service"],
    "Online Backup": ["","No", "Yes", "No internet service"],
    "Device Protection": ["","No", "Yes", "No internet service"],
    "Tech Support": ["","No", "Yes", "No internet service"],
    "Streaming TV": ["","No", "Yes", "No internet service"],
    "Streaming Movies": ["","No", "Yes", "No internet service"],
    "Contract": ["","Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["","No", "Yes"],
    "Payment Method": ["","Mailed check", "Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"],
    "Monthly Charges": None,
    "Total Charges": None,
    "Model Type": ["", "Random Forest", "ANN", "SVM", "XG-Boost"]
}

keys = list(options.keys())
results = {}

with st.container():
    st.write("### 📝 Data Pelanggan")
    for i in range(0, 20, 2):
        col1, col2 = st.columns(2)
        
        label_left = keys[i]
        if options[label_left] is None:
            if "Tenure" in label_left:
                results[label_left] = col1.number_input(label_left, min_value=0, max_value=100, step=1)
            else:
                results[label_left] = col1.number_input(label_left, min_value=0.0, step=0.01, format="%.2f")
        else:
            results[label_left] = col1.selectbox(label_left, options[label_left])


        if i+1 < len(keys):
            label_right = keys[i+1]
            if options[label_right] is None:
                if "Tenure" in label_right:
                    results[label_right] = col2.number_input(label_right, min_value=0, max_value=100, step=1)
                else:
                    results[label_right] = col2.number_input(label_right, min_value=0.0, step=0.01, format="%.2f")
            else:
                results[label_right] = col2.selectbox(label_right, options[label_right])


# 4. PREPROCESSING
def preprocess_input(data_input, encoders, scaler):
    df = pd.DataFrame([data_input])
    if 'Model Type' in df.columns:
        df = df.drop(columns=['Model Type'])

    for col in encoders:
        if col in df.columns:
            le = encoders[col]
            val = df[col].iloc[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = -1 
    
    cols_to_scale = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    try:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    except Exception as e:
        st.error(f"Error Scaling: {e}")
        st.stop()

    df = df.apply(pd.to_numeric, errors='ignore')
    return df


# 5. PREDIKSI + SHAP + GENAI
st.markdown("---")

if st.button("🔍 Prediksi & Analisis"):
    if "" in results.values():
        st.warning("⚠️ Mohon lengkapi semua data form di atas.")
    else:
        try:
            # A. DATA PREPARATION
            df_display = pd.DataFrame([results])
            if 'Model Type' in df_display.columns:
                df_display = df_display.drop(columns=['Model Type'])
            
            with st.expander("Lihat Data Input (Mentah)", expanded=False):
                st.dataframe(df_display)

            df_final = preprocess_input(results, encoders_dict, scaler_obj)
            model_choice = results.get("Model Type")


            # B. PREDIKSI
            if model_choice != "":
                prediksi = 0
                probabilitas = 0.0
                
                if model_choice in ["Random Forest", "SVM", "XG-Boost"]:
                    prediksi = int(results.get("Model Type") == "SVM" and model_svm.predict(df_final)[0] or 
                                   results.get("Model Type") == "XG-Boost" and model_xgb.predict(df_final)[0] or
                                   model_rf.predict(df_final)[0])
                    
                    if model_choice == "Random Forest":
                        probabilitas = model_rf.predict_proba(df_final)[0][1]
                    elif model_choice == "XG-Boost":
                        probabilitas = model_xgb.predict_proba(df_final)[0][1]
                    elif model_choice == "SVM":
                        if hasattr(model_svm, "predict_proba"):
                            probabilitas = model_svm.predict_proba(df_final)[0][1]
                        else:
                            probabilitas = 1.0 if prediksi == 1 else 0.0 
                
                elif model_choice == "ANN":
                # 1. Pastikan model berhasil dimuat
                    if model_ann is None:
                        st.error("Model ANN tidak ditemukan/gagal dimuat.")
                        st.stop()
                    X_input = df_final.values.astype('float32')
                    raw_prediction = model_ann.predict(X_input, verbose=0)
                    probabilitas = float(raw_prediction.flatten()[0])
                    prediksi = 1 if probabilitas > 0.5 else 0

                prob_persen = round(probabilitas * 100, 2)
                st.subheader(f"Hasil Prediksi ({model_choice})")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    if prediksi == 1:
                        st.error("🛑 **PREDIKSI: CHURN**")
                        st.write("Pelanggan berisiko churn.")
                    else:
                        st.success("✅ **PREDIKSI: TIDAK CHURN**")
                        st.write("Pelanggan aman.")
                
                with col_res2:
                    st.metric("Probabilitas Churn", f"{prob_persen:.2f}%")
                    st.progress(int(prob_persen))


                # C. SHAP
                st.markdown("---")
                st.subheader("🕵️ Analisis Faktor (Explainability)")
                
                faktor_utama = []
                detail_faktor = {}

                with st.spinner('Menghitung kontribusi fitur (SHAP)...'):
                    try:
                        explainer = None
                        shap_values = None
                        shap_values_single = None
                        base_value = None

                        # TreeExplainer
                        if model_choice in ["Random Forest", "XG-Boost"]:
                            curr_model = model_rf if model_choice == "Random Forest" else model_xgb
                            explainer = shap.TreeExplainer(curr_model)
                            shap_values = explainer.shap_values(df_final)
                            
                            if isinstance(shap_values, list):
                                shap_churn = shap_values[1] 
                            elif len(np.array(shap_values).shape) == 3:
                                shap_churn = shap_values[:, :, 1]
                            else:
                                shap_churn = shap_values
                                
                            shap_values_single = shap_churn[0]
                            base_value = explainer.expected_value
                            
                            if isinstance(base_value, (list, np.ndarray)):
                                base_value = base_value[1] if len(base_value) > 1 else base_value[0]

                        # KernelExplainer
                        elif model_choice in ["ANN", "SVM"]:
                            def model_predict_wrapper(data):
                                if model_choice == "ANN":
                                    return model_ann.predict(data).flatten()
                                elif model_choice == "SVM":
                                    if hasattr(model_svm, "predict_proba"):
                                        return model_svm.predict_proba(data)[:, 1]
                                    else:
                                        return model_svm.predict(data)

                            background_data = pd.DataFrame(
                                np.zeros((1, df_final.shape[1])), 
                                columns=df_final.columns
                            )

                            explainer = shap.KernelExplainer(model_predict_wrapper, background_data)
                            shap_values = explainer.shap_values(df_final, nsamples=100)
                          

                            if isinstance(shap_values, list):
                                shap_values_single = shap_values[0] 
                            else:
                                shap_values_single = shap_values[0]

                            base_value = explainer.expected_value
                            if isinstance(base_value, (list, np.ndarray)):
                                try:
                                    base_value = base_value[0]
                                except:
                                    pass

                       
                        # VISUALISASI                       
                        if shap_values_single is not None:
                            st.write(f"#### Force Plot ({model_choice})")
                            force_plot = shap.force_plot(
                                base_value,
                                shap_values_single,
                                df_final.values[0],
                                feature_names=df_final.columns,
                                link="logit" if model_choice in ["Random Forest", "XG-Boost", "ANN"] else "identity"
                            )
                            st_shap(force_plot)

                            if len(np.array(shap_values_single).shape) > 1:
                                shap_values_single = np.array(shap_values_single).flatten()

                            abs_shap = np.abs(shap_values_single)
                            top2_idx = np.argsort(abs_shap)[::-1][:2]
                            
                            feature_names = df_final.columns.tolist()
                           
                            faktor_utama = [feature_names[i] for i in top2_idx]
                            detail_faktor = {fitur: df_display[fitur].values[0] for fitur in faktor_utama}

                            st.write("#### Faktor Terpenting")
                            c1, c2 = st.columns(2)
                            for idx, (fitur, val_raw) in enumerate(detail_faktor.items()):
                                idx_feat = feature_names.index(fitur)
                                shap_score = shap_values_single[idx_feat]
                                
                                effect = "Meningkatkan Risiko" if shap_score > 0 else "Menurunkan Risiko"
                                delta_color = "inverse" if shap_score > 0 else "normal"
                                
                                with (c1 if idx == 0 else c2):
                                    st.metric(label=fitur, value=str(val_raw), delta=effect, delta_color=delta_color)

                    except Exception as e:
                        st.warning(f"Gagal memuat SHAP untuk {model_choice}. Error: {e}")
                

                # D. GENERATIVE AI
                st.markdown("---")
                st.subheader("🤖 Rekomendasi Bisnis (AI)")

                if not os.environ.get("GOOGLE_API_KEY"):
                    st.warning("⚠️ **Fitur Analisis AI Non-aktif.** Masukkan Google API Key di Sidebar untuk mengaktifkan.")
                else:
                    with st.spinner("Sedang menganalisis strategi retensi dengan AI..."):
                        try:
                            llm = ChatGoogleGenerativeAI(
                                model="gemini-2.5-flash", 
                                temperature=0.4
                            )
                            
                            system_prompt = f""" Kamu adalah AI Analyst untuk perusahaan telekomunikasi.

                            Tugasmu:
                            1. Jelaskan apakah pelanggan ini churn atau tidak berdasarkan prediksi model.
                            2. Jelaskan fitur apa yang paling mempengaruhi hasil prediksi (berdasarkan SHAP values).
                            3. Buat analisis dengan bahasa manusia, tidak teknis.
                            4. Berikan strategi retensi otomatis yang cocok untuk pelanggan ini.
                            5. Tidak usah menjelaskan selain hal-hal diatas.
                            6. Gunakan bahasa yang mudah dimengerti, jangan gunakan bahasa teknis.
                            7. Gunakan bahasa se manusiawi mungkin, tetapi masih sopan dan formal, serta profesional.

                            Format jawaban:
                            - Kesimpulan Churn (Tampilkan berapa persen probabilitas pengguna akan churn, dan berapa persen pengguna tidak akan churn)
                            - Faktor Utama Penyebab
                            - Rekomendasi Retensi
                            """

                            user_prompt = f"""
                            HASIL PREDIKSI MODEL:
                            - Prediksi: {"CHURN" if prediksi == 1 else "TIDAK CHURN"}
                            - Probabilitas churn: {round(prob_persen)} %

                            FITUR DAN SHAP VALUES (pengaruh terhadap churn):
                            {faktor_utama}

                            Tolong jelaskan:
                            1. Mengapa pelanggan ini diprediksi churn / tidak churn
                            2. Fitur apa yang paling berpengaruh
                            3. Rekomendasi retensi apa yang paling cocok
                            """

                            response = llm.invoke([
                                ("system", system_prompt),
                                ("human", user_prompt)
                            ])
               
                            st.info("💡 **Analisis Cerdas & Strategi:**")
                            st.write(response.content)
                            
                        except Exception as e:
                            st.error(f"Gagal menghubungi AI: {e}")

            else:
                st.warning("Silakan pilih tipe model terlebih dahulu.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan sistem: {e}")