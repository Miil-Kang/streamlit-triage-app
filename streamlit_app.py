import streamlit as st
import pandas as pd
import joblib  # 모델이 .pkl로 저장되어 있다고 가정

st.title("📊 Triage Disposition Prediction")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # 예측 자동 실행
    try:
        model = joblib.load("model.pkl")  # 모델 파일 경로 수정 필요
        predictions = model.predict(df)
        st.write("### 🔮 Prediction Results")
        st.dataframe(predictions)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
