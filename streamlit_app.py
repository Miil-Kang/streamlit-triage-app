import streamlit as st
import pandas as pd
import joblib

st.title("📊 Triage Disposition Prediction")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # 모델 불러오기
    model = joblib.load("model.pkl")  # 모델 경로가 정확한지 확인하세요

    # 예측 버튼
    if st.button("예측하기"):
        predictions = model.predict(df)
        df["예측 결과"] = ["입원" if p == 1 else "퇴원" for p in predictions]
        st.write("📈 예측 결과:")
        st.dataframe(df)
