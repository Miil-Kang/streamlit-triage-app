import streamlit as st
import pandas as pd
import joblib

# 모델 불러오기
model = joblib.load("model.pkl")

st.title("📊 Triage Disposition Prediction")

# CSV 파일 업로드
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # 예측 버튼
    if st.button("예측하기"):
        # 모델 예측
        prediction = model.predict(df)
        df["예측결과"] = ["입원" if p == 1 else "퇴원" for p in prediction]

        st.success("예측 결과:")
        st.dataframe(df)
