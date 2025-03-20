import pickle
import numpy as np
import pandas as pd
import streamlit as st 
from sklearn.preprocessing import StandardScaler,LabelEncoder


def load_model():
    with  open("student_lr_final_model.pkl", 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocesssing_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocesssing_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student performance prediction")
    st.write("Enter your data to get a prediction for your performance")
    
    hour_studied = st.number_input("Hours studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("previous score", min_value=40, max_value=100, value=70)
    extra = st.selectbox("extra curri activity", ["Yes","No"])
    sleeping_hour = st.number_input("sleeping hours", min_value=4, max_value=10 , value=7)
    number_of_paper_solved = st.number_input("number of question paper solved", min_value=0, max_value=10 , value=5)
    
    if st.button("predict-your_score"):
        user_data = {
            "Hours Studied": hour_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleeping_hour,
            "Sample Question Papers Practiced": number_of_paper_solved
        }
        prediction = predict_data(user_data)
        st.success(f"Your prediction result is {prediction}")


if __name__ == "__main__":
    main()
    