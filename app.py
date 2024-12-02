import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Fungsi untuk Mengunggah Data
def upload_data():
    st.title("Upload Dataset")
    st.markdown("Unggah file CSV yang berisi data kesehatan seperti dataset diabetes Pima.")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data berhasil diunggah!")
        st.dataframe(df.head())
        return df
    return None

# 2. Fungsi untuk Exploratory Data Analysis (EDA)
def eda(df):
    st.title("Exploratory Data Analysis (EDA)")

    if df is not None:
        st.write("Data Overview:")
        st.write(df.describe())

        # Visualisasi distribusi
        st.subheader("Distribusi Usia")
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Distribusi Glukosa")
        fig, ax = plt.subplots()
        sns.histplot(df['Glucose'], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Correlations Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# 3. Fungsi untuk Pemodelan dan Prediksi
def modeling(df):
    st.title("Modeling & Prediksi Diabetes")

    if df is not None:
        st.write("Persiapkan data untuk model.")
        
        # Pisahkan fitur dan target
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]
        
        # Bagi data menjadi train dan test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standarisasi data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Prediksi
        y_pred = model.predict(X_test_scaled)

        st.subheader("Evaluasi Model")
        st.write(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap='Blues', ax=ax)
        st.pyplot(fig)

        # Prediksi individual
        st.subheader("Prediksi Individual")
        age = st.number_input("Masukkan usia:", min_value=1, max_value=100)
        glucose = st.number_input("Masukkan kadar glukosa:", min_value=0, max_value=500)
        bmi = st.number_input("Masukkan BMI:", min_value=10.0, max_value=60.0)

        input_data = [[age, glucose, 0, 0, 0, bmi, 0, 0]]  # Model memerlukan semua fitur, set yang tidak digunakan = 0
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction == 0:
            st.write("Prediksi: Tidak Terdiagnosis Diabetes")
        else:
            st.write("Prediksi: Terdiagnosis Diabetes")

# 4. Main Function untuk Menjalankan Aplikasi
def main():
    st.title("Aplikasi Prediksi Diabetes")
    
    # Upload Data
    df = upload_data()
    
    # EDA
    if df is not None:
        eda(df)
    
    # Modeling dan Prediksi
    if df is not None:
        modeling(df)

if __name__ == "__main__":
    main()
