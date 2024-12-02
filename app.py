import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Fungsi untuk Memuat Data Secara Otomatis dari GitHub
def load_data():
    st.title("Dataset Diabetes")
    
    # URL raw GitHub untuk file CSV
    csv_url = "https://raw.githubusercontent.com/Adpt5/streamlit-app-diabetes/master/data/diabetes.csv"
    
    # Memuat data dari URL
    df = pd.read_csv(csv_url)
    
    st.write("Data berhasil dimuat!")
    st.dataframe(df.head())
    
    return df

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

# 4. Fungsi untuk About (Deskripsi Aplikasi)
def about():
    st.title("About")
    st.markdown("""
    **Aplikasi Prediksi Diabetes** ini bertujuan untuk memprediksi apakah seseorang terdiagnosis diabetes atau tidak berdasarkan beberapa parameter medis seperti usia, kadar glukosa, dan BMI.

    Aplikasi ini menggunakan **algoritma Random Forest** untuk melatih model prediksi berdasarkan dataset Diabetes Pima Indian. Model ini akan memberikan prediksi apakah seseorang memiliki risiko diabetes berdasarkan parameter yang dimasukkan.

    **Tentang Pembuat:**
    Nama: **Adipati Sulaiman**
    
    Terima kasih telah menggunakan aplikasi ini!
    """)

# 5. Main Function untuk Menjalankan Aplikasi
def main():
    st.sidebar.title("Menu")
    
    # Sidebar Menu
    menu = st.sidebar.radio("Pilih Menu", ("Home", "About"))
    
    if menu == "Home":
        # Load Data secara otomatis
        df = load_data()
        
        # EDA
        if df is not None:
            eda(df)
        
        # Modeling dan Prediksi
        if df is not None:
            modeling(df)
    
    elif menu == "About":
        about()

if __name__ == "__main__":
    main()
