import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Sayfa Ayarları
st.set_page_config(page_title="Sigorta Tahmin", layout="wide")
st.title("🏥 Sigorta Maliyeti Tahmin Sistemi")

# --- 1. VERİ YÜKLEME ---
# Dosya bulunamazsa hata vermemesi için basit kontrol
import os

if not os.path.exists('insurance.csv'):
    st.error("HATA: 'insurance.csv' dosyası bulunamadı! Lütfen dosyayı app.py yanına koy.")
    st.stop()

df = pd.read.csv("insurance")
# --- 2. ÖN İŞLEME VE MODEL ---
# Veriyi sayısal hale getir
df['sex_encoded'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker_encoded'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], prefix='region', drop_first=True)

# Model değişkenlerini seç
X = df[['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded',
        'region_northwest', 'region_southeast', 'region_southwest']].astype(float)
y = df['charges']

# Modeli eğit
model = LinearRegression()
model.fit(X, y)

# --- 3. KULLANICI ARAYÜZÜ (SOL MENÜ) ---
st.sidebar.header("Bilgileri Giriniz")
age = st.sidebar.slider("Yaş", 18, 80, 25)
bmi = st.sidebar.number_input("BMI (Vücut Kitle İndeksi)", 15.0, 50.0, 25.0)
children = st.sidebar.slider("Çocuk Sayısı", 0, 5, 0)
sex = st.sidebar.selectbox("Cinsiyet", ["Kadın", "Erkek"])
smoker = st.sidebar.selectbox("Sigara Kullanıyor mu?", ["Hayır", "Evet"])
region = st.sidebar.selectbox("Bölge", ["Northeast", "Northwest", "Southeast", "Southwest"])

# --- 4. TAHMİN İŞLEMİ ---
# Girdileri modele uygun hale getir
input_data = pd.DataFrame([{
    'age': age,
    'bmi': bmi,
    'children': children,
    'sex_encoded': 1 if sex == "Erkek" else 0,
    'smoker_encoded': 1 if smoker == "Evet" else 0,
    'region_northwest': 1 if region == "Northwest" else 0,
    'region_southeast': 1 if region == "Southeast" else 0,
    'region_southwest': 1 if region == "Southwest" else 0
}])

# Buton ve Sonuç
if st.button("Maliyeti Hesapla", type="primary"):
    tahmin = model.predict(input_data)[0]
    st.success(f"Tahmini Yıllık Sigorta Primi: ${tahmin:,.2f}")

    if smoker == "Evet":
        st.warning("Not: Sigara kullanımı maliyeti ciddi oranda artırıyor!")

# --- 5. GRAFİKLER ---
st.divider()
st.subheader("Veri Analizi")
col1, col2 = st.columns(2)

with col1:
    st.write("Sigara ve Maliyet İlişkisi")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=df, x='smoker', y='charges', ax=ax1)
    st.pyplot(fig1)

with col2:
    st.write("BMI ve Maliyet İlişkisi")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', ax=ax2)

    st.pyplot(fig2)
