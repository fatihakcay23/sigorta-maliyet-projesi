"""
Sigorta Maliyeti Tahmin Uygulaması
----------------------------------
Bu Streamlit uygulaması, bir kişinin yaş, BMI, çocuk sayısı, cinsiyet,
sigara kullanımı ve yaşadığı bölge bilgilerine göre yıllık sigorta
maliyetini tahmin eden bir Doğrusal Regresyon (Linear Regression) modeli
sunar.

Veri seti: insurance.csv (Kaggle "Medical Cost Personal Dataset")
Yazar: Fatih Akçay
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# 1. SAYFA AYARLARI
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sigorta Maliyeti Tahmin Sistemi",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Sigorta Maliyeti Tahmin Sistemi")
st.caption(
    "Yaş, BMI, sigara kullanımı ve diğer faktörlere göre yıllık sigorta "
    "primini tahmin eden bir makine öğrenmesi uygulaması."
)


# --------------------------------------------------------------------------
# 2. VERİ YÜKLEME (cache'lenir, tekrar tekrar okunmaz)
# --------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = os.path.join(os.path.dirname(__file__), "insurance.csv")
    if not os.path.exists(data_path):
        st.error(
            "HATA: 'insurance.csv' dosyası bulunamadı! "
            "Lütfen dosyayı app.py ile aynı klasöre koyun."
        )
        st.stop()
    return pd.read_csv(data_path)


df_raw = load_data()


# --------------------------------------------------------------------------
# 3. ÖN İŞLEME VE MODEL EĞİTİMİ (cache'lenir, sadece bir kez çalışır)
# --------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "age",
    "bmi",
    "children",
    "sex_encoded",
    "smoker_encoded",
    "region_northwest",
    "region_southeast",
    "region_southwest",
]


@st.cache_resource
def train_model(df: pd.DataFrame):
    data = df.copy()
    data["sex_encoded"] = data["sex"].map({"female": 0, "male": 1})
    data["smoker_encoded"] = data["smoker"].map({"no": 0, "yes": 1})
    data = pd.get_dummies(data, columns=["region"], prefix="region", drop_first=True)

    # Modelin beklediği tüm sütunların var olduğundan emin ol
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = 0

    X = data[FEATURE_COLUMNS].astype(float)
    y = data["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trained_model = LinearRegression()
    trained_model.fit(X_train, y_train)

    predictions = trained_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return trained_model, mae, r2, data


model, mae, r2, df = train_model(df_raw)


# --------------------------------------------------------------------------
# 4. MODEL PERFORMANSI (CV'de göstermek için önemli!)
# --------------------------------------------------------------------------
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Model", "Linear Regression")
metric_col2.metric("R² Skoru (test seti)", f"{r2:.3f}")
metric_col3.metric("Ortalama Mutlak Hata (MAE)", f"${mae:,.0f}")

st.divider()


# --------------------------------------------------------------------------
# 5. KULLANICI ARAYÜZÜ (SOL MENÜ)
# --------------------------------------------------------------------------
st.sidebar.header("Bilgilerinizi Girin")
age = st.sidebar.slider("Yaş", 18, 80, 25)
bmi = st.sidebar.number_input("BMI (Vücut Kitle İndeksi)", 15.0, 50.0, 25.0)
children = st.sidebar.slider("Çocuk Sayısı", 0, 5, 0)
sex = st.sidebar.selectbox("Cinsiyet", ["Kadın", "Erkek"])
smoker = st.sidebar.selectbox("Sigara Kullanıyor mu?", ["Hayır", "Evet"])
region = st.sidebar.selectbox(
    "Bölge", ["Northeast", "Northwest", "Southeast", "Southwest"]
)


# --------------------------------------------------------------------------
# 6. TAHMİN İŞLEMİ
# --------------------------------------------------------------------------
input_data = pd.DataFrame(
    [
        {
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex_encoded": 1 if sex == "Erkek" else 0,
            "smoker_encoded": 1 if smoker == "Evet" else 0,
            "region_northwest": 1 if region == "Northwest" else 0,
            "region_southeast": 1 if region == "Southeast" else 0,
            "region_southwest": 1 if region == "Southwest" else 0,
        }
    ]
)

left, right = st.columns([1, 2])
with left:
    if st.button("Maliyeti Hesapla", type="primary", use_container_width=True):
        tahmin = model.predict(input_data)[0]
        st.success(f"Tahmini Yıllık Sigorta Primi: **${tahmin:,.2f}**")
        if smoker == "Evet":
            st.warning("Not: Sigara kullanımı maliyeti ciddi oranda artırıyor!")


# --------------------------------------------------------------------------
# 7. VERİ ANALİZİ GRAFİKLERİ
# --------------------------------------------------------------------------
st.divider()
st.subheader("Veri Analizi")

col1, col2 = st.columns(2)

with col1:
    st.write("Sigara Kullanımı ve Maliyet İlişkisi")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=df_raw, x="smoker", y="charges", ax=ax1)
    ax1.set_xlabel("Sigara Kullanımı")
    ax1.set_ylabel("Maliyet ($)")
    st.pyplot(fig1)

with col2:
    st.write("BMI ve Maliyet İlişkisi")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df_raw, x="bmi", y="charges", hue="smoker", ax=ax2)
    ax2.set_xlabel("BMI")
    ax2.set_ylabel("Maliyet ($)")
    st.pyplot(fig2)

st.caption(
    "Veri seti: Kaggle 'Medical Cost Personal Dataset' • "
    "Model: scikit-learn LinearRegression"
)
