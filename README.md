# Sigorta Maliyeti Tahmin Sistemi

**Insurance Cost Prediction App** — Doğrusal regresyon ile bireylerin yıllık sağlık sigortası maliyetini tahmin eden, Streamlit ile geliştirilmiş interaktif bir makine öğrenmesi uygulaması.

> Kısa özet (EN): A Streamlit web app that predicts annual medical insurance charges from a person's age, BMI, smoking status, number of children, sex and region, using a Linear Regression model trained with scikit-learn.

<!-- Canlı demo yayınladıktan sonra bu satırı güncelleyin: -->
🔗 **Canlı Demo:** _[Streamlit Cloud linkinizi buraya ekleyin]_

<!-- Uygulamanızdan bir ekran görüntüsü alıp docs/screenshot.png olarak ekleyin, sonra alttaki satırın başındaki # işaretini kaldırın -->
<!-- ![Uygulama Ekran Görüntüsü](docs/screenshot.png) -->

---

##  Proje Hakkında

Bu proje, [Kaggle Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) üzerinde bir regresyon analizi yapılarak geliştirilmiştir. Amaç, bir kişinin;

- Yaşı
- Vücut Kitle İndeksi (BMI)
- Çocuk sayısı
- Cinsiyeti
- Sigara kullanıp kullanmadığı
- Yaşadığı bölge

bilgilerine bakarak yıllık sigorta maliyetini (`charges`) tahmin etmektir. Proje; veri ön işleme, keşifsel veri analizi (EDA), model eğitimi/değerlendirmesi ve sonuçların interaktif bir web arayüzü üzerinden sunulması adımlarını kapsar.
##  Özellikler

-  Kullanıcının kendi bilgilerini girerek anlık tahmin alabildiği interaktif form
-  Sigara kullanımı ve BMI'nin maliyet üzerindeki etkisini gösteren görselleştirmeler
-  Ekranda gösterilen model performans metrikleri (R², MAE)
-  Kategorik değişkenler için one-hot encoding, eğitim/test ayrımı ile doğrulama

##  Kullanılan Teknolojiler

| Katman | Teknoloji |
|---|---|
| Dil | Python |
| Veri işleme | pandas, NumPy |
| Modelleme | scikit-learn (Linear Regression) |
| Görselleştirme | Matplotlib, Seaborn |
| Web arayüzü | Streamlit |
| Analiz / Ödev dosyaları | Jupyter Notebook, R Markdown |

##  Proje Yapısı

```
sigorta-maliyet-projesi/
├── app.py                                  # Streamlit web uygulaması
├── insurance.csv                           # Veri seti (Kaggle)
├── requirements.txt                        # Python bağımlılıkları
├── Regresyon Ödevi Notebook FDHB.ipynb     # EDA ve model geliştirme süreci (Jupyter)
├── Regresyon Analizi Dersi Ödev Dosyası.Rmd  # R ile yapılan regresyon analizi
└── README.md
```

##  Kurulum ve Çalıştırma

```bash
# 1. Depoyu klonlayın
git clone https://github.com/fatihakcay23/sigorta-maliyet-projesi.git
cd sigorta-maliyet-projesi

# 2. (Önerilir) sanal ortam oluşturun
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Bağımlılıkları kurun
pip install -r requirements.txt

# 4. Uygulamayı başlatın
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

##  Model Performansı

Model, verinin %80'i ile eğitilip %20'lik test seti üzerinde değerlendirilmiştir:

| Metrik | Değer |
|---|---|
| R² Skoru | Uygulama içinde canlı olarak hesaplanır |
| MAE (Ortalama Mutlak Hata) | Uygulama içinde canlı olarak hesaplanır |

> Not: En büyük etkiyi sigara kullanımı değişkeni yaratmaktadır; sigara içenlerde tahmini maliyet belirgin şekilde artmaktadır.

##  Geliştirme Fikirleri

- [ ] Doğrusal regresyona ek olarak Random Forest / Gradient Boosting gibi modellerle karşılaştırma
- [ ] Model performansını iyileştirmek için özellik mühendisliği (ör. yaş × sigara etkileşimi)
- [ ] Streamlit Community Cloud üzerinde canlı yayına alma
- [ ] Birim testleri ekleme
##  Geliştirici

**Fatih Akçay**
[GitHub](https://github.com/fatihakcay23)

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
