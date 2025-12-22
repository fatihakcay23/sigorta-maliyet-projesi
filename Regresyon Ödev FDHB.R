# ==============================================================================
# SİGORTA MALİYETİ REGRESYON ANALİZİ (R VERSİYONU)
# ==============================================================================

# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ
# Eğer yüklü değillerse install.packages("paket_adi") ile yükleyiniz.
library(tidyverse)    # Veri manipülasyonu ve okuma (pandas karşılığı)
library(ggplot2)      # Görselleştirme (matplotlib/seaborn karşılığı)
library(corrplot)     # Korelasyon matrisi görseli
library(caTools)      # Train-Test Split için
library(car)          # VIF testi için
library(lmtest)       # Breusch-Pagan ve Durbin-Watson testleri için
install.packages("fastDummies")
library(fastDummies)  # One-Hot Encoding için

# ==============================================================================
# 2. VERİ YÜKLEME VE ÖN İNCELEME
# ==============================================================================
# Dosya yolunu kendi bilgisayarına göre ayarla
df <- read.csv(file.choose())
cat("Veri Seti Boyutu:", dim(df), "\n")
print(head(df))

# Eksik Veri Kontrolü
cat("\n--- Eksik Veri Sayıları ---\n")
print(colSums(is.na(df)))

# ==============================================================================
# 3. VERİ ÖN İŞLEME (PREPROCESSING)
# ==============================================================================

# A) Aykırı Değer (Outlier) Analizi - IQR Yöntemi
numeric_cols <- c("age", "bmi", "children", "charges")

cat("\n--- Aykırı Değer Analizi (IQR) ---\n")
for(col in numeric_cols) {
  Q1 <- quantile(df[[col]], 0.25)
  Q3 <- quantile(df[[col]], 0.75)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  
  outliers <- df[[col]][df[[col]] < lower | df[[col]] > upper]
  cat(col, ":", length(outliers), "adet aykırı değer bulundu.\n")
}

# B) Encoding (Kategorik Dönüşümler)
# Python'daki mantığın aynısını uyguluyoruz:
# Sex: female=0, male=1
# Smoker: no=0, yes=1
df <- df %>%
  mutate(
    sex_encoded = ifelse(sex == "male", 1, 0),
    smoker_encoded = ifelse(smoker == "yes", 1, 0)
  )

# Region için One-Hot Encoding (drop_first = TRUE mantığıyla)
# Python'da get_dummies(drop_first=True) kullanılmıştı.
# R'da fastDummies paketi ile yapıyoruz.
df <- dummy_cols(df, select_columns = "region", remove_first_dummy = TRUE)

# Sütun isimlerini düzeltelim (boşluk veya tire varsa)
names(df) <- make.names(names(df)) 
# Not: Python'da region_northwest olurken R'da region_northwest şeklinde gelebilir.

# ==============================================================================
# 4. KEŞİFSEL VERİ ANALİZİ (EDA)
# ==============================================================================

# Korelasyon Matrisi
# Sadece sayısal sütunları seçelim (Python notebook'undaki gibi)
selected_cols <- c("age", "bmi", "children", "charges", "sex_encoded", "smoker_encoded")
cor_matrix <- cor(df[, selected_cols])

# Görselleştirme
corrplot(cor_matrix, method = "color", type = "upper", 
         addCoef.col = "black", tl.col = "black", title = "Korelasyon Matrisi")

# Smoker vs Charges Boxplot (ggplot2 ile)
ggplot(df, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Sigara Kullanımına Göre Maliyet Dağılımı")

# ==============================================================================
# 5. MODEL KURULUMU (TRAIN-TEST SPLIT & LINEAR REGRESSION)
# ==============================================================================

# Train-Test Split (%80 Train, %20 Test)
set.seed(42) # Python'daki random_state=42 ile aynı olması için
split <- sample.split(df$charges, SplitRatio = 0.8)
train_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

cat("\nTrain Set Boyutu:", nrow(train_set), "\n")
cat("Test Set Boyutu:", nrow(test_set), "\n")

# Model Değişkenlerini Seçme
# Python'daki 'X' değişkenleri
model_formula <- charges ~ age + bmi + children + sex_encoded + smoker_encoded + 
  region_northwest + region_southeast + region_southwest

# Modeli Eğitme (OLS)
model <- lm(model_formula, data = train_set)

# Model Özeti (Statsmodels summary çıktısının R karşılığı)
cat("\n--- Model Özeti ---\n")
summary(model)

# ==============================================================================
# 6. MODEL VARSAYIMLARININ KONTROLÜ
# ==============================================================================

# A) VIF (Çoklu Bağıntı) Analizi
cat("\n--- VIF Değerleri ---\n")
vif_values <- vif(model)
print(vif_values)
if(any(vif_values > 5)) cat("UYARI: 5'ten büyük VIF değeri var!\n") else cat("Multicollinearity sorunu görünmüyor.\n")

# B) Normallik Testi (Shapiro-Wilk)
residuals <- residuals(model)
shapiro_res <- shapiro.test(residuals[1:5000]) # R'da max 5000 örneğe izin verilir
cat("\n--- Shapiro-Wilk Normallik Testi ---\n")
print(shapiro_res)

# C) Heteroskedastisite Testi (Breusch-Pagan)
cat("\n--- Breusch-Pagan Testi ---\n")
print(bptest(model))

# D) Otokorelasyon (Durbin-Watson)
cat("\n--- Durbin-Watson Testi ---\n")
print(dwtest(model))

# Varsayım Grafikleri (Residual Plots)
par(mfrow = c(2, 2)) # 4 grafiği tek ekrana sığdır
plot(model)
par(mfrow = c(1, 1)) # Geri düzelt

# ==============================================================================
# 7. MODEL PERFORMANSI VE TAHMİN
# ==============================================================================

# Test Seti Üzerinde Tahmin
predictions <- predict(model, newdata = test_set)

# Metrik Hesaplamaları
rmse_val <- sqrt(mean((test_set$charges - predictions)^2))
mae_val <- mean(abs(test_set$charges - predictions))
r2_val <- summary(lm(predictions ~ test_set$charges))$r.squared # Basit R2 hesabı

cat("\n--- Test Seti Performansı ---\n")
cat("RMSE:", round(rmse_val, 2), "\n")
cat("MAE :", round(mae_val, 2), "\n")
cat("R²  :", round(r2_val, 4), "\n")