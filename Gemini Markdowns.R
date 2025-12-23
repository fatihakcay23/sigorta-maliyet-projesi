# ==============================================================================
# SİGORTA MALİYETİ REGRESYON ANALİZİ
# ==============================================================================

# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ
# ------------------------------------------------------------------------------
library(tidyverse)
library(ggplot2)
library(corrplot)
library(caTools)
library(car)
library(lmtest)
# install.packages("fastDummies") # Eğer yüklü değilse bu satırı çalıştırın
library(fastDummies)



# ==============================================================================
# 2. VERİ YÜKLEME VE ÖN İNCELEME
# ==============================================================================

df <- read.csv(file.choose())
cat("Veri Seti Boyutu:", dim(df), "\n")
print(head(df))


# Analize başlamadan önce veriyi R ortamına aktarıyoruz. file.choose() komutu, 
# dosya yolunu elle yazmak yerine bir pencere açıp dosyayı seçmeni sağlar. 
# dim() komutuyla satır ve sütun sayısını kontrol edip verinin büyüklüğünü anlıyoruz. 
# head() ile de verinin ilk 6 satırına bakarak yapının düzgün yüklenip yüklenmediğini teyit ediyoruz.

cat("\n--- Eksik Veri Sayıları ---\n")
print(colSums(is.na(df)))


# Veri setinde boş (NA) değerler olup olmadığını kontrol ediyoruz. 
# Eğer boş değerler varsa model hata verebilir veya yanlış sonuçlar üretebilir. 
# colSums(is.na(df)) kodu, her sütundaki eksik veri sayısını toplayıp bize gösterir.

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


# Veri setindeki aşırı uç değerleri (outliers) tespit ediyoruz. 
# IQR (Interquartile Range) yöntemi istatistikte en yaygın kullanılan yöntemdir. 
# Verinin orta %50'lik kısmını baz alarak alt ve üst sınırlar belirlenir. 
# Bu sınırların dışında kalan değerler "aykırı" kabul edilir. 
# Aykırı değerler, regresyon doğrusunu kendisine çekerek modelin genel başarısını düşürebilir.

# B) Encoding (Kategorik Dönüşümler)
df <- df %>%
  mutate(
    sex_encoded = ifelse(sex == "male", 1, 0),
    smoker_encoded = ifelse(smoker == "yes", 1, 0)
  )


# Makine öğrenmesi modelleri metin (string) verilerle matematiksel işlem yapamaz. 
# Bu yüzden "Cinsiyet" ve "Sigara Kullanımı" gibi iki seçenekli kategorik verileri sayıya çeviriyoruz. 
# Erkek için 1, Kadın için 0; Sigara içen için 1, içmeyen için 0 ataması yaparak 
# bilgisayarın anlayacağı formata getiriyoruz (Label Encoding).

df <- dummy_cols(df, select_columns = "region", remove_first_dummy = TRUE)
names(df) <- make.names(names(df)) 


# "Bölge" (region) gibi ikiden fazla seçeneği olan kategorik veriler için One-Hot Encoding yapıyoruz. 
# Her bölge için ayrı bir sütun oluşturulur (0 veya 1 değeri alan). 
# remove_first_dummy = TRUE yapmamızın sebebi "Dummy Variable Trap" (Kukla Değişken Tuzağı) denilen durumdan kaçınmaktır. 
# Yani 4 bölge varsa 3 sütun yeterlidir; hepsi 0 ise kişi 4. bölgededir demektir.

# ==============================================================================
# 4. KEŞİFSEL VERİ ANALİZİ (EDA)
# ==============================================================================

selected_cols <- c("age", "bmi", "children", "charges", "sex_encoded", "smoker_encoded")
cor_matrix <- cor(df[, selected_cols])

corrplot(cor_matrix, method = "color", type = "upper", 
         addCoef.col = "black", tl.col = "black", title = "Korelasyon Matrisi")


# Hangi değişkenlerin birbirleriyle ve en önemlisi hedef değişkenimiz olan "charges" (maliyet) 
# ile ilişkili olduğunu görmek için korelasyon matrisine bakıyoruz. 
# Renkli harita üzerinde 1'e yakın değerler pozitif, -1'e yakın değerler negatif ilişkiyi gösterir. 
# Örneğin sigara içme durumu ile maliyet arasında yüksek bir korelasyon görmeyi bekleriz.

ggplot(df, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Sigara Kullanımına Göre Maliyet Dağılımı")


# Sigara kullanımının maliyet üzerindeki etkisini görsel olarak kanıtlamak için Kutu Grafiği (Boxplot) çizdiriyoruz. 
# Eğer sigara içenlerin kutusu, içmeyenlere göre çok daha yukardaysa, 
# bu değişkenin model için çok önemli (belirleyici) bir özellik olduğunu anlarız.

# ==============================================================================
# 5. MODEL KURULUMU (TRAIN-TEST SPLIT & LINEAR REGRESSION)
# ==============================================================================

set.seed(42)
split <- sample.split(df$charges, SplitRatio = 0.8)
train_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

cat("\nTrain Set Boyutu:", nrow(train_set), "\n")
cat("Test Set Boyutu:", nrow(test_set), "\n")


# Modelin başarısını ölçebilmek için elimizdeki veriyi ikiye bölüyoruz. 
# %80'ini (Train Set) modele "öğretmek" için, kalan %20'sini (Test Set) ise modelin hiç görmediği 
# veriler üzerinde ne kadar iyi tahmin yaptığını "sınamak" için ayırıyoruz. 
# set.seed(42) komutu, her çalıştırdığımızda aynı rastgele ayrımı yapmasını sağlar (tekrar edilebilirlik için).

model_formula <- charges ~ age + bmi + children + sex_encoded + smoker_encoded + 
  region_northwest + region_southeast + region_southwest

model <- lm(model_formula, data = train_set)

cat("\n--- Model Özeti ---\n")
summary(model)


# Çoklu Doğrusal Regresyon (Multiple Linear Regression) modelimizi kuruyoruz. 
# "charges" hedef değişkenimizdir; yaş, bmi, çocuk sayısı vb. ise onu tahmin etmek için kullandığımız özelliklerdir. 
# lm() fonksiyonu ile modeli eğitiyoruz. summary(model) çıktısı bize modelin istatistiksel detaylarını verir: 
# Hangi değişkenler anlamlı (P-value < 0.05), model veriyi ne kadar açıklıyor (R-squared) gibi kritik bilgiler burada yer alır.

# ==============================================================================
# 6. MODEL VARSAYIMLARININ KONTROLÜ
# ==============================================================================

# A) VIF (Çoklu Bağıntı) Analizi
cat("\n--- VIF Değerleri ---\n")
vif_values <- vif(model)
print(vif_values)
if(any(vif_values > 5)) cat("UYARI: 5'ten büyük VIF değeri var!\n") else cat("Multicollinearity sorunu görünmüyor.\n")


# Regresyon modelinin güvenilir olması için değişkenlerin birbirleriyle çok yüksek ilişkili olmaması gerekir. 
# VIF (Variance Inflation Factor) testi bunu ölçer. 
# Eğer VIF değeri 5 veya 10'un üzerindeyse "Çoklu Bağlantı" (Multicollinearity) sorunu var demektir; 
# bu da modelin katsayılarının güvenilmez olmasına yol açar.

# B) Normallik Testi (Shapiro-Wilk)
residuals <- residuals(model)
shapiro_res <- shapiro.test(residuals[1:5000]) 
cat("\n--- Shapiro-Wilk Normallik Testi ---\n")
print(shapiro_res)


# Modelin yaptığı hataların (artıkların) Normal Dağılıma (Çan Eğrisi) uyup uymadığını test ediyoruz. 
# İdeal bir regresyon modelinde hataların rastgele ve normal dağılması beklenir. 
# Eğer p-value < 0.05 çıkarsa hatalar normal dağılmıyor demektir, bu da modelin bazı bilgileri yakalayamadığını gösterebilir.

# C) Heteroskedastisite Testi (Breusch-Pagan)
cat("\n--- Breusch-Pagan Testi ---\n")
print(bptest(model))


# Hataların varyansının sabit olup olmadığını kontrol ediyoruz (Homoscedasticity varsayımı). 
# Eğer varyans sabit değilse (Heteroskedastisite), modelin tahmin hataları bazı durumlarda (örneğin yüksek fiyatlarda) 
# sistematik olarak artıyor demektir. Bu testin sonucunda p-value < 0.05 ise varyans sabit değildir, yani bir sorun vardır.

# D) Otokorelasyon (Durbin-Watson)
cat("\n--- Durbin-Watson Testi ---\n")
print(dwtest(model))


# Hataların birbirleriyle ilişkili olup olmadığını (Otokorelasyon) test ederiz. 
# Özellikle zaman serisi verilerinde önemlidir ama burada da satırlar arası bağımlılık olmaması için bakılır. 
# Durbin-Watson değeri 2 civarındaysa otokorelasyon yoktur (istenen durum). 
# 0 veya 4'e yakınsa hatalar birbirini etkiliyor demektir.

par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))


# İstatistiksel testlerin yanı sıra, modelin varsayımlarını grafiklerle (Residual Plots) gözle kontrol ediyoruz. 
# Özellikle "Residuals vs Fitted" grafiğinde kırmızı çizginin düz olması ve noktaların rastgele dağılması istenir.

# ==============================================================================
# 7. MODEL PERFORMANSI VE TAHMİN
# ==============================================================================

predictions <- predict(model, newdata = test_set)

rmse_val <- sqrt(mean((test_set$charges - predictions)^2))
mae_val <- mean(abs(test_set$charges - predictions))
r2_val <- summary(lm(predictions ~ test_set$charges))$r.squared

cat("\n--- Test Seti Performansı ---\n")
cat("RMSE:", round(rmse_val, 2), "\n")
cat("MAE :", round(mae_val, 2), "\n")
cat("R²  :", round(r2_val, 4), "\n")


# Son aşamada, ayırdığımız Test seti üzerinde modelin gerçek performansını ölçüyoruz. 
# RMSE (Hata Kareler Ortalamasının Karekökü) ve MAE (Ortalama Mutlak Hata), modelin tahminlerinin 
# gerçek değerden ortalama ne kadar saptığını (örneğin kaç dolar yanıldığını) gösterir. 
# R² (R-kare) ise modelin test setindeki değişkenliği ne kadar iyi açıkladığını gösteren 0 ile 1 arası bir puandır. 
# Yüksek R² ve düşük hata (RMSE/MAE) istenir.