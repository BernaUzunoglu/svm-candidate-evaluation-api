# İşe Alımda Aday Seçimi: SVM ile Başvuru Değerlendirme

Bu proje, yazılım geliştirici pozisyonu için başvuran adayların, tecrübe yılı ve teknik sınav puanına göre işe alınıp alınmayacağını tahmin eden bir makine öğrenimi modeli içerir.

## 🎯 Proje Aşamaları

### 1. Veri Üretimi ve Ön İşleme
- Faker kütüphanesi ile gerçekçi aday verileri üretimi
- Her aday için detaylı profil bilgileri:
  - Kişisel bilgiler (ad-soyad, doğum tarihi)
  - Eğitim bilgileri (üniversite, bölüm, mezuniyet yılı)
  - İş tecrübesi ve teknik puan
- Veri ölçeklendirme ve normalizasyon

### 2. Model Geliştirme
- SVM (Support Vector Machine) modeli eğitimi
- Lineer kernel kullanımı
- Model performans değerlendirmesi:
  - Accuracy score
  - Confusion matrix
  - Classification report
- Karar sınırı görselleştirmesi

### 3. API Geliştirme (FastAPI)
- RESTful API endpoints:
  - `/predict`: Aday değerlendirmesi
  - `/train`: Model yeniden eğitimi
  - `/docs`: Swagger UI dokümantasyonu
- Özellikler:
  - Otomatik model yükleme/kaydetme
  - Güven skoru hesaplama
  - Hata yönetimi
  - Veri doğrulama

## 📊 Model Kriterleri

- **İşe Alınmaz (1)**:
  - Tecrübesi 2 yıldan az VE
  - Teknik puanı 60'tan düşük
- **İşe Alınır (0)**:
  - Diğer tüm durumlar

## 🛠️ Proje Yapısı

```
.
├── data/               # Veri dosyaları
│   ├── candidate_data.csv    # Aday verileri
│   ├── model.joblib         # Kayıtlı model
│   └── decision_boundary.png # Karar sınırı görseli
├── src/               # Kaynak kodlar
│   ├── data/         # Veri işleme modülleri
│   │   └── generate_data.py  # Veri üretimi
│   ├── models/       # Model eğitimi ve değerlendirme
│   │   └── train_model.py    # Model işlemleri
│   ├── api/          # API modülleri
│   │   ├── app.py           # FastAPI uygulaması
│   │   └── test_client.py   # API test istemcisi
│   └── main.py       # Ana uygulama
├── notebooks/         # Jupyter notebook'lar
├── tests/            # Test dosyaları
├── requirements.txt  # Bağımlılıklar
└── README.md         # Proje dokümantasyonu
```

## 🚀 Kurulum ve Kullanım

1. **Gerekli paketleri yükleyin**:
```bash
pip install -r requirements.txt
```

2. **API'yi başlatın**:
```bash
python src/api/app.py
```

3. **Test istemcisini çalıştırın**:
```bash
python src/api/test_client.py
```

4. **API'yi test edin**:
- Tarayıcıda `http://localhost:8000/docs` adresine gidin
- Swagger UI üzerinden endpoint'leri test edin

## 📈 Model Performansı

Model performansı şu metriklerle değerlendirilir:
- Accuracy (Doğruluk)
- Precision (Kesinlik)
- Recall (Duyarlılık)
- F1-Score

## 🔄 Model Yeniden Eğitimi

Model, aşağıdaki durumlarda yeniden eğitilebilir:
- Yeni veri eklendiğinde
- Model performansı düştüğünde
- `/train` endpoint'i çağrıldığında

## 📱 API Kullanım Örnekleri

### Tahmin Yapma
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "tecrube_yili": 3.5,
    "teknik_puan": 75.0
})
```

### Modeli Yeniden Eğitme
```python
response = requests.post("http://localhost:8000/train")
```

## 🔮 Geliştirme Alanları

1. **Model İyileştirmeleri**:
   - Farklı kernel'ler deneme (`rbf`, `poly`)
   - Hiperparametre optimizasyonu
   - Cross-validation

2. **Özellik Mühendisliği**:
   - Yeni özellikler ekleme
   - Feature importance analizi
   - Outlier tespiti

3. **API Geliştirmeleri**:
   - Authentication ekleme
   - Rate limiting
   - Logging sistemi
   - Monitoring

4. **Frontend Geliştirme**:
   - Web arayüzü
   - Dashboard
   - Görselleştirmeler 