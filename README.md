# İşe Alımda Aday Seçimi: SVM ile Başvuru Değerlendirme

Bu proje, yazılım geliştirici pozisyonu için başvuran adayların, tecrübe yılı ve teknik sınav puanına göre işe alınıp alınmayacağını tahmin eden bir makine öğrenimi modeli içerir.

---
## 🚀 Canlı Demo

Projeye ait Streamlit uygulamasını canlı olarak incelemek için aşağıdaki bağlantıya tıklayabilirsiniz:

👉 [Yazılım Geliştirici Aday Değerlendirme Uygulaması – Canlı Demo](https://svm-candidate-evaluation-api.streamlit.app/)
---

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

## 📁 Proje Yapısı

Proje aşağıdaki gibi yapılandırılmıştır. Modülerlik, okunabilirlik ve sürdürülebilirlik göz önünde bulundurularak organize edilmiştir.


```
svm-candidate-evaluation-api/
│
├── data/                              # Model ve veri dosyalarının saklandığı dizin
│   ├── best_model_linear.joblib       # En iyi performansı veren modelin kaydı
│   ├── candidate_data.csv             # Üretilen örnek aday verileri
│   └── model.joblib                   # Alternatif model dosyası
│
├── docs/                              # Proje dokümantasyonları
│   └── Project_Documentation.md       # Proje ile ilgili açıklamalar ve kullanım bilgileri
│
├── src/                               # Tüm kaynak kodları içeren ana klasör
│   ├── api/                           # FastAPI tabanlı REST API endpointleri
│   │   ├── app.py                     # Uygulamanın ana API dosyası
│   │   └── test_client.py             # API testleri için basit istemci
│   │
│   ├── data/                          # Veri üretme işlemlerini içeren modül
│   │   └── generate_data.py           # Aday verisi oluşturan fonksiyonlar
│   │
│   ├── models/                        # Model eğitimi, değerlendirmesi ve kaydı
│   │   └── train_model.py             # SVM modellerini eğiten ve kaydeden kodlar
│   │
│   └── results/                       # Model değerlendirme çıktıları
│       ├── all_results.csv            # Tüm modellerin karşılaştırmalı sonuçları
│       ├── decision_boundary_*.png    # Modellerin karar sınırlarını görselleştiren grafikler
│       └── *_report.json              # Her model için detaylı değerlendirme metrikleri
│
├── config.py                          # Genel konfigürasyonları içeren dosya
├── main.py                            # Opsiyonel olarak ana çalıştırma dosyası
├── .env                               # Ortam değişkenlerini tanımlayan dosya
├── .gitignore                         # Git tarafından takip edilmeyecek dosyalar
├── README.md                          # Projeye genel bakış ve kullanım yönergeleri
├── requirements.txt                   # Gerekli Python paketleri
└── streamlit_app.py                   # Streamlit tabanlı web arayüzü

```

## 🚀 Kurulum ve Kullanım

1. **Gerekli paketleri yükleyin**:
```bash
pip install -r requirements.txt
```

2. **FastAPI'yi başlatın**:
```bash
uvicorn src.api.app:app --reload
```
- API dokümantasyonuna erişmek için: `http://localhost:8000/docs`
- Swagger UI üzerinden endpoint'leri test edin

3. **Streamlit uygulamasını başlatın**:
```bash
streamlit run streamlit_app.py
```
- Web arayüzüne erişmek için: `http://localhost:8501`

4. **Test istemcisini çalıştırın**:
```bash
python src/api/test_client.py
```

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

---
## 🎥 Uygulama Arayüzü (PDF)

Bu projeye ait Streamlit tabanlı görsel kullanıcı arayüzünü aşağıdaki PDF dokümanından inceleyebilirsiniz:

📄 [Streamlit Demo Görselleri (PDF)](https://github.com/BernaUzunoglu/svm-candidate-evaluation-api/blob/main/assets/streamlit-ui-demo.pdf)

---