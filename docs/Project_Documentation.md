# İşe Alımda Aday Seçimi: SVM ile Başvuru Değerlendirme

Bu proje, bir teknoloji firmasında insan kaynakları (HR) ekibinin, yazılım geliştirici pozisyonu için başvuran adayları değerlendirmesine yardımcı olacak bir **Makine Öğrenmesi (SVM)** tabanlı tahmin sistemidir.

## 🎯 Amaç

Adayların:
- `tecrube_yili`: Toplam yazılım geliştirme deneyimi (0–10 yıl)
- `teknik_puan`: Teknik sınavdan aldığı puan (0–100)

verilerine göre **işe alınıp alınmayacağını** tahmin eden bir model oluşturmak.

---
## 🚀 Canlı Demo

Projeye ait Streamlit uygulamasını canlı olarak incelemek için aşağıdaki bağlantıya tıklayabilirsiniz:

👉 [Yazılım Geliştirici Aday Değerlendirme Uygulaması – Canlı Demo](https://svm-candidate-evaluation-api.streamlit.app/)


## 🏷️ Etiketleme Kriteri (Kural Tabanlı)

- **Tecrübe < 2 yıl** **ve** **Teknik Puan < 60** ise:  
  🔴 `etiket = 1` (İşe Alınmadı)
  
- Diğer tüm durumlar:  
  🟢 `etiket = 0` (İşe Alındı)

---

## 📊 Kullanılan Yöntemler

- 📦 Veri üretimi: `random` (veya `Faker`)
- 📏 Veri ölçekleme: `StandardScaler`
- 🤖 Model: `SVC(kernel='linear')`
- 📈 Başarı değerlendirme: `accuracy_score`, `confusion_matrix`, `classification_report`
- 🧩 Görselleştirme: `matplotlib` ile karar sınırı çizimi

---

## 🔧 Proje Adımları

1. **200+ aday verisi** üretildi (Faker ile).
2. Kural tabanlı olarak etiketlendi.
3. Eğitim ve test setine ayrıldı (`train_test_split`).
4. Veriler `StandardScaler` ile normalize edildi.
5. `SVC(kernel='linear')` ile model eğitildi.
6. Karar sınırı matplotlib ile çizildi.
7. Kullanıcıdan giriş alınıp tahmin yapıldı.
8. Model başarı metrikleri hesaplandı.


## 🎥 Uygulama Arayüzü (PDF)

Bu projeye ait Streamlit tabanlı görsel kullanıcı arayüzünü aşağıdaki PDF dokümanından inceleyebilirsiniz:

📄 [Streamlit Demo Görselleri (PDF)](https://github.com/BernaUzunoglu/svm-candidate-evaluation-api/blob/main/assets/streamlit-ui-demo.pdf)

---