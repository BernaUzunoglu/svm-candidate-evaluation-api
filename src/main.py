import os
from src.data.generate_data import generate_candidate_data, save_data
from src.models.train_model import (
    load_and_preprocess_data,
    train_svm_models,
    evaluate_and_save_models,
    plot_decision_boundary,
    predict_candidate
)

def main():
    # Veri dizinini oluştur
    os.makedirs('data', exist_ok=True)
    
    # Veri oluştur ve kaydet
    print("Veri oluşturuluyor...")
    data = generate_candidate_data()
    save_data(data)
    
    # Veriyi yükle ve ön işle
    print("\nVeri yükleniyor ve ön işleniyor...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Modeli eğit
    print("\nModel eğitiliyor...")
    model = train_svm_models(X_train, y_train)
    
    # Modeli değerlendir
    print("\nModel değerlendiriliyor...")
    evaluate_and_save_models(model, X_test, y_test)
    
    # Karar sınırını görselleştir
    print("\nKarar sınırı görselleştiriliyor...")
    plot_decision_boundary(model, X_train, y_train, scaler)
    
    # Kullanıcıdan girdi al ve tahmin yap
    while True:
        print("\nYeni bir aday için tahmin yapmak için bilgileri girin:")
        try:
            tecrube_yili = float(input("Tecrübe yılı (0-10): "))
            teknik_puan = float(input("Teknik puan (0-100): "))
            
            if not (0 <= tecrube_yili <= 10 and 0 <= teknik_puan <= 100):
                print("Hata: Geçersiz değer aralığı!")
                continue
            
            prediction = predict_candidate(model, scaler, tecrube_yili, teknik_puan)
            result = "İşe alınmaz" if prediction == 1 else "İşe alınır"
            print(f"\nTahmin Sonucu: {result}")
            
        except ValueError:
            print("Hata: Geçerli bir sayı girin!")
        
        devam = input("\nBaşka bir tahmin yapmak ister misiniz? (e/h): ")
        if devam.lower() != 'e':
            break

if __name__ == "__main__":
    main() 