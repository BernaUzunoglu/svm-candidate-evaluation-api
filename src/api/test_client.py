import requests
import json

BASE_URL = "http://localhost:8000"

def test_predict():
    # Test verisi
    test_data = {
        "tecrube_yili": 3.5,
        "teknik_puan": 75.0
    }
    
    # Tahmin isteği gönder
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    
    # Sonucu yazdır
    print("\nTahmin Sonucu:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

def test_train():
    # Modeli yeniden eğit
    response = requests.post(f"{BASE_URL}/train")
    
    # Sonucu yazdır
    print("\nEğitim Sonucu:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    print("API Test İstemcisi")
    print("=================")
    
    while True:
        print("\n1. Tahmin Yap")
        print("2. Modeli Yeniden Eğit")
        print("3. Çıkış")
        
        choice = input("\nSeçiminiz (1-3): ")
        
        if choice == "1":
            test_predict()
        elif choice == "2":
            test_train()
        elif choice == "3":
            break
        else:
            print("Geçersiz seçim!") 