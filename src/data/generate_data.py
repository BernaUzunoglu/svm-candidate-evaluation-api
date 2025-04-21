import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from src.config import Config

def generate_candidate_data(num_samples=200):
    """
    Aday verilerini oluşturur.
    
    Args:
        num_samples (int): Oluşturulacak örnek sayısı
        
    Returns:
        pd.DataFrame: Aday verilerini içeren DataFrame
    """
    fake = Faker('tr_TR')
    Faker.seed(42)  # Tekrarlanabilirlik için
    
    # Boş listeler oluştur
    ad_soyad = []
    dogum_tarihi = []
    universite = []
    bolum = []
    mezuniyet_yili = []
    tecrube_yili = []
    teknik_puan = []
    
    # Her aday için veri üret
    for _ in range(num_samples):
        # Kişisel bilgiler
        ad_soyad.append(fake.name())
        dogum_tarihi.append(fake.date_of_birth(minimum_age=22, maximum_age=45))
        
        # Eğitim bilgileri
        universite.append(fake.company() + " Üniversitesi")
        bolum.append(fake.random_element(elements=(
            "Bilgisayar Mühendisliği",
            "Yazılım Mühendisliği",
            "Elektrik-Elektronik Mühendisliği",
            "Bilgisayar Bilimleri",
            "Matematik Mühendisliği"
        )))
        
        # Mezuniyet yılı (2010-2023 arası)
        mezuniyet = fake.random_int(min=2010, max=2023)
        mezuniyet_yili.append(mezuniyet)
        
        # Tecrübe yılı (0-10 arası)
        tecrube = fake.random_int(min=0, max=10)
        tecrube_yili.append(tecrube)
        
        # Teknik puan (0-100 arası)
        # Tecrübe ve mezuniyet yılına göre puanı etkile
        base_score = fake.random_int(min=30, max=90)
        experience_bonus = tecrube * 2  # Her yıl için 2 puan bonus
        graduation_bonus = (2023 - mezuniyet) * 1  # Her yıl için 1 puan bonus
        final_score = min(100, base_score + experience_bonus + graduation_bonus)
        teknik_puan.append(final_score)
    
    # Etiketleri belirle
    etiket = np.where((np.array(tecrube_yili) < 2) & (np.array(teknik_puan) < 60), 1, 0)
    
    # DataFrame oluştur
    data = pd.DataFrame({
        'ad_soyad': ad_soyad,
        'dogum_tarihi': dogum_tarihi,
        'universite': universite,
        'bolum': bolum,
        'mezuniyet_yili': mezuniyet_yili,
        'tecrube_yili': tecrube_yili,
        'teknik_puan': teknik_puan,
        'etiket': etiket
    })
    
    return data

def save_data(data, filename='candidate_data.csv'):
    """
    Veriyi CSV dosyasına kaydeder.
    
    Args:
        data (pd.DataFrame): Kaydedilecek veri
        filename (str): Dosya adı
    """
    data.to_csv(Config.PROJECT_ROOT / f'data/{filename}', index=False)
    print(f"Veri başarıyla kaydedildi: {filename}")

if __name__ == "__main__":
    # Veri oluştur ve kaydet
    data = generate_candidate_data()
    save_data(data,"candidate_data_v1.csv")