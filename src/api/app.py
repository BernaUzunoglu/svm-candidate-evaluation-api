import os
import sys
import joblib
import numpy as np
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sklearn.metrics import accuracy_score
from src.models.train_model import train_svm_models, predict_candidate
from src.data.generate_data import generate_candidate_data, save_data

# Proje kök dizinini Python path'ine ekle
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

app = FastAPI(
    title="İşe Alım Aday Değerlendirme API",
    description="Yazılım geliştirici adaylarının işe alım değerlendirmesi için SVM tabanlı API",
    version="1.0.0"
)

# Model ve scaler'ı global olarak tut
model = None
scaler = None

class CandidateInput(BaseModel):
    tecrube_yili: float
    teknik_puan: float

class CandidateResponse(BaseModel):
    prediction: int
    result: str
    confidence: float

class TrainingResponse(BaseModel):
    message: str
    accuracy: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    global model, scaler
    # Veri dizinini oluştur
    os.makedirs('data', exist_ok=True)
    print("📌 Startup event başladı...")

    # Eğer model dosyası yoksa, yeni model oluştur
    if not os.path.exists('data/best_model_linear.joblib'):
        # Veri oluştur
        data = generate_candidate_data()
        save_data(data)

        # Modeli eğit
        from src.models.train_model import load_and_preprocess_data
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
        model = train_svm_models(X_train, y_train)

        # Modeli kaydet
        joblib.dump((model, scaler), 'data/best_model_linear.joblib')
        print("✅ Model ---------------------")
    else:
        # Kayıtlı modeli yükle
        model, scaler = joblib.load('data/best_model_linear.joblib')
        print("✅ Model yüklendi ------------------.")

@app.get("/")
async def root():
    return {
        "message": "İşe Alım Aday Değerlendirme API'ye Hoş Geldiniz",
        "endpoints": {
            "/predict": "Aday değerlendirmesi yapmak için",
            "/train": "Modeli yeniden eğitmek için",
            "/docs": "API dokümantasyonu için"
        }
    }

@app.post("/predict", response_model=CandidateResponse)
async def predict(candidate: CandidateInput):
    try:
        # Girdi kontrolü
        if not (0 <= candidate.tecrube_yili <= 10 and 0 <= candidate.teknik_puan <= 100):
            raise HTTPException(
                status_code=400,
                detail="Geçersiz değer aralığı! Tecrübe yılı 0-10, teknik puan 0-100 arası olmalıdır."
            )
        
        # Tahmin yap
        prediction = predict_candidate(model, scaler, candidate.tecrube_yili, candidate.teknik_puan)
        
        # Güven skorunu hesapla
        X = np.array([[candidate.tecrube_yili, candidate.teknik_puan]])
        X_scaled = scaler.transform(X)
        confidence = abs(model.decision_function(X_scaled)[0])
        
        return {
            "prediction": int(prediction),
            "result": "İşe alınmaz" if prediction == 1 else "İşe alınır",
            "confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train_model():
    try:
        global model, scaler

        from src.models.train_model import (
            load_and_preprocess_data,
            train_svm_models,
            evaluate_and_save_models,
            save_best_model_as_pickle
        )

        # Veri oluştur ve kaydet
        data = generate_candidate_data()
        save_data(data)

        # Veriyi yükle ve ölçekle
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

        # Tüm modelleri eğit
        models = train_svm_models(X_train, y_train)

        # Değerlendirme ve kayıt
        output_dir = 'src/results'
        evaluate_and_save_models(models, X_test, y_test, output_dir)
        save_best_model_as_pickle(models, X_test, y_test, scaler, 'data')

        # En iyi modeli yükle
        model, scaler = joblib.load('data/best_model_linear.joblib')  # dosya adını dinamikleştirebilirsin
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            "message": "Model başarıyla yeniden eğitildi",
            "accuracy": accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 