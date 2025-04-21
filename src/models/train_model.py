import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from src.config import Config
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_and_preprocess_data(filename='candidate_data.csv'):
    data = pd.read_csv(Config.PROJECT_ROOT / f'data/{filename}')
    X = data[['tecrube_yili', 'teknik_puan']]
    y = data['etiket']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_svm_models(X_train, y_train, kernels=None):
    if kernels is None:
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']

    models = {}
    for kernel in kernels:
        model = SVC(kernel=kernel)
        model.fit(X_train, y_train)
        models[kernel] = model

    return models

def evaluate_and_save_models(models, X_test, y_test, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)

        result = {
            'kernel': name,
            'accuracy': acc,
            'confusion_matrix': cm,
            'classification_report': report
        }

        summary.append(result)

        with open(output_path / f'{name}_report.json', 'w') as f:
            json.dump(result, f, indent=4)

    pd.DataFrame(summary).to_csv(output_path / 'all_results.csv', index=False)
    print(f"Tüm çıktılar '{output_path}' klasörüne kaydedildi.")

def plot_decision_boundary(model, X, y, scaler, kernel_name, output_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.6)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    try:
        Z = model.decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
                   alpha=0.5, linestyles=['--', '-', '--'])
    except:
        pass

    plt.xlabel('Tecrübe Yılı (Ölçeklendirilmiş)')
    plt.ylabel('Teknik Puan (Ölçeklendirilmiş)')
    plt.title(f'SVM Karar Sınırı - {kernel_name}')
    plot_path = Path(output_dir) / f'decision_boundary_{kernel_name}.png'
    plt.savefig(plot_path)
    plt.close()

def save_best_model_as_pickle(models, X_test, y_test, scaler, output_dir):
    best_acc = -1
    best_model = None
    best_name = ''

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    if best_model:
        pkl_path = Path(output_dir) / f'best_model_{best_name}.joblib'
        # joblib.dump(best_model, pkl_path)
        joblib.dump((best_model, scaler), pkl_path)

        print(f"✅ En iyi model ({best_name}, accuracy={best_acc:.4f}) olarak kaydedildi: {pkl_path}")

def predict_candidate(model, scaler, tecrube_yili, teknik_puan):
    X = np.array([[tecrube_yili, teknik_puan]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return prediction

if __name__ == "__main__":
    output_dir = Config.PROJECT_ROOT / 'src/results'

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    models = train_svm_models(X_train, y_train)
    evaluate_and_save_models(models, X_test, y_test, output_dir)

    for name, model in models.items():
        plot_decision_boundary(model, X_train, y_train, scaler, name, output_dir)

    save_best_model_as_pickle(models, X_test, y_test, Config.PROJECT_ROOT / 'data')
