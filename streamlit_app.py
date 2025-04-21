import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
from src.config import Config
import subprocess
import threading
import time

st.set_page_config(page_title="Aday Değerlendirme", page_icon="👩‍💻", layout="wide")

def run_fastapi():
    subprocess.Popen(["uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", "8000"])

# FastAPI’yi ayrı bir thread olarak başlat (bir kez çalışsın)
@st.cache_resource
def start_fastapi_once():
    threading.Thread(target=run_fastapi, daemon=True).start()
    time.sleep(2)  # API'nin ayağa kalkmasını bekle (2 saniye yeterli olur genelde)

# Başlatmayı tetikle
start_fastapi_once()

#  Kenarlardan ortalamak için custom padding (CSS)
st.markdown("""
    <style>
        .block-container {
            padding-left: 15rem;
            padding-right: 15rem;
        }
    </style>
""", unsafe_allow_html=True)

# Giriş Başlığı
st.title("👩‍💻 Yazılım Geliştirici Aday Değerlendirme")

# Sekmeler
tab1, tab2, tab3, tab4 = st.tabs(["📊 Tahmin", "🛠️ Model Eğitimi", "🔎 İstatistikler", "🧭 Karar Sınırı Grafikleri"])

# --- TAB 1: Tahmin ---
with tab1:
    st.subheader("Aday Tahmin Formu")
    tecrube = st.slider("Tecrübe Yılı (0-10)", 0.0, 10.0, step=0.5)
    teknik_puan = st.slider("Teknik Puan (0-100)", 0.0, 100.0, step=1.0)

    if st.button("Tahmin Et"):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json={
                "tecrube_yili": tecrube,
                "teknik_puan": teknik_puan
            })
            result = response.json()
            if response.status_code == 200:
                st.success(f"✅ Tahmin Sonucu: **{result['result']}**")
                st.info(f"🔢 Güven Skoru: `{result['confidence']:.2f}`")
            else:
                st.error(result["detail"])
        except Exception as e:
            st.error(f"API bağlantı hatası: {e}")

# --- TAB 2: Model Eğitimi ---
with tab2:
    st.subheader("Modeli Yeniden Eğit")
    if st.button("Modeli Eğit ve Doğruluğu Göster"):
        try:
            response = requests.post("http://127.0.0.1:8000/train")
            result = response.json()
            if response.status_code == 200:
                accuracy = result["accuracy"]
                st.success("✅ Model başarıyla eğitildi.")
                st.info(f"🎯 Doğruluk: `{accuracy:.2%}`")
                fig, ax = plt.subplots()
                ax.bar(["Accuracy"], [accuracy], color="green")
                ax.set_ylim(0, 1)
                ax.set_ylabel("Başarım")
                st.pyplot(fig)
            else:
                st.error(result["detail"])
        except Exception as e:
            st.error(f"API bağlantı hatası: {e}")

# --- TAB 3: Örnek Veri ---
with tab3:
    st.subheader("📊 Örnek Veri Dağılımı ve Analizi")

    try:
        df = pd.read_csv("data/candidate_data.csv")
        df["etiket_label"] = df["etiket"].map({0: "İşe Alınır", 1: "İşe Alınmaz"})

        # Temel istatistikler
        st.markdown("### 📌 Temel İstatistikler")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tecrübe Ort.", f"{df['tecrube_yili'].mean():.2f}")
            st.metric("Min Tecrübe", f"{df['tecrube_yili'].min()}")
        with col2:
            st.metric("Teknik Ort.", f"{df['teknik_puan'].mean():.2f}")
            st.metric("Max Teknik", f"{df['teknik_puan'].max()}")
        with col3:
            st.metric("Toplam Aday", len(df))

        # Etiket Dağılımı (Pie benzeri: Donut tarzı bar chart)
        st.markdown("### 🎯 Etiket Dağılımı")
        etiket_counts = df["etiket_label"].value_counts().reset_index()
        etiket_counts.columns = ["Durum", "Sayısı"]

        fig = px.pie(etiket_counts, names="Durum", values="Sayısı")
        st.plotly_chart(fig)

        # Teknik Puan Histogram (Bar chart alternatifi)
        st.markdown("### 🧠 Teknik Puan Dağılımı")
        teknik_binler = pd.cut(df["teknik_puan"], bins=10)
        bin_sayim = teknik_binler.value_counts().sort_index()

        # Bin aralıklarını string'e çevir ve df haline getir
        bin_df = pd.DataFrame({
            "Puan Aralığı": bin_sayim.index.astype(str),
            "Aday Sayısı": bin_sayim.values
        }).set_index("Puan Aralığı")

        st.bar_chart(bin_df)

        # Tecrübe vs Teknik Puan (Plotly ile scatter plot)
        st.markdown("### 🧪 Tecrübe vs Teknik Puan")

        import plotly.express as px

        scatter_fig = px.scatter(
            df,
            x="tecrube_yili",
            y="teknik_puan",
            color="etiket_label",
            labels={"etiket_label": "Durum"},
            title="Tecrübe ve Teknik Puan İlişkisi"
        )
        st.plotly_chart(scatter_fig)

        # Mezuniyet Yılına Göre Etiket Dağılımı (Bar chart)
        st.markdown("### 🎓 Mezuniyet Yılına Göre İşe Alınma")

        # Hazırlık
        mezun_df = df.groupby(["mezuniyet_yili", "etiket_label"]).size().unstack(fill_value=0)
        mezun_df.index = mezun_df.index.astype(int)
        mezun_df = mezun_df.sort_index()
        mezun_df.columns.name = None  # kolon başlığı kaldır

        # Melt (long format) - plotly bunu sever
        mezun_long = mezun_df.reset_index().melt(id_vars="mezuniyet_yili", var_name="Durum", value_name="Aday Sayısı")

        # Plotly ile çizim
        fig = px.line(mezun_long,
                      x="mezuniyet_yili",
                      y="Aday Sayısı",
                      color="Durum",
                      markers=True,
                      title="Mezuniyet Yılına Göre İşe Alınma")

        fig.update_layout(xaxis=dict(tickformat=".0f"))  # float tarihleri engelle

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Veri analizinde hata: {e}")

# --- TAB 4: Karar Sınırı Görselleri ---
with tab4:
    # --- METRİK TABLOSU ---
    st.markdown("### 📊 Karşılaştırmalı Performans Tablosu")

    kernel_reports = {
        "Linear": "linear_report.json",
        "Poly": "poly_report.json",
        "RBF": "rbf_report.json",
        "Sigmoid": "sigmoid_report.json"
    }

    summary_data = []
    for kernel, path in kernel_reports.items():
        with open(Config.PROJECT_ROOT / f"src/results/{path}", "r") as file:
            report = json.load(file)
            summary_data.append({
                "Kernel": kernel,
                "Accuracy": report["accuracy"],
                "Precision (Macro)": report["classification_report"]["macro avg"]["precision"],
                "Recall (Macro)": report["classification_report"]["macro avg"]["recall"],
                "F1 Score (Macro)": report["classification_report"]["macro avg"]["f1-score"]
            })

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(
        df_summary.style
        .format({
            "Accuracy": "{:.3f}",
            "Precision (Macro)": "{:.3f}",
            "Recall (Macro)": "{:.3f}",
            "F1 Score (Macro)": "{:.3f}"
        })
        .background_gradient(cmap="YlGnBu", axis=0)
    )

    # --- KARAR SINIRI GÖRSELLERİ ---
    st.markdown("### 🧭 Kernel Türlerine Göre Karar Sınırı Görselleri")

    col1, col2 = st.columns(2)
    with col1:
        st.image("src/results/decision_boundary_linear.png", caption="Linear Kernel")
        st.markdown(
            "**Linear Kernel:** Düz bir çizgiyle sınıfları ayırır. Veri doğrusal ayrılabiliyorsa hızlı ve etkilidir. "
            "**Veri setimiz için oldukça yeterli görünüyor.**")
    with col2:
        st.image("src/results/decision_boundary_rbf.png", caption="RBF Kernel")
        st.markdown(
            "**RBF Kernel:** Karmaşık, eğrisel karar sınırları çizebilir. Verinin yapısını öğrenmede çok başarılıdır. "
            "**Veri setimiz mükemmel sonuçlar vermiş.**")

    col3, col4 = st.columns(2)
    with col3:
        st.image("src/results/decision_boundary_poly.png", caption="Polynomial Kernel")
        st.markdown("**Polynomial Kernel:** Veriyi polinom fonksiyonlarla dönüştürerek ayırmaya çalışır. "
                    "Verinin doğasına uygun değilse precision düşebilir.")
    with col4:
        st.image("src/results/decision_boundary_sigmoid.png", caption="Sigmoid Kernel")
        st.markdown("**Sigmoid Kernel:** Genellikle sinir ağlarında benzerlik ölçmek için kullanılır. "
                    "**Veri setimizde etiket 1 sınıfını hiç öğrenememiş.**")

    # --- SONUÇ / ÖNERİ ---
    st.markdown("### ✅ Sonuç ve Öneri")

    st.success("""
        Veriye ve sonuçlara bakıldığında:

        - Hem **Linear** hem **RBF** kernel modeli %100 başarı ile sınıflandırma yapmış.
        - Ancak, RBF kernel daha esnek karar sınırı çizdiği için veri karmaşıklaştıkça daha avantajlı hale gelir.
        - **Şu anda düşük veri çeşitliliği olduğu için Linear kernel ideal.** 
        Daha fazla veri geldiğinde RBF kernel tercih edilebilir.

        > 🎯 **Şu an önerilen kernel: Linear**  
        """)

