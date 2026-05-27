# 🌤️ JakU — Dashboard Kualitas Udara DKI Jakarta

Platform monitoring kualitas udara DKI Jakarta berbasis machine learning **XGBoost**, dibangun dengan **Streamlit**. Mengikuti pipeline CRISP-DM dari notebook penelitian.

## 📂 Struktur Project

```
JakU/
├── app.py                     # Aplikasi Streamlit utama (4 halaman)
├── train_model.py             # Script training XGBoost dari Data_ISPU.csv
├── requirements.txt
├── .streamlit/config.toml
├── assets/
│   └── logo.svg               # Logo JakU
├── data/                      # Data dummy untuk dashboard
│   ├── ispu_dummy.csv
│   ├── wilayah_dummy.csv
│   ├── prediksi_dummy.csv
│   └── edukasi_dummy.csv
└── models/                    # Model terlatih (.pkl) - output dari notebook
    ├── model_xgboost.pkl      # ⭐ Model utama untuk prediksi
    ├── model_random_forest.pkl
    ├── model_svm.pkl
    ├── label_encoder.pkl
    ├── standard_scaler.pkl
    ├── fitur_polutan.pkl
    └── metrik.pkl
```

## 🚀 Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔁 Mengganti Model dengan Hasil Notebook Asli

File `.pkl` di folder `models/` saat ini di-generate dari `train_model.py` (synthetic data berbasis rules ISPU). Untuk memakai model dari notebook `ISPU_Klasifikasi_CRISP_DM_FIXED__4_.ipynb`, ada **2 opsi**:

### Opsi 1 — Copy langsung dari Colab
Setelah sel `joblib.dump(...)` di notebook dijalankan, download 6 file ini dari Colab:
- `model_xgboost.pkl`
- `model_random_forest.pkl`
- `model_svm.pkl`
- `label_encoder.pkl`
- `standard_scaler.pkl`
- `fitur_polutan.pkl`

Lalu **timpa** file di folder `models/`.

### Opsi 2 — Training ulang dengan data Anda
1. Letakkan `Data_ISPU.csv` (semicolon-delimited) di folder `data/`
2. Jalankan: `python train_model.py`
3. File `.pkl` akan otomatis ter-generate di folder `models/`

## ☁️ Deploy ke Streamlit Community Cloud

1. Push repo ke GitHub (sertakan folder `models/` agar `.pkl` ikut)
2. Buka [share.streamlit.io](https://share.streamlit.io/)
3. New app → pilih repo → main file: `app.py`
4. Deploy

> **Catatan:** Jika `.pkl` terlalu besar untuk Git biasa, gunakan **Git LFS** atau letakkan file di release GitHub, lalu download saat runtime.

## ✨ Fitur

| Halaman | Konten |
|---|---|
| **Dashboard** | Ringkasan ISPU DKI Jakarta, peta wilayah Folium, prediksi 7 hari, tren ISPU |
| **Detail Wilayah** | Tab per kota (Pusat, Utara, Barat, Selatan, Timur, Kep. Seribu) |
| **Simulasi Prediksi ISPU** | Slider 6 polutan + preset → prediksi via XGBoost |
| **Edukasi & Insight** | 5 kategori ISPU, dampak kesehatan, sumber polusi, tips |

### Popup "Informasi Polutan"
Muncul ketika user klik:
- **"Lihat penjelasan polutan"** di Dashboard
- **"Lihat penjelasan polutan"** di Detail Wilayah
- **"Info"** (di samping "Komposisi Polutan") di Simulasi Prediksi ISPU

## 🧠 Integrasi Model XGBoost

Fungsi `prediksi_ispu_xgboost()` di `app.py` mengikuti pipeline `prediksi_ispu()` dari notebook (cell 70):

1. Susun input 6 polutan dalam urutan yang sama dengan training:
   `pm_sepuluh, pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida`
2. Load `model_xgboost.pkl` + `label_encoder.pkl`
3. `model.predict(input)` → inverse_transform → kategori (`BAIK` / `SEDANG` / `TIDAK SEHAT`)
4. Confidence dari `predict_proba()`
