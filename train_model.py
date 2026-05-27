"""
============================================================
TRAINING SCRIPT - MODEL XGBOOST UNTUK JAKU
============================================================
Script ini mereplikasi pipeline dari notebook ISPU_Klasifikasi_CRISP_DM_FIXED.ipynb
untuk menghasilkan file .pkl yang dipakai oleh app.py.

Cara pakai:
    1. Letakkan file Data_ISPU.csv di folder data/
    2. Jalankan: python train_model.py
    3. File .pkl akan tersimpan di folder models/

Jika file Data_ISPU.csv tidak tersedia, script ini akan men-generate
synthetic training data berdasarkan rules ISPU PERMEN LHK No. 14 Tahun 2020
sehingga app.py tetap bisa berjalan untuk demo deployment.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ============================================================
# KONFIGURASI
# ============================================================
DATA_PATH = 'data/Data_ISPU.csv'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

FITUR_POLUTAN = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida',
                 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']


# ============================================================
# LOAD ATAU GENERATE DATA
# ============================================================
def load_or_generate_data():
    """Load data nyata jika ada, jika tidak generate synthetic."""
    if os.path.exists(DATA_PATH):
        print(f'✓ Memuat data dari {DATA_PATH}')
        df_raw = pd.read_csv(DATA_PATH, sep=';')
        return df_raw

    print('⚠ File Data_ISPU.csv tidak ditemukan, men-generate synthetic data')
    print('  berdasarkan rules ISPU PERMEN LHK No. 14 Tahun 2020...\n')

    np.random.seed(42)
    n_per_class = 1000

    # BAIK: kondisi udara bersih
    baik = pd.DataFrame({
        'pm_sepuluh':        np.random.uniform(5, 50, n_per_class),
        'pm_duakomalima':    np.random.uniform(2, 15, n_per_class),
        'sulfur_dioksida':   np.random.uniform(2, 50, n_per_class),
        'karbon_monoksida':  np.random.uniform(0.1, 4, n_per_class),
        'ozon':              np.random.uniform(5, 100, n_per_class),
        'nitrogen_dioksida': np.random.uniform(5, 70, n_per_class),
        'kategori': 'BAIK'
    })

    # SEDANG: kondisi udara moderat
    sedang = pd.DataFrame({
        'pm_sepuluh':        np.random.uniform(50, 150, n_per_class),
        'pm_duakomalima':    np.random.uniform(15, 55, n_per_class),
        'sulfur_dioksida':   np.random.uniform(50, 180, n_per_class),
        'karbon_monoksida':  np.random.uniform(4, 8, n_per_class),
        'ozon':              np.random.uniform(100, 235, n_per_class),
        'nitrogen_dioksida': np.random.uniform(70, 150, n_per_class),
        'kategori': 'SEDANG'
    })

    # TIDAK SEHAT: kondisi udara buruk
    tidak_sehat = pd.DataFrame({
        'pm_sepuluh':        np.random.uniform(150, 350, n_per_class // 2),
        'pm_duakomalima':    np.random.uniform(55, 150, n_per_class // 2),
        'sulfur_dioksida':   np.random.uniform(180, 400, n_per_class // 2),
        'karbon_monoksida':  np.random.uniform(8, 15, n_per_class // 2),
        'ozon':              np.random.uniform(235, 400, n_per_class // 2),
        'nitrogen_dioksida': np.random.uniform(150, 300, n_per_class // 2),
        'kategori': 'TIDAK SEHAT'
    })

    df = pd.concat([baik, sedang, tidak_sehat], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ============================================================
# PIPELINE TRAINING
# ============================================================
def main():
    df = load_or_generate_data()

    # Konversi ke numerik
    for col in FITUR_POLUTAN:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter kategori valid
    df = df[df['kategori'].isin(['BAIK', 'SEDANG', 'TIDAK SEHAT'])].copy()

    # Imputasi missing dengan median
    for col in FITUR_POLUTAN:
        df[col] = df[col].fillna(df[col].median())

    # Hapus outlier dengan IQR
    mask = pd.Series([True] * len(df), index=df.index)
    for col in FITUR_POLUTAN:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = mask & (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    df = df[mask].copy()

    print(f'Jumlah data setelah pembersihan: {len(df)} baris')
    print(f'Distribusi kategori:\n{df["kategori"].value_counts()}\n')

    # Split
    X = df[FITUR_POLUTAN].copy()
    y = df['kategori'].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Scaling (untuk SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ----------------------------------------------------
    # XGBOOST - model utama untuk JakU
    # ----------------------------------------------------
    print('Melatih XGBoost...')
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
    }
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='mlogloss',
                      use_label_encoder=False),
        param_grid=xgb_params, cv=skf, scoring='accuracy', n_jobs=-1, verbose=0
    )
    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_
    xgb_acc = accuracy_score(y_test, xgb_best.predict(X_test))
    print(f'  Best params : {xgb_grid.best_params_}')
    print(f'  Test acc    : {xgb_acc:.4f}\n')

    # ----------------------------------------------------
    # RANDOM FOREST (cadangan)
    # ----------------------------------------------------
    print('Melatih Random Forest...')
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f'  Test acc    : {rf_acc:.4f}\n')

    # ----------------------------------------------------
    # SVM (cadangan)
    # ----------------------------------------------------
    print('Melatih SVM...')
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    print(f'  Test acc    : {svm_acc:.4f}\n')

    # ----------------------------------------------------
    # SIMPAN SEMUA ARTEFAK
    # ----------------------------------------------------
    joblib.dump(xgb_best, os.path.join(MODELS_DIR, 'model_xgboost.pkl'))
    joblib.dump(rf,       os.path.join(MODELS_DIR, 'model_random_forest.pkl'))
    joblib.dump(svm,      os.path.join(MODELS_DIR, 'model_svm.pkl'))
    joblib.dump(le,       os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    joblib.dump(scaler,   os.path.join(MODELS_DIR, 'standard_scaler.pkl'))
    joblib.dump(FITUR_POLUTAN, os.path.join(MODELS_DIR, 'fitur_polutan.pkl'))

    # Simpan metrik untuk ditampilkan di halaman dashboard
    metrik = {
        'XGBoost': float(xgb_acc),
        'Random Forest': float(rf_acc),
        'SVM': float(svm_acc),
        'feature_importance': dict(zip(
            FITUR_POLUTAN,
            [float(v) for v in xgb_best.feature_importances_]
        ))
    }
    joblib.dump(metrik, os.path.join(MODELS_DIR, 'metrik.pkl'))

    print('═' * 60)
    print('✓ Semua artefak tersimpan di folder models/')
    print('═' * 60)


if __name__ == '__main__':
    main()
