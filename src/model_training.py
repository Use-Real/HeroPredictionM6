# src/model_training.py
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

def preprocess_data(data):
    """
    Preprocessing khusus untuk dataset hero dengan penanganan NaN
    """
    # Konversi kolom persentase yang masih berbentuk string
    def convert_percentage(col):
        return pd.to_numeric(col.str.rstrip('%'), errors='coerce') / 100

    # Pilih fitur yang relevan
    features = [
        'T_Picked',      # Total dipilih
        'T_Wins',        # Total menang
        'T_Loses',       # Total kalah
        'BS_Picked',     # Picked di Best Stage
        'BS_Wins',       # Menang di Best Stage
        'RS_Picked',     # Picked di Regular Stage
        'RS_Wins',       # Menang di Regular Stage
        'T_Banned'       # Total banned
    ]

    # Pilih target
    target_columns = [
        'T_WinRate',     # Win Rate Total
        'T_PickPercentage'  # Pick Percentage Total
    ]

    # Konversi target ke numerik
    for col in target_columns:
        data[col] = convert_percentage(data[col])

    # Encode hero names
    le = LabelEncoder()
    data['Hero_Encoded'] = le.fit_transform(data['Hero'])
    features.append('Hero_Encoded')

    # Pilih target (misalnya T_WinRate)
    target = 'T_WinRate'

    # Siapkan X dan y
    X = data[features]
    y = data[target]

    # Tangani NaN dengan imputer
    imputer_X = SimpleImputer(strategy='median')
    imputer_y = SimpleImputer(strategy='median')

    # Imputer untuk fitur
    X_imputed = imputer_X.fit_transform(X)
    
    # Imputer untuk target
    y_imputed = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_imputed, test_size=0.2, random_state=42
    )

    # Scaling fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def train_model(filepath):
    """
    Fungsi utama training model
    """
    # Baca dataset
    data = pd.read_csv(filepath)

    # Preprocessing
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(data)

    # Pilih model - RandomForestRegressor untuk prediksi win rate
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n===== HASIL TRAINING =====")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Simpan model dan scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/winrate_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')

    # Fitur penting
    feature_names = [
        'T_Picked', 'T_Wins', 'T_Loses', 'BS_Picked', 
        'BS_Wins', 'RS_Picked', 'RS_Wins', 'T_Banned', 'Hero_Encoded'
    ]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nImportance Fitur:")
    print(feature_importance)

    return {
        'model': model,
        'mse': mse,
        'r2': r2,
        'feature_importance': feature_importance
    }

# Jalankan training jika file dieksekusi langsung
if __name__ == "__main__":
    filepath = r'data/M5_World_Championship.csv'
    train_model(filepath)