import pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset"""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Data cleaning"""
    # Hapus kolom yang tidak diperlukan
    columns_to_drop = ['Index', 'Played By', 'Played With', 'Played Vs.']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Konversi kolom persentase
    percentage_columns = [
        'T_WinRate', 'T_PickPercentage', 
        'T_BansPercentage', 'T_PicksBansPercentage'
    ]
    
    for col in percentage_columns:
        df[col] = pd.to_numeric(df[col].str.strip('%'), errors='coerce')
    
    return df

def feature_engineering(df):
    """Membuat fitur tambahan"""
    # Contoh: Rasio ban terhadap pick
    df['ban_pick_ratio'] = df['T_BansPercentage'] / df['T_PickPercentage']
    
    # One-hot encoding untuk Roles
    df_encoded = pd.get_dummies(df, columns=['Roles'])
    
    return df_encoded

def prepare_data_for_model(df):
    """Mempersiapkan data untuk pemodelan"""
    # Pilih fitur
    features = [
        'T_PickPercentage', 
        'T_BansPercentage', 
        'ban_pick_ratio'
    ]
    
    # Tambahkan kolom one-hot encoding roles
    role_columns = [col for col in df.columns if col.startswith('Roles_')]
    features.extend(role_columns)
    
    X = df[features]
    y = df['T_WinRate']
    
    return X, y