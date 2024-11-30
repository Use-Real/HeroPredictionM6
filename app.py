import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Fungsi untuk memuat dan membersihkan data
def load_data():
    # Baca data
    data = pd.read_csv('data/M5_World_Championship.csv')
    
    # Fungsi konversi pick rate
    def convert_pickrate(value):
        try:
            if isinstance(value, str):
                value = value.replace('%', '').strip()
                if value == '-':
                    return np.nan
                return float(value)
            return value
        except:
            return np.nan
    
    # Konversi kolom Pick Rate
    data['T_PickPercentage'] = data['T_PickPercentage'].apply(convert_pickrate)
    data = data.dropna(subset=['T_PickPercentage'])
    
    return data

# Fungsi untuk mendapatkan top 10 hero berdasarkan role
def get_top_10_heroes_by_role(data, role):
    role_data = data[data['Roles'].str.contains(role, case=False)]
    return role_data.nlargest(10, 'T_PickPercentage')

# Fungsi utama Streamlit
def main():
    st.title('Hero Pick Prediction for M6 Championship Mobile Legend')
    
    # Muat data
    data = load_data()
    
    # Ekstrak unique roles
    roles = data['Roles'].str.split(', ', expand=True).stack().unique()
    selected_role = st.selectbox('Pilih Role Hero', sorted(roles))
    
    # Ambil 10 hero teratas
    top_heroes = get_top_10_heroes_by_role(data, selected_role)
    
    # Grafik horizontal bar
    fig = px.bar(
        top_heroes, 
        x='T_PickPercentage', 
        y='Hero', 
        orientation='h',
        title=f'{selected_role} Heroes',
        labels={'T_PickPercentage': 'Pick Rate (%)', 'Hero': 'Hero Name'}
    )
    
    # Styling grafik
    fig.update_layout(
        height=600,
        width=800,
        title_font_size=20
    )
    
    # Tampilkan grafik
    st.plotly_chart(fig)

# Jalankan aplikasi
if __name__ == '__main__':
    main()