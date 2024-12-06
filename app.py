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

    # Sidebar menu
    st.sidebar.title('Menu')
    menu_options = ['Prediksi Hero', 'Dataset', 'Deskripsi Dataset']
    selected_menu = st.sidebar.selectbox('Pilih Menu', menu_options)
    
    # Muat data
    data = load_data()
    
    if selected_menu == 'Prediksi Hero':
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
    
    elif selected_menu == 'Dataset':
        st.subheader('Dataset Mentah')
        st.dataframe(data)

    elif selected_menu == 'Deskripsi Dataset':
        st.subheader('About Dataset')
        st.markdown(
            """
            Kejuaraan Dunia M5 adalah ajang final dan Kejuaraan Dunia Musim Kompetitif MLBB 2023 yang diselenggarakan oleh Moonton.
            Jika Anda tertarik dengan data E-Sports, saya sarankan Anda untuk memeriksa https://liquipedia.net/
            Kumpulan Data ini berisi statistik hero yang digunakan dalam Kejuaraan Dunia M5
            “Mobile Legends: Bang Bang” (MLBB) adalah gim arena pertempuran daring multipemain (MOBA) seluler gratis yang dikembangkan dan diterbitkan oleh Moonton, anak perusahaan ByteDance. Dirilis pada tahun 2016, gim ini semakin populer di seluruh dunia, terutama di Asia Tenggara.

            Dalam game ini, dua tim yang terdiri dari lima pemain saling bertanding secara real-time. Pertandingan berlangsung cepat, dengan minimal 10 detik perjodohan dan 10 menit pertandingan. Gameplaynya melibatkan pertarungan di tiga jalur untuk merebut menara musuh dan mempertahankan menara mereka sendiri. Tidak ada pelatihan hero atau aspek bayar untuk bermain, dan hasilnya ditentukan oleh keterampilan, kemampuan, dan strategi. Setiap pemain mengendalikan karakter unik, yang disebut Hero, dengan kemampuan dan sifat yang unik. Para hero tersebut didefinisikan oleh enam peran: Tank, Marksman, Assassin, Fighter, Mage, dan Support. Peran-peran ini menentukan tanggung jawab pemain untuk tim mereka masing-masing. Setelah sukses, Moonton telah memasuki kancah esports dengan membuat beberapa turnamen regional, yang dikenal sebagai Mobile Legends: Bang Bang Professional League (MPL), yang berfungsi sebagai kualifikasi untuk Kejuaraan Dunia Mobile Legends.

            """
        )

# Jalankan aplikasi
if __name__ == '__main__':
    main()
