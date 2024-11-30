# main.py
from src.model_training import train_model

def main():
    # Gunakan path absolut ke dataset
    filepath = r'H:\My Drive\0.KULIAH\SEMESTER 5\DATA SCIENCE\LAPORAN\Pertemuan 13\data\M5_World_Championship.csv'
    
    # Jalankan training
    result = train_model(filepath)
    
    if result:
        print("Training berhasil!")
        print(f"MSE: {result['mse']}")
        print(f"R2 Score: {result['r2']}")
        
        print("\nImportance Fitur:")
        print(result['feature_importance'])

if __name__ == "__main__":
    main()