import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def preprocess_ispu(
    input_path: str,
    output_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    
    # ===============================
    # 1. LOAD DATA
    # ===============================
    df = pd.read_csv(input_path)

    # ===============================
    # 2. DROP KOLOM TIDAK PERLU
    # ===============================
    if 'No' in df.columns:
        df.drop(columns=['No'], inplace=True)

    # ===============================
    # 3. BUAT TARGET ISPU_KATEGORI
    # ===============================
    if 'ISPU_Kategori' not in df.columns and 'ISPU' in df.columns:
        bins = [0, 50, 100, 200, 300, float('inf')]
        labels = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
        df['ISPU_Kategori'] = pd.cut(
            df['ISPU'], bins=bins, labels=labels, include_lowest=True
        )

    # ===============================
    # 4. MISSING VALUES
    # ===============================
    df = df.dropna()

    # ===============================
    # 5. DUPLICATES
    # ===============================
    df = df.drop_duplicates()

    # ===============================
    # 6. SIMPAN DATA PREPROCESSING
    # ===============================
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "DataISPU_preprocessing.csv"
    df.to_csv(output_file, index=False)

    # ===============================
    # 7. PREPARE TRAIN TEST
    # ===============================
    feature_cols = ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']
    X = df[feature_cols]
    y = df['ISPU_Kategori']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# AGAR BISA DIJALANKAN LANGSUNG
# ============================================================
if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[1]

    INPUT_DATA = BASE_DIR  / "DataISPU_raw.csv"
    OUTPUT_DATA = BASE_DIR / "DataISPU_preprocessing"

    preprocess_ispu(
        input_path=INPUT_DATA,
        output_path=OUTPUT_DATA
    )

    print("Preprocessing selesai. Data siap digunakan.")
