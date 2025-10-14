import pandas as pd
from pycaret.classification import setup, get_config

# 1️⃣ Leer dataset crudo
df = pd.read_csv("data/raw/telco_churn.csv")

# 2️⃣ Configurar PyCaret para limpieza
setup(
    data=df,
    target='churn',             # columna objetivo
    imputation_type='simple',   # imputación básica
    numeric_imputation='mean',  # reemplaza nulos numéricos por la media
    categorical_imputation='mode', # reemplaza nulos categóricos por la moda
    normalize=True,             # normaliza variables numéricas
    normalize_method='zscore',  # método estándar
    remove_outliers=False,      # (opcional)
    preprocess=True,            # activa pipeline completo
    session_id=123,
    verbose=False,
    html=False
)

# 3️⃣ Extraer dataset transformado desde la configuración de PyCaret
data_transformed = get_config('dataset_transformed')

# 4️⃣ Guardar dataset limpio (usá rutas relativas desde raíz)
output_path = "data/processed/telco_churn_clean.csv"
data_transformed.to_csv(output_path, index=False)

print(f"✅ Dataset limpio guardado en: {output_path}")
