import yaml
import json
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# 1️⃣ Leer parámetros desde params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# --- Secciones del YAML ---
paths = params["path"]
model_cfg = params["model"]
split_cfg = params["split"]

# --- Variables de configuración ---
input_path = paths["clean_data"]
model_path = paths["model_path"]
metrics_path = paths["metrics_path"]

# --- Parámetros del modelo y del split ---
C = model_cfg.get("C", 1.0)
max_iter = model_cfg.get("max_iter", 100)
solver = model_cfg.get("solver", "lbfgs")
test_size = split_cfg.get("test_size", 0.2)
random_state = split_cfg.get("random_state", 42)


# 2️⃣ Cargar dataset
df = pd.read_csv(input_path)

# Eliminar columnas irrelevantes si existen
cols_to_drop = [col for col in ["customer_id", "CustomerID", "id"] if col in df.columns]
df = df.drop(columns=cols_to_drop, errors="ignore")

# 3️⃣ Separar X e y
target_col = "churn"
X = df.drop(columns=[target_col])
y = df[target_col]

# 4️⃣ Codificar variables categóricas automáticamente
X = pd.get_dummies(X, drop_first=True)

# 5️⃣ Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# 6️⃣ Configurar MLflow remoto (usa tus variables de entorno)
mlflow.set_tracking_uri("https://dagshub.com/edisonjef/Telco.mlflow")
mlflow.set_experiment("Telco")

# Crear un nombre legible para el experimento
run_name = f"LogisticRegression_C{C}_iter{max_iter}_{solver}"

# 7️⃣ Iniciar run de MLflow
with mlflow.start_run(run_name=run_name):
    # Loguear todos los parámetros del modelo (desde YAML)
    for key, value in model_cfg.items():
        mlflow.log_param(f"model_{key}", value)

    for key, value in split_cfg.items():
        mlflow.log_param(f"split_{key}", value)

    # 8️⃣ Entrenar el modelo
    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver)
    model.fit(X_train, y_train)

    # 9️⃣ Evaluar métricas
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Loguear métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # 🔹 Guardar modelo local y loguearlo como artefacto
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    # 🔹 Guardar métricas locales y loguearlas como artefacto
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)
    mlflow.log_artifact(metrics_path)

# ✅ Mensaje final
print("✅ Modelo: LogisticRegression")
print(f"✅ Parámetros -> C: {C}, max_iter: {max_iter}, solver: {solver}")
print(f"✅ Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
print(f"✅ Modelo guardado en: {model_path}")
print(f"✅ Métricas guardadas en: {metrics_path}")



