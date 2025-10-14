import yaml
import json
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 1️⃣ Leer parámetros desde params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

paths = params["path"]
model_cfg = params["model"]
split_cfg = params["split"]

# Variables de configuración
input_path = paths["raw_data"]
model_path = paths["model_path"]
metrics_path = paths["metrics_path"]

C = model_cfg["C"]
max_iter = model_cfg["max_iter"]
solver = model_cfg["solver"]

test_size = split_cfg["test_size"]
random_state = split_cfg["random_state"]

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
mlflow.set_experiment("Telco_Experiments")

with mlflow.start_run():
    # Registrar parámetros
    mlflow.log_params(model_cfg)

    # 7️⃣ Entrenar modelo
    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver)
    model.fit(X_train, y_train)

    # 8️⃣ Evaluar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 9️⃣ Registrar métricas y modelo en MLflow
    mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
    # se comenta porque da error en la version actual de mlflow
    # mlflow.sklearn.log_model(model, "model")

    # 10️⃣ Guardar localmente (para DVC)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)

print(f"✅ Modelo guardado en: {model_path}")
print(f"✅ Métricas guardadas en: {metrics_path}")

