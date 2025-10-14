# src/train.py
import yaml, json, mlflow, mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# 1️⃣ Leer parámetros desde params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# --- Secciones ---
paths = params["path"]
model_cfg = params["model"]
split_cfg = params["split"]

# --- Variables ---
input_path = paths["raw_data"]
model_path = paths["model_path"]
metrics_path = paths["metrics_path"]

C = model_cfg["C"]
max_iter = model_cfg["max_iter"]
solver = model_cfg["solver"]

test_size = split_cfg["test_size"]
random_state = split_cfg["random_state"]

# 2️⃣ Leer dataset
df = pd.read_csv(input_path)
X = df.drop(columns=["churn"])
y = df["churn"]

# 3️⃣ Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# 4️⃣ Configurar MLflow (DagsHub)
mlflow.set_tracking_uri("https://github.com/edisonfernandezjef/TelcoVision.mlflow")
mlflow.set_experiment("TelcoVision_Experiments")

# 5️⃣ Entrenar y registrar
with mlflow.start_run():
    mlflow.log_params(model_cfg)

    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
    mlflow.sklearn.log_model(model, "model")

    # 6️⃣ Guardar artefactos locales (DVC)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)

print(f"✅ Modelo guardado en: {model_path}")
print(f"✅ Métricas guardadas en: {metrics_path}")
