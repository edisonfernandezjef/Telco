import yaml
import json
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Modelos soportados
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


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
mlflow.set_experiment("Telco_Experiments")

# 7️⃣ Seleccionar modelo desde params.yaml
model_type = model_cfg.get("type", "LogisticRegression")

# Crear un nombre legible para el experimento
run_name = f"{model_type}_run"

# 8️⃣ Iniciar run de MLflow
with mlflow.start_run(run_name=run_name):
    # Loguear tipo de modelo
    mlflow.log_param("model_type", model_type)

    # Loguear todos los parámetros del modelo (plano)
    for key, value in model_cfg.items():
        mlflow.log_param(f"model_{key}", str(value))

    # Loguear parámetros de split
    for key, value in split_cfg.items():
        mlflow.log_param(f"split_{key}", str(value))

    # Selección del modelo
    if model_type == "LogisticRegression":
        model = LogisticRegression(
            C=model_cfg.get("C", 1.0),
            max_iter=model_cfg.get("max_iter", 100),
            solver=model_cfg.get("solver", "lbfgs")
        )

    elif model_type == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(
            criterion=model_cfg.get("criterion", "gini"),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            random_state=model_cfg.get("random_state", random_state)
        )

    elif model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", None),
            random_state=model_cfg.get("random_state", random_state)
        )

    elif model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(
            n_neighbors=model_cfg.get("n_neighbors", 5)
        )

    elif model_type == "SVC":
        model = SVC(
            kernel=model_cfg.get("kernel", "rbf"),
            C=model_cfg.get("C", 1.0),
            gamma=model_cfg.get("gamma", "scale"),
            probability=True
        )

    else:
        raise ValueError(f"Modelo '{model_type}' no está soportado actualmente.")

    # 9️⃣ Entrenar el modelo
    model.fit(X_train, y_train)

    # 🔍 Evaluar métricas
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 🔢 Loguear métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # 1️⃣0️⃣ Guardar modelo local y loguear artefactos
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)
    mlflow.log_artifact(metrics_path)

# Mensaje final
print(f"✅ Modelo entrenado: {model_type}")
print(f"✅ Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
print(f"✅ Modelo guardado en: {model_path}")
print(f"✅ Métricas guardadas en: {metrics_path}")


