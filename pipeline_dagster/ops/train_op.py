from dagster import op
import subprocess

@op
def train_op(cleaned_data_path: str):
    """Ejecuta el entrenamiento del modelo."""
    print(f"🤖 Entrenando modelo con datos de {cleaned_data_path}...")
    subprocess.run(["python", "src/train.py"], check=True)
    print("✅ Entrenamiento completado")

