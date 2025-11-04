# telcovision_dagster/ops/data_prep_op.py
from dagster import op
import subprocess

@op
def data_prep_op():
    """Ejecuta el stage de limpieza de datos (DVC o Python)."""
    print("Ejecutando limpieza de datos...")
    subprocess.run(["python", "src/data_prep.py"], check=True)
    print("✅ Limpieza completada")
    return "data/processed/clean_data.csv"  # valor simbólico