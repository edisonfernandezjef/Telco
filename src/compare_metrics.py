import json
import sys

# Leer métricas actuales (PR)
with open("metrics/metrics.json") as f:
    new_metrics = json.load(f)

# Leer métricas de main
with open("metrics/metrics_main.json") as f:
    old_metrics = json.load(f)

new_f1 = new_metrics["f1_score"]
old_f1 = old_metrics["f1_score"]

print(f"F1 en main: {old_f1}")
print(f"F1 en PR:   {new_f1}")

# Si la métrica empeora → fallar CI
if new_f1 <= old_f1:
    print("❌ El modelo del PR es PEOR o IGUAL que el de main. Fallando CI.")
    sys.exit(1)

print("✔️ El modelo es MEJOR. CI OK.")
sys.exit(0)
