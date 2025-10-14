# 📡 TelcoVision — Predicción de Churn

**Instituto:** ISTEA  
**Materia:** Laboratorio de Minería de Datos
**Profesor:** Diego Mosquera
**Carrera:** Ciencia de Datos e Inteligencia Artificial  
**Autores:** Edison Fernandez, David Wuscovi, Luis Fuentes  
**Año:** 2025  

---

## 🎯 Objetivo del Proyecto

La empresa ficticia **TelcoVision** busca reducir la rotación de clientes (*churn*).  
A partir de los datos de uso de servicios, información demográfica y métodos de pago, se desarrolló un **pipeline reproducible de Machine Learning** que predice si un cliente se dará de baja (`churn = 1`) o no (`churn = 0`).

Este proyecto simula el trabajo real de un equipo MLOps aplicando:
- Versionado de datos con **DVC**
- Experimentación con **MLflow / DVC Experiments**
- Automatización con **GitHub Actions**
- Trazabilidad completa en **DagsHub**

---

## 🧱 Estructura del Proyecto

```bash
├── data/
│   ├── raw/              # Datos originales (telco_churn.csv)
│   ├── processed/        # Datos limpios generados por el pipeline
│
├── src/
│   ├── data_prep.py      # Limpieza y preprocesamiento
│   ├── train.py          # Entrenamiento del modelo
│   ├── evaluate.py       # (opcional) Métricas y visualizaciones
│
├── models/               # Modelos entrenados versionados con DVC
│
├── .github/workflows/
│   └── ci.yaml           # Workflow CI/CD automatizado
│
├── params.yaml           # Parámetros de los experimentos
├── dvc.yaml              # Definición del pipeline de DVC
├── dvc.lock              # Estado actual del pipeline
├── requirements.txt      # Dependencias del entorno
├── README.md             # Documentación del proyecto
└── ENTREGA_FINAL.md      # (bonus) Detalles y reflexiones finales
