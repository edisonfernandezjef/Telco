import os

print("User:", os.getenv("MLFLOW_TRACKING_USERNAME"))
print("Token:", os.getenv("MLFLOW_TRACKING_PASSWORD"))
