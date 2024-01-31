from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from minio import Minio
from minio.error import S3Error
import joblib
import os

# Define the request model for training
class TrainRequest(BaseModel):
    model_type: str
    hyperparameters: dict

# Initialize FastAPI app
app = FastAPI()

# Initialize Minio client
minio_client = Minio(
    "minio:9000",
    access_key="minio1234",
    secret_key="minio1234",
    secure=False
)

# Function to save a model to Minio
def save_model_to_minio(model, model_id):
    try:
        minio_client.make_bucket("models")
    except S3Error:
        pass  # Bucket already exists
    model_path = f"model_{model_id}.joblib"
    joblib.dump(model, model_path)
    minio_client.fput_object("models", model_path, model_path)

# Function to load a model from Minio
def load_model_from_minio(model_id):
    model_path = f"model_{model_id}.joblib"
    minio_client.fget_object("models", model_path, model_path)
    return joblib.load(model_path)

# Dictionary to store trained models
models = {}

@app.post("/train/")
async def train_model(train_request: TrainRequest):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if train_request.model_type == "random_forest":
        model = RandomForestClassifier(**train_request.hyperparameters)
    elif train_request.model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**train_request.hyperparameters)
    else:
        raise HTTPException(status_code=400, detail="Unsupported model type")

    model.fit(X_train, y_train)
    model_id = str(len(models))
    models[model_id] = model
    save_model_to_minio(model, model_id)  # Save model to Minio
    return {"message": "Model trained", "model_id": model_id}

@app.get("/models/")
async def list_models():
    return list(models.keys())

@app.post("/predict/{model_id}")
async def predict(model_id: str, data: list):
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = load_model_from_minio(model_id)  # Load model from Minio
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
