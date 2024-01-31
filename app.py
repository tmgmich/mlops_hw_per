# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib
import os


# Определение модели запроса для обучения
class TrainRequest(BaseModel):
    model_type: str
    hyperparameters: dict


app = FastAPI()

# Словарь для хранения обученных моделей
models = {}


@app.post("/train/")
async def train_model(train_request: TrainRequest):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    if train_request.model_type == "random_forest":
        model = RandomForestClassifier(**train_request.hyperparameters)
    elif train_request.model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**train_request.hyperparameters)
    else:
        raise HTTPException(status_code=400, detail="Unsupported model type")

    model.fit(X_train, y_train)
    model_id = str(len(models))
    models[model_id] = model
    joblib.dump(model, f"model_{model_id}.joblib")
    return {"message": "Model trained", "model_id": model_id}


@app.get("/models/")
async def list_models():
    return list(models.keys())


@app.post("/predict/{model_id}")
async def predict(model_id: str, data: list):
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = models[model_id]
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

