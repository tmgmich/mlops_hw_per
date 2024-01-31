import pytest
from moto import mock_s3
#import moto
from minio import Minio
import app2_mlflow
from fastapi.testclient import TestClient

# Фикстура для тестового клиента FastAPI
@pytest.fixture
def client():
    with TestClient(app2_mlflow) as client:
        yield client

# Фикстура для мокирования S3
@pytest.fixture
def mock_s3_bucket():
    with mock_s3():
        # Инициализация мокированного S3 клиента
        minio_client = Minio(
            "localhost:9000",
            access_key="minio1234",
            secret_key="minio1234",
            secure=False
        )

        # Создание мокированного bucket
        minio_client.make_bucket("models")
        yield minio_client

        # Удаление bucket после теста
        for obj in minio_client.list_objects("models"):
            minio_client.remove_object("models", obj.object_name)
        minio_client.remove_bucket("models")

# Тестирование эндпоинта обучения модели с мокированным Minio
def test_train_model(client, mock_s3_bucket):
    response = client.post("/train/", json={
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": 10}
    })
    assert response.status_code == 200
    assert "model trained" in response.json()["message"]

# Тестирование эндпоинта списка моделей
def test_list_models(client):
    response = client.get("/models/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
