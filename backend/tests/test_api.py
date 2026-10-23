from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_invalid_product_type() -> None:
    response = client.post(
        "/api/predict",
        json={
            "type": "X",
            "air_temperature": 300,
            "process_temperature": 310,
            "rotational_speed": 1500,
            "torque": 40,
            "tool_wear": 100,
        },
    )
    assert response.status_code == 422


def test_predict_returns_category_and_scores() -> None:
    response = client.post(
        "/api/predict",
        json={
            "type": "M",
            "air_temperature": 300,
            "process_temperature": 310,
            "rotational_speed": 1500,
            "torque": 40,
            "tool_wear": 100,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_category"] in {
        "No Failure",
        "Heat Dissipation Failure",
        "Overstrain Failure",
        "Power Failure",
        "Tool Wear Failure",
    }
    assert result["scores"]
    assert abs(sum(score["score"] for score in result["scores"]) - 1.0) < 1e-8


def test_predict_allows_values_outside_demo_range_with_notice() -> None:
    response = client.post(
        "/api/predict",
        json={
            "type": "M",
            "air_temperature": 305,
            "process_temperature": 310,
            "rotational_speed": 1500,
            "torque": 40,
            "tool_wear": 100,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert "Air temperature [K]" in result["outside_training_range"]
    assert result["range_notice"]
