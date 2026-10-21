from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
EXAMPLE = {
    "Inquiry_Date": "2026-09-15", "Customer_Budget_SAR": 120000, "Preferred_Vehicle_Type": "SUV",
    "Lead_Source": "Referral", "Website_Visits_Last_30_Days": 4, "Previous_Inquiries": 2,
    "Financing_Preference": "Cash", "Brand": "Toyota", "Car_Model": "RAV4", "Vehicle_Type": "SUV",
    "Model_Year": 2022, "Mileage_KM": 45000, "Fuel_Type": "Petrol", "Listed_Price_SAR": 112000,
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["purchase_probability"] <= 1
    assert body["will_purchase"] == (body["purchase_probability"] >= body["threshold"])


def test_over_budget_lowers_probability():
    cheap = client.post("/predict", json=EXAMPLE).json()["purchase_probability"]
    pricey = client.post("/predict", json={**EXAMPLE, "Listed_Price_SAR": 180000}).json()["purchase_probability"]
    assert pricey < cheap


def test_batch():
    r = client.post("/predict/batch", json=[EXAMPLE, {**EXAMPLE, "Lead_Source": "TikTok"}])
    assert r.status_code == 200 and len(r.json()) == 2


def test_validation_errors():
    assert client.post("/predict", json={**EXAMPLE, "Fuel_Type": "Diesel"}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "Customer_Budget_SAR": -5}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "Model_Year": 2030}).status_code == 422


def test_explain_and_web_app():
    body = client.post("/predict/explain", json=EXAMPLE).json()
    assert body["purchase_probability"] == client.post("/predict", json=EXAMPLE).json()["purchase_probability"]
    assert body["factors"] and {"factor", "value", "impact"} <= body["factors"][0].keys()
    assert "Purchase Likelihood" in client.get("/").text
