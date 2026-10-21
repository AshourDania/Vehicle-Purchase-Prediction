"""Vehicle Purchase Prediction API.

Run:  uvicorn app.main:app --reload
Then open http://127.0.0.1:8000 for the web app, or http://127.0.0.1:8000/docs for the API docs.
"""
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.features import build_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "vehicle_purchase_model_v2.joblib"
bundle = joblib.load(MODEL_PATH)
pipeline, threshold, feature_columns = bundle["pipeline"], bundle["threshold"], bundle["feature_columns"]

VehicleType = Literal["Hatchback", "Pickup", "SUV", "Sedan"]
LeadSource = Literal["Google Ads", "Haraj", "Instagram", "Referral", "Snapchat",
                     "TikTok", "Walk-in", "Website", "X (Twitter)"]

app = FastAPI(
    title="Vehicle Purchase Prediction API",
    description="Predicts whether a car inquiry will turn into a purchase within 30 days.",
    version="2.0.0",
)


class Inquiry(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "Inquiry_Date": "2026-09-15", "Customer_Budget_SAR": 120000,
        "Preferred_Vehicle_Type": "SUV", "Lead_Source": "Referral",
        "Website_Visits_Last_30_Days": 4, "Previous_Inquiries": 2,
        "Financing_Preference": "Cash", "Brand": "Toyota", "Car_Model": "RAV4",
        "Vehicle_Type": "SUV", "Model_Year": 2022, "Mileage_KM": 45000,
        "Fuel_Type": "Petrol", "Listed_Price_SAR": 112000,
    }]})

    Inquiry_Date: date
    Customer_Budget_SAR: float = Field(gt=0)
    Preferred_Vehicle_Type: VehicleType
    Lead_Source: LeadSource
    Website_Visits_Last_30_Days: int = Field(ge=0)
    Previous_Inquiries: int = Field(ge=0)
    Financing_Preference: Literal["Cash", "Financing", "Undecided"]
    Brand: str = Field(min_length=1, description="Brands seen in training: BYD, Chevrolet, Ford, Honda, "
                                                 "Hyundai, Kia, Lexus, MG, Nissan, Toyota")
    Car_Model: Optional[str] = Field(default=None, description="Accepted for records; not used by the v2 model")
    Vehicle_Type: VehicleType
    Model_Year: int = Field(ge=1990)
    Mileage_KM: float = Field(ge=0)
    Fuel_Type: Literal["Electric", "Hybrid", "Petrol"]
    Listed_Price_SAR: float = Field(gt=0)

    @model_validator(mode="after")
    def check_model_year(self):
        if self.Model_Year > self.Inquiry_Date.year + 1:
            raise ValueError("Model_Year cannot be more than one year after Inquiry_Date")
        return self


class Prediction(BaseModel):
    purchase_probability: float
    will_purchase: bool
    threshold: float
    model_version: str


def _predict(inquiries: list[Inquiry]) -> list[Prediction]:
    rows = build_features(pd.DataFrame([i.model_dump() for i in inquiries]))
    probs = pipeline.predict_proba(rows[feature_columns])[:, 1]
    return [Prediction(purchase_probability=round(float(p), 4), will_purchase=bool(p >= threshold),
                       threshold=threshold, model_version=bundle["model_version"]) for p in probs]


# Plain-language names for the factors shown in the web app.
FACTOR_LABELS = {
    "Customer_Budget_SAR": "Customer budget", "Website_Visits_Last_30_Days": "Website visits",
    "Previous_Inquiries": "Previous inquiries", "Mileage_KM": "Mileage", "Listed_Price_SAR": "Listed price",
    "Budget_Gap_SAR": "Price above/below budget", "Preference_Match": "Matches preferred type",
    "Vehicle_Age": "Vehicle age", "Log_Budget_Ratio": "Price vs budget", "Over_Budget": "Over budget",
    "Preferred_Vehicle_Type": "Preferred type", "Lead_Source": "Lead source",
    "Financing_Preference": "Payment method", "Brand": "Brand", "Vehicle_Type": "Vehicle type",
    "Fuel_Type": "Fuel type",
}


class Factor(BaseModel):
    factor: str
    value: str
    impact: float  # log-odds contribution vs an average inquiry; positive raises the probability


class Explanation(Prediction):
    factors: list[Factor]


def _explain(inquiry: Inquiry) -> list[Factor]:
    """Per-feature contributions of the logistic regression, relative to an average inquiry."""
    pre = pipeline.named_steps["preprocessor"]
    coef = pipeline.named_steps["classifier"].coef_[0]
    row = build_features(pd.DataFrame([inquiry.model_dump()]))
    t = pre.transform(row[feature_columns])
    x = t.toarray().ravel() if hasattr(t, "toarray") else np.asarray(t).ravel()
    names = pre.get_feature_names_out()
    totals: dict[str, float] = {}
    for name, c, v in zip(names, coef, x):
        group, col = name.split("__", 1)
        if group == "numeric":
            totals[col] = totals.get(col, 0.0) + c * v  # scaled value is already centred on the mean
    enc = pre.named_transformers_["categorical"]
    start = len(pre.named_transformers_["numeric"].feature_names_in_)
    for col, cats in zip(enc.feature_names_in_, enc.categories_):
        block = coef[start:start + len(cats)]
        active = x[start:start + len(cats)]
        totals[col] = float(block @ active - block.mean())
        start += len(cats)
    raw = row.iloc[0]
    merged = {  # the price/budget features overlap, so report them as one factor
        "Log_Budget_Ratio": totals.pop("Log_Budget_Ratio") + totals.pop("Over_Budget") + totals.pop("Budget_Gap_SAR"),
        **totals,
    }
    def shown(col):
        if col == "Log_Budget_Ratio":
            return f"{raw.Listed_Price_SAR / raw.Customer_Budget_SAR:.0%} of budget"
        if col == "Preference_Match":
            return "Yes" if raw.Preference_Match else "No"
        v = raw[col]
        return f"{v:,.0f}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
    factors = [Factor(factor=FACTOR_LABELS[c], value=shown(c), impact=round(float(v), 3)) for c, v in merged.items()]
    return sorted(factors, key=lambda f: abs(f.impact), reverse=True)


@app.get("/", include_in_schema=False)
def web_app():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {"model_version": bundle["model_version"], "threshold": threshold,
            "features": feature_columns, "sklearn_version": bundle["sklearn_version"],
            "evaluation": bundle["evaluation"]}


# Shown pre-filled on the /docs page ("Try it out").
EXAMPLE = Inquiry.model_config["json_schema_extra"]["examples"][0]


@app.post("/predict", response_model=Prediction)
def predict(inquiry: Annotated[Inquiry, Body(openapi_examples={"sample": {"summary": "Sample inquiry", "value": EXAMPLE}})]):
    return _predict([inquiry])[0]


@app.post("/predict/explain", response_model=Explanation)
def predict_explain(inquiry: Annotated[Inquiry, Body(openapi_examples={"sample": {"summary": "Sample inquiry", "value": EXAMPLE}})]):
    """Prediction plus the factors that pushed it up or down (used by the web app)."""
    return Explanation(**_predict([inquiry])[0].model_dump(), factors=_explain(inquiry))


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(inquiries: Annotated[list[Inquiry], Body(openapi_examples={"sample": {
        "summary": "Two inquiries", "value": [EXAMPLE, {**EXAMPLE, "Lead_Source": "TikTok", "Financing_Preference": "Undecided"}]}})]):
    if not 1 <= len(inquiries) <= 1000:
        raise HTTPException(status_code=422, detail="Send between 1 and 1000 inquiries")
    return _predict(inquiries)
