"""Feature engineering shared by training and the API, so both compute features identically."""
import numpy as np
import pandas as pd

NUMERIC = [
    "Customer_Budget_SAR", "Website_Visits_Last_30_Days", "Previous_Inquiries",
    "Mileage_KM", "Listed_Price_SAR", "Budget_Gap_SAR", "Preference_Match",
    "Vehicle_Age", "Log_Budget_Ratio", "Over_Budget",
]
CATEGORICAL = [
    "Preferred_Vehicle_Type", "Lead_Source", "Financing_Preference",
    "Brand", "Vehicle_Type", "Fuel_Type",
]
FEATURES = NUMERIC + CATEGORICAL


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to raw inquiry rows (same columns as the Excel sheet)."""
    df = df.copy()
    df["Inquiry_Date"] = pd.to_datetime(df["Inquiry_Date"])
    df["Budget_Gap_SAR"] = df["Listed_Price_SAR"] - df["Customer_Budget_SAR"]
    df["Preference_Match"] = (
        df["Preferred_Vehicle_Type"].str.strip().str.lower()
        == df["Vehicle_Type"].str.strip().str.lower()
    ).astype(int)
    df["Vehicle_Age"] = (df["Inquiry_Date"].dt.year - df["Model_Year"]).clip(lower=0)
    df["Log_Budget_Ratio"] = np.log(df["Listed_Price_SAR"] / df["Customer_Budget_SAR"])
    df["Over_Budget"] = (df["Listed_Price_SAR"] > df["Customer_Budget_SAR"]).astype(int)
    return df
