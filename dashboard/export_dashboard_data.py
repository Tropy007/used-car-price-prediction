"""Export the small, pre-aggregated tables behind the Tableau and Power BI dashboards.

Run from the repository root:  python dashboard/export_dashboard_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)
def save(df, name):
    df.to_csv(OUT / name, index=False)
    print(name); print(df.to_string(index=False)); print()
INR_PER_USD = 83
raw = pd.read_csv("https://raw.githubusercontent.com/manishkr1754/CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv", index_col=0)
cars = raw.drop_duplicates().query("km_driven <= 500_000").copy()
cars["price_usd"] = (cars["selling_price"] / INR_PER_USD).round(0)
top = cars["brand"].value_counts().head(10).index
b = cars[cars["brand"].isin(top)].groupby("brand")["price_usd"].median().sort_values(ascending=False).reset_index()
b["Brand"] = [f"{i:02d}. {x}" for i, x in enumerate(b["brand"], 1)]
save(b[["Brand", "price_usd"]].rename(columns={"price_usd": "Median price (USD)"}), "cars_by_brand.csv")
age = cars[cars["vehicle_age"] <= 15].groupby("vehicle_age")["price_usd"].median().reset_index()
age["Vehicle age (years)"] = age["vehicle_age"].map(lambda a: f"{a:02d}")
save(age[["Vehicle age (years)", "price_usd"]].rename(columns={"price_usd": "Median price (USD)"}), "cars_by_age.csv")
ft = cars.groupby(["fuel_type", "transmission_type"])["price_usd"].median().reset_index()
ft = ft[ft["fuel_type"].isin(["Petrol", "Diesel", "CNG"])]
save(ft.rename(columns={"fuel_type": "Fuel type", "transmission_type": "Transmission", "price_usd": "Median price (USD)"}), "cars_fuel_transmission.csv")
km = pd.cut(cars["km_driven"], [0, 25_000, 50_000, 75_000, 100_000, 150_000, 500_000],
            labels=["1. Under 25k km", "2. 25-50k km", "3. 50-75k km", "4. 75-100k km", "5. 100-150k km", "6. 150k+ km"])
save(cars.groupby(km, observed=True)["price_usd"].median().reset_index().rename(columns={"km_driven": "Kilometres driven", "price_usd": "Median price (USD)"}), "cars_by_km.csv")
# Model results below come from the held-out test set in notebooks/used_car_price_prediction.ipynb
save(pd.DataFrame({"metric": ["Listings", "Median price", "Price model R²", "Typical error (MAE)", "Premium classifier F1", "Premium accuracy"],
                   "value": [f"{len(cars):,}", f"${cars.price_usd.median():,.0f}", "0.857", "$1,740", "0.889", "94.7%"]}), "cars_kpis.csv")
