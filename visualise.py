import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress
import matplotlib.ticker as ticker

os.makedirs("plots", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")

df = pd.read_csv("data/final_combined_data.csv")

df_clean = df.dropna(subset=["avg_car_value", "avg_price_per_sqft"])

df_grouped = df_clean.groupby("location").agg({
    "avg_car_value": "median",
    "avg_price_per_sqft": "median",
    "filename": "count"
}).rename(columns={"filename": "num_images"}).reset_index()

df_grouped = df_grouped[df_grouped["avg_price_per_sqft"] < 100000]

df_grouped["log_car_value"] = np.log1p(df_grouped["avg_car_value"])
df_grouped["log_house_value"] = np.log1p(df_grouped["avg_price_per_sqft"])

# -------------------------------
# 1. Distribution plots
# -------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_grouped["avg_car_value"], kde=True)
plt.title("Distribution of Average Car Value")
plt.xlabel("Average Car Value")

plt.subplot(1, 2, 2)
sns.histplot(df_grouped["avg_price_per_sqft"], kde=True)
plt.title("Distribution of Property Price per Sq Ft")
plt.xlabel("Price per Sq Ft (₹)")

plt.tight_layout()
plt.savefig("plots/distributions.png", dpi=300)
plt.close()

# -------------------------------
# 2. Log scatter plot
# -------------------------------
plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df_grouped,
    x="log_car_value",
    y="log_house_value",
    size="num_images",
    sizes=(50, 250),
    alpha=0.7,
    legend=False
)

sns.regplot(
    data=df_grouped,
    x="log_car_value",
    y="log_house_value",
    scatter=False
)

slope, intercept, r_value, p_value, std_err = linregress(
    df_grouped["log_car_value"],
    df_grouped["log_house_value"]
)

plt.text(
    0.05,
    0.95,
    f"R² = {r_value**2:.3f}\np = {p_value:.3g}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white")
)

for _, row in df_grouped.iterrows():
    plt.text(
        row["log_car_value"],
        row["log_house_value"],
        row["location"],
        fontsize=8
    )

plt.title("Car Value vs Property Price per Sq Ft")
plt.xlabel("Log Average Car Value")
plt.ylabel("Log Property Price per Sq Ft")

plt.tight_layout()
plt.savefig("plots/car_vs_property_scatter.png", dpi=300)
plt.close()

# -------------------------------
# 3. Correlation heatmap
# -------------------------------
corr = df_grouped[[
    "avg_car_value",
    "avg_price_per_sqft",
    "num_images"
]].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=300)
plt.close()

# -------------------------------
# 4. Boxplot by property decile
# -------------------------------
df_grouped["property_decile"] = pd.qcut(
    df_grouped["avg_price_per_sqft"],
    5,
    labels=["D1", "D2", "D3", "D4", "D5"]
)

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_grouped,
    x="property_decile",
    y="avg_car_value"
)

plt.title("Car Value by Property Price Group")
plt.xlabel("Property Price Group")
plt.ylabel("Average Car Value")
plt.tight_layout()
plt.savefig("plots/car_value_by_property_group.png", dpi=300)
plt.close()

print("Saved all plots to: plots/")
