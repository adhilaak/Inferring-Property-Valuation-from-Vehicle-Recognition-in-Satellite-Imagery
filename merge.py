import pandas as pd

car_data = pd.read_csv("data/enriched_car_data.csv")
house_data = pd.read_csv("data/location_house_prices.csv")


def clean_location(x):
    return str(x).strip().lower()


car_data["location"] = car_data["location"].apply(clean_location)
house_data["location"] = house_data["location"].apply(clean_location)

car_data = car_data[car_data["location"] != ""]
house_data = house_data[house_data["location"] != ""]


print("\nLocations in car_data:", sorted(car_data["location"].unique()))
print("Locations in house_data:", sorted(house_data["location"].unique()))


merged = car_data.merge(
    house_data,
    on="location",
    how="left",
    suffixes=("", "_house")
)


unmatched = merged[merged["avg_price_per_sqft"].isna()]
print("\nLocations with missing house prices after merge:")
print(unmatched["location"].unique())


merged.to_csv("data/final_combined_data.csv", index=False)

print("\nSaved final combined dataset to: data/final_combined_data.csv")
