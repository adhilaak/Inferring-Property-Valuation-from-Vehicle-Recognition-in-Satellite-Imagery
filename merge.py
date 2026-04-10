import pandas as pd
import re


car_data = pd.read_csv("data/enriched_car_data.csv")
house_data = pd.read_csv("data/location_house_prices.csv")


def clean_postcode(p):
    return re.sub(r"\s+", " ", str(p)).strip().upper()


car_data["postcode"] = car_data["postcode"].apply(clean_postcode)
house_data["postcode"] = house_data["postcode"].apply(clean_postcode)


car_data = car_data[car_data["postcode"] != ""]
house_data = house_data[house_data["postcode"] != ""]


print("\nPostcodes in car_data:", sorted(car_data["postcode"].unique()))
print("Postcodes in house_data:", sorted(house_data["postcode"].unique()))


merged = car_data.merge(house_data, on="postcode", how="left", suffixes=("", "_house"))


unmatched = merged[merged["avg_house_price"].isna()]
print("\nPostcodes with missing house prices after merge:")
print(unmatched["postcode"].unique())


merged.to_csv("data/final_combined_data.csv", index=False)
print("\nSaved final combined dataset with house prices to: data/final_combined_data.csv")
