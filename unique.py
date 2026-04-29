import pandas as pd

df = pd.read_csv("data/image_location_map.csv")
print("Unique locations in map file:", df["location"].nunique())

df = pd.read_csv("data/location_house_prices.csv")
print("Locations with house prices:", df["location"].nunique())

df = pd.read_csv("data/final_combined_data.csv")
print("Locations in final dataset:", df["location"].nunique())
