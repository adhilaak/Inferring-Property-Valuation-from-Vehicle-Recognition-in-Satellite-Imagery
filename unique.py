import pandas as pd

df = pd.read_csv("data/image_location_map.csv")
print("Unique postcodes in map file:", df["postcode"].nunique())

df = pd.read_csv("data/location_house_prices.csv")
print(" Postcodes with house prices:", df["postcode"].nunique())

df = pd.read_csv("data/final_combined_data.csv")  # adjust path if needed
print("Postcodes in final dataset:", df["postcode"].nunique())
