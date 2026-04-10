import requests
import pandas as pd
import os
import re


land_registry = pd.read_csv("data/pp-2023.csv", header=None)
land_registry.columns = [
    "transaction_id", "price", "date", "postcode", "property_type", "new_build",
    "tenure", "address_1", "address_2", "street", "locality", "town", "district",
    "county", "ppd_category", "record_status"
]
land_registry["postcode"] = land_registry["postcode"].astype(str).str.strip()
land_registry["price"] = pd.to_numeric(land_registry["price"], errors="coerce")


def extract_clean_postcode(text):
    text = text.replace("_", " ")  
    match = re.search(r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b', text.upper())
    return match.group(0).strip() if match else None



def get_average_price(postcode):
    if not postcode or pd.isna(postcode) or len(postcode) < 4:
        return None
    try:
        matches = land_registry[land_registry["postcode"].str.startswith(postcode[:4])]
        if matches.empty:
            return None
        return int(matches["price"].median())
    except Exception as e:
        print(f"Failed to get price for postcode '{postcode}': {e}")
        return None

existing_file = "data/location_house_prices.csv"
if os.path.exists(existing_file):
    existing_df = pd.read_csv(existing_file)
    processed_postcodes = set(existing_df["postcode"].dropna().unique())
else:
    existing_df = pd.DataFrame()
    processed_postcodes = set()


mapping_df = pd.read_csv("data/image_location_map.csv")
mapping_df = mapping_df.dropna(subset=["postcode"])
mapping_df["postcode"] = mapping_df["postcode"].astype(str).str.strip()

location_postcode_price = []

for _, row in mapping_df.iterrows():
    loc = row["location"]
    raw_postcode = row["postcode"]
    cleaned_postcode = extract_clean_postcode(raw_postcode)

    if not cleaned_postcode:
        print(f" Could not extract valid postcode from: {raw_postcode}")
        continue

    print(f"Cleaned postcode: {cleaned_postcode}")

    if cleaned_postcode in processed_postcodes:
        print(f"Skipping already processed postcode: {cleaned_postcode}")
        continue

    avg_price = get_average_price(cleaned_postcode)
    print(f"{cleaned_postcode}: avg_price={avg_price}")
    location_postcode_price.append({
        "location": loc,
        "postcode": cleaned_postcode,
        "avg_house_price": avg_price
    })


new_df = pd.DataFrame(location_postcode_price)
final_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["postcode"])
final_df.to_csv(existing_file, index=False)
print(" Saved postcode + house price data to:", existing_file)
