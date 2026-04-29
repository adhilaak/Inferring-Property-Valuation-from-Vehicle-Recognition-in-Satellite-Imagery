import pandas as pd
import os


housing_df = pd.read_csv("data/house_prices.csv")

housing_df["Location"] = housing_df["Location"].astype(str).str.strip().str.lower()
housing_df["Price"] = pd.to_numeric(housing_df["Price"], errors="coerce")
housing_df["Area"] = pd.to_numeric(housing_df["Area"], errors="coerce")


housing_df["price_per_sqft"] = housing_df["Price"] / housing_df["Area"]


def get_average_price(location):
    if not location or pd.isna(location):
        return None

    location = str(location).strip().lower()

    try:
        matches = housing_df[housing_df["Location"] == location]

        if matches.empty:
            return None

        return float(matches["price_per_sqft"].median())

    except Exception as e:
        print(f"Failed to get price for '{location}': {e}")
        return None


existing_file = "data/location_house_prices.csv"

if os.path.exists(existing_file):
    existing_df = pd.read_csv(existing_file)
    processed_locations = set(existing_df["location"].dropna().unique())
else:
    existing_df = pd.DataFrame()
    processed_locations = set()


mapping_df = pd.read_csv("data/image_location_map.csv")
mapping_df = mapping_df.dropna(subset=["location"])

location_price_rows = []

for _, row in mapping_df.iterrows():
    loc = row["location"]
    clean_loc = str(loc).strip().lower()

    if clean_loc in processed_locations:
        print(f"Skipping already processed: {clean_loc}")
        continue

    avg_price = get_average_price(clean_loc)
    print(f"{clean_loc}: avg_price={avg_price}")

    location_price_rows.append({
        "location": clean_loc,
        "avg_price_per_sqft": avg_price
    })


new_df = pd.DataFrame(location_price_rows)

final_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["location"])

final_df.to_csv(existing_file, index=False)

print("Saved location house price data to:", existing_file)
