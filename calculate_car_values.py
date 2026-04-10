import pandas as pd
from car_brand_values import brand_to_value


df = pd.read_csv("data/detailed_car_brand_predictions.csv")


df["predicted_brand"] = df["predicted_brand"].str.title()


df["car_value"] = df["predicted_brand"].str.lower().map(brand_to_value).fillna(0)



avg_by_image = df.groupby("filename")["car_value"].mean().reset_index()
avg_by_image.rename(columns={"car_value": "avg_car_value"}, inplace=True)


avg_by_location = df.groupby("location")["car_value"].mean().reset_index()
avg_by_location.rename(columns={"car_value": "avg_car_value_location_avg"}, inplace=True)


df = df.merge(avg_by_image, on="filename", how="left")
df = df.merge(avg_by_location, on="location", how="left")


df.to_csv("data/enriched_car_data.csv", index=False)
print("Saved enriched data to: data/enriched_car_data.csv")
