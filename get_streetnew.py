import os
import time
import csv
from datetime import datetime
from streetview import search_panoramas, get_panorama, get_panorama_meta

from locations_config import locations, step_count, step_size, step_direction

output_folder = "images"
mapping_file = "data/image_location_map.csv"
GOOGLE_MAPS_API_KEY = "YOUR_API_KEY"

os.makedirs(output_folder, exist_ok=True)
os.makedirs("data", exist_ok=True)

existing_images = set()
if os.path.exists(mapping_file):
    with open(mapping_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            existing_images.add(row["location"])


def generate_coordinates(lat, lon, steps, step_size, direction):
    coords = []
    for i in range(steps):
        if direction == "lat":
            coords.append((lat + i * step_size, lon))
        else:
            coords.append((lat, lon + i * step_size))
    return coords


write_header = not os.path.exists(mapping_file)

with open(mapping_file, "a", newline="") as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=["city", "state", "location", "latitude", "longitude"]
    )

    if write_header:
        writer.writeheader()

    for city, data in locations.items():
        state = data["state"]
        start_lat = data["lat"]
        start_lon = data["lon"]

        coords_list = generate_coordinates(
            start_lat,
            start_lon,
            step_count,
            step_size,
            step_direction
        )

        for idx, (lat, lon) in enumerate(coords_list):
            location_id = f"{city.lower()}_location_{idx+1}"

            if location_id in existing_images:
                print(f"Skipping existing: {location_id}")
                continue

            panos = search_panoramas(lat=lat, lon=lon)
            if not panos:
                continue

            dated_panos = []

            for pano in panos:
                try:
                    meta = get_panorama_meta(pano.pano_id, api_key=GOOGLE_MAPS_API_KEY)
                    capture_date = datetime.strptime(meta.date, "%Y-%m") if meta.date else datetime.min
                    dated_panos.append((pano, capture_date))
                except:
                    continue

            if not dated_panos:
                continue

            dated_panos.sort(key=lambda x: x[1], reverse=True)
            latest_pano = dated_panos[0][0]

            try:
                image = get_panorama(latest_pano.pano_id)
            except:
                continue

            output_filename = f"{location_id}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            image.save(output_path, "JPEG")

            writer.writerow({
                "city": city.lower(),
                "state": state.lower(),
                "location": location_id,
                "latitude": lat,
                "longitude": lon
            })

            print(f"Saved: {output_filename}")

            time.sleep(1)
