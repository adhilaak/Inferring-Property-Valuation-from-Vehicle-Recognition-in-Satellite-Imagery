import os
import time
import csv
from datetime import datetime
from streetview import search_panoramas, get_panorama, get_panorama_meta
from PIL import Image

# === CONFIGURATION ===
from locations_config import locations, step_count, step_size, step_direction

output_folder = "images"
mapping_file = "data/image_location_map.csv"
GOOGLE_MAPS_API_KEY = "AIzaSyBDLs5QwO7zuokPLMlhdztG2EGSi_N-GOw"

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
    writer = csv.DictWriter(csvfile, fieldnames=["postcode", "location", "latitude", "longitude"])
    if write_header:
        writer.writeheader()

    for postcode, (start_lat, start_lon) in locations.items():
        coords_list = generate_coordinates(start_lat, start_lon, step_count, step_size, step_direction)

        for idx, (lat, lon) in enumerate(coords_list):
            location_id = f"{postcode}_location_{idx+1}"
            if location_id in existing_images:
                print(f"🟡 Skipping existing: {location_id}")
                continue

            print(f"\n {postcode} - Step {idx+1}: ({lat:.6f}, {lon:.6f})")
            panos = search_panoramas(lat=lat, lon=lon)
            if not panos:
                print(" No panoramas found.")
                continue

            dated_panos = []
            for pano in panos:
                try:
                    meta = get_panorama_meta(pano.pano_id, api_key=GOOGLE_MAPS_API_KEY)
                    if meta.date:
                        capture_date = datetime.strptime(meta.date, "%Y-%m")
                    else:
                        capture_date = datetime.min
                    dated_panos.append((pano, capture_date))
                except Exception as e:
                    print(f"Skipping invalid pano metadata: {e}")
                    continue

            if not dated_panos:
                print("No valid panoramas after metadata check.")
                continue

            dated_panos.sort(key=lambda x: x[1], reverse=True)
            latest_pano = dated_panos[0][0]
            latest_date = dated_panos[0][1]

            print(f" Pano ID: {latest_pano.pano_id} ({latest_date.strftime('%Y-%m')})")

       
            image = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    image = get_panorama(latest_pano.pano_id)
                    break
                except Exception as e:
                    print(f"Download failed (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        print(" Skipping due to repeated download failures.")
                        image = None
                    else:
                        time.sleep(2)

            if image is None:
                continue

            output_filename = f"{location_id}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            image.save(output_path, "JPEG")
            print(f" Saved: {output_filename}")

            writer.writerow({
                "postcode": postcode,
                "location": location_id,
                "latitude": lat,
                "longitude": lon
            })

            time.sleep(1)  # Avoid rate-limiting
