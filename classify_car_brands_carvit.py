import torch
import os
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from transformers import ViTFeatureExtractor, ViTForImageClassification
import pandas as pd
import torch.nn.functional as F

# Configuration
IMAGE_FOLDER = "images/perspectives"
ANNOTATED_FOLDER = "annotated_detections"
MODEL_NAME = "yolov5s"
YOLO_CONFIDENCE_THRESHOLD = 0.5
CARVIT_CONFIDENCE_THRESHOLD = 0.5
CSV_CONFIDENCE_THRESHOLD = 0.9
MIN_BOX_SIZE = 100
OUTPUT_CSV = "data/detailed_car_brand_predictions.csv"

# Loading Models
yolo = torch.hub.load('ultralytics/yolov5', MODEL_NAME)
feature_extractor = ViTFeatureExtractor.from_pretrained("abdusah/CarViT")
carvit_model = ViTForImageClassification.from_pretrained("abdusah/CarViT")

# 
if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
    existing_df = pd.read_csv(OUTPUT_CSV)
    processed_files = set(existing_df["filename"].unique())
else:
    existing_df = pd.DataFrame()
    processed_files = set()

detailed_rows = []
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

for filename in os.listdir(IMAGE_FOLDER):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    if filename in processed_files:
        print(f"⏩ Skipping already processed image: {filename}")
        continue

    annotated_path = os.path.join(ANNOTATED_FOLDER, f"annotated_{filename}")
    image_path = os.path.join(IMAGE_FOLDER, filename)
    print(f"\n🔍 Processing: {filename}")
    location_name = filename.split(".")[0]
    postcode = location_name.split("_location_")[0].replace("_", " ").upper()

    try:
        full_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f" Failed to open image: {filename} ({e})")
        continue

    width, height = full_img.size
    draw = ImageDraw.Draw(full_img)

    font_size = max(36, height // 20)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    results = yolo(image_path)
    detections = results.pandas().xyxy[0]
    detections = detections[detections['confidence'] > YOLO_CONFIDENCE_THRESHOLD]
    car_detections = detections[detections['name'] == 'car']

    if len(car_detections) == 0:
        print("    No strong car detections.")
        continue

    for i, row in car_detections.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])

        if (xmax - xmin < MIN_BOX_SIZE) or (ymax - ymin < MIN_BOX_SIZE):
            print(f"   Car #{i+1} too small, skipped.")
            continue

        cropped_car = full_img.crop((xmin, ymin, xmax, ymax)).resize((384, 384))
        inputs = feature_extractor(images=cropped_car, return_tensors="pt")
        with torch.no_grad():
            outputs = carvit_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        confidence = probs.max().item()

        if confidence >= CARVIT_CONFIDENCE_THRESHOLD:
            predicted_idx = probs.argmax(-1).item()
            predicted_brand = carvit_model.config.id2label[predicted_idx]
            print(f"   Car #{i+1}: {predicted_brand} (confidence: {confidence:.2f})")

            label = f"{predicted_brand} ({confidence*100:.1f}%)"
            box_color = "red"
            text_color = "white"
            line_width = max(4, height // 250)

            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=box_color, width=line_width)
            text_bbox = draw.textbbox((xmin, ymin), label, font=font)
            draw.rectangle(text_bbox, fill=box_color)
            draw.text((text_bbox[0], text_bbox[1]), label, font=font, fill=text_color)

            if confidence >= CSV_CONFIDENCE_THRESHOLD:
                detailed_rows.append({
                    "location": location_name,
                    "filename": filename,
                    "car_number": i+1,
                    "predicted_brand": predicted_brand,
                    "confidence": confidence,
                    "postcode": postcode
                })
        else:
            print(f"  Car #{i+1} prediction too weak ({confidence:.2f}), skipped.")

    full_img.save(annotated_path)
    print(f" Saved annotated image: {annotated_path}")

# Saving output to CSV
new_df = pd.DataFrame(detailed_rows)
df_combined = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()
df_combined.to_csv(OUTPUT_CSV, index=False)
print("\nSaved updated high-confidence predictions to:", OUTPUT_CSV)
