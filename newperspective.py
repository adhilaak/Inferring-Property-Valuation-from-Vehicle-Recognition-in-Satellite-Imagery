import os
import cv2
import numpy as np


INPUT_FOLDER = "images"
OUTPUT_FOLDER = "images/perspectives"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


FOV = 60        
HEIGHT = 1200     
WIDTH = 1800
YAW_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
PITCH = 0

def equirectangular_to_perspective(img, fov, yaw_deg, pitch_deg, height, width):
    h_eq, w_eq = img.shape[:2]
    fov_rad = np.deg2rad(fov)

    x = np.linspace(-np.tan(fov_rad / 2), np.tan(fov_rad / 2), width)
    y = np.linspace(-np.tan(fov_rad / 2), np.tan(fov_rad / 2), height)
    xv, yv = np.meshgrid(x, y)
    zv = np.ones_like(xv)

    vecs = np.stack([xv, yv, zv], axis=-1)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)

    def rot_matrix(yaw, pitch):
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ])
        return Ry @ Rx

    R = rot_matrix(yaw_deg, pitch_deg)
    dirs = vecs @ R.T

    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(np.clip(dirs[..., 1], -1, 1))

    u = (lon / np.pi + 1) / 2 * w_eq
    v = (lat / (np.pi / 2) + 1) / 2 * h_eq

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_WRAP)


existing_perspectives = set(os.listdir(OUTPUT_FOLDER))


for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(INPUT_FOLDER, filename)
        panorama = cv2.imread(input_path)
        if panorama is None or panorama.size == 0:
            print(f"Skipping unreadable or corrupt file: {filename}")
            continue

        base_name = os.path.splitext(filename)[0]
        for yaw in YAW_ANGLES:
            out_filename = f"{base_name}_yaw_{yaw}.jpg"
            out_path = os.path.join(OUTPUT_FOLDER, out_filename)

            if out_filename in existing_perspectives:
                print(f"⏩ Skipping existing perspective: {out_filename}")
                continue

            try:
                perspective = equirectangular_to_perspective(panorama, FOV, yaw, PITCH, HEIGHT, WIDTH)
                cv2.imwrite(out_path, perspective)
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Failed to process {out_filename}: {e}")
