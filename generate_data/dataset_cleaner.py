import os
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from model_objects.YOLOv8 import FaceDetector

register_heif_opener()


def resize_with_padding(img, target_size=224):
    """Resizes image maintaining aspect ratio and adds black padding."""
    h, w = img.shape[:2]
    # Calculate the scaling factor to fit the target size
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image without distortion
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Center the resized image on the canvas
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def clean_folder(target_folder, target_size=224,
                 model_path=r"C:\Users\yoavt\PycharmProjects\final_projact\models\yolov8n-face.pt"):
    try:
        detector = FaceDetector(model_path=model_path, conf_threshold=0.5)
    except Exception as e:
        print(f"❌ Failed to load detector: {e}")
        return

    files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.heic'))]

    for filename in files:
        img_path = os.path.join(target_folder, filename)

        # HEIC to BGR Conversion
        if filename.lower().endswith(('.heic', '.heif')):
            pil_img = Image.open(img_path).convert("RGB")
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.imread(img_path)

        if frame is None: continue

        results = detector.detect(frame, expand_ratio=0.20)

        if len(results) > 0:
            face_crop = results[0]["crop"]

            # --- 🛠️ FIX: Use Padding instead of squashing ---
            final_img = resize_with_padding(face_crop, target_size=target_size)

            # Save as JPG for consistency
            new_filename = os.path.splitext(filename)[0] + ".jpg"
            save_path = os.path.join(target_folder, new_filename)

            cv2.imwrite(save_path, final_img)

            # Remove original if it was HEIC to avoid duplicates
            if filename.lower().endswith('.heic'):
                os.remove(img_path)

            print(f"✅ Padded & Saved: {new_filename}")
        else:
            print(f"⚠️ No face detected in {filename}")


if __name__ == "__main__":
    MY_FOLDER = r"C:\Users\yoavt\PycharmProjects\final_projact\data\resnet_dataset\test"
    clean_folder(MY_FOLDER)