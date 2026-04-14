import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import glob
import json

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Testing on: {device}")


# ---  HELPER: Padding Function (Matches your Clean script) ---
def resize_with_padding(img, target_size=224):
    """Maintain aspect ratio and add black borders to prevent squashing."""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def get_class_names(mapping_path):
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    # Sort by index to match training folder order
    return [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]


def load_final_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Must match my trained layers

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# 4. UPDATED TRANSFORMS (Must be 224x224 to match my training)
test_tfms = transforms.Compose([
    # We no longer need Resize here if we use the padding function below
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def batch_test(folder_path, model, class_names):
    images = glob.glob(os.path.join(folder_path, "*.*"))
    images = [i for i in images if i.lower().endswith(('.jpg', '.png', '.jpeg', '.heic'))]

    if not images:
        print(f"No images found in {folder_path}")
        return

    print(f"Testing {len(images)} images | Resolution: 224x224\n")
    print(f"{'FILENAME':<25} | {'PREDICTION':<15} | {'CONFIDENCE'}")
    print("-" * 60)

    for path in images:
        img = cv2.imread(path)
        if img is None: continue

        # --- FIX: Apply Padding so the face isn't squashed ---
        img_padded = resize_with_padding(img, target_size=224)

        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = test_tfms(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

        name = os.path.basename(path)
        label = class_names[preds.item()]
        print(f"{name:<25} | {label:<15} | {conf.item() * 100:.1f}%")


if __name__ == "__main__":
    TEST_DIR = r"C:\Users\yoavt\PycharmProjects\final_projact\data\resnet dataset\real_test\test_for_gate"
    MODEL_FILE = r"C:\Users\yoavt\PycharmProjects\final_projact\models\resnet18_2.pt"  #trained weights
    MAPPING_FILE = r"C:\Users\yoavt\PycharmProjects\final_projact\models\class_mapping.json"  # folder map

    try:
        classes = get_class_names(MAPPING_FILE)
        my_model = load_final_model(MODEL_FILE, len(classes))
        batch_test(TEST_DIR, my_model, classes)
    except Exception as e:
        print(f"Error: {e}")