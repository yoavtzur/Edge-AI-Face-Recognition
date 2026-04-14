import cv2
import os
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
SOURCE_DIR = r"C:\Users\yoavt\PycharmProjects\final_projact\generate_data\FFHQ_128X128"
ROOT_DIR = r"C:\Users\yoavt\PycharmProjects\final_projact\data\gate_dataset"
TARGET_SIZE = 128
NUM_IMAGES_TO_USE = 10000
TRAIN_RATIO = 0.8


def create_gate_folders():
    for split in ["train", "val"]:
        for c in ["normal", "low_light", "motion_blur", "low_res"]:
            path = os.path.join(ROOT_DIR, split, c)
            os.makedirs(path, exist_ok=True)


# --- FIX 3: SMART RESIZE (No Black Bars) ---
def smart_resize(img, target_size=128):
    """
    Resizes image to target_size square using the best interpolation.
    """
    h, w = img.shape[:2]
    if h < target_size or w < target_size:
        interp = cv2.INTER_CUBIC  # Sharp upscale
    else:
        interp = cv2.INTER_AREA  # Clean downscale
    return cv2.resize(img, (target_size, target_size), interpolation=interp)


def letterbox_resize(img, target_size=TARGET_SIZE, color=(0, 0, 0)):
    """Resizes image while maintaining aspect ratio and adding padding."""
    h, w = img.shape[:2]
    # Calculate the ratio to fit the image into the target size
    r = min(target_size[0] / h, target_size[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    # Resize keeping the proportions
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Calculate padding needed to reach target_size
    dw = target_size[0] - new_unpad[0]
    dh = target_size[1] - new_unpad[1]
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    # Add black bars
    return cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)


# --- FIX 1: DIRTY NORMAL (The most important fix) ---
def make_webcam_noise(img):
    """
    Makes 'Normal' images look like a webcam/phone camera.
    Adds slight grain and softness so the model accepts real life as 'Normal'.
    """
    # 1. Subtle Blur (Lens softness)
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # 2. Subtle Grain (Sensor Noise)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Add noise only to Value channel (looks like ISO grain, not confetti)
    noise = np.random.normal(0, 3, v.shape).astype(np.int16)
    v_noisy = np.clip(v.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img_noisy = cv2.merge([h, s, v_noisy])
    return cv2.cvtColor(img_noisy, cv2.COLOR_HSV2BGR)


# --- FIX 2: REALISTIC LOW RES (Soft, not Blocky) ---
def make_low_res(img):
    """
    Hybrid Approach: Pixelate -> Then Soften.
    Looks like real video compression (Zoom/Teams).
    """
    h, w = img.shape[:2]

    # 1. Downscale
    scale = random.uniform(0.3, 0.5)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. Upscale with NEAREST (Pixelate)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. Soften the blocks (Blur)
    soft_pixelated = cv2.GaussianBlur(pixelated, (3, 3), 0)

    return soft_pixelated


def make_low_light(img):
    """
    Focuses on underexposure safely by manipulating the V channel.
    Prevents the red color shift and maintains realistic skin tones.
    """
    # 1. Convert to HSV FIRST so we can isolate brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 2. Apply Gamma heavily, but ONLY to the V channel
    gamma = random.uniform(2.0, 4.0)
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    v_dark = cv2.LUT(v, look_up_table)

    # 3. Add Realistic Lens Softness (Subtle blur to the dark image)
    if random.random() > 0.5:
        v_dark = cv2.GaussianBlur(v_dark, (3, 3), 0)

    # 4. Add sensor grain/noise
    noise = np.random.normal(0, random.uniform(1, 3), v_dark.shape).astype(np.int16)
    v_noisy = np.clip(v_dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 5. Merge the original colors back with the newly darkened brightness
    return cv2.cvtColor(cv2.merge([h, s, v_noisy]), cv2.COLOR_HSV2BGR)


def make_motion_blur(img):
    """
    Creates a true directional linear motion blur (camera shake/movement)
    rather than just a generic 2D filter.
    """
    kernel_size = random.choice([7, 9, 11, 15])

    # Create an empty kernel
    kernel = np.zeros((kernel_size, kernel_size))

    # Draw a horizontal line in the middle
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize
    kernel = kernel / kernel_size

    # Rotate the line to a random angle
    angle = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)

    return cv2.filter2D(img, -1, kernel)


def generate():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Could not find {SOURCE_DIR}")
        return

    create_gate_folders()

    all_images = list(Path(SOURCE_DIR).glob("*.png")) + list(Path(SOURCE_DIR).glob("*.jpg"))
    random.shuffle(all_images)
    selected_images = all_images[:NUM_IMAGES_TO_USE]
    split_point = int(len(selected_images) * TRAIN_RATIO)

    print(f"--- Generating REAL-TIME Data from {len(selected_images)} images ---")

    for idx, img_path in enumerate(tqdm(selected_images)):
        current_split = "train" if idx < split_point else "val"
        img = cv2.imread(str(img_path))
        if img is None: continue

        # Resize first (Full Square)
        base_img = smart_resize(img, target_size=TARGET_SIZE)
        filename = f"{idx}_{img_path.name}"
        base_path = os.path.join(ROOT_DIR, current_split)

        # SAVE CLASSES
        # 1. Normal (With NOISE added!)
        cv2.imwrite(os.path.join(base_path, "normal", filename), make_webcam_noise(base_img))

        # 2. Bad Classes
        cv2.imwrite(os.path.join(base_path, "low_light", filename), make_low_light(base_img))
        cv2.imwrite(os.path.join(base_path, "motion_blur", filename), make_motion_blur(base_img))
        cv2.imwrite(os.path.join(base_path, "low_res", filename), make_low_res(base_img))

    print(f"\n Done! Real-time data generated at: {ROOT_DIR}")
    print("Now run train_gate.py again.")


if __name__ == "__main__":
    generate()