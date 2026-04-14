import cv2
import os
import time
from model_objects.YOLOv8 import FaceDetector

# --- CONFIGURATION ---
# We will save these directly into your Gate's training folder
BASE_TRAIN_DIR = r"C:\Users\yoavt\PycharmProjects\final_projact\data\resnet dataset\omer_picture_224"
MODEL_YOLO = r"C:\Users\yoavt\PycharmProjects\final_projact\models\yolov8n-face.pt"

# Ensure folders exist
FOLDERS = {
    'normal': os.path.join(BASE_TRAIN_DIR, "normal"),
    'low_light': os.path.join(BASE_TRAIN_DIR, "low_light"),
    'motion_blur': os.path.join(BASE_TRAIN_DIR, "motion_blur"),
    'low_res': os.path.join(BASE_TRAIN_DIR, "low_res")
}
for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)


def smart_resize(img, target_size=224):
    """Ensures the face crop matches your synthetic FFHQ data exactly."""
    h, w = img.shape[:2]
    interp = cv2.INTER_CUBIC if (h < target_size or w < target_size) else cv2.INTER_AREA
    return cv2.resize(img, (target_size, target_size), interpolation=interp)


def main():
    print("Initializing Camera and YOLO...")
    detector = FaceDetector(model_path=MODEL_YOLO)

    # Open webcam (Change 0 to 1 or 2 if using external USB camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- CONTROLS ---")
    print("Press 'N' to save a NORMAL face")
    print("Press 'D' to save a DARK/LOW LIGHT face")
    print("Press 'B' to save a BLURRY face")
    print("Press 'R' to save a LOW RES face ")
    print("Press 'Q' to QUIT")
    print("------------------\n")

    counts = {'normal': 0, 'low_light': 0, 'motion_blur': 0, 'low_res': 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect face
        results = detector.detect(frame, expand_ratio=0.30)

        display_frame = frame.copy()
        face_to_save = None

        if results:
            # Grab the first face found
            res = results[0]
            face_crop = res["crop"]
            coords = res["coords"]

            # Format exactly like your training data
            face_to_save = smart_resize(face_crop, target_size=224)

            # Draw a box so you know it's tracking you
            cv2.rectangle(display_frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
            cv2.putText(display_frame, "Face Locked - Ready to Save", (coords[0], coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show instructions on screen
        cv2.putText(display_frame,
                    f"Saved -> N: {counts['normal']} | D: {counts['low_light']} | B: {counts['motion_blur']} | R: {counts['low_res']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Gate Data Collector", display_frame)

        # Key listeners
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Only save if a face is currently detected
        if face_to_save is not None:
            timestamp = int(time.time() * 1000)

            if key == ord('n'):
                path = os.path.join(FOLDERS['normal'], f"real_norm_{timestamp}.jpg")
                cv2.imwrite(path, face_to_save)
                counts['normal'] += 1
                print("Saved Normal Face")

            elif key == ord('d'):
                path = os.path.join(FOLDERS['low_light'], f"dark_{timestamp}.jpg")
                cv2.imwrite(path, face_to_save)
                counts['low_light'] += 1
                print("Saved Dark Face")

            elif key == ord('b'):
                path = os.path.join(FOLDERS['motion_blur'], f"blur_{timestamp}.jpg")
                cv2.imwrite(path, face_to_save)
                counts['motion_blur'] += 1
                print("Saved Blurry Face")

            elif key == ord('r'):
                # Saving the unmodified face_to_save for manual processing later
                path = os.path.join(FOLDERS['low_res'], f"low_res_{timestamp}.jpg")
                cv2.imwrite(path, face_to_save)
                counts['low_res'] += 1
                print("Saved Low-Res Face (Original Image)")

    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection complete!")


if __name__ == "__main__":
    main()
