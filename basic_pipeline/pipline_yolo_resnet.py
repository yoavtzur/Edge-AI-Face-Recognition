import os
import cv2
import torch
import numpy as np
import json
from model_objects.YOLOv8 import FaceDetector
from model_objects.ResNet import build_model

import time
# ---  CONFIGURATION ---/
MODEL_PATHS = {
    "yolo": r"C:\Users\yoavt\PycharmProjects\final_projact\models\yolov8n-face.pt",
    "resnet": r"C:\Users\yoavt\PycharmProjects\final_projact\models\resnet18_3.pt",
    "mapping": r"C:\Users\yoavt\PycharmProjects\final_projact\models\class_mapping.json"
}


class BaselinePipeline:
    def __init__(self):
        # 1. Device Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Initializing Baseline (No Gate) on: {self.device}")

        # 2. Face Detector (YOLOv8)
        self.detector = FaceDetector(model_path=MODEL_PATHS["yolo"])

        # 3. Identification Model (ResNet-18)
        # Load class names
        with open(MODEL_PATHS["mapping"], 'r') as f:
            self.classes = [k for k, v in sorted(json.load(f).items(), key=lambda x: x[1])]

        # Load Model
        self.id_model = build_model(num_classes=len(self.classes))
        self.id_model.load_state_dict(torch.load(MODEL_PATHS["resnet"], map_location=self.device))
        self.id_model.to(self.device).eval()

        # 4. Normalization Tensors (Standard ImageNet Stats) ( number more fasr on jetson)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # 5. Standard Resizer (Padding)
    def smart_resize(self, img, target_size=224):
        h, w = img.shape[:2]

        # Determine if we are upscaling or downscaling
        if h < target_size or w < target_size:
            # We are enlarging the image
            # INTER_CUBIC provides the best smoothness for faces
            interp = cv2.INTER_CUBIC
        else:
            # We are shrinking the image
            # INTER_AREA is best for maintaining detail when downscaling
            interp = cv2.INTER_AREA

        return cv2.resize(img, (target_size, target_size), interpolation=interp)

    # =========================================================
    #  FUNCTION 1: TEST ON FOLDER (Batch Processing)
    # =========================================================
    def run_on_folder(self, folder_path, output_folder="results_baseline"):
        """Loops through images and runs ONLY Yolo -> ResNet"""

        # --- Cleanup Old Results ---
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            print(f" Clearing old results in '{output_folder}'...")
            for old_file in os.listdir(output_folder):
                try:
                    file_path = os.path.join(output_folder, old_file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except:
                    pass

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f" Processing {len(image_files)} images from: {folder_path}")

        for filename in image_files:
            full_path = os.path.join(folder_path, filename)
            frame = cv2.imread(full_path)
            if frame is None: continue

            # Run the Logic
            processed_frame = self._process_frame(frame)

            # Save Result
            save_path = os.path.join(output_folder, f"baseline_{filename}")
            cv2.imwrite(save_path, processed_frame)
            print(f"Saved: {save_path}")

        print(f"\n Baseline Test Complete. Check '{output_folder}'")

    # =========================================================
    #  FUNCTION 2: TEST ON CAMERA (Real-Time)
    # =========================================================
    def run_on_camera(self):
        """Runs the pipeline on the webcam feed."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(" Error: Could not open camera.")
            return

        print(" Camera Live. Press 'q' to quit ")

        # Optional: Set resolution to HD
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Run the Logic
            processed_frame = self._process_frame(frame)

            # Show Result
            cv2.imshow("Baseline (No Agents)", processed_frame)

            # Exit logic
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # =========================================================
    #  SHARED LOGIC (The actual pipeline)
    # =========================================================
    def _process_frame(self, frame):
        """Contains the core YOLO -> ResNet logic used by both functions."""

        t_start = time.time()
        # STEP 1: Detect Faces (YOLO)
        results = self.detector.detect(frame, expand_ratio=0.30)

        for res in results:
            face = res["crop"]
            coords = res["coords"]

            # STEP 2: Preprocess for ResNet
            # Resize to 224x224 with padding (Letterbox)
            input_face = self.smart_resize(face, target_size= 224)

            # BGR to RGB (ResNet was trained on RGB)
            input_face_rgb = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)

            # Convert to Tensor & Normalize
            tensor = torch.from_numpy(np.array(input_face_rgb)).permute(2, 0, 1).float().unsqueeze(0).to(
                self.device) / 255.0
            tensor = (tensor - self.mean) / self.std

            # STEP 3: Identify
            with torch.no_grad():
                output = self.id_model(tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                id_conf, pred = torch.max(prob, 1)

            # Threshold Logic
            if id_conf.item() < 0.40:
                person_name = "Unknown"
            else:
                person_name = self.classes[pred.item()]


            t_end = time.time()
            total_time = t_end - t_start
            print(f"️ Total Pipeline Time: {total_time:.4f} sec")
            # STEP 4: Draw Results
            id_percent = id_conf.item() * 100
            label = f"{person_name} ({id_percent:.1f}%)"

            # Color: Red for 'other' or 'Unknown', Green for Yoav/Omer
            color = (0, 255, 0) if person_name not in ["other", "Unknown"] else (0, 0, 255)

            # Draw Box
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)

            # Draw Text (Smart Positioning)
            # If box is too close to top, put text inside
            text_y = coords[1] + 25 if coords[1] < 30 else coords[1] - 10

            # Add a black background for text readability
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (coords[0], text_y - h - 5), (coords[0] + w, text_y + 5), (0, 0, 0), -1)

            cv2.putText(frame, label, (coords[0], text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame


# =========================================================
#  MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    system = BaselinePipeline()

    # --- OPTION 1: Run on Folder ---
    #test_folder_path = r"C:\Users\Your0124\pycharm_project_test\data\resnet_dataset\testpipline"
    #system.run_on_folder(test_folder_path)

    # --- OPTION 2: Run on Camera ---
    system.run_on_camera()