import os
import cv2
import torch
import numpy as np
import json
import time
import csv
from model_objects.YOLOv8 import FaceDetector
from model_objects.ResNet import build_model
from model_objects.gate import AdaptiveGate
from restoration_agents.low_light_agent import DynamicLowLightAgent
from restoration_agents.motion_blur_agent import MotionBlurAgent
from restoration_agents.low_res_agent import SuperResAgent
from generate_data.generate_data_for_gate import smart_resize

# --- CONFIGURATION ---
TEST_FOLDER = r"C:\Users\yoavt\PycharmProjects\final_projact\data\gate_dataset\test"
OUTPUT_CSV = "pipeline_comparison2.csv"


MODEL_PATHS = {
    "yolo": "models/yolov8n-face.pt",
    "resnet": "models/id_classifier_resnet18.pt",
    "gate": "models/gate_model_best_2.pth",
    "mapping": "models/class_mapping.json"
}


class PipelineBenchmark:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Benchmark on: {self.device} ")

        # 1. Shared Models
        print("Loading Models...")
        self.detector = FaceDetector(model_path=MODEL_PATHS["yolo"])
        self.gate = AdaptiveGate(model_path=MODEL_PATHS["gate"])

        # Agents
        self.low_light_agent = DynamicLowLightAgent()
        self.motion_blur_agent = MotionBlurAgent()
        self.super_res_agent = SuperResAgent()
        self.smart_resize = smart_resize

        # ResNet Identification
        with open(MODEL_PATHS["mapping"], 'r') as f:
            self.classes = [k for k, v in sorted(json.load(f).items(), key=lambda x: x[1])]

        self.id_model = build_model(num_classes=len(self.classes))

        # Load weights safely
        try:
            self.id_model.load_state_dict(torch.load(MODEL_PATHS["resnet"], map_location=self.device))
        except:
            print("cant load resnet...")

        self.id_model.to(self.device).eval()

        # Normalization Stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def run_inference_resnet(self, face_img):
        """
        Runs the ResNet identification part only.
        Returns: (Predicted Name, Confidence Score)
        """
        # Preprocess
        input_face = self.smart_resize(face_img, target_size=224)
        input_face_rgb = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(np.array(input_face_rgb)).permute(2, 0, 1).float().unsqueeze(0).to(
            self.device) / 255.0
        tensor = (tensor - self.mean) / self.std

        # Predict
        with torch.no_grad():
            output = self.id_model(tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

        # Unknown Threshold
        if conf.item() < 0.40:
            return "Unknown", conf.item()
        return self.classes[pred.item()], conf.item()

    def run_benchmark(self):
        headers = [
            "Image", "Face_Index","BASE_Name",
             "BASE_Conf", "BASE_Time(ms)","Gate Decision",
            "GATE_Name", "GATE_Conf", "GATE_Time(ms)",
            "Conf_Gain", "Time_Cost(ms)"
        ]

        image_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
        print(f"Testing {len(image_files)} Images ")

        with open(OUTPUT_CSV, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for filename in image_files:
                path = os.path.join(TEST_FOLDER, filename)
                frame = cv2.imread(path)
                if frame is None: continue

                results = self.detector.detect(frame, expand_ratio=0.30)

                # LOOP THROUGH ALL FACES (Just like main.py)
                for res in results:
                    face = res["crop"]
                    coords = res["coords"]

                    # --- PATH A: BASELINE ---
                    t0 = time.time()
                    base_name, base_conf = self.run_inference_resnet(face)
                    base_time = (time.time() - t0) * 1000

                    # --- PATH B: INTEGRATED GATE ---
                    t2 = time.time()
                    gate_score, quality = self.gate.process(face)

                    if quality == "low_light":
                        fixed_face = self.low_light_agent.process(face)
                    elif quality == "low_res":
                        fixed_face = self.super_res_agent.process(face)
                    elif quality == "motion_blur":
                        fixed_face = self.motion_blur_agent.process(face)
                    else:
                        fixed_face = face

                    gate_name, gate_conf = self.run_inference_resnet(fixed_face)
                    gate_time = (time.time() - t2) * 1000

                    # --- SAVE RESULTS ---
                    conf_diff = (gate_conf - base_conf) * 100
                    time_diff = gate_time - base_time

                    print(
                        f"[{filename[:10]} (Face {coords})] {quality.upper():<10} | Base: {base_conf:.2f} | Gate: {gate_conf:.2f}")

                    writer.writerow([
                        filename, coords,base_name,
                         f"{base_conf:.4f}", f"{base_time:.2f}",quality,
                        gate_name, f"{gate_conf:.4f}", f"{gate_time:.2f}",
                        f"{conf_diff:.2f}", f"{time_diff:.2f}"
                    ])

        print(f"\n Benchmark Complete! Results saved to: {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    benchmark = PipelineBenchmark()
    benchmark.run_benchmark()
