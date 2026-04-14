import os
import cv2
from model_objects.gate import AdaptiveGate


def run_gate_pipeline():
    # 1. SETUP
    folder_path = r"C:\Users\yoavt\PycharmProjects\final_projact\data\resnet dataset\real_test\test_for_gate"
    model_path = r"C:\Users\yoavt\PycharmProjects\final_projact\models\gate_model_best_2.pth"

    gate = AdaptiveGate(model_path=model_path)

    # 2. TRACKING
    # Initialize counts for each class to show at the end
    summary_counts = {cls: 0 for cls in gate.classes}
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]
    total = len(image_files)

    print(f"-- Processing {total} images through the Gate ---")
    print(f"{'IMAGE NAME':<25} | {'DECISION':<12} | {'CONFIDENCE'}")
    print("-" * 60)

    # 3. THE LOOP
    for filename in image_files:
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is None: continue
        # This call handles ALL the resizing and padding to 128x128
        confidence, decision = gate.process(img)

        # Update our summary counter
        summary_counts[decision] += 1

        print(f"{filename:<25} | {decision:<12} | {confidence:.1f}%")

    # 4. FINAL CLASS SUMMARY
    print("\n" + "=" * 35)
    print("      GATE RESULTS SUMMARY")
    print("=" * 35)
    for cls_name, count in summary_counts.items():
        print(f"{cls_name.upper():<15}: {count} images")
    print("-" * 35)
    print(f"TOTAL PROCESSED: {total}")
    print("=" * 35)


if __name__ == "__main__":
    run_gate_pipeline()