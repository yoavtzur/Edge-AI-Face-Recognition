import cv2
import os
import numpy as np


class SuperResAgent:
    def __init__(self, model_path="models/ESPCN_x3.pb", scale=3, algo_name="espcn"):
        """
        Initializes the Super Resolution Agent with support for multiple models.
        algo_name options: 'fsrcnn', 'espcn', 'lapsrn', 'edsr'
        """
        self.scale = scale
        self.model_path = model_path

        # 1. Verify Model Exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[SuperRes] CRITICAL ERROR: Model file not found at '{self.model_path}'."
            )

        # 2. Load OpenCV Super Resolution
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()

            # Read the model file
            self.sr.readModel(self.model_path)

            # Set the specific algorithm and scale
            # This string MUST match the model type exactly ("espcn")
            self.sr.setModel(algo_name, scale)

            print(f"[SuperRes] Success! Loaded {algo_name.upper()} model (Scale x{scale})")

        except Exception as e:
            raise RuntimeError(f"[SuperRes] Failed to load model. Error: {e}")

    def process(self, face_crop):
        """
        Input: Low-res face crop (BGR numpy array).
        Output: High-res AI-upscaled face crop.
        """
        if face_crop is None or face_crop.size == 0:
            return face_crop

        # 1. Run AI Inference
        # This will now use the sharper model (ESPCN/LapSRN)
        result = self.sr.upsample(face_crop)

        # Apply a tiny bit of sharpening after AI upscale
        # This helps "pop" the edges of eyes and mouths
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        result = cv2.filter2D(result, -1, kernel)

        return result


def main():
    print(" Testing Better Super Resolution")

    # algo_name must be 'espcn' if using ESPCN_x3.pb
    try:
        agent = SuperResAgent(model_path=r"C:\Users\yoavt\PycharmProjects\final_projact\models\ESPCN_x3.pb", scale=3, algo_name="espcn")
    except Exception as e:
        print(e)
        return

    # Load  low-res image...
    img_path = r"C:\Users\yoavt\PycharmProjects\final_projact\data\resnet dataset\real_test\test_for_gate\real_lowrez_11.jpg"
    if not os.path.exists(img_path):
        print("Image not found")
        return

    original = cv2.imread(img_path)

    # Process
    upscaled = agent.process(original)

    # Visualization Code
    h, w = upscaled.shape[:2]
    original_resized = cv2.resize(original, (w, h),
                                  interpolation=cv2.INTER_NEAREST)  #Nearest for "pixelated" look comparison

    cv2.imshow("Original (Pixelated) vs ESPCN (Sharp)", np.hstack((original_resized, upscaled)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()