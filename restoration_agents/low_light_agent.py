import os
import cv2
import numpy as np


class DynamicLowLightAgent:
    def __init__(self):
        pass

    def _get_dynamic_params(self, l_channel):
        avg_brightness = np.mean(l_channel)
        print("avg_brightness:", avg_brightness)

        # X-axis: The brightness thresholds from your original code
        brightness_levels = [0, 20, 30, 40, 80, 100, 255]

        # Y-axis: The parameters you want at those exact brightness levels
        clip_limits = [4.0, 4.0, 3.0, 3.0, 2.0, 1.0, 0.0]
        gammas = [0.70, 0.70, 0.80, 0.85, 1.0, 1.0, 1.0]
        denoises = [15, 15, 12, 10, 5, 0, 0]

        # np.interp smoothly calculates the exact value based on where avg_brightness falls
        clip_limit = np.interp(avg_brightness, brightness_levels, clip_limits)
        gamma = np.interp(avg_brightness, brightness_levels, gammas)

        # Denoise needs to be an integer for the filter
        denoise = int(np.interp(avg_brightness, brightness_levels, denoises))

        return clip_limit, gamma, denoise

    def process(self, image):

        if image is None: return None
        #LAB - format for allows the agent to isolate the L (Lightness) channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clip, gamma, denoise = self._get_dynamic_params(l)

        # --- Early Exit Logic ---
        if clip == 0 and gamma == 1.0 and denoise == 0:
            return image  # Return the original BGR image without any math
        # -----------------------------

        # 1. Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))  # Smaller grid often better for faces
        enhanced_l = clahe.apply(l)

        # 2. Apply Gamma Correction to the L channel BEFORE converting to BGR
        # This prevents the color shift in skin tones.
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_l = cv2.LUT(enhanced_l, table)

        # 3. Proportionally boost saturation to prevent the "ghost" look
        # If we brighten the image, we must slightly scale the A and B channels towards the edges
        if clip > 1.0 or gamma < 1.0:
            # Shift A and B channels to center 0, scale, then shift back to 128
            a = cv2.addWeighted(a, 1.15, np.zeros_like(a), 0, -128 * 0.15)
            b = cv2.addWeighted(b, 1.15, np.zeros_like(b), 0, -128 * 0.15)

        # 4. Merge back to BGR
        merged = cv2.merge((enhanced_l, a, b))
        final_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # 5. Fast, Edge-Preserving Denoising (Bilateral Filter instead of NL-Means)
        if denoise > 0:
            # Adjust d, sigmaColor, sigmaSpace depending on image size
            final_bgr = cv2.bilateralFilter(final_bgr, d=5, sigmaColor=25, sigmaSpace=25)

        return final_bgr


def main():
        print("Testing Low Light Agent")

        # 1. Initialize the Agent
        agent = DynamicLowLightAgent()

        # 2. Load a dark image
        img_path = r"C:\Users\yoavt\PycharmProjects\final_projact\data\resnet dataset\real_test\test_for_gate\real_dark_17.jpg"

        if not os.path.exists(img_path):
            print(f" Error: Image not found at {img_path}")
            return

        original = cv2.imread(img_path)

        # 3. Run the Agent
        result = agent.process(original)

        # 4. Show Side-by-Side Comparison
        # We stack them horizontally (Left: Original, Right: Fixed)
        comparison = np.hstack((original, result))

        # Optional: Resize if the image is too big for screen
        h, w = comparison.shape[:2]
        if w > 1500:
            scale = 1500 / w
            comparison = cv2.resize(comparison, None, fx=scale, fy=scale)

        cv2.imshow("Left: Original (Dark) | Right: Agent Result (Enhanced)", comparison)

        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()