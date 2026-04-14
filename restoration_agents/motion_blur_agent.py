import cv2
import numpy as np
import os

class MotionBlurAgent:
    def __init__(self, blur_threshold=250.0):

        #Initializes the Deblurring Agent using a Gaussian Unsharp Mask.
        self.blur_threshold = blur_threshold  # Lower = blurrier

    def get_blur_score(self, face_crop):
        # The Laplacian operator calculates the 2nd derivative of the image
        # This highlights regions of rapid intensity change (edges)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def process(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return face_crop

        # 1. Measure the blur score
        score = self.get_blur_score(face_crop)
        print(f"Blur Score: {score:.2f}")

        # 2. Check if fix is needed
        if score < self.blur_threshold:
            # DYNAMIC CALCULATION:
            # We want 'amount' to increase as 'score' decreases.
            # Example: if threshold is 100 and score is 20, we need a lot of sharpening.
            # If score is 80, we only need a little bit.

            # This formula scales amount based on the 'gap' between threshold and score
            raw_amount = (self.blur_threshold / max(score, 1.0)) * 0.5
            dynamic_amount = min(raw_amount, 2.5)

            print(f"Applying Dynamic Sharpness: {dynamic_amount:.2f}")

            # 3. Apply the fix using the dynamic amount
            blurred = cv2.GaussianBlur(face_crop, (5, 5), 1.0)
            # Result = Original + (Amount X Details).
            sharpened = cv2.addWeighted(face_crop, 1.0 + dynamic_amount, blurred, -dynamic_amount, 0)
            return sharpened

        return face_crop

def main():
    print("---  Testing Motion Blur Agent on Real Data ---")

    # 1. Initialize the Agent
    agent = MotionBlurAgent()

    # 2. Path to your REAL blurry image
    img_path = r"C:\Users\yoavt\PycharmProjects\final_projact\data\gate_dataset\test\78754992-8693-4f2a-b38d-81d28ed01cc4.JPG"

    if not os.path.exists(img_path):
        print(f" Error: Image not found at {img_path}")
        return

    # 3. Load the real blurry image
    blurry_img = cv2.imread(img_path)
    if blurry_img is None: return

    # 4. Run the Agent (Fix the real blur)
    result = agent.process(blurry_img)

    # 5. Visual Comparison (Real Blur vs Fixed)
    h, w = blurry_img.shape[:2]
    # Resize only for display purposes
    display_blurry = cv2.resize(blurry_img, (224, 224))
    display_fixed = cv2.resize(result, (224, 224))

    # Add Labels
    cv2.putText(display_blurry, "before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_fixed, "Agent Fixed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Stack Side-by-Side
    comparison = np.hstack((display_blurry, display_fixed))

    print(" Displaying results... (Press any key to close)")
    cv2.imshow("Real Motion Blur Correction", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()