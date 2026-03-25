import cv2
import numpy as np
import os

def create_sample_media():
    print("Creating sample office image...")
    # Create a dummy image. Let's make it a clear image (brightness 150)
    img_good = np.full((400, 400, 3), 150, dtype=np.uint8)
    cv2.imwrite("sample_office.jpg", img_good)
    
    print("Creating sample blurry image...")
    # Create a blurry image
    img_blur = np.full((400, 400, 3), 150, dtype=np.uint8)
    img_blur = cv2.GaussianBlur(img_blur, (51, 51), 0)
    cv2.imwrite("sample_other.jpg", img_blur)

if __name__ == "__main__":
    create_sample_media()
    
    print("\n--- Testing Verifier ---")
    try:
        from verifier import MediaVerifier
        verifier = MediaVerifier()
        
        print("\nVerifying sample_office.jpg:")
        res1 = verifier.verify_media("sample_office.jpg")
        print(f"Score: {res1['score']}, Decision: {res1['decision']}")
        print(f"Reasoning: {res1['reasoning']}")
        print("Details:", res1['details'])
        
        print("\nVerifying sample_other.jpg:")
        res2 = verifier.verify_media("sample_other.jpg")
        print(f"Score: {res2['score']}, Decision: {res2['decision']}")
        print(f"Reasoning: {res2['reasoning']}")
        print("Details:", res2['details'])
    except Exception as e:
        print(f"Error during verification: {e}")
