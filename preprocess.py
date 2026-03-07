import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def deskew_image(image):
    """Detect and correct image skew"""
    # Find all non-zero points (text)
    coords = np.column_stack(np.where(image > 0))
    
    if len(coords) < 10:  # Not enough points to detect skew
        print("  Not enough text to detect skew")
        return image
    
    # Calculate skew angle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    print(f"  Detected skew angle: {angle:.2f} degrees")
    
    # Only deskew if angle is significant
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        print(f"  Corrected rotation by {angle:.2f} degrees")
        return rotated
    else:
        print("  No significant skew detected")
        return image

def preprocess_with_progress(image_path):
    """Preprocess image and show each step using matplotlib"""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        img = np.array(Image.open(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Step 1: Original
    plt.figure(figsize=(8,4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Step 1/5: Original")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # Step 2: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(8,4))
    plt.imshow(gray, cmap='gray')
    plt.title("Step 2/5: Grayscale")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # Step 3: Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    plt.figure(figsize=(8,4))
    plt.imshow(denoised, cmap='gray')
    plt.title("Step 3/5: Denoised")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # Step 4: Deskew
    print("🔄 Checking for skew...")
    deskewed = deskew_image(denoised)
    plt.figure(figsize=(8,4))
    plt.imshow(deskewed, cmap='gray')
    plt.title("Step 4/5: Deskewed")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # Step 5: Binary
    _, binary = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure(figsize=(8,4))
    plt.imshow(binary, cmap='gray')
    plt.title("Step 5/5: Binary (Ready for OCR)")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    return binary


def preprocess_for_easyocr(image_path):
    """Optimized preprocessing specifically for EasyOCR"""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        img = np.array(Image.open(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Show original
    plt.figure(figsize=(8,4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Step 1/4: Original")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # STEP 1: Convert to grayscale (keep for visualization)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(8,4))
    plt.imshow(gray, cmap='gray')
    plt.title("Step 2/4: Grayscale")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # STEP 2: Gentle denoising (preserve edges)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)  # Reduced from 30
    plt.figure(figsize=(8,4))
    plt.imshow(denoised, cmap='gray')
    plt.title("Step 3/4: Light Denoising")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # STEP 3: Deskew if needed
    print("🔄 Checking for skew...")
    coords = np.column_stack(np.where(denoised > 0))
    if len(coords) > 10:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        print(f"  Detected angle: {angle:.2f} degrees")
        if abs(angle) > 0.5:
            (h, w) = denoised.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            denoised = cv2.warpAffine(denoised, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            print("  Corrected rotation")
        else:
            print("  No significant skew")
    
    plt.figure(figsize=(8,4))
    plt.imshow(denoised, cmap='gray')
    plt.title("Step 4/4: Final (Ready for EasyOCR)")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # Return GRAYSCALE, not binary!
    return denoised