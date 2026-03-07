#!/usr/bin/env python3
import sys
import cv2
import easyocr
from pathlib import Path
from preprocess import preprocess_with_progress, preprocess_for_easyocr

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input_image> <output_file>")
        sys.exit(1)
    
    input_path, output_path = sys.argv[1], sys.argv[2]
    
    print(f"📄 Processing: {input_path}")
    
    # Debug: Check if file exists
    if not Path(input_path).exists():
        print(f"❌ File does not exist: {input_path}")
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)
    
    print(f"✅ File found: {Path(input_path).absolute()}")
    
    # Read image
    image = preprocess_with_progress(input_path)
    image_easy = preprocess_for_easyocr(input_path)

    if image is None or image_easy is None:
        print(f"❌ OpenCV could not read the image")
        sys.exit(1)
    
    # Initialize OCR
    reader = easyocr.Reader(['en'])
    
    # Run OCR
    result = reader.readtext(image)
    result_easy = reader.readtext(image_easy)

    # Extract text
    text = '\n'.join([item[1] for item in result]) # type: ignore
    text_easy = '\n'.join([item[1] for item in result_easy]) # type: ignore
    # Save output
    Path(output_path).write_text(text, encoding='utf-8')
    Path(output_path.replace('.txt', '_easy.txt')).write_text(text_easy, encoding='utf-8')

    print(f"✅ Done -> {output_path}")

if __name__ == "__main__":
    main()