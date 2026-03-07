# Simple OCR Pipeline

## Overview

This is a simple OCR pipeline demonstrating the flow of OCR(Optical Character Recognition).
This uses EasyOCR pre-trained model to recognize text from a picture of printed text paper after heuristic preprocessing of grayscaling, denoising, deskewing and binarization.

## Run

Clone the git repo, install dependencies and use run.py to run the project

```bash
git clone https://github.com/coderanger428/ai-ocr-pipeline.git
pip install -r requirements.txt
python run.py <image_path> <output_txt_path>
```

## Future Improvements

Use a different model or delicate preprocessing pipeline to enhance the results
