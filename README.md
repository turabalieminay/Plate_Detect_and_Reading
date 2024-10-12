# Plate_Detect_and_Reading

## License Plate Detection and OCR System

This project demonstrates a license plate detection and recognition system using a custom-trained YOLOv8 model and Tesseract OCR. The project includes two scripts: one for detecting and recognizing the license plate text and the other for performing plate detection only.

### Script 1: License Plate Detection and OCR

This script integrates **YOLOv8** for detecting license plates and **Tesseract** for Optical Character Recognition (OCR) to extract the text from the detected license plates.

#### Workflow:
1. The image is processed using the **YOLOv8 model** to detect license plates.
2. For each detected license plate, the bounding box coordinates are used to crop the plate from the image.
3. The cropped plate image is then converted to grayscale for better OCR results.
4. **Tesseract** is applied to the grayscale plate image to read and extract the text.
5. The detected plate text, along with the bounding box coordinates, is displayed on the image.
6. The processed image shows both the bounding box around the detected plate and the extracted plate text.

This approach enables real-time detection and recognition of license plates in various images, making it useful for tasks such as traffic monitoring and parking management.

### Script 2: License Plate Detection Only

In this script, the focus is purely on detecting license plates using the **YOLOv8 model**.

#### Workflow:
1. The image is processed using the **YOLOv8 model** to detect license plates.
2. Bounding boxes are drawn around the detected plates to highlight their location in the image.
3. No OCR is performed in this script—its sole function is plate detection.

This script is useful when only detection is needed without recognizing the text on the plates. It can be a good starting point for building more complex systems or for visualization purposes.

### Dependencies
To run these scripts, you will need the following:
- Python 3.x
- OpenCV (`pip install opencv-python`)
- YOLOv8 (`pip install ultralytics`)
- Tesseract OCR (`pip install pytesseract` and download from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract))
- Pillow (`pip install pillow`)
