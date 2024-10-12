from ultralytics import YOLO
import cv2
from PIL import Image
import pytesseract

# Tesseract'ın bilgisayarında yüklü olduğu yolu belirt
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\aytur\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# YOLOv8 modelini yükle
model = YOLO('last.pt')

# Görüntü dosya yolu (örneğin: 'C:/Users/aytur/Desktop/plate_detect/inference/image.jpg')
image_path = 'inference/x5_1103506790srf.jpg'

# OCR sonuçlarını kaydetmek için bir liste oluştur
ocr_results = []

# Görüntü üzerinde plaka tespiti ve OCR işlemi
def detect_and_read_plate(image_path):
    # Görüntüyü yükle
    img = cv2.imread(image_path)
    
    # YOLO ile plaka tespiti
    results = model.predict(source=img)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding box koordinatları (xmin, ymin, xmax, ymax)
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
            
            # Tespit edilen plakayı kırp
            plate = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            
            # OCR işlemi için plakayı gri tonlamaya çevir
            gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            
            # OCR işlemi için PIL formatına dönüştür
            pil_image = Image.fromarray(gray_plate)
            
            # OCR ile plakayı oku
            plate_text = pytesseract.image_to_string(pil_image, config='--psm 8')
            
            # OCR sonucu ve plaka koordinatlarını kaydet
            ocr_results.append((plate_text, (xmin, ymin, xmax, ymax)))
            
            # Tespit edilen plakayı ve OCR sonucunu göster
            print(f"Tespit edilen plaka: {plate_text}")
            
            # Plaka çerçevesini ve metni görüntüye ekle
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(img, plate_text.strip(), (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
    # Sonucu göster
    cv2.imshow('Detected Plates', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Belirtilen görüntüde plaka tespiti ve okuma işlemi
detect_and_read_plate(image_path)
