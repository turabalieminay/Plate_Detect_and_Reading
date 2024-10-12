from ultralytics import YOLO
import cv2

# YOLOv8 modelini yükle
model = YOLO('last.pt')  # last.pt model dosyanızın bulunduğu yolu belirtin

# Tespit yapılacak görüntünün dosya yolu (örneğin: 'C:/Users/aytur/Desktop/plate_detect/inference/image.jpg')
image_path = 'inference/x5_1103506790srf.jpg'

# Görüntü üzerinde plaka tespiti
def detect_plate(image_path):
    # Görüntüyü yükle
    img = cv2.imread(image_path)
    
    # YOLO ile plaka tespiti
    results = model.predict(source=img)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding box koordinatları (xmin, ymin, xmax, ymax)
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
            
            # Tespit edilen plakayı kare içine al (Görüntü üzerinde çerçeve çiz)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(img, "Plate", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    # Sonucu göster
    cv2.imshow('Detected Plates', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Belirtilen görüntüde plaka tespiti
detect_plate(image_path)

