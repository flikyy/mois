import pymurapi
import cv2
import numpy as np
from ultralytics import YOLO
import time

# 1. Инициализация нейросети и симулятора
# Используем модель nano для максимального FPS в реальном времени
model = YOLO('yolov8n.pt') 
auv = pymurapi.mur_init()

def process_image(img):
    """
    Обработка кадра: захват любого обнаруженного объекта в рамку.
    """
    # Запуск детекции без фильтрации по классам
    # conf=0.3 — порог уверенности (чем ниже, тем больше объектов захватит)
    results = model.predict(img, conf=0.3, verbose=False)
    
    annotated_frame = img.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Получаем координаты рамки (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Отрисовка универсальной рамки (зеленая, без названия класса)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Расчет центра объекта для навигации
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Вместо названия класса выводим "Object" и уверенность
            conf = float(box.conf[0])
            cv2.putText(annotated_frame, f"Object {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return annotated_frame

def main():
    print("Алгоритм захвата объектов запущен. Для выхода нажмите 'q'.")
    prev_time = 0

    while True:
        # Захват изображения с фронтальной камеры[cite: 3]
        frame = auv.get_image_front()
        
        if frame is not None:
            # Обработка
            output_img = process_image(frame)
            
            # Расчет FPS[cite: 3]
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            cv2.putText(output_img, f"FPS: {int(fps)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Вывод через OpenCV (исправляет AttributeError со скриншота image_56fafa.png)[cite: 3]
            cv2.imshow("Object Detection Tool", output_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        time.sleep(0.01)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
