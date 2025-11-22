import cv2
import mss
import numpy as np
import pygetwindow as gw
from ultralytics import YOLO

# --- НАСТРОЙКИ ---

# 1. ПУТЬ К ТВОЕЙ ОБУЧЕННОЙ МОДЕЛИ
#    Я взял его прямо из твоего лога. Проверь, что он верный.
MODEL_PATH = r'runs/detect/yolov8n_wow_bobber/weights/best.pt'

# 2. Заголовок окна игры
WINDOW_TITLE = "World of Warcraft"

# --- КОНЕЦ НАСТРОЕК ---

def main():
    print("--- Детектор поплавков запущен ---")
    
    # 1. Загружаем нашу обученную модель
    print(f"Загрузка модели из: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Убедитесь, что путь к файлу 'best.pt' указан верно.")
        return
        
    print("Модель успешно загружена.")

    # 2. Находим окно игры
    try:
        game_window = gw.getWindowsWithTitle(WINDOW_TITLE)[0]
    except IndexError:
        print(f"ОШИБКА: Окно '{WINDOW_TITLE}' не найдено!")
        return

    print(f"Окно '{game_window.title}' найдено. Начинаем захват...")
    print("\nНажмите 'Q' в окне детектора, чтобы выйти.")
    
    # 3. Основной цикл детекции
    with mss.mss() as sct:
        while True:
            # Определяем область захвата (все окно игры)
            monitor = {
                "top": game_window.top, 
                "left": game_window.left, 
                "width": game_window.width, 
                "height": game_window.height
            }
            
            # Захватываем кадр
            img = np.array(sct.grab(monitor))
            
            # Конвертируем BGRA (от mss) в BGR (для OpenCV)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # 4. САМАЯ ГЛАВНАЯ ЧАСТЬ: Отправляем кадр в модель
            #    stream=True - это оптимизация для видеопотока
            results = model(img_bgr, stream=True, verbose=False)

            # 5. Обрабатываем результаты
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Получаем координаты прямоугольника
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Получаем уверенность модели в результате
                    conf = round(float(box.conf[0]), 2)
                    
                    # Рисуем прямоугольник и подпись на кадре
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_bgr, f"bobber {conf}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    print(f"Поплавок найден! Уверенность: {conf}. Координаты: ({x1}, {y1})")

            # 6. Показываем результат в окне
            cv2.imshow("WoW Fishing Detector", img_bgr)

            # Выход из цикла по нажатию клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("\n--- Работа детектора завершена ---")

if __name__ == "__main__":
    main()