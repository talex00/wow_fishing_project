import os
import time
from datetime import datetime
import keyboard
import mss
import pygetwindow as gw

# --- НАСТРОЙКИ ---
WINDOW_TITLE = "World of Warcraft"
SAVE_FOLDER = "fishing_dataset_sorted"
TRIGGER_KEY = "page up"
EXIT_KEY = "esc"
# --- КОНЕЦ НАСТРОЕК ---


def take_screenshot(window_info):
    """
    Делает скриншот указанного окна и сохраняет его с параметрами в имени.
    ВАЖНО: Экземпляр mss создается внутри функции, чтобы избежать проблем с потоками.
    """
    # Получаем актуальные размеры окна на момент вызова
    width = window_info.width
    height = window_info.height
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    filename = f"{width}x{height}_{timestamp}.png"
    full_path = os.path.join(SAVE_FOLDER, filename)
    
    monitor = {
        "top": window_info.top, 
        "left": window_info.left, 
        "width": width, 
        "height": height
    }
    
    # Создаем экземпляр mss прямо здесь, в том же потоке, где он будет использован.
    with mss.mss() as sct:
        # Захватываем изображение
        sct_img = sct.grab(monitor)
        
        # Сохраняем изображение в файл
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=full_path)
    
    print(f"Скриншот сохранен: {full_path}")


def main():
    """
    Основная функция: находит окно, настраивает окружение и запускает прослушивание.
    """
    print("--- Умный сборщик скриншотов запущен (v2, исправлена ошибка потоков) ---")
    
    print(f"Ищем окно с заголовком, содержащим: '{WINDOW_TITLE}'...")
    try:
        game_window = gw.getWindowsWithTitle(WINDOW_TITLE)[0]
    except IndexError:
        print(f"\nОШИБКА: Окно с заголовком '{WINDOW_TITLE}' не найдено!")
        print("Пожалуйста, убедитесь, что игра запущена и заголовок в настройках указан верно.")
        return

    print(f"Окно найдено! -> '{game_window.title}'. Текущее разрешение: {game_window.width}x{game_window.height}")
    game_window.activate()
    time.sleep(0.5)

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
        print(f"Создана папка для сохранения: {SAVE_FOLDER}")
        
    print(f"\nНажмите '{TRIGGER_KEY.upper()}' чтобы сделать скриншот окна игры.")
    print(f"Нажмите '{EXIT_KEY.upper()}' чтобы завершить работу скрипта.\n")
    
    # Обратите внимание, мы больше не создаем 'sct' здесь.
    # Мы просто передаем объект окна в нашу функцию.
    keyboard.add_hotkey(TRIGGER_KEY, lambda: take_screenshot(game_window))
    
    keyboard.wait(EXIT_KEY)

    print("\n--- Работа скрипта завершена ---")


if __name__ == "__main__":
    main()