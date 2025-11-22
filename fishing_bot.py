import mss
import numpy as np
import pygetwindow as gw
from ultralytics import YOLO
import pydirectinput
import time
import keyboard
import soundcard as sc
import cv2
import os
import json  # <--- Для сохранения настроек

# Отключаем лишние логи
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import librosa

# ==============================================================================
# --- НАСТРОЙКИ ПО УМОЛЧАНИЮ ---
# ==============================================================================
CONFIG_FILE = 'bot_config.json'
VISUAL_MODEL_PATH = r'visual_model.pt'
AUDIO_MODEL_PATH  = r'audio_model.keras'
WINDOW_TITLE = "World of Warcraft"

# Клавиши
FISHING_KEY = '1'
START_STOP_KEY = 'f8'
CALIBRATE_KEY = 'f9'  # <--- Новая клавиша для настройки
EXIT_KEY = 'esc'

# Константы
CONFIDENCE_THRESHOLD = 0.65
SEARCH_TIMEOUT_SEC = 15
LISTEN_TIMEOUT_SEC = 20
AI_CONFIDENCE_THRESHOLD = 0.8

# Глобальные переменные
bot_running = False
# Значение по умолчанию, если конфига нет (будет перезаписано)
current_proximity_threshold = 0.5 

# ==============================================================================

def load_config():
    """Загружает порог громкости из файла."""
    global current_proximity_threshold
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                current_proximity_threshold = data.get('proximity_threshold', 0.5)
                print(f"[КОНФИГ] Настройки загружены. Порог громкости: {current_proximity_threshold:.2f}")
        except Exception as e:
            print(f"[КОНФИГ] Ошибка чтения конфига: {e}")
    else:
        print("[КОНФИГ] Файл настроек не найден. Использую значения по умолчанию.")

def save_config(threshold):
    """Сохраняет порог громкости в файл."""
    global current_proximity_threshold
    
    # !!! ИСПРАВЛЕНИЕ: Принудительно превращаем numpy-число в обычный float
    threshold = float(threshold)
    
    current_proximity_threshold = threshold
    data = {'proximity_threshold': threshold}
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f)
        print(f"[КОНФИГ] Настройки сохранены! Новый порог: {threshold:.2f}")
    except Exception as e:
        print(f"[КОНФИГ] Не удалось сохранить настройки: {e}")

def toggle_bot_state():
    global bot_running
    bot_running = not bot_running
    state = "ЗАПУЩЕН" if bot_running else "ОСТАНОВЛЕН"
    print(f"\n========== БОТ {state} ==========")

def calibrate_sound_threshold(audio_model):
    """
    Мастер настройки звука.
    Просит пользователя сделать всплеск, замеряет его и выставляет порог.
    """
    print("\n--------------------------------------------------")
    print("   РЕЖИМ КАЛИБРОВКИ ЗВУКА")
    print("--------------------------------------------------")
    print("1. Перейдите в игру.")
    print("2. Забросьте удочку.")
    print("3. Ждите, пока произойдет всплеск.")
    print("   (Я буду слушать 15 секунд и искать САМЫЙ ГРОМКИЙ всплеск)")
    
    print("\nНачинаю прослушивание через 3...", end="", flush=True); time.sleep(1)
    print(" 2...", end="", flush=True); time.sleep(1)
    print(" 1...", end="", flush=True); time.sleep(1)
    print(" СЛУШАЮ! Сделайте всплеск!")
    
    max_splash_volume = 0.0
    start_time = time.time()
    
    # Настройки аудио (как в основном боте)
    TARGET_SR = 22050
    BUFFER_SIZE = int(TARGET_SR * 1.0)
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    
    try:
        default_speaker = sc.default_speaker()
        loopback_mic = sc.get_microphone(id=str(default_speaker.id), include_loopback=True)
        CHUNK_SIZE = int(TARGET_SR * 0.1)

        with loopback_mic.recorder(samplerate=TARGET_SR) as mic:
            while time.time() - start_time < 15: # Слушаем 15 секунд
                new_data = mic.record(numframes=CHUNK_SIZE)
                new_data = np.mean(new_data, axis=1)
                
                # Обновляем буфер для ИИ
                audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
                audio_buffer[-CHUNK_SIZE:] = new_data
                
                # Считаем громкость
                current_vol = np.linalg.norm(new_data) * 10
                
                # Проверяем ИИ, является ли это всплеском
                # (Чтобы не калиброваться на случайный шум машины)
                input_data = preprocess_audio_chunk(audio_buffer, sr=TARGET_SR)
                is_splash = False
                if input_data is not None:
                    prediction = audio_model(input_data, training=False)[0][0]
                    if prediction > 0.6: # Чуть снизим порог для калибровки
                        is_splash = True
                
                if is_splash:
                    print(f"   > Услышал всплеск! Громкость: {current_vol:.2f}")
                    if current_vol > max_splash_volume:
                        max_splash_volume = current_vol
                
    except Exception as e:
        print(f"Ошибка калибровки: {e}")
        return

    if max_splash_volume == 0.0:
        print("\n[ОШИБКА] Я не услышал ни одного всплеска за 15 секунд.")
        print("Попробуйте снова.")
    else:
        # ГЛАВНАЯ МАГИЯ:
        # Мы устанавливаем порог на уровне 40% от громкости ТВОЕГО всплеска.
        # Твой всплеск (100%) пройдет.
        # Чужой всплеск (обычно тише 50%) будет отсеян.
        new_threshold = max_splash_volume * 0.4
        
        print(f"\nМаксимальная громкость вашего всплеска: {max_splash_volume:.2f}")
        print(f"Рекомендуемый порог (40%): {new_threshold:.2f}")
        save_config(new_threshold)
        print("--------------------------------------------------")
        print("Калибровка завершена. Можете запускать бота (F8).")


def preprocess_audio_chunk(audio_data, sr=22050):
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return np.expand_dims(mfccs_processed, axis=0)
    except:
        return None

def find_bobber(model, window):
    start_time = time.time()
    with mss.mss() as sct:
        while time.time() - start_time < SEARCH_TIMEOUT_SEC:
            monitor = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
            img = np.array(sct.grab(monitor))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            results = model(img_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)
            best_box = None; max_conf = 0
            for box in results[0].boxes:
                if box.conf[0] > max_conf:
                    max_conf = box.conf[0]
                    best_box = box
            
            if best_box is not None:
                x1, y1, x2, y2 = [int(i) for i in best_box.xyxy[0]]
                center_x = window.left + (x1 + x2) // 2
                center_y = window.top + (y1 + y2) // 2
                print(f"   [ЗРЕНИЕ] Поплавок найден! (Conf: {max_conf:.2f})")
                return (center_x, center_y)
            time.sleep(0.5)
    return None

def listen_for_splash_ai(audio_model, timeout):
    """Гибридный слух с использованием настройки из конфига."""
    global current_proximity_threshold # Используем глобальную переменную
    
    start_time = time.time()
    TARGET_SR = 22050
    BUFFER_SIZE = int(TARGET_SR * 1.0)
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    
    try:
        default_speaker = sc.default_speaker()
        loopback_mic = sc.get_microphone(id=str(default_speaker.id), include_loopback=True)
        CHUNK_SIZE = int(TARGET_SR * 0.1) 
        
        print(f"   [СЛУХ] Жду клева... (Порог дальности: {current_proximity_threshold:.2f})")

        with loopback_mic.recorder(samplerate=TARGET_SR) as mic:
            while time.time() - start_time < timeout:
                new_data = mic.record(numframes=CHUNK_SIZE)
                new_data = np.mean(new_data, axis=1) 
                
                audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
                audio_buffer[-CHUNK_SIZE:] = new_data
                
                # 1. Фильтр громкости (Proximity Check)
                current_vol = np.linalg.norm(new_data) * 10
                if current_vol < current_proximity_threshold:
                    continue # Слишком тихо/далеко

                # 2. ИИ проверка
                input_data = preprocess_audio_chunk(audio_buffer, sr=TARGET_SR)
                if input_data is not None:
                    prediction = audio_model(input_data, training=False)[0][0]
                    if prediction > AI_CONFIDENCE_THRESHOLD:
                        print(f"\n   [СЛУХ] !!! ПОКЛЕВКА !!! (Vol: {current_vol:.2f} | AI: {prediction:.2f})")
                        return True
    except Exception as e:
        print(f"\n[ОШИБКА СЛУХА] {e}"); return False
    return False

def loot_bobber(x, y):
    original_pos = pydirectinput.position()
    print(f"   [ДЕЙСТВИЕ] Лутаем!")
    pydirectinput.moveTo(x, y, duration=0.1)
    pydirectinput.rightClick()
    time.sleep(0.5)
    pydirectinput.moveTo(original_pos[0], original_pos[1], duration=0.1)

def main():
    print("--- WoW AI Fishing Bot v6.0 (Auto-Config) ---")
    print("Инициализация...")
    
    # 0. Загружаем настройки
    load_config()

    try:
        vision_model = YOLO(VISUAL_MODEL_PATH)
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        print(">> Модели загружены.")
    except Exception as e:
        print(f"\nОШИБКА ЗАГРУЗКИ: {e}"); return

    keyboard.add_hotkey(START_STOP_KEY, toggle_bot_state)
    
    # Добавляем клавишу для калибровки
    keyboard.add_hotkey(CALIBRATE_KEY, lambda: calibrate_sound_threshold(audio_model))
    
    try:
        game_window = gw.getWindowsWithTitle(WINDOW_TITLE)[0]
    except IndexError:
        print(f"ОШИБКА: Окно '{WINDOW_TITLE}' не найдено!"); return
    
    print(f"\nУправление:")
    print(f"  [{START_STOP_KEY.upper()}] - Старт / Стоп")
    print(f"  [{CALIBRATE_KEY.upper()}] - КАЛИБРОВКА ЗВУКА (Нажать перед первым использованием)")
    print(f"  [{EXIT_KEY.upper()}] - Выход")

    while not keyboard.is_pressed(EXIT_KEY):
        if not bot_running:
            time.sleep(0.1)
            continue
            
        print("\n--- Цикл ---")
        print("[1] Заброс...")
        game_window.activate(); time.sleep(0.2)
        pydirectinput.press(FISHING_KEY); time.sleep(2.5) 
        
        print("[2] Поиск...")
        bobber_coords = find_bobber(vision_model, game_window)
        
        if bobber_coords is None:
            print("   Не нашел. Повтор."); continue
        
        print(f"[3] Слушаю...")
        was_splash = listen_for_splash_ai(audio_model, LISTEN_TIMEOUT_SEC)
        
        if was_splash:
            loot_bobber(bobber_coords[0], bobber_coords[1])
            time.sleep(1.0) 
        else:
            print("   Тишина. Повтор.")
        
        time.sleep(1.5)
    print("\n--- Пока! ---")

if __name__ == "__main__":
    main()