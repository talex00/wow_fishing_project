import soundcard as sc
import numpy as np

# --- НАСТРОЙКИ ---
# Начни с этого значения и подбирай. Шкала громкости будет примерно от 0.0 до 10.0
THRESHOLD = 10
# --- КОНЕЦ НАСТРОЕК ---

print("--- Тест громкости v2 (SoundCard - правильный метод) ---")

try:
    # 1. Находим устройство вывода по умолчанию (наши колонки/наушники)
    default_speaker = sc.default_speaker()
    print(f"Найдено устройство вывода по умолчанию: '{default_speaker.name}'")

    # 2. Ищем СООТВЕТСТВУЮЩИЙ ему микрофон в режиме loopback
    #    Мы передаем ID колонок, чтобы найти именно их "виртуальный микрофон"
    loopback_mic = sc.get_microphone(id=str(default_speaker.id), include_loopback=True)
    print(f"Найден соответствующий loopback-микрофон: '{loopback_mic.name}'")

    print("\nЗапустите игру и начните рыбачить. Следите за значениями громкости.")
    print(f"Текущий порог для теста: {THRESHOLD}")
    print("Нажмите Ctrl+C для выхода.")

    # 3. Запускаем запись именно с этого loopback-микрофона
    with loopback_mic.recorder(samplerate=48000) as mic:
        while True:
            audio_data = mic.record(numframes=1024)
            volume = np.linalg.norm(audio_data) * 10
            
            print(f"Текущая громкость: {volume:.2f}", end='\r')

            if volume > THRESHOLD:
                print(f"\n!!! ОБНАРУЖЕН ПИК ГРОМКОСТИ: {volume:.2f} !!!")
                
except KeyboardInterrupt:
    print("\n--- Тест завершен ---")
except Exception as e:
    print(f"\n[ОШИКА АУДИО] {e}")
    print("Не удалось найти или запустить loopback-устройство.")
    print("Убедитесь, что в Windows выбрано устройство вывода звука по умолчанию.")