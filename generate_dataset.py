import os
import numpy as np
import librosa
import soundfile as sf
from glob import glob

# --- НАСТРОЙКИ ---
SOURCE_SPLASH = 'audio_source/splash'
SOURCE_NOISE = 'audio_source/noise'

OUTPUT_SPLASH = 'dataset_audio/splash'
OUTPUT_NOISE = 'dataset_audio/noise'

TARGET_SR = 22050  # Стандарт для обучения
DURATION = 1.0     # Длительность всех файлов (1 сек)

# Сколько вариаций создавать для каждого ВСПЛЕСКА
AUGMENTATIONS_PER_SPLASH = 200 
# --- КОНЕЦ НАСТРОЕК ---

def ensure_dirs():
    os.makedirs(OUTPUT_SPLASH, exist_ok=True)
    os.makedirs(OUTPUT_NOISE, exist_ok=True)

def pad_or_trim(audio, sr, duration):
    """Приводит аудио к ровно 1 секунде."""
    target_len = int(sr * duration)
    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        return audio[start : start + target_len]
    elif len(audio) < target_len:
        padding = target_len - len(audio)
        return np.pad(audio, (0, padding), 'constant')
    return audio

def augment_audio(audio, sr, is_noise=False):
    """Создает измененную версию звука."""
    # Для шума мы делаем изменения чуть мягче, чтобы он оставался узнаваемым
    # Для всплеска - чуть агрессивнее
    
    # 1. Изменение высоты тона (Pitch)
    steps = np.random.uniform(-2, 2) if not is_noise else np.random.uniform(-1, 1)
    try:
        audio_aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
    except:
        audio_aug = audio # Если сбой (бывает на очень коротких звуках)
    
    # 2. Изменение скорости (Time Stretch)
    rate = np.random.uniform(0.8, 1.2)
    try:
        audio_aug = librosa.effects.time_stretch(audio_aug, rate=rate)
    except:
        pass

    # 3. Шум и громкость
    noise_amp = 0.005 * np.random.uniform() * np.amax(audio_aug)
    audio_aug = audio_aug + noise_amp * np.random.normal(size=audio_aug.shape[0])
    
    gain = np.random.uniform(0.7, 1.3)
    audio_aug = audio_aug * gain
    
    return audio_aug

def process_files():
    ensure_dirs()
    print("--- Генерация сбалансированного датасета ---")
    
    # Ищем файлы всех популярных форматов
    extensions = ['*.ogg', '*.wav', '*.mp3']
    
    splash_files = []
    for ext in extensions:
        splash_files.extend(glob(os.path.join(SOURCE_SPLASH, ext)))
        
    noise_files = []
    for ext in extensions:
        noise_files.extend(glob(os.path.join(SOURCE_NOISE, ext)))

    if not splash_files:
        print("ОШИБКА: Нет файлов всплесков!")
        return
    if not noise_files:
        print("ОШИБКА: Нет файлов шума!")
        return

    # --- 1. Генерируем ВСПЛЕСКИ ---
    print(f"Найдено исходных всплесков: {len(splash_files)}")
    total_splashes_target = len(splash_files) * AUGMENTATIONS_PER_SPLASH
    print(f"Цель: создать {total_splashes_target} примеров всплеска.")
    
    count_splash = 0
    for filepath in splash_files:
        try:
            audio, sr = librosa.load(filepath, sr=TARGET_SR)
            base_audio = pad_or_trim(audio, sr, DURATION)
            filename = os.path.basename(filepath).split('.')[0]
            
            # Сохраняем оригинал
            sf.write(os.path.join(OUTPUT_SPLASH, f"{filename}_orig.wav"), base_audio, sr)
            
            # Генерируем вариации
            for i in range(AUGMENTATIONS_PER_SPLASH):
                aug_audio = augment_audio(base_audio, sr, is_noise=False)
                aug_audio = pad_or_trim(aug_audio, sr, DURATION)
                sf.write(os.path.join(OUTPUT_SPLASH, f"{filename}_aug_{i}.wav"), aug_audio, sr)
                count_splash += 1
        except Exception as e:
            print(f"Ошибка {filepath}: {e}")

    # --- 2. Генерируем ШУМЫ (С балансировкой) ---
    print(f"\nНайдено исходных шумов: {len(noise_files)}")
    
    # Считаем, сколько нужно копий каждого шума, чтобы догнать количество всплесков
    # Пример: нужно 600 всплесков, у нас 17 шумов. 600 / 17 = ~35 копий каждого шума.
    augs_per_noise = int(total_splashes_target / len(noise_files))
    print(f"Балансировка: Буду создавать по {augs_per_noise} вариаций для каждого файла шума.")

    count_noise = 0
    for filepath in noise_files:
        try:
            audio, sr = librosa.load(filepath, sr=TARGET_SR)
            base_audio = pad_or_trim(audio, sr, DURATION) # Обрезаем или дополняем исходный шум
            filename = os.path.basename(filepath).split('.')[0]

            # Генерируем вариации шума
            for i in range(augs_per_noise):
                aug_audio = augment_audio(base_audio, sr, is_noise=True)
                aug_audio = pad_or_trim(aug_audio, sr, DURATION)
                sf.write(os.path.join(OUTPUT_NOISE, f"{filename}_aug_{i}.wav"), aug_audio, sr)
                count_noise += 1
                
        except Exception as e:
            print(f"Ошибка {filepath}: {e}")

    print(f"\n--- ИТОГ ---")
    print(f"Всплесков создано: {count_splash}")
    print(f"Шумов создано:     {count_noise}")
    
    if abs(count_splash - count_noise) > 100:
        print("Предупреждение: Небольшой дисбаланс, но для начала пойдет.")
    else:
        print("Баланс идеальный!")

if __name__ == "__main__":
    process_files()