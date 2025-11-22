import os
import numpy as np
import librosa
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# --- НАСТРОЙКИ ---
DATASET_PATH = 'dataset_audio'
SPLASH_DIR = os.path.join(DATASET_PATH, 'splash')
NOISE_DIR = os.path.join(DATASET_PATH, 'noise')

MODEL_SAVE_PATH = 'bobber_audio_model.keras'

# Параметры обработки звука (должны совпадать с генератором)
SAMPLE_RATE = 22050
DURATION = 1.0
# --- КОНЕЦ НАСТРОЕК ---

def extract_features(filepath):
    """Превращает аудиофайл в массив чисел (MFCC)."""
    try:
        # Загружаем аудио
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        
        # Гарантируем длину 1 сек (на случай сбоев генератора)
        target_len = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
        else:
            audio = audio[:target_len]

        # Вычисляем MFCC (это как "отпечаток пальца" для звука)
        # n_mfcc=40 - берем 40 ключевых характеристик
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Усредняем значения по времени, чтобы получить один вектор на весь файл
        # Это сильно упрощает модель и делает ее быстрее
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        return mfccs_processed
    except Exception as e:
        print(f"Ошибка при чтении {filepath}: {e}")
        return None

def load_data():
    print("--- Загрузка и обработка данных ---")
    features = []
    labels = []

    # 1. Загружаем ВСПЛЕСКИ (метка 1)
    splash_files = glob(os.path.join(SPLASH_DIR, '*.wav'))
    print(f"Обработка {len(splash_files)} файлов всплеска...")
    for file in splash_files:
        data = extract_features(file)
        if data is not None:
            features.append(data)
            labels.append(1) # 1 = Всплеск

    # 2. Загружаем ШУМЫ (метка 0)
    noise_files = glob(os.path.join(NOISE_DIR, '*.wav'))
    print(f"Обработка {len(noise_files)} файлов шума...")
    for file in noise_files:
        data = extract_features(file)
        if data is not None:
            features.append(data)
            labels.append(0) # 0 = Шум

    return np.array(features), np.array(labels)

def main():
    # 1. Подготовка данных
    X, y = load_data()
    
    # Перемешиваем данные
    X, y = shuffle(X, y, random_state=42)
    
    # Разделяем на обучение (80%) и тест (20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nДанные подготовлены: {len(X_train)} для обучения, {len(X_val)} для проверки.")

    # 2. Создание нейросети
    # Используем простую полносвязную сеть (Dense), так как входные данные простые
    model = tf.keras.models.Sequential([
        # Входной слой (40 нейронов, так как у нас 40 MFCC признаков)
        tf.keras.layers.Input(shape=(40,)),
        
        # Скрытые слои
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2), # Защита от переобучения
        
        tf.keras.layers.Dense(32, activation='relu'),
        
        # Выходной слой (1 нейрон: 0 или 1)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 3. Обучение
    print("\n--- Начало обучения ---")
    history = model.fit(X_train, y_train, 
                        epochs=15,            # 15 эпох обычно достаточно для такой простой задачи
                        batch_size=32, 
                        validation_data=(X_val, y_val))

    # 4. Оценка и сохранение
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"\nТочность на проверке: {accuracy*100:.2f}%")
    
    model.save(MODEL_SAVE_PATH)
    print(f"Модель сохранена в файл: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()