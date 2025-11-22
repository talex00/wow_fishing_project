import os
import random
import shutil

# --- НАСТРОЙКИ ---
BASE_PATH = 'dataset'
IMAGES_SRC = os.path.join(BASE_PATH, 'images')
LABELS_SRC = os.path.join(BASE_PATH, 'labels')

TRAIN_PATH_IMG = os.path.join(BASE_PATH, 'train', 'images')
TRAIN_PATH_LBL = os.path.join(BASE_PATH, 'train', 'labels')
VAL_PATH_IMG = os.path.join(BASE_PATH, 'val', 'images')
VAL_PATH_LBL = os.path.join(BASE_PATH, 'val', 'labels')

VAL_PERCENT = 0.2 
# --- КОНЕЦ НАСТРОЕК ---

def create_dirs():
    os.makedirs(TRAIN_PATH_IMG, exist_ok=True)
    os.makedirs(TRAIN_PATH_LBL, exist_ok=True)
    os.makedirs(VAL_PATH_IMG, exist_ok=True)
    os.makedirs(VAL_PATH_LBL, exist_ok=True)

def split_dataset():
    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
    # Получаем список всех файлов, кроме 'classes.txt'
    all_labels = [f for f in os.listdir(LABELS_SRC) if f.endswith('.txt') and f != 'classes.txt']
    
    random.shuffle(all_labels)

    num_val = int(len(all_labels) * VAL_PERCENT)
    val_files = all_labels[:num_val]
    train_files = all_labels[num_val:]
    
    # Теперь расчет будет верным (на основе 178 файлов)
    print(f"Всего файлов для обработки: {len(all_labels)}")
    print(f"Для обучения: {len(train_files)}")
    print(f"Для валидации: {len(val_files)}")

    copy_files(train_files, TRAIN_PATH_IMG, TRAIN_PATH_LBL)
    copy_files(val_files, VAL_PATH_IMG, VAL_PATH_LBL)
    
    print("\nКопирование завершено!")

def copy_files(file_list, dest_img_path, dest_lbl_path):
    for lbl_file in file_list:
        img_file = lbl_file.replace('.txt', '.png')
        
        # Проверим на всякий случай, существует ли картинка, прежде чем копировать
        if os.path.exists(os.path.join(IMAGES_SRC, img_file)):
            shutil.copy(os.path.join(LABELS_SRC, lbl_file), dest_lbl_path)
            shutil.copy(os.path.join(IMAGES_SRC, img_file), dest_img_path)
        else:
            print(f"ВНИМАНИЕ: Для метки {lbl_file} не найдено изображение {img_file}. Пропускаем.")


if __name__ == "__main__":
    create_dirs()
    split_dataset()