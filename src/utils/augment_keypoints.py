import os
import numpy as np
from scipy.interpolate import interp1d
import random

SOURCE_DIR = 'data/extracted_keypoints'
TARGET_DIR = 'data/extracted_keypoints_aug'
LABELS = ['baris', 'pendet', 'margapati']
NUM_FEATURES = 99
MAX_FRAMES = 50
AUG_PER_FILE = 3  # Jumlah augmentasi per file

def jitter(data, sigma=0.01):
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + noise

def time_warp(data):
    # Interpolasi ke frame lebih pendek lalu stretch ke MAX_FRAMES lagi
    original_len = data.shape[0]
    new_len = random.randint(int(0.8 * original_len), int(1.2 * original_len))
    new_len = min(max(new_len, 10), 2 * MAX_FRAMES)

    f = interp1d(np.linspace(0, 1, original_len), data, axis=0, kind='linear')
    warped = f(np.linspace(0, 1, new_len))

    # Potong atau padding ke MAX_FRAMES
    if warped.shape[0] > MAX_FRAMES:
        warped = warped[:MAX_FRAMES]
    else:
        padding = np.tile(warped[-1:], (MAX_FRAMES - warped.shape[0], 1))
        warped = np.vstack((warped, padding))
    return warped

def mirror(data):
    # Flip sumbu X (diasumsikan dimensi x berada di setiap 3 indeks pertama dari setiap keypoint)
    reshaped = data.reshape(-1, 33, 3)
    reshaped[:, :, 0] *= -1  # Flip X
    return reshaped.reshape(-1, 99)

def augment_and_save():
    for label in LABELS:
        src_folder = os.path.join(SOURCE_DIR, label)
        tgt_folder = os.path.join(TARGET_DIR, label)
        os.makedirs(tgt_folder, exist_ok=True)

        for fname in os.listdir(src_folder):
            if not fname.endswith('.npy'):
                continue

            data = np.load(os.path.join(src_folder, fname))
            base_name = os.path.splitext(fname)[0]

            # Simpan original (dengan padding jika perlu)
            if data.shape[0] < MAX_FRAMES:
                padding = np.tile(data[-1:], (MAX_FRAMES - data.shape[0], 1))
                data = np.vstack((data, padding))
            elif data.shape[0] > MAX_FRAMES:
                data = data[:MAX_FRAMES]

            np.save(os.path.join(tgt_folder, base_name + '_orig.npy'), data)

            for i in range(AUG_PER_FILE):
                aug_data = data.copy()
                choice = random.choice(['jitter', 'timewarp', 'mirror'])

                if choice == 'jitter':
                    aug_data = jitter(aug_data)
                elif choice == 'timewarp':
                    aug_data = time_warp(aug_data)
                elif choice == 'mirror':
                    aug_data = mirror(aug_data)

                aug_name = f"{base_name}_{choice}_{i}.npy"
                np.save(os.path.join(tgt_folder, aug_name), aug_data)

        print(f"✅ Augmentasi selesai untuk kelas {label}")

if __name__ == "__main__":
    augment_and_save()
