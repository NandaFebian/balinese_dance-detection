import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# ========================
# Konfigurasi & Parameter
# ========================
LABELS = ['baris', 'pendet', 'margapati']
SAVE_DIR = 'src/model'
os.makedirs(SAVE_DIR, exist_ok=True)

# ========================
# Load Dataset
# ========================
X_train = np.load('data/result_processing/X_train.npy')
y_train = np.load('data/result_processing/y_train.npy')
X_test = np.load('data/result_processing/X_test.npy')
y_test = np.load('data/result_processing/y_test.npy')

print("✅ Dataset dimuat:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# ========================
# Parameter Model
# ========================
sequence_length = X_train.shape[1]  # 50
num_features = X_train.shape[2]     # 99
num_classes = y_train.shape[1]      # 3

# ========================
# Bangun Model LSTM
# ========================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ========================
# Callbacks
# ========================
checkpoint = ModelCheckpoint(
    os.path.join(SAVE_DIR, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ========================
# Training
# ========================
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop]
)
end_time = time.time()

print(f"\n✅ Training selesai dalam {end_time - start_time:.2f} detik")
print(f"📁 Model terbaik disimpan di {SAVE_DIR}/best_model.h5")

# ========================
# Evaluasi Akhir
# ========================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🧪 Evaluasi Test Set:\nLoss = {loss:.4f} | Akurasi = {accuracy:.4f}")

# ========================
# Visualisasi Akurasi & Loss
# ========================
plt.figure(figsize=(12, 5))

# Plot Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Akurasi per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_plot.png'))
plt.show()

# ========================
# Confusion Matrix & Report
# ========================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=LABELS))

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
plt.show()
