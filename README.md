
# 🕺 Balinese Dance Detection

Sebuah sistem deteksi gerakan tari Bali berbasis video, yang menggunakan **Convolutional Neural Network (CNN)** untuk mengklasifikasi gerakan dasar penari wanita (wajah, badan, kaki), kemudian **menganalisis pola gerakan** menggunakan algoritma **Suffix Tree**.

---

## 🎯 Tujuan Proyek

- Mengklasifikasi frame video tari Bali ke dalam kelas gerakan dasar (face, full-body, leg) menggunakan CNN (misalnya AlexNet).
- Menganalisis pola urutan gerakan untuk mendeteksi **kemiripan koreografi** antara video tari yang berbeda.
- Mendukung preservasi budaya melalui analisis kuantitatif pola gerak tari Bali.

---

## 🧪 Dataset

- Video ritmik tari Bali, terbagi kategori:
  - **Face**: 2 gerakan dasar
  - **Body**: 11 gerakan dasar
  - **Legs**: 4 gerakan dasar
- Frame diolah menjadi dataset gambar, dilabeli manual (ground truth). Preprocessing termasuk cropping dan binarisasi.

---

## 🧠 Metodologi

1. **Preprocessing**:  
   - Crop wajah (dlib), badan, dan kaki dari video  
   - Convert ke grayscale, thresholding lokal untuk binarisasi

2. **Klasifikasi (CNN)**  
   - Arsitektur: AlexNet/trained CNN  
   - Output: prediksi kelas dari masing-masing part

3. **Pattern Analysis**  
   - Konversi hasil label ke string kode  
   - Gunakan **Suffix Tree** untuk mencari pola subsequence dan menghitung kesamaan koreografi

---

## 🛠️ Instalasi

```bash
git clone https://github.com/NandaFebian/balinese_dance-detection.git
cd balinese_dance-detection
pip install -r requirements.txt
```

**Dependencies** utama:
- `opencv-python`
- `torch` atau `tensorflow` (sesuai implementasi CNN)
- `dlib`
- `numpy`, `suffix_trees`, dll.

---

## ▶️ Usage

```bash
python detect.py \
  --video path/to/video.mp4 \
  --model path/to/cnn_model.pth \
  --output predictions.txt
```

**Pipeline**:
1. Frame video diproses, dicrop ke tiga bagian.
2. Model CNN memprediksi tiap frame.
3. Hasil prediksi disimpan sebagai urutan kode.
4. Suffix Tree menganalisis pola kemunculan subsequence antar video.

---

## 📊 Hasil & Evaluasi

- **Akurasi Tinggi**:  
  - Face: ~95%  
  - Body: ~85%  
  - Legs: ~92%  
- **Suffix Tree** berhasil mengidentifikasi subsequence gerakan yang sering muncul, mendukung analisis kemiripan dan struktur koreografi.

---

## 🧩 Struktur Repo

```
├── data/                 # contoh video & frame dataset
├── models/               # model CNN hasil training
│   └── alexnet_*.pth
├── detect.py             # script utama deteksi & pattern analysis
├── utils.py              # helper functions (preprocessing, suffix tree)
├── requirements.txt
└── README.md             # dokumentasi ini
```

---

## 🔀 Pengembangan Selanjutnya

- Tambahkan **real-time detection** dari webcam
- Tingkatkan akurasi dengan arsitektur CNN modern
- Analisis koreografi multi-penari/durasi panjang
- Integrasi front-end (streamlit, web app) untuk visualisasi

---

## 👥 Tim & Referensi

- Dataset dasar: Rudy Hendrawan et al. – “Video Dataset of Balinese Dance Basic Movement”  
- Paper: I Komang Hendra Trinium Jaya et al. – CNN + Suffix Tree untuk koreografi tari Bali  
- Repo ini oleh: [NandaFebian](https://github.com/NandaFebian)

---

## 📝 Lisensi

Distribusi: _MIT License_  
Gunakan dan modifikasi bebas untuk penelitian dan penyebaran ilmu.
