# Edge Detection for Face Image Segmentation

 # Jawaban

```python
# Install dan import library
!pip install opencv-python-headless
from google.colab import files
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Fungsi untuk menampilkan dua gambar
def show_two_images(img1, title1, img2, title2, cmap='gray'):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray' if len(img1.shape)==2 else None)
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap=cmap)
    plt.title(title2)
    plt.axis('off')
    plt.show()

# Upload gambar
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Baca gambar asli
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("Wajah tidak terdeteksi.")
else:
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        break

    # Preprocessing dan segmentasi (edge detection only)
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_face, (200, 200))
    blurred = cv2.medianBlur(resized, 5)
    edges = cv2.Canny(blurred, 100, 200)

    # Tampilkan hanya wajah asli dan hasil deteksi tepi
    show_two_images(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB), 'Foto Asli (ROI Wajah)',
                    edges, 'Hasil Deteksi Tepi')
```
**Output :** <br>
![thanos](https://github.com/user-attachments/assets/4a8f4f18-f291-4068-9339-ec7590e39e22)
