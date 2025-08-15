# OCR Mods - Kurulum Talimatları

Bu doküman, OCR Mods sisteminin kurulumu için detaylı talimatları içerir.

## Sistem Gereksinimleri

### Donanım
- **RAM**: En az 4GB (8GB önerilen)
- **GPU**: NVIDIA GPU (CUDA desteği ile) - Opsiyonel ama önerilen
- **Disk Alanı**: En az 2GB boş alan

### Yazılım
- **Python**: 3.8 veya üzeri
- **CUDA**: 11.0 veya üzeri (GPU kullanımı için)
- **Tesseract OCR**: Sistem seviyesinde kurulum gerekli

## Adım Adım Kurulum

### 1. Python Kurulumu

#### Windows
1. [Python.org](https://www.python.org/downloads/) adresinden Python 3.8+ indirin
2. Kurulum sırasında "Add Python to PATH" seçeneğini işaretleyin
3. Kurulumu tamamlayın

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### macOS
```bash
# Homebrew ile
brew install python3

# veya Python.org'dan indirin
```

### 2. Tesseract OCR Kurulumu

#### Windows
1. [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) adresinden indirin
2. Kurulum sırasında "Add to PATH" seçeneğini işaretleyin
3. Kurulum yolunu not edin (örn: `C:\Program Files\Tesseract-OCR\tesseract.exe`)

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-tur
```

#### macOS
```bash
brew install tesseract
```

### 3. CUDA Kurulumu (GPU Kullanımı İçin)

#### Windows
1. [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) indirin
2. Kurulumu tamamlayın
3. Sistem PATH'ine CUDA yolunu ekleyin

#### Linux
```bash
# Ubuntu/Debian için
sudo apt install nvidia-cuda-toolkit
```

### 4. Proje Kurulumu

1. **Projeyi klonlayın veya indirin**
```bash
git clone <repository-url>
cd OCRMods
```

2. **Sanal ortam oluşturun**
```bash
# Windows
python -m venv ocr_env
ocr_env\Scripts\activate

# Linux/macOS
python3 -m venv ocr_env
source ocr_env/bin/activate
```

3. **Gerekli paketleri yükleyin**
```bash
pip install -r requirements.txt
```

### 5. Tesseract Yolunu Ayarlama

Windows'ta Tesseract yolunu ayarlamak için:

```python
# main.py dosyasında TesseractOCR sınıfını başlatırken:
tesseract = TesseractOCR(tesseract_path="C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
```

## Test ve Doğrulama

### 1. Sistem Testi
```bash
python test_ocr.py
```

### 2. Ana Program Testi
```bash
python main.py
```

## Sorun Giderme

### Yaygın Hatalar

#### 1. Tesseract Bulunamadı
**Hata**: `tesseract is not installed or it's not in your PATH`

**Çözüm**:
- Tesseract'ın kurulu olduğundan emin olun
- PATH'e eklendiğinden emin olun
- Windows'ta tam yolu belirtin

#### 2. CUDA Hatası
**Hata**: `CUDA not available`

**Çözüm**:
- CUDA Toolkit'in kurulu olduğundan emin olun
- GPU sürücülerinin güncel olduğundan emin olun
- CPU modunda çalıştırmayı deneyin

#### 3. Bellek Hatası
**Hata**: `Out of memory`

**Çözüm**:
- Daha küçük görüntüler kullanın
- Batch size'ı azaltın
- GPU belleğini kontrol edin

#### 4. Import Hatası
**Hata**: `ModuleNotFoundError`

**Çözüm**:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### GPU Kullanımı

GPU kullanımını kontrol etmek için:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Performans Optimizasyonu

### 1. GPU Kullanımı
- NVIDIA GPU'nuz varsa CUDA kurulumunu yapın
- EasyOCR ve PaddleOCR GPU modunda çalışacaktır

### 2. Bellek Optimizasyonu
- Büyük görüntüleri işlemeden önce yeniden boyutlandırın
- Batch processing kullanın

### 3. Model Seçimi
- **Tesseract**: Hızlı, CPU'da iyi çalışır
- **EasyOCR**: Çok dilli, GPU'da hızlı
- **PaddleOCR**: Yüksek doğruluk, GPU'da çok hızlı

## Destek

Sorun yaşarsanız:
1. Test script'ini çalıştırın: `python test_ocr.py`
2. Log dosyalarını kontrol edin
3. GitHub Issues'da sorun bildirin

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.
