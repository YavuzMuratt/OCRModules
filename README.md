# OCR Mods - Modüler OCR Sistemi

Bu proje, farklı OCR modellerini (TesseractOCR, EasyOCR, PaddleOCR) kullanarak görüntü işleme yapabilen modüler bir sistemdir.

## Özellikler

- **3 Farklı OCR Modeli**: TesseractOCR, EasyOCR, PaddleOCR 3.x
- **GPU Desteği**: Tüm modeller GPU üzerinde çalışır
- **Hız Ölçümü**: OCR işlem sürelerini ölçer
- **Görüntü Ön İşleme**: Opsiyonel görüntü iyileştirme
- **ROI Seçimi**: İnteraktif ROI (Region of Interest) seçimi
- **Otomatik Sonuç Kaydetme**: Tarih ve model bilgisi ile organize edilmiş sonuçlar
- **Loglama**: Detaylı işlem logları
- **Bounding Box Görselleştirme**: OCR sonuçlarını görsel olarak işaretler

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Tesseract OCR'ı sisteminize kurun:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

## Kullanım

Ana script'i çalıştırın:
```bash
python main.py
```

## Proje Yapısı

```
OCRMods/
├── main.py                 # Ana script
├── requirements.txt        # Gerekli paketler
├── images/                 # İşlenecek görseller
├── modules/
│   ├── __init__.py
│   ├── ocr_models/        # OCR modelleri
│   ├── preprocessor.py    # Görüntü ön işleme
│   ├── roi_selector.py    # ROI seçimi
│   ├── timer.py           # Hız ölçümü
│   └── logger.py          # Loglama
└── results/               # Sonuçlar (otomatik oluşturulur)
```

## Modüller

### OCR Modelleri
- **TesseractOCR**: Klasik ve güvenilir OCR
- **EasyOCR**: Çok dilli destek
- **PaddleOCR**: Yüksek doğruluk

### Yardımcı Modüller
- **Preprocessor**: Görüntü iyileştirme
- **ROI Selector**: Bölge seçimi
- **Timer**: Performans ölçümü
- **Logger**: Detaylı loglama
