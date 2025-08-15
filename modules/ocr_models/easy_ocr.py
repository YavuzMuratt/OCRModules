import cv2
import numpy as np
import easyocr
from typing import List, Dict, Any
from .base_ocr import BaseOCR

class EasyOCRModel(BaseOCR):
    """EasyOCR modeli"""
    
    def __init__(self, languages: List[str] = None, gpu: bool = True):
        super().__init__("EasyOCR")
        self.languages = languages or ['en']
        self.gpu = gpu
        self.reader = None
        
    def initialize(self, **kwargs):
        """EasyOCR modelini başlatır"""
        try:
            # GPU kullanımını ayarla
            gpu = kwargs.get('gpu', self.gpu)
            
            # Reader'ı başlat
            self.reader = easyocr.Reader(self.languages, gpu=gpu)
            
            # Test için basit bir OCR işlemi yap
            test_image = np.zeros((100, 100), dtype=np.uint8)
            self.reader.readtext(test_image)
            
            # Model değişkenini set et (BaseOCR.is_model_ready() için)
            self.model = self.reader
            self.is_initialized = True
            print(f"✓ {self.model_name} başarıyla başlatıldı (GPU: {gpu})")
            
        except Exception as e:
            print(f"✗ {self.model_name} başlatılamadı: {str(e)}")
            self.is_initialized = False
            
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Görüntüden metin çıkarır"""
        if not self.is_model_ready():
            raise RuntimeError(f"{self.model_name} henüz başlatılmamış!")
            
        try:
            # Görüntüyü ön işle
            processed_image = self.preprocess_for_model(image)
            
            # OCR işlemi
            results = self.reader.readtext(processed_image)
            
            # Sonuçları formatla
            formatted_results = []
            for (bbox, text, confidence) in results:
                # Bbox formatını düzenle
                bbox_array = np.array(bbox)
                x1, y1 = bbox_array[0]
                x2, y2 = bbox_array[2]
                
                result = {
                    'text': text,
                    'confidence': confidence * 100,  # Yüzdeye çevir
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'bbox_xywh': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                }
                formatted_results.append(result)
                
            return self.postprocess_results(formatted_results)
            
        except Exception as e:
            print(f"EasyOCR hatası: {str(e)}")
            return []
            
    def draw_results(self, image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Sonuçları görüntü üzerine çizer"""
        result_image = image.copy()
        
        for result in results:
            bbox = result.get('bbox', [])
            text = result.get('text', '')
            confidence = result.get('confidence', 0)
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                
                # Bounding box çiz
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Metin ve güven bilgisini yaz
                label = f"{text} ({confidence:.1f}%)"
                cv2.putText(result_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
        return result_image
        
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """EasyOCR için özel ön işleme"""
        # EasyOCR genellikle ham görüntülerle iyi çalışır
        # Sadece temel iyileştirmeler yap
        if len(image.shape) == 3:
            # BGR'den RGB'ye çevir
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return rgb_image
        else:
            # Gri tonlamayı RGB'ye çevir
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return rgb_image
