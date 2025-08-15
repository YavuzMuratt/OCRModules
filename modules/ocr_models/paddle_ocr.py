import cv2
import numpy as np
import os
import time
from typing import List, Dict, Any
from .base_ocr import BaseOCR

class PaddleOCRModel(BaseOCR):
    """PaddleOCR modeli - Doğrudan import ile çalışır"""
    
    def __init__(self, use_gpu: bool = True, lang: str = 'en'):
        super().__init__("PaddleOCR")
        self.use_gpu = use_gpu
        self.lang = lang
        self.ocr = None
        self.is_initialized = False
        
    def initialize(self, **kwargs):
        """PaddleOCR'ı başlatır"""
        try:
            print("PaddleOCR başlatılıyor...")
            
            # PaddleOCR import et
            from paddleocr import PaddleOCR
            
            # CPU modunda başlat (CUDA DLL sorunu nedeniyle)
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, device="cpu")
            
            # Test için basit bir OCR işlemi yap
            test_image = np.zeros((100, 100), dtype=np.uint8)
            self.ocr.ocr(test_image)
            
            # Model değişkenini set et (BaseOCR.is_model_ready() için)
            self.model = self.ocr
            self.is_initialized = True
            print("✓ PaddleOCR başarıyla başlatıldı (CPU)")
            return True
            
        except Exception as e:
            print(f"✗ PaddleOCR başlatma hatası: {str(e)}")
            print("PaddleOCR kullanılamıyor, diğer modeller kullanılabilir.")
            return False
            
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Görüntüden metin çıkarır"""
        if not self.is_initialized:
            raise RuntimeError(f"{self.model_name} henüz başlatılmamış!")
            
        try:
            # Görüntüyü ön işle
            processed_image = self.preprocess_for_model(image)
            
            # OCR işlemi
            results = self.ocr.ocr(processed_image)
            
            # Sonuçları formatla
            formatted_results = []
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, confidence) = line
                    
                    # Bbox formatını düzenle
                    bbox_array = np.array(bbox)
                    x_coords = bbox_array[:, 0]
                    y_coords = bbox_array[:, 1]
                    
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    result = {
                        'text': text,
                        'confidence': confidence * 100,
                        'bbox': [x1, y1, x2, y2],
                        'bbox_xywh': [x1, y1, x2 - x1, y2 - y1]
                    }
                    formatted_results.append(result)
                    
            return self.postprocess_results(formatted_results)
            
        except Exception as e:
            print(f"PaddleOCR hatası: {str(e)}")
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
                
                # Poligon çiz
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                cv2.polylines(result_image, [pts], isClosed=True, color=(255, 128, 0), thickness=2)
                
                # Metin ve güven bilgisini yaz
                label = f"{text} ({confidence:.1f}%)"
                cv2.putText(result_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
                
        return result_image
        
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """PaddleOCR için özel ön işleme"""
        return image.copy()