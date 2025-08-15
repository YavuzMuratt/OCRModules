import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any
from .base_ocr import BaseOCR

class TesseractOCR(BaseOCR):
    """Tesseract OCR modeli"""
    
    def __init__(self, tesseract_path: str = None):
        super().__init__("TesseractOCR")
        self.tesseract_path = tesseract_path
        
    def initialize(self, **kwargs):
        """Tesseract modelini başlatır"""
        try:
            # Tesseract yolunu ayarla
            if self.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                
            # Test için basit bir OCR işlemi yap
            test_image = np.zeros((100, 100), dtype=np.uint8)
            pytesseract.image_to_string(test_image)
            
            # Model değişkenini set et (BaseOCR.is_model_ready() için)
            self.model = pytesseract
            self.is_initialized = True
            print(f"✓ {self.model_name} başarıyla başlatıldı")
            
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
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            # Sonuçları formatla
            results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i])
                
                # Boş metinleri ve düşük güvenilirlikli sonuçları filtrele
                if text and confidence > 0:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    result = {
                        'text': text,
                        'confidence': confidence,
                        'bbox': [x, y, x + w, y + h],
                        'bbox_xywh': [x, y, w, h]
                    }
                    results.append(result)
                    
            return self.postprocess_results(results)
            
        except Exception as e:
            print(f"Tesseract OCR hatası: {str(e)}")
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
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Metin ve güven bilgisini yaz
                label = f"{text} ({confidence:.1f}%)"
                cv2.putText(result_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        return result_image
        
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """Tesseract için özel ön işleme"""
        # Gri tonlama
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Gürültü azaltma
        denoised = cv2.medianBlur(gray, 3)
        
        # Kontrast artırma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
