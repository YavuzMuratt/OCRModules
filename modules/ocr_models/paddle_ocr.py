import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Any
from .base_ocr import BaseOCR

class PaddleOCRModel(BaseOCR):
    """PaddleOCR modeli"""
    
    def __init__(self, use_gpu: bool = True, lang: str = 'en'):
        super().__init__("PaddleOCR")
        self.use_gpu = use_gpu
        self.lang = lang
        self.ocr = None
        
    def initialize(self, **kwargs):
        """PaddleOCR modelini başlatır"""
        try:
            # GPU kullanımını ayarla
            use_gpu = kwargs.get('use_gpu', self.use_gpu)
            lang = kwargs.get('lang', self.lang)
            
            # PaddleOCR'ı yeni API ile başlat (arkadaşınızın scriptindeki gibi)
            device = "gpu:0" if use_gpu else "cpu"
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=device,
                lang=lang
            )
            
            # Test için basit bir OCR işlemi yap
            test_image = np.zeros((100, 100), dtype=np.uint8)
            # Yeni API ile test
            try:
                self.ocr.predict(input=test_image)
            except:
                # Eski API ile test
                self.ocr.ocr(test_image)
            
            # Model değişkenini set et (BaseOCR.is_model_ready() için)
            self.model = self.ocr
            self.is_initialized = True
            print(f"✓ {self.model_name} başarıyla başlatıldı (Device: {device}, Lang: {lang})")
            
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
            
            # Önce yeni API'yi dene
            try:
                predict_result = self.ocr.predict(input=processed_image)
                formatted_results = self._parse_predict_result(predict_result)
            except:
                # Eski API'yi dene
                results = self.ocr.ocr(processed_image, cls=True)
                formatted_results = self._parse_legacy_result(results)
                        
            return self.postprocess_results(formatted_results)
            
        except Exception as e:
            print(f"PaddleOCR hatası: {str(e)}")
            return []
            
    def _parse_predict_result(self, predict_result: List) -> List[Dict[str, Any]]:
        """Yeni predict() API sonuçlarını parse eder"""
        formatted_results = []
        
        try:
            for res in predict_result:
                # OCRResult objesinden metin ve koordinatları çıkar
                if hasattr(res, 'text_lines') and isinstance(res.text_lines, (list, tuple)):
                    for line in res.text_lines:
                        if isinstance(line, dict):
                            text = line.get('transcription') or line.get('text') or line.get('label', '')
                            confidence = line.get('confidence', 0.0) * 100
                            bbox = line.get('bbox', [])
                            
                            if text and bbox:
                                # Bbox formatını düzenle
                                bbox_array = np.array(bbox)
                                x_coords = bbox_array[:, 0]
                                y_coords = bbox_array[:, 1]
                                
                                x1, y1 = int(min(x_coords)), int(min(y_coords))
                                x2, y2 = int(max(x_coords)), int(max(y_coords))
                                
                                result = {
                                    'text': str(text),
                                    'confidence': confidence,
                                    'bbox': [x1, y1, x2, y2],
                                    'bbox_xywh': [x1, y1, x2 - x1, y2 - y1]
                                }
                                formatted_results.append(result)
                                
        except Exception as e:
            print(f"Predict result parsing hatası: {str(e)}")
            
        return formatted_results
        
    def _parse_legacy_result(self, results: List) -> List[Dict[str, Any]]:
        """Eski ocr() API sonuçlarını parse eder"""
        formatted_results = []
        
        try:
            if results and results[0]:
                for line in results[0]:
                    if line:
                        bbox, (text, confidence) = line
                        
                        # Bbox formatını düzenle
                        bbox_array = np.array(bbox)
                        x_coords = bbox_array[:, 0]
                        y_coords = bbox_array[:, 1]
                        
                        x1, y1 = int(min(x_coords)), int(min(y_coords))
                        x2, y2 = int(max(x_coords)), int(max(y_coords))
                        
                        result = {
                            'text': text,
                            'confidence': confidence * 100,  # Yüzdeye çevir
                            'bbox': [x1, y1, x2, y2],
                            'bbox_xywh': [x1, y1, x2 - x1, y2 - y1]
                        }
                        formatted_results.append(result)
                        
        except Exception as e:
            print(f"Legacy result parsing hatası: {str(e)}")
            
        return formatted_results
            
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
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Metin ve güven bilgisini yaz
                label = f"{text} ({confidence:.1f}%)"
                cv2.putText(result_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
        return result_image
        
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """PaddleOCR için özel ön işleme"""
        # PaddleOCR genellikle ham görüntülerle iyi çalışır
        # Sadece temel iyileştirmeler yap
        if len(image.shape) == 3:
            # BGR'den RGB'ye çevir
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return rgb_image
        else:
            # Gri tonlamayı RGB'ye çevir
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return rgb_image
