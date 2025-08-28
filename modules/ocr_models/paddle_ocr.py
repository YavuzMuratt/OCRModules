import cv2
import numpy as np
import os
import time
from typing import List, Dict, Any
from .base_ocr import BaseOCR

def check_gpu_availability():
    """GPU kullanılabilirliğini kontrol eder"""
    try:
        import paddle
        # PaddlePaddle GPU desteğini kontrol et
        if paddle.is_compiled_with_cuda():
            print("✓ PaddlePaddle CUDA desteği mevcut")
            # GPU sayısını kontrol et
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count > 0:
                print(f"✓ {gpu_count} GPU bulundu")
                return True, gpu_count
            else:
                print("⚠ CUDA desteği var ama GPU bulunamadı")
                return False, 0
        else:
            print("✗ PaddlePaddle CUDA desteği yok")
            return False, 0
    except Exception as e:
        print(f"⚠ GPU kontrol hatası: {e}")
        return False, 0

class PaddleOCRModel(BaseOCR):
    """PaddleOCR modeli - GPU hızlandırma desteği ile"""
    
    def __init__(self, use_gpu: bool = True, lang: str = 'en'):
        super().__init__("PaddleOCR")
        self.use_gpu = use_gpu
        self.lang = lang
        self.ocr = None
        self.is_initialized = False
        self.gpu_available = False
        self.gpu_count = 0
        
    def initialize(self, **kwargs):
        """PaddleOCR'ı başlatır"""
        try:
            print("PaddleOCR başlatılıyor...")
            
            # GPU kullanılabilirliğini kontrol et
            if self.use_gpu:
                self.gpu_available, self.gpu_count = check_gpu_availability()
            
            # PaddleOCR import et
            from paddleocr import PaddleOCR
            
            # GPU kullanılabilirse GPU ile başlat
            if self.gpu_available and self.use_gpu:
                print(f"GPU modunda başlatılıyor (GPU:0)...")
                self.ocr = PaddleOCR(
                    use_angle_cls=True, 
                    lang=self.lang, 
                    device="gpu:0",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False
                )
            else:
                print("CPU modunda başlatılıyor...")
                self.ocr = PaddleOCR(
                    use_angle_cls=True, 
                    lang=self.lang, 
                    device="cpu",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False
                )
            
            # Test için basit bir OCR işlemi yap
            test_image = np.zeros((100, 100), dtype=np.uint8)
            self.ocr.ocr(test_image)
            
            # Model değişkenini set et (BaseOCR.is_model_ready() için)
            self.model = self.ocr
            self.is_initialized = True
            
            device_info = "GPU" if (self.gpu_available and self.use_gpu) else "CPU"
            print(f"✓ PaddleOCR başarıyla başlatıldı ({device_info})")
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
            
            # OCR işlemi - PaddleOCR 3.x predict() API'sini kullan
            try:
                # Yeni predict() API'sini dene
                results = self.ocr.predict(input=processed_image)
                return self._process_predict_results(results)
            except AttributeError:
                # Eski ocr() API'sini kullan
                results = self.ocr.ocr(processed_image)
                return self._process_legacy_results(results)
            
        except Exception as e:
            print(f"PaddleOCR hatası: {str(e)}")
            return []
    
    def _process_predict_results(self, results) -> List[Dict[str, Any]]:
        """PaddleOCR 3.x predict() sonuçlarını işler"""
        formatted_results = []
        
        try:
            # Her OCR sonucunu işle
            for res in results:
                try:
                    # OCRResult objesinden metinleri çıkar
                    if hasattr(res, "rec_texts") and hasattr(res, "rec_scores") and hasattr(res, "rec_boxes"):
                        texts = res.rec_texts
                        scores = res.rec_scores
                        boxes = res.rec_boxes
                        
                        for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
                            if text and text.strip():
                                result = {
                                    'text': text.strip(),
                                    'confidence': float(score) * 100,
                                    'bbox': box,
                                    'bbox_xywh': [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                                }
                                formatted_results.append(result)
                    
                    # Alternatif: to_dict() metodunu kullan
                    elif hasattr(res, "to_dict"):
                        data = res.to_dict()
                        if "rec_texts" in data and "rec_scores" in data and "rec_boxes" in data:
                            texts = data["rec_texts"]
                            scores = data["rec_scores"]
                            boxes = data["rec_boxes"]
                            
                            for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
                                if text and text.strip():
                                    result = {
                                        'text': text.strip(),
                                        'confidence': float(score) * 100,
                                        'bbox': box,
                                        'bbox_xywh': [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                                    }
                                    formatted_results.append(result)
                                    
                except Exception as res_error:
                    print(f"Sonuç işleme hatası: {res_error}")
                    continue
                    
        except Exception as e:
            print(f"Predict sonuç işleme hatası: {e}")
            
        return self.postprocess_results(formatted_results)
    
    def _process_legacy_results(self, results) -> List[Dict[str, Any]]:
        """Eski PaddleOCR ocr() API sonuçlarını işler"""
        formatted_results = []
        
        try:
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
                    
        except Exception as e:
            print(f"Legacy sonuç işleme hatası: {e}")
            
        return self.postprocess_results(formatted_results)
        
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
    
    def get_device_info(self) -> str:
        """Kullanılan cihaz bilgisini döndürür"""
        if self.gpu_available and self.use_gpu:
            return f"GPU:0 (CUDA)"
        else:
            return "CPU"