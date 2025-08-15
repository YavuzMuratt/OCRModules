import time
from typing import Dict, Any

class PerformanceTimer:
    """OCR işlemlerinin performansını ölçen sınıf"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.ocr_time = 0.0
        self.preprocessing_time = 0.0
        self.total_time = 0.0
        
    def start(self):
        """Zamanlayıcıyı başlatır"""
        self.start_time = time.time()
        
    def end(self):
        """Zamanlayıcıyı durdurur ve toplam süreyi hesaplar"""
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        
    def start_ocr(self):
        """OCR işlemi için zamanlayıcıyı başlatır"""
        self.ocr_start = time.time()
        
    def end_ocr(self):
        """OCR işlemini durdurur ve süreyi hesaplar"""
        self.ocr_end = time.time()
        self.ocr_time = self.ocr_end - self.ocr_start
        
    def start_preprocessing(self):
        """Ön işleme için zamanlayıcıyı başlatır"""
        self.preprocessing_start = time.time()
        
    def end_preprocessing(self):
        """Ön işlemeyi durdurur ve süreyi hesaplar"""
        self.preprocessing_end = time.time()
        self.preprocessing_time = self.preprocessing_end - self.preprocessing_start
        
    def get_timing_stats(self) -> Dict[str, Any]:
        """Tüm zamanlama istatistiklerini döndürür"""
        return {
            'total_time': round(self.total_time, 3),
            'ocr_time': round(self.ocr_time, 3),
            'preprocessing_time': round(self.preprocessing_time, 3),
            'other_time': round(self.total_time - self.ocr_time - self.preprocessing_time, 3)
        }
        
    def reset(self):
        """Tüm zamanlayıcıları sıfırlar"""
        self.start_time = None
        self.end_time = None
        self.ocr_time = 0.0
        self.preprocessing_time = 0.0
        self.total_time = 0.0
