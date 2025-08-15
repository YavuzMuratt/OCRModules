from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Optional

class BaseOCR(ABC):
    """OCR modelleri için temel sınıf"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self, **kwargs):
        """Modeli başlatır"""
        pass
        
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Görüntüden metin çıkarır"""
        pass
        
    @abstractmethod
    def draw_results(self, image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Sonuçları görüntü üzerine çizer"""
        pass
        
    def is_model_ready(self) -> bool:
        """Modelin hazır olup olmadığını kontrol eder"""
        return self.is_initialized and self.model is not None
        
    def get_model_name(self) -> str:
        """Model adını döndürür"""
        return self.model_name
        
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """Model için görüntü ön işleme (override edilebilir)"""
        return image.copy()
        
    def postprocess_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sonuçları son işleme (override edilebilir)"""
        return results
