import cv2
import numpy as np
from typing import List, Tuple, Optional

class ImagePreprocessor:
    """Görüntü ön işleme sınıfı"""
    
    def __init__(self):
        self.applied_steps = []
        
    def reset_steps(self):
        """Uygulanan adımları sıfırlar"""
        self.applied_steps = []
        
    def get_applied_steps(self) -> List[str]:
        """Uygulanan adımları döndürür"""
        return self.applied_steps.copy()
        
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü gri tonlamaya çevirir"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.applied_steps.append("Gri tonlama dönüşümü")
            return gray
        return image
        
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """Gaussian blur uygular"""
        blurred = cv2.GaussianBlur(image, kernel_size, 0)
        self.applied_steps.append(f"Gaussian blur (kernel: {kernel_size})")
        return blurred
        
    def apply_median_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Median blur uygular"""
        blurred = cv2.medianBlur(image, kernel_size)
        self.applied_steps.append(f"Median blur (kernel: {kernel_size})")
        return blurred
        
    def apply_bilateral_filter(self, image: np.ndarray, d: int = 15, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
        """Bilateral filter uygular (kenar koruyucu)"""
        filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        self.applied_steps.append(f"Bilateral filter (d={d}, sigma_color={sigma_color}, sigma_space={sigma_space})")
        return filtered
        
    def apply_adaptive_threshold(self, image: np.ndarray, max_value: int = 255, 
                                adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                threshold_type: int = cv2.THRESH_BINARY, 
                                block_size: int = 11, c: int = 2) -> np.ndarray:
        """Adaptive threshold uygular"""
        thresholded = cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, c)
        self.applied_steps.append(f"Adaptive threshold (block_size={block_size}, c={c})")
        return thresholded
        
    def apply_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """Otsu threshold uygular"""
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.applied_steps.append("Otsu threshold")
        return thresholded
        
    def apply_morphology(self, image: np.ndarray, operation: str = 'open', kernel_size: int = 3) -> np.ndarray:
        """Morphological işlemler uygular"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if operation == 'open':
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            self.applied_steps.append(f"Morphological opening (kernel: {kernel_size}x{kernel_size})")
        elif operation == 'close':
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            self.applied_steps.append(f"Morphological closing (kernel: {kernel_size}x{kernel_size})")
        elif operation == 'dilate':
            result = cv2.dilate(image, kernel, iterations=1)
            self.applied_steps.append(f"Dilation (kernel: {kernel_size}x{kernel_size})")
        elif operation == 'erode':
            result = cv2.erode(image, kernel, iterations=1)
            self.applied_steps.append(f"Erosion (kernel: {kernel_size}x{kernel_size})")
        else:
            return image
            
        return result
        
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """Kontrast artırır"""
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        self.applied_steps.append(f"Kontrast artırma (alpha={alpha}, beta={beta})")
        return enhanced
        
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """CLAHE (Contrast Limited Adaptive Histogram Equalization) uygular"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        self.applied_steps.append(f"CLAHE (clip_limit={clip_limit}, tile_grid={tile_grid_size})")
        return enhanced
        
    def remove_noise(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """Gürültü azaltma"""
        if method == 'gaussian':
            return self.apply_gaussian_blur(image)
        elif method == 'median':
            return self.apply_median_blur(image)
        elif method == 'bilateral':
            return self.apply_bilateral_filter(image)
        else:
            return image
            
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü keskinleştirir"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        self.applied_steps.append("Keskinleştirme")
        return sharpened
        
    def auto_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Otomatik ön işleme pipeline'ı"""
        self.reset_steps()
        
        # Gri tonlama
        if len(image.shape) == 3:
            image = self.convert_to_grayscale(image)
            
        # Gürültü azaltma
        image = self.remove_noise(image, method='bilateral')
        
        # Kontrast artırma
        image = self.apply_clahe(image)
        
        # Keskinleştirme
        image = self.sharpen_image(image)
        
        return image
        
    def preprocess_for_ocr(self, image: np.ndarray, steps: List[str] = None) -> np.ndarray:
        """OCR için özel ön işleme"""
        if steps is None:
            steps = ['grayscale', 'noise_reduction', 'contrast_enhancement']
            
        self.reset_steps()
        processed_image = image.copy()
        
        for step in steps:
            if step == 'grayscale':
                processed_image = self.convert_to_grayscale(processed_image)
            elif step == 'noise_reduction':
                processed_image = self.remove_noise(processed_image, method='bilateral')
            elif step == 'contrast_enhancement':
                processed_image = self.apply_clahe(processed_image)
            elif step == 'threshold':
                processed_image = self.apply_otsu_threshold(processed_image)
            elif step == 'morphology':
                processed_image = self.apply_morphology(processed_image, operation='open')
                
        return processed_image
