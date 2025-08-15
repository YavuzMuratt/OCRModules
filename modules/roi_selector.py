import cv2
import numpy as np
from typing import Tuple, Optional, List

class ROISelector:
    """ROI (Region of Interest) seçimi için sınıf"""
    
    def __init__(self):
        self.roi_coords = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.window_name = "ROI Seçimi"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback fonksiyonu"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
    def select_roi_interactive(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """İnteraktif ROI seçimi"""
        # Görüntüyü kopyala
        display_image = image.copy()
        
        # Pencere oluştur
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("ROI seçimi için:")
        print("- Mouse ile dikdörtgen çizin")
        print("- Enter tuşuna basın (seçimi onaylamak için)")
        print("- ESC tuşuna basın (iptal etmek için)")
        
        while True:
            # Görüntüyü kopyala
            temp_image = display_image.copy()
            
            # Mevcut seçimi çiz
            if self.start_point and self.end_point:
                cv2.rectangle(temp_image, self.start_point, self.end_point, (0, 255, 0), 2)
                
                # Koordinatları göster
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
                cv2.putText(temp_image, f"ROI: ({x}, {y}, {w}, {h})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Görüntüyü göster
            cv2.imshow(self.window_name, temp_image)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter
                if self.start_point and self.end_point:
                    # ROI koordinatlarını hesapla
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
                    
                    # Minimum boyut kontrolü
                    if w > 10 and h > 10:
                        self.roi_coords = (x, y, w, h)
                        cv2.destroyAllWindows()
                        return self.roi_coords
                    else:
                        print("ROI çok küçük! Daha büyük bir alan seçin.")
                else:
                    print("Önce bir ROI seçin!")
                    
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
                
    def select_roi_auto(self, image: np.ndarray, method: str = 'center') -> Optional[Tuple[int, int, int, int]]:
        """Otomatik ROI seçimi"""
        height, width = image.shape[:2]
        
        if method == 'center':
            # Merkez bölgesi
            center_x, center_y = width // 2, height // 2
            roi_size = min(width, height) // 3
            x = center_x - roi_size // 2
            y = center_y - roi_size // 2
            w = roi_size
            h = roi_size
            
        elif method == 'top_half':
            # Üst yarı
            x, y = 0, 0
            w, h = width, height // 2
            
        elif method == 'bottom_half':
            # Alt yarı
            x, y = 0, height // 2
            w, h = width, height // 2
            
        elif method == 'left_half':
            # Sol yarı
            x, y = 0, 0
            w, h = width // 2, height
            
        elif method == 'right_half':
            # Sağ yarı
            x, y = width // 2, 0
            w, h = width // 2, height
            
        else:
            return None
            
        self.roi_coords = (x, y, w, h)
        return self.roi_coords
        
    def apply_roi(self, image: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """ROI'yi görüntüye uygular"""
        x, y, w, h = roi_coords
        return image[y:y+h, x:x+w]
        
    def draw_roi_on_image(self, image: np.ndarray, roi_coords: Tuple[int, int, int, int], 
                         color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """ROI'yi görüntü üzerine çizer"""
        x, y, w, h = roi_coords
        result_image = image.copy()
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
        
        # ROI bilgilerini yaz
        cv2.putText(result_image, f"ROI: ({x}, {y}, {w}, {h})", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_image
        
    def get_roi_coords(self) -> Optional[Tuple[int, int, int, int]]:
        """Mevcut ROI koordinatlarını döndürür"""
        return self.roi_coords
        
    def reset_roi(self):
        """ROI'yi sıfırlar"""
        self.roi_coords = None
        self.start_point = None
        self.end_point = None
        self.drawing = False
