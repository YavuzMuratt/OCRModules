import logging
import os
from datetime import datetime
from typing import Dict, Any, List

class OCRLogger:
    """OCR işlemleri için loglama sınıfı"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.logger = None
        self.log_file = None
        
    def setup_logger(self, model_name: str, use_preprocessing: bool = False, use_roi: bool = False):
        """Logger'ı kurar ve log dosyasını oluşturur"""
        # Sonuç klasörü adını oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{model_name}"
        
        if use_preprocessing:
            folder_name += "_preprocessed"
        if use_roi:
            folder_name += "_roi"
            
        # Sonuç klasörünü oluştur
        result_folder = os.path.join(self.results_dir, folder_name)
        os.makedirs(result_folder, exist_ok=True)
        
        # Log dosyası yolu
        self.log_file = os.path.join(result_folder, "ocr_results.log")
        
        # Logger'ı yapılandır
        self.logger = logging.getLogger(f"OCR_{model_name}")
        self.logger.setLevel(logging.INFO)
        
        # Dosya handler'ı
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler'ı
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Handler'ları ekle
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        return result_folder
        
    def log_image_info(self, image_path: str, image_size: tuple):
        """Görüntü bilgilerini loglar"""
        if self.logger:
            self.logger.info(f"İşlenen görüntü: {image_path}")
            self.logger.info(f"Görüntü boyutu: {image_size[0]}x{image_size[1]}")
            
    def log_preprocessing_info(self, preprocessing_steps: List[str]):
        """Ön işleme bilgilerini loglar"""
        if self.logger and preprocessing_steps:
            self.logger.info("Uygulanan ön işleme adımları:")
            for step in preprocessing_steps:
                self.logger.info(f"  - {step}")
                
    def log_roi_info(self, roi_coords: tuple):
        """ROI bilgilerini loglar"""
        if self.logger and roi_coords:
            x, y, w, h = roi_coords
            self.logger.info(f"ROI koordinatları: x={x}, y={y}, width={w}, height={h}")
            
    def log_ocr_results(self, results: List[Dict[str, Any]], timing_stats: Dict[str, Any]):
        """OCR sonuçlarını loglar"""
        if self.logger:
            self.logger.info("=" * 50)
            self.logger.info("OCR SONUÇLARI")
            self.logger.info("=" * 50)
            
            # Zamanlama bilgileri
            self.logger.info("PERFORMANS İSTATİSTİKLERİ:")
            for key, value in timing_stats.items():
                self.logger.info(f"  {key}: {value} saniye")
                
            # OCR sonuçları
            self.logger.info(f"\nBULUNAN METİN SAYISI: {len(results)}")
            self.logger.info("\nMETİN SONUÇLARI:")
            
            for i, result in enumerate(results, 1):
                text = result.get('text', '')
                confidence = result.get('confidence', 0)
                bbox = result.get('bbox', [])
                
                self.logger.info(f"\n{i}. Metin:")
                self.logger.info(f"  İçerik: {text}")
                self.logger.info(f"  Güven: {confidence:.2f}%")
                self.logger.info(f"  Bounding Box: {bbox}")
                
    def log_error(self, error_message: str):
        """Hata mesajlarını loglar"""
        if self.logger:
            self.logger.error(f"HATA: {error_message}")
            
    def log_warning(self, warning_message: str):
        """Uyarı mesajlarını loglar"""
        if self.logger:
            self.logger.warning(f"UYARI: {warning_message}")
            
    def log_summary(self, total_images: int, successful_images: int):
        """İşlem özetini loglar"""
        if self.logger:
            self.logger.info("=" * 50)
            self.logger.info("İŞLEM ÖZETİ")
            self.logger.info("=" * 50)
            self.logger.info(f"Toplam görüntü: {total_images}")
            self.logger.info(f"Başarılı işlem: {successful_images}")
            self.logger.info(f"Başarı oranı: {(successful_images/total_images)*100:.1f}%")
            
    def close(self):
        """Logger'ı kapatır"""
        if self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
