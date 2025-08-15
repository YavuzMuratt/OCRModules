#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Mods - Modüler OCR Sistemi
Ana script
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Modülleri import et
from modules.timer import PerformanceTimer
from modules.logger import OCRLogger
from modules.preprocessor import ImagePreprocessor
from modules.roi_selector import ROISelector
from modules.ocr_models.tesseract_ocr import TesseractOCR
from modules.ocr_models.easy_ocr import EasyOCRModel

class OCRMods:
    """Ana OCR sistemi sınıfı"""
    
    def __init__(self):
        self.timer = PerformanceTimer()
        self.preprocessor = ImagePreprocessor()
        self.roi_selector = ROISelector()
        self.logger = None
        
        # OCR modelleri
        self.ocr_models = {
            'tesseract': TesseractOCR(),
            'easyocr': EasyOCRModel(),
            'paddleocr': 'paddleocr_runner'  # Özel script ile çalışır
        }
        
        # Sonuçlar klasörü
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def initialize_models(self, selected_model: str = None):
        """OCR modellerini başlatır"""
        print("OCR modelleri başlatılıyor...")
        
        if selected_model and selected_model in self.ocr_models:
            # Sadece seçilen modeli başlat
            model = self.ocr_models[selected_model]
            model.initialize()
        else:
            # Tüm modelleri başlat
            for name, model in self.ocr_models.items():
                print(f"\n{name.upper()} başlatılıyor...")
                model.initialize()
                
    def get_available_models(self) -> List[str]:
        """Kullanılabilir modelleri döndürür"""
        return list(self.ocr_models.keys())
        
    def select_model(self) -> str:
        """Kullanıcıdan model seçimi alır"""
        print("\n" + "="*50)
        print("OCR MODEL SEÇİMİ")
        print("="*50)
        
        available_models = self.get_available_models()
        
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model.upper()}")
            
        while True:
            try:
                choice = input(f"\nModel seçin (1-{len(available_models)}): ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(available_models):
                    selected_model = available_models[choice_idx]
                    print(f"Seçilen model: {selected_model.upper()}")
                    
                    # PaddleOCR seçildiğinde özel script'i çalıştır
                    if selected_model == 'paddleocr':
                        print("\nPaddleOCR seçildi! Özel PaddleOCR script'i başlatılıyor...")
                        self.run_paddleocr_script()
                        return None  # Ana döngüden çık
                    
                    return selected_model
                else:
                    print("Geçersiz seçim! Tekrar deneyin.")
            except ValueError:
                print("Lütfen bir sayı girin!")
                
    def run_paddleocr_script(self):
        """PaddleOCR özel script'ini çalıştırır"""
        try:
            import subprocess
            script_path = "paddleocr_runner.py"
            
            if not os.path.exists(script_path):
                print(f"✗ {script_path} bulunamadı!")
                return
                
            print("PaddleOCR Runner başlatılıyor...")
            print("=" * 50)
            
            # Script'i çalıştır
            result = subprocess.run([sys.executable, script_path])
            
            if result.returncode == 0:
                print("\nPaddleOCR Runner başarıyla tamamlandı!")
            else:
                print(f"\nPaddleOCR Runner hatası: {result.returncode}")
                
        except Exception as e:
            print(f"PaddleOCR script çalıştırma hatası: {str(e)}")
                
    def ask_preprocessing(self) -> bool:
        """Kullanıcıya ön işleme sorar"""
        print("\n" + "="*50)
        print("GÖRÜNTÜ ÖN İŞLEME")
        print("="*50)
        
        while True:
            choice = input("Görüntü ön işleme uygulansın mı? (e/h): ").strip().lower()
            if choice in ['e', 'evet', 'y', 'yes']:
                return True
            elif choice in ['h', 'hayır', 'n', 'no']:
                return False
            else:
                print("Lütfen 'e' veya 'h' girin!")
                
    def ask_roi_selection(self) -> bool:
        """Kullanıcıya ROI seçimi sorar"""
        print("\n" + "="*50)
        print("ROI SEÇİMİ")
        print("="*50)
        
        while True:
            choice = input("ROI (Region of Interest) seçilsin mi? (e/h): ").strip().lower()
            if choice in ['e', 'evet', 'y', 'yes']:
                return True
            elif choice in ['h', 'hayır', 'n', 'no']:
                return False
            else:
                print("Lütfen 'e' veya 'h' girin!")
                
    def select_roi_method(self) -> str:
        """ROI seçim yöntemini belirler"""
        print("\nROI seçim yöntemi:")
        print("1. İnteraktif seçim (mouse ile)")
        print("2. Otomatik seçim (merkez)")
        
        while True:
            choice = input("Yöntem seçin (1-2): ").strip()
            if choice == '1':
                return 'interactive'
            elif choice == '2':
                return 'auto'
            else:
                print("Geçersiz seçim! Tekrar deneyin.")
                
    def load_images(self, images_dir: str = "images") -> List[str]:
        """Görüntü dosyalarını yükler"""
        if not os.path.exists(images_dir):
            print(f"Hata: {images_dir} klasörü bulunamadı!")
            return []
            
        # Desteklenen görüntü formatları
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        image_files = []
        for file in os.listdir(images_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(images_dir, file))
                
        if not image_files:
            print(f"Hata: {images_dir} klasöründe görüntü dosyası bulunamadı!")
            return []
            
        print(f"{len(image_files)} görüntü dosyası bulundu.")
        return sorted(image_files)
        
    def process_single_image(self, image_path: str, selected_model: str, 
                           use_preprocessing: bool, use_roi: bool, roi_method: str = None) -> Dict[str, Any]:
        """Tek bir görüntüyü işler"""
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Görüntü yüklenemedi: {image_path}")
                
            original_image = image.copy()
            image_size = image.shape[:2]
            
            # Logger'ı ayarla
            if self.logger:
                self.logger.log_image_info(image_path, image_size)
                
            # Timer'ı başlat
            self.timer.start()
            
            # ROI seçimi
            roi_coords = None
            if use_roi:
                self.timer.start_preprocessing()
                
                if roi_method == 'interactive':
                    roi_coords = self.roi_selector.select_roi_interactive(image)
                else:
                    roi_coords = self.roi_selector.select_roi_auto(image)
                    
                if roi_coords:
                    image = self.roi_selector.apply_roi(image, roi_coords)
                    if self.logger:
                        self.logger.log_roi_info(roi_coords)
                        
                self.timer.end_preprocessing()
                
            # Görüntü ön işleme
            if use_preprocessing:
                self.timer.start_preprocessing()
                image = self.preprocessor.preprocess_for_ocr(image)
                preprocessing_steps = self.preprocessor.get_applied_steps()
                if self.logger:
                    self.logger.log_preprocessing_info(preprocessing_steps)
                self.timer.end_preprocessing()
                
            # OCR işlemi
            self.timer.start_ocr()
            ocr_model = self.ocr_models[selected_model]
            results = ocr_model.extract_text(image)
            self.timer.end_ocr()
            
            # Timer'ı durdur
            self.timer.end()
            
            # ROI koordinatlarını düzelt
            if roi_coords:
                roi_x, roi_y, roi_w, roi_h = roi_coords
                # OCR sonuçlarındaki koordinatları ROI offset'ine göre düzelt
                for result in results:
                    if 'bbox' in result:
                        # Bounding box koordinatlarını ROI offset'ine göre düzelt
                        x1, y1, x2, y2 = result['bbox']
                        result['bbox'] = [x1 + roi_x, y1 + roi_y, x2 + roi_x, y2 + roi_y]
                        
                    if 'bbox_xywh' in result:
                        # Bbox_xywh formatını da düzelt
                        x, y, w, h = result['bbox_xywh']
                        result['bbox_xywh'] = [x + roi_x, y + roi_y, w, h]
            
            # Sonuçları görselleştir
            if roi_coords:
                # ROI'yi orijinal görüntü üzerine çiz
                original_image = self.roi_selector.draw_roi_on_image(original_image, roi_coords)
                
            # OCR sonuçlarını orijinal görüntü üzerine çiz
            result_image = ocr_model.draw_results(original_image, results)
            
            # Zamanlama istatistikleri
            timing_stats = self.timer.get_timing_stats()
            
            # Logger'a sonuçları yaz
            if self.logger:
                self.logger.log_ocr_results(results, timing_stats)
                
            return {
                'success': True,
                'image_path': image_path,
                'results': results,
                'result_image': result_image,
                'timing_stats': timing_stats,
                'roi_coords': roi_coords
            }
            
        except Exception as e:
            error_msg = f"Görüntü işleme hatası ({image_path}): {str(e)}"
            print(error_msg)
            if self.logger:
                self.logger.log_error(error_msg)
                
            return {
                'success': False,
                'image_path': image_path,
                'error': str(e)
            }
            
    def save_results(self, result_folder: str, image_path: str, result_data: Dict[str, Any]):
        """Sonuçları kaydeder"""
        try:
            # Görüntü adını al
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Sonuç görüntüsünü kaydet
            if result_data.get('success') and 'result_image' in result_data:
                result_image_path = os.path.join(result_folder, f"{image_name}_result.png")
                cv2.imwrite(result_image_path, result_data['result_image'])
                print(f"Sonuç görüntüsü kaydedildi: {result_image_path}")
                
            # Zamanlama istatistiklerini kaydet
            if 'timing_stats' in result_data:
                timing_file = os.path.join(result_folder, f"{image_name}_timing.txt")
                with open(timing_file, 'w', encoding='utf-8') as f:
                    f.write("ZAMANLAMA İSTATİSTİKLERİ\n")
                    f.write("=" * 30 + "\n")
                    for key, value in result_data['timing_stats'].items():
                        f.write(f"{key}: {value} saniye\n")
                        
        except Exception as e:
            print(f"Sonuç kaydetme hatası: {str(e)}")
            
    def run(self):
        """Ana çalıştırma fonksiyonu"""
        print("OCR Mods - Modüler OCR Sistemi")
        print("=" * 50)
        
        # Model seçimi
        selected_model = self.select_model()
        
        # PaddleOCR seçildiyse ana döngüden çık
        if selected_model is None:
            return
        
        # Modelleri başlat
        self.initialize_models(selected_model)
        
        # Ön işleme seçimi
        use_preprocessing = self.ask_preprocessing()
        
        # ROI seçimi
        use_roi = self.ask_roi_selection()
        roi_method = None
        if use_roi:
            roi_method = self.select_roi_method()
            
        # Logger'ı ayarla
        self.logger = OCRLogger(self.results_dir)
        result_folder = self.logger.setup_logger(selected_model, use_preprocessing, use_roi)
        
        # Görüntüleri yükle
        image_files = self.load_images()
        if not image_files:
            return
            
        print(f"\n{len(image_files)} görüntü işlenecek...")
        
        # Her görüntüyü işle
        successful_count = 0
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] İşleniyor: {os.path.basename(image_path)}")
            
            result_data = self.process_single_image(
                image_path, selected_model, use_preprocessing, use_roi, roi_method
            )
            
            if result_data['success']:
                successful_count += 1
                self.save_results(result_folder, image_path, result_data)
                print(f"✓ Başarılı: {len(result_data['results'])} metin bulundu")
            else:
                print(f"✗ Başarısız: {result_data.get('error', 'Bilinmeyen hata')}")
                
        # Özet
        if self.logger:
            self.logger.log_summary(len(image_files), successful_count)
            
        print(f"\n" + "="*50)
        print("İŞLEM TAMAMLANDI")
        print("="*50)
        print(f"Toplam görüntü: {len(image_files)}")
        print(f"Başarılı işlem: {successful_count}")
        print(f"Başarı oranı: {(successful_count/len(image_files))*100:.1f}%")
        print(f"Sonuçlar: {result_folder}")
        
        # Logger'ı kapat
        if self.logger:
            self.logger.close()

def main():
    """Ana fonksiyon"""
    try:
        ocr_system = OCRMods()
        ocr_system.run()
    except KeyboardInterrupt:
        print("\n\nİşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\nBeklenmeyen hata: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
