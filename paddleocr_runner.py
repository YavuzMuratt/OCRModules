#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR Runner - Modüler OCR Sistemi için PaddleOCR Özel Script'i
Tüm modüler özellikleri içerir: Timer, Logger, Preprocessor, ROI Selector
GPU hızlandırma desteği ile
"""

import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Any

# Modülleri import et
from modules.timer import PerformanceTimer
from modules.logger import OCRLogger
from modules.preprocessor import ImagePreprocessor
from modules.roi_selector import ROISelector

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

class PaddleOCRRunner:
    """PaddleOCR için özel runner sınıfı - GPU hızlandırma desteği ile"""
    
    def __init__(self):
        self.timer = PerformanceTimer()
        self.preprocessor = ImagePreprocessor()
        self.roi_selector = ROISelector()
        self.logger = None
        self.ocr = None
        self.gpu_available = False
        self.gpu_count = 0
        
        # Sonuçlar klasörü
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def initialize_paddleocr(self):
        """PaddleOCR'ı başlatır"""
        try:
            print("PaddleOCR başlatılıyor...")
            
            # GPU kullanılabilirliğini kontrol et
            self.gpu_available, self.gpu_count = check_gpu_availability()
            
            # PaddleOCR import et
            from paddleocr import PaddleOCR
            
            # GPU kullanılabilirse GPU ile başlat
            if self.gpu_available:
                print(f"GPU modunda başlatılıyor (GPU:0)...")
                self.ocr = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="gpu:0"
                )
            else:
                print("CPU modunda başlatılıyor...")
                self.ocr = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu"
                )
            
            # Test için basit bir OCR işlemi yap (güvenli şekilde)
            test_image_path = "test_image.png"
            test_image = np.zeros((100, 100, 3), dtype=np.uint8) * 255
            cv2.imwrite(test_image_path, test_image)
            
            try:
                results = self.ocr.predict(input=test_image_path)
                # Sonuçları kontrol et (boş olabilir)
                if results is not None:
                    print("✓ PaddleOCR test başarılı")
                else:
                    print("✓ PaddleOCR başlatıldı (test sonucu boş)")
                # Test dosyasını sil
                if os.path.exists(test_image_path):
                    os.remove(test_image_path)
            except Exception as test_error:
                print(f"⚠ Test hatası (görmezden geliniyor): {test_error}")
                # Test dosyasını sil
                if os.path.exists(test_image_path):
                    os.remove(test_image_path)
            
            device_info = "GPU" if self.gpu_available else "CPU"
            print(f"✓ PaddleOCR başarıyla başlatıldı ({device_info})")
            return True
            
        except Exception as e:
            print(f"✗ PaddleOCR başlatma hatası: {str(e)}")
            print("PaddleOCR kullanılamıyor!")
            return False
            
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
        
    def extract_text_with_paddleocr(self, image_path: str, roi_image: np.ndarray = None, result_folder: str = None) -> List[Dict[str, Any]]:
        """PaddleOCR ile metin çıkarır (predict API kullanarak)"""
        try:
            # ROI varsa geçici dosya olarak kaydet
            temp_image_path = image_path
            temp_file_created = False
            
            if roi_image is not None:
                # Geçici dosya oluştur - orijinal dosya adını koru
                temp_dir = os.path.join(self.results_dir, "temp_roi")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Orijinal dosya adını koru (PaddleOCR'ın kendi çıktıları için)
                original_name = os.path.basename(image_path)
                temp_image_path = os.path.join(temp_dir, original_name)
                
                # ROI görüntüsünü kaydet
                cv2.imwrite(temp_image_path, roi_image)
                temp_file_created = True
                print(f"DEBUG: ROI görüntüsü geçici dosyaya kaydedildi: {temp_image_path}")
            
            # PaddleOCR 3.x predict() API'sini kullan
            predict_result = self.ocr.predict(input=temp_image_path)
            
            # Debug için sonuç formatını yazdır
            print(f"DEBUG: PaddleOCR predict sonuç tipi: {type(predict_result)}")
            if predict_result and len(predict_result) > 0:
                print(f"DEBUG: İlk sonuç tipi: {type(predict_result[0])}")
            
            # Sonuçları formatla
            formatted_results = []
            
            # Her OCR sonucunu işle
            for res in predict_result:
                try:
                    # OCRResult objesinden metinleri çıkar
                    if hasattr(res, "rec_texts") and hasattr(res, "rec_scores") and hasattr(res, "rec_boxes"):
                        # PaddleOCR 3.x formatı: rec_texts, rec_scores, rec_boxes
                        texts = res.rec_texts
                        scores = res.rec_scores
                        boxes = res.rec_boxes
                        
                        print(f"DEBUG: Bulunan metin sayısı: {len(texts)}")
                        
                        for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
                            if text and text.strip():  # Boş metinleri atla
                                result = {
                                    'text': text.strip(),
                                    'confidence': float(score) * 100,
                                    'bbox': box,  # [x1, y1, x2, y2] formatında
                                    'bbox_xywh': [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                                }
                                formatted_results.append(result)
                                print(f"DEBUG: Metin {i+1}: '{text}' (Güven: {score:.3f})")
                    
                    # Alternatif: to_dict() metodunu kullan
                    elif hasattr(res, "to_dict"):
                        try:
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
                        except Exception as dict_error:
                            print(f"DEBUG: to_dict() hatası: {dict_error}")
                    
                    # Sonuçları kaydet (görsel ve JSON)
                    try:
                        if hasattr(res, "save_to_img") and hasattr(res, "save_to_json"):
                            # Sonuç klasörünü kullan (result_folder parametresi varsa)
                            save_dir = result_folder if result_folder else os.path.join(self.results_dir, "temp_paddle_results")
                            os.makedirs(save_dir, exist_ok=True)
                            
                            # Görsel ve JSON kaydet
                            res.save_to_img(save_dir)
                            res.save_to_json(save_dir)
                            print(f"DEBUG: PaddleOCR sonuçları kaydedildi: {save_dir}")
                    except Exception as save_error:
                        print(f"DEBUG: Sonuç kaydetme hatası: {save_error}")
                        
                except Exception as res_error:
                    print(f"DEBUG: Sonuç işleme hatası: {res_error}")
                    continue
            
            print(f"DEBUG: Toplam formatlanmış sonuç: {len(formatted_results)}")
            
            # Geçici dosyayı temizle
            if temp_file_created and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                    print(f"DEBUG: Geçici dosya temizlendi: {temp_image_path}")
                except Exception as cleanup_error:
                    print(f"DEBUG: Geçici dosya temizleme hatası: {cleanup_error}")
            
            return formatted_results
            
        except Exception as e:
            print(f"PaddleOCR predict hatası: {str(e)}")
            
            # Hata durumunda da geçici dosyayı temizle
            if temp_file_created and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except:
                    pass
            
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
        
    def process_single_image(self, image_path: str, use_preprocessing: bool, 
                           use_roi: bool, roi_method: str = None, result_folder: str = None) -> Dict[str, Any]:
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
            
            # ROI seçimi ve görüntü işleme
            roi_coords = None
            processed_image = image.copy()
            
            if use_roi:
                self.timer.start_preprocessing()
                
                if roi_method == 'interactive':
                    roi_coords = self.roi_selector.select_roi_interactive(image)
                else:
                    roi_coords = self.roi_selector.select_roi_auto(image)
                    
                if roi_coords:
                    processed_image = self.roi_selector.apply_roi(image, roi_coords)
                    if self.logger:
                        self.logger.log_roi_info(roi_coords)
                        
                self.timer.end_preprocessing()
                
            # Görüntü ön işleme
            if use_preprocessing:
                self.timer.start_preprocessing()
                processed_image = self.preprocessor.preprocess_for_ocr(processed_image)
                preprocessing_steps = self.preprocessor.get_applied_steps()
                if self.logger:
                    self.logger.log_preprocessing_info(preprocessing_steps)
                self.timer.end_preprocessing()
                
            # OCR işlemi
            self.timer.start_ocr()
            # ROI veya ön işleme uygulandıysa işlenmiş görüntüyü kullan
            if use_roi or use_preprocessing:
                results = self.extract_text_with_paddleocr(image_path, roi_image=processed_image, result_folder=result_folder)
            else:
                results = self.extract_text_with_paddleocr(image_path, result_folder=result_folder)
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
            result_image = self.draw_results(original_image, results)
            
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
        print("PaddleOCR Runner - Modüler OCR Sistemi")
        print("=" * 50)
        
        # PaddleOCR'ı başlat
        if not self.initialize_paddleocr():
            return
        
        # Ön işleme seçimi
        use_preprocessing = self.ask_preprocessing()
        
        # ROI seçimi
        use_roi = self.ask_roi_selection()
        roi_method = None
        if use_roi:
            roi_method = self.select_roi_method()
            
        # Logger'ı ayarla
        self.logger = OCRLogger(self.results_dir)
        result_folder = self.logger.setup_logger("paddleocr", use_preprocessing, use_roi)
        
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
                image_path, use_preprocessing, use_roi, roi_method, result_folder
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
        runner = PaddleOCRRunner()
        runner.run()
    except KeyboardInterrupt:
        print("\n\nİşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\nBeklenmeyen hata: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
