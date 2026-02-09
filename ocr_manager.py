"""
Gestionnaire singleton pour les modèles OCR
Initialise EasyOCR et Tesseract une seule fois
"""
import easyocr
import time


class OCRManager:
    """Singleton pour gérer les modèles OCR"""
    
    _instance = None
    _reader = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCRManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialise le manager (appelé une seule fois)"""
        if not OCRManager._initialized:
            print("Initialisation des modèles OCR (une seule fois)...")
            start = time.time()
            
            # Initialiser EasyOCR
            OCRManager._reader = easyocr.Reader(["en", "ar"], gpu=False)
            
            elapsed = time.time() - start
            print(f"Modèles OCR chargés en {elapsed:.2f}s")
            OCRManager._initialized = True
    
    @classmethod
    def get_reader(cls):
        """Retourne l'instance EasyOCR (partagée)"""
        if cls._reader is None:
            cls()  # Force l'initialisation
        return cls._reader
    
    @classmethod
    def is_ready(cls):
        """Vérifie si les modèles sont prêts"""
        return cls._initialized


# Instance globale
_ocr_manager = OCRManager()


def get_easyocr_reader():
    """
    Fonction utilitaire pour obtenir le reader EasyOCR
    Utilise toujours la même instance
    """
    return OCRManager.get_reader()
