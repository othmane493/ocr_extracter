"""
Extracteur spécialisé pour les nouvelles CIN marocaines
"""
import cv2
import numpy as np
from typing import List, Dict
from cin_extractor_base import CINExtractor


class CINNewExtractor(CINExtractor):
    """Extracteur pour les nouvelles cartes d'identité (CIN New)"""
    
    def __init__(self, template_path: str, image_path: str, debug: bool = True):
        super().__init__(template_path, image_path, debug)
        self.cin_type = "CIN_NEW"
    
    def preprocess_zone(self, zone: np.ndarray) -> np.ndarray:
        """
        Prétraitement pour Tesseract - CIN New
        Contraste élevé et binarisation Otsu
        
        Args:
            zone: Zone d'image à prétraiter
            
        Returns:
            Zone prétraitée en noir et blanc
        """
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        
        # Augmentation du contraste
        gray = cv2.addWeighted(gray, 2.0, gray, -0.5, 0)
        
        # Binarisation avec Otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def preprocess_zone_easyocr(self, zone: np.ndarray) -> np.ndarray:
        """
        Prétraitement pour EasyOCR - CIN New
        Retourne la zone originale (EasyOCR gère bien les nouvelles CIN)
        
        Args:
            zone: Zone d'image à prétraiter
            
        Returns:
            Zone originale
        """
        return zone
    
    def extract_text_tesseract(self, zone: np.ndarray) -> List[Dict]:
        """
        Extraction de texte avec Tesseract pour CIN New
        
        Args:
            zone: Zone prétraitée
            
        Returns:
            Liste de blocs de texte avec confiance
        """
        try:
            # Import local pour éviter la dépendance si non utilisé
            from utils.ocr_utils import extract_text_tesseract
            return extract_text_tesseract(zone)
        except ImportError:
            # Fallback si le module n'existe pas
            return []
    
    def get_confidence_threshold(self) -> int:
        """
        Seuil de confiance pour CIN New
        Les nouvelles CIN ont généralement une meilleure qualité
        
        Returns:
            Seuil de confiance (80%)
        """
        return 80


# Fonction utilitaire pour créer une instance rapidement
def create_cin_new_extractor(image_path: str, 
                            template_path: str = "config/cin_new_template.json",
                            debug: bool = True) -> CINNewExtractor:
    """
    Créé une instance de CINNewExtractor
    
    Args:
        image_path: Chemin vers l'image CIN
        template_path: Chemin vers le template (par défaut: config/cin_new_template.json)
        debug: Active le mode debug
        
    Returns:
        Instance de CINNewExtractor
    """
    return CINNewExtractor(template_path, image_path, debug)
