"""
Extracteur spécialisé pour les anciennes CIN marocaines
"""
import cv2
import numpy as np
from typing import List, Dict
from cin_extractor_base import CINExtractor


class CINOldExtractor(CINExtractor):
    """Extracteur pour les anciennes cartes d'identité (CIN Old)"""
    
    def __init__(self, template_path: str, image_path: str, debug: bool = True):
        super().__init__(template_path, image_path, debug)
        self.cin_type = "CIN_OLD"
        self.pivot_img_path = "enhanced_cin.jpg"
    
    def preprocess_zone(self, zone: np.ndarray) -> np.ndarray:
        """
        Prétraitement pour Tesseract - CIN Old
        Optimisé pour les anciennes CIN avec qualité plus faible
        
        Args:
            zone: Zone d'image à prétraiter
            
        Returns:
            Zone prétraitée en noir et blanc
        """
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        
        # Contraste adapté pour anciennes CIN
        gray = cv2.addWeighted(gray, 1.90, gray, -0.6, 0)
        
        # Binarisation avec seuil ajusté
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Sauvegarde optionnelle pour debug
        if self.debug:
            cv2.imwrite(self.pivot_img_path, thresh)
        
        return thresh
    
    def preprocess_zone_easyocr(self, zone: np.ndarray) -> np.ndarray:
        """
        Prétraitement pour EasyOCR - CIN Old
        Agrandissement et amélioration du contraste pour anciennes CIN
        
        Args:
            zone: Zone d'image à prétraiter
            
        Returns:
            Zone prétraitée et agrandie
        """
        # Agrandissement 3x pour améliorer la qualité
        img_big = cv2.resize(zone, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)
        
        # Égalisation adaptative de l'histogramme (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Augmentation du contraste sans casser le texte
        sharpened = cv2.addWeighted(gray, 2.1, gray, -0.5, 0)
        # Binarisation avec Otsu
        _, thresh = cv2.threshold(gray, 146, 100, cv2.MORPH_DIAMOND)
        cv2.imwrite("test.jpg", thresh)
        return thresh
    
    def preprocessing_alternative(self, img: np.ndarray) -> np.ndarray:
        """
        Méthode alternative de prétraitement (non utilisée par défaut)
        Garde le texte noir net et estompe le fond
        
        Args:
            img: Image source
            
        Returns:
            Image avec texte net et fond estompé
        """
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Contraste élevé pour renforcer le texte noir
        alpha = 2.0
        beta = -180
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Créer un masque pour les pixels foncés (texte)
        _, mask = cv2.threshold(enhanced, 70, 255, cv2.THRESH_BINARY)
        
        # Inverser le masque pour le fond
        mask_inv = cv2.bitwise_not(mask)
        
        # Garder le texte noir net
        text_only = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        
        # Estomper le fond
        background = cv2.bitwise_and(gray, gray, mask=mask_inv)
        background = cv2.GaussianBlur(background, (5, 5), 0)
        
        # Combiner texte net + fond estompé
        final = cv2.add(text_only, background)
        
        # Enregistrer le résultat
        cv2.imwrite(self.pivot_img_path, final)
        return cv2.imread(self.pivot_img_path)
    
    def extract_text_tesseract(self, zone: np.ndarray) -> List[Dict]:
        """
        Extraction de texte avec Tesseract pour CIN Old
        
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
        Seuil de confiance pour CIN Old
        Les anciennes CIN ont une qualité plus faible, donc seuil plus bas
        
        Returns:
            Seuil de confiance (60%)
        """
        return 60


# Fonction utilitaire pour créer une instance rapidement
def create_cin_old_extractor(image_path: str,
                            template_path: str = "config/cin_old_template.json",
                            debug: bool = True) -> CINOldExtractor:
    """
    Créé une instance de CINOldExtractor
    
    Args:
        image_path: Chemin vers l'image CIN
        template_path: Chemin vers le template (par défaut: config/cin_old_template.json)
        debug: Active le mode debug
        
    Returns:
        Instance de CINOldExtractor
    """
    return CINOldExtractor(template_path, image_path, debug)
