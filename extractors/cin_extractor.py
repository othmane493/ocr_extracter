"""
Extracteur unifié pour les CIN (Old et New)
Intègre l'architecture POO existante
"""
import os
import sys
import time

# Import de l'architecture CIN existante
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from cin_detector import extract_cin


class CINExtractor:
    """
    Classe wrapper pour l'extraction de CIN
    Utilise l'architecture POO existante
    """
    
    def __init__(self):
        """Initialise l'extracteur CIN"""
        print("Initialisation de l'extracteur CIN")
    
    def extract(self, image_path, cin_type):
        """
        Extrait les données d'une CIN
        
        Args:
            image_path: Chemin vers l'image
            cin_type: Type de CIN ('cin_old' ou 'cin_new')
            
        Returns:
            Dictionnaire contenant les données extraites
        """
        start_time = time.time()
        
        # Conversion du type pour compatibilité avec le détecteur
        if cin_type == 'cin_old':
            detected_type = 'OLD'
        elif cin_type == 'cin_new':
            detected_type = 'NEW'
        else:
            raise ValueError(f"Type de CIN invalide: {cin_type}")
        
        print(f"Extraction de {cin_type} depuis: {image_path}")
        
        # Extraction avec l'architecture existante
        try:
            # Forcer le type spécifié (pas de détection auto)
            data = extract_cin(
                image_path, 
                cin_type=detected_type,
                debug=False  # Désactiver le mode debug pour production
            )
            
            extraction_time = time.time() - start_time
            print(f"Extraction {cin_type} terminée en {extraction_time:.2f}s")
            
            return data
            
        except Exception as e:
            print(f"Erreur lors de l'extraction {cin_type}: {e}")
            raise
