"""
Extracteur unifié pour les CIN (Old et New)
Utilise l'architecture POO existante via extract_cin
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
    Wrapper pour l'extraction de CIN
    Utilise la fonction extract_cin qui gère déjà tout
    """

    def __init__(self):
        """Initialise l'extracteur CIN"""
        pass

    def extract(self, image_path, cin_type):
        """
        Extrait les données d'une CIN

        Args:
            image_path: Chemin vers l'image
            cin_type: Type de CIN ('cin_old' ou 'cin_new')

        Returns:
            Dictionnaire contenant les données extraites
        """

        # Conversion du type
        if cin_type == 'cin_old':
            detected_type = 'OLD'
        elif cin_type == 'cin_new':
            detected_type = 'NEW'
        else:
            raise ValueError(f"Type de CIN invalide: {cin_type}")

        # Utiliser extract_cin qui fait déjà tout le travail
        data = extract_cin(
            image_path,
            cin_type=detected_type,
            debug=False  # Pas de debug en production
        )

        return data