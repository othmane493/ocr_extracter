"""
Classe de base pour l'extraction de données des CIN marocaines
Contient toutes les méthodes communes aux deux types de CIN
"""
import json
import cv2
import numpy as np
import easyocr
import re
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List, Any
from pathlib import Path

from json_transformer import is_arabic


class CINExtractor(ABC):
    """Classe de base abstraite pour l'extraction de CIN"""

    def __init__(self, template_path: str, image_path: str, debug: bool = True):
        """
        Initialise l'extracteur de CIN

        Args:
            template_path: Chemin vers le fichier template JSON
            image_path: Chemin vers l'image à traiter
            debug: Si True, génère une image de debug avec les zones
        """
        self.template_path = template_path
        self.image_path = image_path
        self.debug = debug
        self.debug_image_path = "debug_zones.png"

        # Utiliser le reader EasyOCR partagé du singleton
        if not hasattr(CINExtractor, '_reader') or CINExtractor._reader is None:
            try:
                # Essayer d'utiliser le singleton
                import sys
                import os
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)

                from ocr_manager import get_easyocr_reader
                CINExtractor._reader = get_easyocr_reader()
            except (ImportError, Exception):
                # Fallback si le singleton n'est pas disponible
                CINExtractor._reader = easyocr.Reader(["en", "ar"], gpu=False, verbose=False)

        self.reader = CINExtractor._reader
        self.template = None
        self.img = None

    def load_template(self) -> Dict:
        """Charge le fichier template JSON"""
        with open(self.template_path, encoding="utf-8") as f:
            self.template = json.load(f)
        return self.template

    def load_image(self) -> np.ndarray:
        """Charge et redimensionne l'image selon le template"""
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise ValueError(f"Image non trouvée: {self.image_path}")

        # Redimensionner si nécessaire
        if self.template and "width" in self.template and "height" in self.template:
            target_size = (self.template["width"], self.template["height"])
            if (self.img.shape[1], self.img.shape[0]) != target_size:
                self.img = cv2.resize(self.img, target_size)

        return self.img

    @staticmethod
    def safe_crop(img: np.ndarray, rule: Dict) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Découpe une zone de l'image de manière sécurisée

        Args:
            img: Image source
            rule: Dictionnaire avec x, y, w, h (coordonnées normalisées)

        Returns:
            Tuple (zone_découpée, (x, y, width, height))
        """
        h, w = img.shape[:2]

        x = int(rule["x"] * w)
        y = int(rule["y"] * h)
        ww = int(rule["w"] * w)
        hh = int(rule["h"] * h)

        # Sécurisation des limites
        x = max(0, x)
        y = max(0, y)
        ww = min(ww, w - x)
        hh = min(hh, h - y)

        return img[y:y + hh, x:x + ww], (x, y, ww, hh)

    @staticmethod
    def normalize_date(text: str) -> str:
        """
        Reconstruit une date au format DD.MM.YYYY à partir de n'importe quel texte OCR

        Args:
            text: Texte brut extrait par OCR

        Returns:
            Date formatée ou texte original si invalide
        """
        digits = re.findall(r"\d", text)

        if len(digits) != 8:
            return text  # Date invalide, retourner tel quel

        return "{}{}.{}{}.{}{}{}{}".format(*digits)

    @staticmethod
    def filter_text_by_strictness(text: str) -> bool:
        """
        Filtre le texte pour ne garder que les caractères valides
        Autorise lettres (latin + arabe), chiffres, et certains symboles

        Args:
            text: Texte à filtrer

        Returns:
            True si le texte est valide
        """
        text = str(text).strip()
        return bool(re.fullmatch(r"[A-Za-z0-9\u0600-\u06FF\s\./-]+", text))

    def easyocr_text(self, zone: np.ndarray) -> str:
        """
        Extrait le texte d'une zone avec EasyOCR

        Args:
            zone: Zone d'image à traiter

        Returns:
            Texte extrait
        """
        txt = self.reader.readtext(zone, detail=0)
        if isinstance(txt, list):
            return " ".join(txt)
        return str(txt)

    @staticmethod
    def is_wrong_lang(field: str, text: str) -> bool:
        if field.endswith("_ar") and not is_arabic(text):
            return True
        if field.endswith("_fr") and is_arabic(text):
            return True
        return False

    @staticmethod
    def cmp_phonetic_lang(field: str, text: str, results: Dict,
                        compare_func=None, confidence: float = 0) -> float | bool | int | Any:
        """
        Détermine si l'OCR doit être rejoué avec EasyOCR
        Vérifie la cohérence entre les versions AR et FR

        Args:
            field: Nom du champ
            text: Texte extrait
            results: Résultats déjà extraits
            compare_func: Fonction de comparaison AR/FR
            threshold: Seuil de similarité

        Returns:
            True si l'OCR doit être rejoué
        """
        if not field.endswith("_ar") or compare_func is None:
            return confidence

        fr_field = field.replace("_ar", "_fr")
        if fr_field in results and results[fr_field]:
            try:
                cmp = compare_func(text, results[fr_field])
                return cmp.get("score", 0)
            except Exception:
                return confidence

        return confidence

    @staticmethod
    def reorder_identity_fields(data: Dict) -> Dict:
        """
        Réorganise les champs par entité métier
        Ordre: prenom (fr, ar) -> nom (fr, ar) -> lieu_naissance (fr, ar) -> reste

        Args:
            data: Dictionnaire brut des résultats

        Returns:
            Dictionnaire réorganisé
        """
        ordered = {}

        groups = ["prenom", "nom", "lieu_naissance"]

        # Champs groupés fr / ar
        for g in groups:
            fr_key = f"{g}_fr"
            ar_key = f"{g}_ar"

            if ar_key in data:
                ordered[ar_key] = data[ar_key]
            if fr_key in data:
                ordered[fr_key] = data[fr_key]

        # Le reste (dates, cin, etc.)
        for k, v in data.items():
            if k not in ordered:
                ordered[k] = v

        return ordered

    def create_debug_image(self, debug_img: np.ndarray, results: Dict) -> None:
        """
        Sauvegarde l'image de debug avec les zones annotées

        Args:
            debug_img: Image annotée
            results: Résultats de l'extraction
        """
        if self.debug:
            cv2.imwrite(self.debug_image_path, debug_img)
            print(f"🟢 Image debug générée : {self.debug_image_path}")

    # =============================
    # MÉTHODES ABSTRAITES À IMPLÉMENTER
    # =============================

    @abstractmethod
    def preprocess_zone(self, zone: np.ndarray) -> np.ndarray:
        """
        Prétraitement spécifique d'une zone avant OCR Tesseract
        À implémenter dans les classes filles
        """
        pass

    @abstractmethod
    def preprocess_zone_easyocr(self, zone: np.ndarray) -> np.ndarray:
        """
        Prétraitement spécifique d'une zone avant OCR EasyOCR
        À implémenter dans les classes filles
        """
        pass

    @abstractmethod
    def extract_text_tesseract(self, zone: np.ndarray) -> List[Dict]:
        """
        Extraction de texte avec Tesseract
        À implémenter dans les classes filles
        """
        pass

    @abstractmethod
    def get_confidence_threshold(self) -> int:
        """
        Retourne le seuil de confiance pour décider de rejouer l'OCR
        À implémenter dans les classes filles
        """
        pass

    # =============================
    # EXTRACTION PRINCIPALE
    # =============================

    def extract(self, compare_name_func=None) -> Dict:
        """
        Méthode principale d'extraction

        Args:
            compare_name_func: Fonction optionnelle pour comparer les noms AR/FR

        Returns:
            Dictionnaire des champs extraits
        """
        # Chargement
        self.load_template()
        self.load_image()

        debug_img = self.img.copy()
        results = {}

        # Parcours de tous les champs du template
        for field, rule in self.template["fields"].items():
            zone, (x, y, w, h) = self.safe_crop(self.img, rule)

            # Annotation debug
            if self.debug:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug_img, field, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Extraction avec Tesseract
            zone_preprocessed = self.preprocess_zone(zone)
            blocks = self.extract_text_tesseract(zone_preprocessed)

            # Filtrage des blocs
            blocks = [b for b in blocks if self.filter_text_by_strictness(b.get("text", ""))]

            if blocks:
                ocr_text = " ".join(b.get("text", "") for b in blocks)
                confidence = sum(b.get("confidence", 0) for b in blocks) / len(blocks)
                if self.is_wrong_lang(field, ocr_text):
                    zone_retry = self.preprocess_zone_easyocr(zone)
                    ocr_text = self.easyocr_text(zone_retry)
                # Retry avec EasyOCR si confiance faible
                elif field.endswith("_ar") or confidence < self.get_confidence_threshold():
                    cmp_tesseract = self.cmp_phonetic_lang(field, ocr_text, results, compare_name_func, int(confidence/100))
                    zone_retry = self.preprocess_zone_easyocr(zone)
                    ocr_easyocr_text = self.easyocr_text(zone_retry)
                    cmp_easyocr = self.cmp_phonetic_lang(field, ocr_easyocr_text, results, compare_name_func)
                    ocr_text = ocr_easyocr_text if cmp_tesseract < cmp_easyocr else ocr_text
            else:
                # Pas de texte trouvé avec Tesseract, utiliser EasyOCR
                zone_retry = self.preprocess_zone_easyocr(zone)
                ocr_text = self.easyocr_text(zone_retry)

            # Normalisation des dates
            if "date" in field:
                ocr_text = self.normalize_date(ocr_text)

            results[field] = ocr_text

        # Sauvegarde debug
        self.create_debug_image(debug_img, results)

        # Réorganisation des champs
        results = self.reorder_identity_fields(results)

        return results