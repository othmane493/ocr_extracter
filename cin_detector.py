"""
Détecteur automatique de type de CIN et point d'entrée unifié
"""
import cv2
import numpy as np
from utils.similarity import compare_name_ar_fr
from typing import Dict, Tuple, Optional
from cin_new_extractor import CINNewExtractor
from cin_old_extractor import CINOldExtractor


class CINTypeDetector:
    """Détecteur automatique du type de CIN (nouvelle ou ancienne)"""
    
    # Caractéristiques distinctives
    # Les plages sont en BGR (Blue, Green, Red) - OpenCV format
    OLD_CIN_INDICATORS = {
        # Anciennes CIN: bande supérieure verdâtre/jaunâtre (plus de G que R)
        "dominant_color_range": [(100, 150, 100), (200, 255, 200)],  # Teinte verte/jaune
        "aspect_ratio_range": (1.4, 1.8),
    }

    NEW_CIN_INDICATORS = {
        # Nouvelles CIN: bande supérieure rose/rouge (plus de R que G et B)
        "dominant_color_range": [(140, 140, 180), (240, 200, 255)],  # Teinte rose/rouge
        "aspect_ratio_range": (1.4, 1.8),
    }

    @staticmethod
    def get_dominant_color(img: np.ndarray) -> Tuple[int, int, int]:
        """
        Calcule la couleur dominante de l'image
        Se concentre sur la bande supérieure (20% du haut) qui est le vrai discriminant

        Args:
            img: Image BGR

        Returns:
            Tuple (B, G, R) de la couleur dominante
        """
        # Extraire la bande supérieure (20% du haut de l'image)
        # C'est là que se trouve la différence rose vs vert/jaune
        h, w = img.shape[:2]
        top_band = img[0:int(h * 0.20), :]

        # Réduire la taille pour accélérer le calcul
        small = cv2.resize(top_band, (100, 20))

        # Calculer la moyenne des couleurs
        avg_color_per_row = np.average(small, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        return tuple(map(int, avg_color))

    @staticmethod
    def get_aspect_ratio(img: np.ndarray) -> float:
        """
        Calcule le ratio largeur/hauteur

        Args:
            img: Image

        Returns:
            Ratio largeur/hauteur
        """
        h, w = img.shape[:2]
        return w / h if h > 0 else 0

    @staticmethod
    def color_in_range(color: Tuple[int, int, int],
                      color_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> bool:
        """
        Vérifie si une couleur est dans une plage donnée

        Args:
            color: Couleur à tester (B, G, R)
            color_range: Plage [(B_min, G_min, R_min), (B_max, G_max, R_max)]

        Returns:
            True si la couleur est dans la plage
        """
        min_color, max_color = color_range
        return all(min_val <= c <= max_val
                  for c, min_val, max_val in zip(color, min_color, max_color))

    @classmethod
    def detect_cin_type(cls, image_path: str) -> str:
        """
        Détecte automatiquement le type de CIN (OLD ou NEW)
        Se base principalement sur la couleur de la bande supérieure

        Args:
            image_path: Chemin vers l'image

        Returns:
            "OLD" ou "NEW"
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")

        h, w = img.shape[:2]

        # Extraire la bande supérieure (25% du haut) pour analyse
        top_band = img[0:int(h * 0.25), :]

        # Analyser la couleur dominante de la bande supérieure
        dominant_color = cls.get_dominant_color(img)  # Utilise déjà la bande supérieure
        aspect_ratio = cls.get_aspect_ratio(img)

        # Score pour chaque type
        old_score = 0
        new_score = 0

        # 1. Test couleur dominante (poids: 3 points)
        if cls.color_in_range(dominant_color, cls.OLD_CIN_INDICATORS["dominant_color_range"]):
            old_score += 3
            print(f"   ✓ Couleur correspond à OLD (vert/jaune)")

        if cls.color_in_range(dominant_color, cls.NEW_CIN_INDICATORS["dominant_color_range"]):
            new_score += 3
            print(f"   ✓ Couleur correspond à NEW (rose/rouge)")

        # 2. Analyse HSV de la bande supérieure (poids: 4 points)
        hsv_band = cv2.cvtColor(top_band, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv_band[:, :, 0])
        avg_saturation = np.mean(hsv_band[:, :, 1])

        # Anciennes CIN : teinte jaune-vert (20-60 en HSV) avec saturation faible
        # Nouvelles CIN : peuvent avoir diverses teintes mais R>G dans la bande rose
        if 20 <= avg_hue <= 60 and avg_saturation < 60:
            old_score += 4
            print(f"   ✓ Teinte HSV {avg_hue:.1f} + Saturation faible → OLD (jaune-vert)")
        elif avg_hue >= 140 or avg_hue <= 10:
            new_score += 4
            print(f"   ✓ Teinte HSV {avg_hue:.1f} → NEW (rose-rouge)")

        # 3. Test du ratio Rouge/Vert dans la bande supérieure (poids: 5 points - CRITÈRE PRINCIPAL)
        # Nouvelles CIN ont plus de rouge, anciennes ont plus de vert
        # C'est le critère le plus fiable !
        b, g, r = dominant_color
        if r > g and r > b:  # Plus de rouge
            new_score += 5
            print(f"   ✓ R>G>B ({r}>{g}>{b}) → NEW [CRITÈRE PRINCIPAL]")
        elif g >= r:  # Plus de vert ou égal
            old_score += 5
            print(f"   ✓ G≥R ({g}≥{r}) → OLD [CRITÈRE PRINCIPAL]")

        # 4. Test saturation (poids: 2 points)
        # Les nouvelles CIN ont tendance à être plus saturées (rose vif)
        if avg_saturation > 100:
            new_score += 2
            print(f"   ✓ Saturation élevée ({avg_saturation:.0f}) → NEW")
        elif avg_saturation < 80:
            old_score += 2
            print(f"   ✓ Saturation faible ({avg_saturation:.0f}) → OLD")

        # Décision finale
        print(f"\n📊 Scores finaux: OLD={old_score}, NEW={new_score}")
        print(f"   Couleur dominante BGR: {dominant_color}")
        print(f"   Aspect ratio: {aspect_ratio:.2f}")
        print(f"   Teinte HSV: {avg_hue:.1f}")
        print(f"   Saturation: {avg_saturation:.1f}")

        # En cas d'égalité, privilégier NEW (plus récent)
        if new_score > old_score:
            return "NEW"
        elif old_score > new_score:
            return "OLD"
        else:
            # Égalité: utiliser la teinte comme arbitre
            print(f"   ⚖️  Égalité! Utilisation de la teinte comme arbitre")
            return "OLD" if 20 <= avg_hue <= 90 else "NEW"


class UnifiedCINExtractor:
    """Point d'entrée unifié pour l'extraction de CIN"""

    DEFAULT_TEMPLATES = {
        "NEW": "config/cin_new_template.json",
        "OLD": "config/cin_old_template.json"
    }

    def __init__(self, image_path: str,
                 cin_type: Optional[str] = None,
                 template_path: Optional[str] = None,
                 debug: bool = True):
        """
        Initialise l'extracteur unifié

        Args:
            image_path: Chemin vers l'image CIN
            cin_type: Type de CIN ("OLD" ou "NEW", auto-détecté si None)
            template_path: Chemin vers le template (auto si None)
            debug: Active le mode debug
        """
        self.image_path = image_path
        self.debug = debug

        # Détection automatique du type si non fourni
        if cin_type is None:
            self.cin_type = CINTypeDetector.detect_cin_type(image_path)
            print(f"✨ Type détecté automatiquement: CIN {self.cin_type}")
        else:
            self.cin_type = cin_type.upper()

        # Détermination du template
        if template_path is None:
            self.template_path = self.DEFAULT_TEMPLATES.get(self.cin_type)
            if self.template_path is None:
                raise ValueError(f"Type de CIN inconnu: {self.cin_type}")
        else:
            self.template_path = template_path

        # Création de l'extracteur approprié
        if self.cin_type == "NEW":
            self.extractor = CINNewExtractor(self.template_path, self.image_path, self.debug)
        elif self.cin_type == "OLD":
            self.extractor = CINOldExtractor(self.template_path, self.image_path, self.debug)
        else:
            raise ValueError(f"Type de CIN non supporté: {self.cin_type}")

        print(f"🔧 Extracteur initialisé: {self.extractor.__class__.__name__}")

    def extract(self, compare_name_func=None) -> Dict:
        """
        Lance l'extraction des données

        Args:
            compare_name_func: Fonction optionnelle pour comparer les noms AR/FR

        Returns:
            Dictionnaire des champs extraits
        """
        print(f"🚀 Extraction en cours avec {self.extractor.__class__.__name__}...")
        return self.extractor.extract(compare_name_func)

    @staticmethod
    def extract_from_image(image_path: str,
                          cin_type: Optional[str] = None,
                          compare_name_func=None,
                          debug: bool = True) -> Dict:
        """
        Méthode statique pratique pour extraire directement d'une image

        Args:
            image_path: Chemin vers l'image
            cin_type: Type de CIN (auto-détecté si None)
            compare_name_func: Fonction de comparaison AR/FR
            debug: Mode debug

        Returns:
            Dictionnaire des champs extraits
        """
        extractor = UnifiedCINExtractor(image_path, cin_type, debug=debug)
        return extractor.extract(compare_name_func)


# Fonction de commodité pour l'import simple
def extract_cin(image_path: str,
               cin_type: Optional[str] = None,
               debug: bool = True) -> Dict:
    """
    Fonction simple pour extraire les données d'une CIN

    Args:
        image_path: Chemin vers l'image
        cin_type: "OLD", "NEW" ou None (auto-détection)
        compare_name_func: Fonction de comparaison AR/FR (optionnel)
        debug: Active le mode debug

    Returns:
        Dictionnaire des champs extraits

    Exemple:
        >>> data = extract_cin("images/cin_new.png")
        >>> print(data["nom_fr"])
    """
    return UnifiedCINExtractor.extract_from_image(
        image_path,
        cin_type,
        compare_name_ar_fr,
        debug
    )