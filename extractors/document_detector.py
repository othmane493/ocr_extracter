"""
Détecteur automatique universel de documents
Utilise la détection CIN existante de cin_detector.py
"""
import time

import cv2
import numpy as np
import pytesseract
from difflib import SequenceMatcher


def detect_document_type(image_path: str) -> str:
    """
    Détecte automatiquement le type de document

    Args:
        image_path: Chemin vers l'image

    Returns:
        Type de document: 'cin_old', 'cin_new', 'carte_grise_recto', 'carte_grise_verso'
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")

    # Étape 1: Détecter si c'est une CIN ou une Carte Grise
    doc_category = _detect_category(img)
    print(f"Catégorie détectée: {doc_category}")

    if doc_category == 'cin':
        # Utiliser la détection CIN existante qui fonctionne parfaitement
        start = time.time()
        from cin_detector import CINTypeDetector
        cin_type = CINTypeDetector.detect_cin_type(image_path)
        print("detector_time")
        print(time.time() - start)
        result = 'cin_old' if cin_type == 'OLD' else 'cin_new'
        print(f"Type CIN détecté: {result}")
        return result
    else:
        # Détection Carte Grise Recto vs Verso
        result = _detect_carte_grise_type(img)
        print(f"Type Carte Grise détecté: {result}")
        return result


def _detect_category(img: np.ndarray) -> str:
    """
    Détecte si c'est une CIN ou une Carte Grise
    Utilise plusieurs critères robustes
    """
    h, w = img.shape[:2]

    # Critère 1: Analyse de la bande supérieure colorée (CIN)
    top_band = img[0:int(h * 0.20), :]
    avg_color = np.mean(top_band, axis=(0, 1))
    b, g, r = avg_color

    # CIN ont une bande colorée prononcée (rose ou verte)
    color_variance = max(r, g, b) - min(r, g, b)

    # Critère 2: Ratio de l'image
    aspect_ratio = w / h

    # Critère 3: OCR rapide
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 148, 253, cv2.THRESH_BINARY_INV)

    cin_score = 0
    cg_score = 0

    # Score basé sur couleur de bande supérieure
    if color_variance > 25:  # Bande colorée (ajusté de 40 à 25)
        cin_score += 3
    else:
        cg_score += 1

    # Score basé sur ratio (CIN plus proche de 1.6, CG aussi mais analyse couleur différencie)
    if 1.5 <= aspect_ratio <= 1.7:
        # Les deux peuvent avoir ce ratio, ne pas scorer
        pass

    try:
        # Extraction ciblée sur bande supérieure pour les mots clés
        top_text = pytesseract.image_to_string(
            thresh[0:int(h * 0.3), :],
            lang='fra+ara',
            config='--psm 6'
        )
        top_text_lower = top_text.lower()

        # Mots clés très spécifiques CIN (dans bande supérieure)
        cin_keywords = [
            'carte nationale',
            'البطاقة الوطنية',
            'للتعريف',
            'identité',
            "d'identité"
        ]

        # Mots clés Carte Grise (généralement pas en haut)
        cg_keywords = [
            'royaume du maroc',
            'المملكة المغربية',
            'immatriculation',
            'التسجيل'
        ]

        # Score pour CIN (poids fort si dans bande supérieure)
        for kw in cin_keywords:
            if kw in top_text_lower:
                cin_score += 4

        # Analyse texte complet
        full_text = pytesseract.image_to_string(thresh, lang='fra+ara', config='--psm 6')
        full_text_lower = full_text.lower()

        # Mots spécifiques CIN dans texte complet
        if 'carte nationale' in full_text_lower or 'البطاقة الوطنية' in full_text_lower:
            cin_score += 5

        # Mots spécifiques Carte Grise
        if 'propriétaire' in full_text_lower or 'المالك' in full_text_lower:
            cg_score += 3
        if 'marque' in full_text_lower or 'الاسم التجاري' in full_text_lower:
            cg_score += 3
        if 'genre' in full_text_lower and 'النوع' in full_text_lower:
            cg_score += 3

    except Exception as e:
        # En cas d'erreur OCR, se baser sur la couleur
        pass

    # Décision finale
    print(f"Scores détection catégorie: CIN={cin_score}, Carte_Grise={cg_score}, Color_variance={color_variance:.1f}")

    if cin_score > cg_score:
        return 'cin'
    elif cg_score > cin_score:
        return 'carte_grise'
    else:
        # En cas d'égalité, utiliser la variance de couleur
        result = 'cin' if color_variance > 20 else 'carte_grise'
        print(f"Égalité des scores, décision par couleur: {result}")
        return result


def _detect_carte_grise_type(img: np.ndarray) -> str:
    """
    Détecte si c'est une Carte Grise Recto ou Verso
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 148, 253, cv2.THRESH_BINARY_INV)

    try:
        text = pytesseract.image_to_string(thresh, lang='fra+ara', config='--psm 6')

        # Champs spécifiques au Recto
        recto_fields = [
            'Numéro d\'immatriculation',
            'Propriétaire',
            'Adresse',
            'Usage',
            'Mutation',
            'المالك',
            'العنوان'
        ]

        # Champs spécifiques au Verso
        verso_fields = [
            'Marque',
            'Type',
            'Genre',
            'Modèle',
            'chassis',
            'cylindres',
            'Puissance fiscale',
            'الاسم التجاري',
            'الصنف',
            'النوع'
        ]

        recto_score = 0
        verso_score = 0

        for field in recto_fields:
            if _field_found(text, field):
                recto_score += 1

        for field in verso_fields:
            if _field_found(text, field):
                verso_score += 1

        return 'carte_grise_recto' if recto_score > verso_score else 'carte_grise_verso'

    except Exception:
        return 'carte_grise_recto'


def _field_found(text: str, field: str, threshold: float = 0.6) -> bool:
    """Vérifie si un champ est présent dans le texte"""
    text_lower = text.lower()
    field_lower = field.lower()

    if field_lower in text_lower:
        return True

    words = text_lower.split()
    for word in words:
        if len(word) > 3:
            ratio = SequenceMatcher(None, word, field_lower).ratio()
            if ratio >= threshold:
                return True

    return False