"""
Détecteur de document
Ordre :
1) carte grise
2) CIN
"""

import re
import cv2
import pytesseract
import unicodedata

from cin_detector import CINTypeDetector


def detect_document_type(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image non chargée")

    # 1️⃣ DETECT CARTE GRISE
    cg = detect_carte_grise(img)

    if cg:
        return cg

    # 2️⃣ DETECT CIN
    side = CINTypeDetector.detect_big_photo_side(image_path)

    if side == "left":
        return "cin_new"

    if side == "right":
        return "cin_old"

    raise ValueError("Document inconnu")


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_keyword_groups(text, keyword_groups):
    score = 0
    for group in keyword_groups:
        if any(keyword in text for keyword in group):
            score += 1
    return score


def resize_for_fast_ocr(img, max_width=1400):
    h, w = img.shape[:2]
    if w <= max_width:
        return img

    ratio = max_width / float(w)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def prepare_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ocr_fast(img):
    """
    OCR rapide :
    - resize si image trop grande
    - 1 seul passage principal
    """
    if img is None or img.size == 0:
        return ""

    img = resize_for_fast_ocr(img, max_width=1400)
    gray = prepare_gray(img)

    # léger nettoyage, peu coûteux
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # seuillage principal unique
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    try:
        return pytesseract.image_to_string(
            th,
            lang="fra+ara",
            config="--oem 3 --psm 6"
        )
    except Exception:
        return ""


def ocr_fallback(img):
    """
    Fallback seulement si l'OCR rapide n'a pas donné assez d'indices.
    """
    if img is None or img.size == 0:
        return ""

    img = resize_for_fast_ocr(img, max_width=1400)
    gray = prepare_gray(img)

    texts = []

    versions = [
        gray,
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )
    ]

    for version in versions:
        try:
            text = pytesseract.image_to_string(
                version,
                lang="fra+ara",
                config="--oem 3 --psm 6"
            )
            if text and text.strip():
                texts.append(text)
        except Exception:
            pass

    return "\n".join(texts)


def detect_carte_grise(img):
    recto_words = [
        ["proprietaire", "propriétaire"],
        ["adresse"],
        ["usage"],
        ["mutation"],
        ["immatriculation"],
        ["premiere mise en circulation", "première mise en circulation"],
        ["date mutation", "date de mutation"],
        ["المالك"],
        ["العنوان"],
        ["الاستعمال"]
    ]

    verso_words = [
        ["marque"],
        ["genre"],
        ["modele", "modèle"],
        ["type carburant", "carburant"],
        ["nombre de cylindres", "cylindres"],
        ["puissance fiscale"],
        ["poids"],
        ["places assises", "nombre de places"],
        ["ptac"],
        ["ptra"],
        ["numero dans la serie du type", "numero de serie", "serie du type"]
    ]

    # 1) OCR rapide
    fast_text = normalize_text(ocr_fast(img))

    recto_fast = count_keyword_groups(fast_text, recto_words)
    verso_fast = count_keyword_groups(fast_text, verso_words)

    # décision immédiate si score clair
    if verso_fast >= 2 and verso_fast > recto_fast:
        return "carte_grise_verso"

    if recto_fast >= 2 and recto_fast >= verso_fast:
        return "carte_grise_recto"

    # 2) fallback seulement si le premier passage n'est pas suffisant
    full_text = normalize_text(ocr_fallback(img))

    recto = count_keyword_groups(full_text, recto_words)
    verso = count_keyword_groups(full_text, verso_words)

    if verso >= 2 and verso > recto:
        return "carte_grise_verso"

    if recto >= 2 and recto >= verso:
        return "carte_grise_recto"

    if verso >= 1 and recto == 0:
        return "carte_grise_verso"

    if recto >= 1 and verso == 0:
        return "carte_grise_recto"

    return None