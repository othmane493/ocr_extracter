"""
Détecteur de document
Ordre :
1) carte grise
2) CIN
"""

import cv2
import pytesseract

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



def detect_carte_grise(img):

    text = ocr(img).lower()

    recto_words = [
        "propriétaire",
        "adresse",
        "usage",
        "mutation",
        "المالك"
    ]

    verso_words = [
        "marque",
        "genre",
        "modele",
        "modèle",
        "type carburant",
        "nombre de cylindres",
        "puissance fiscale",
        "poids"
    ]

    recto=0
    verso=0

    for w in recto_words:
        if w in text:
            recto+=1

    for w in verso_words:
        if w in text:
            verso+=1


    if verso >=3:
        return "carte_grise_verso"

    if recto >=2:
        return "carte_grise_recto"

    return None



def ocr(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    try:

        return pytesseract.image_to_string(
            th,
            lang="fra+ara",
            config="--psm 6"
        )

    except:

        return ""