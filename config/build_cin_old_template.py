import cv2
import json
import re

from config.build_cin_new_template import Y_TOLERENCE
from utils.ocr_utils import extract_text_tesseract_pos, group_blocks_by_line

# Configuration images
IMAGE_PATH = "../images/cin-old.jpg"
PIVOT_IMG_PATH = "enhanced_cin.jpg"
DEBUG_IMG = "debug_cin_old_template.png"
OUTPUT = "cin_old_template.json"
X_TOLERENCE = 20
Y_TOLERENCE = 5
def normalize(val, maxv):
    """Normalise une valeur par rapport à la dimension max"""
    return round(val / maxv, 4)

def contains_arabic(text):
    """Détecte si le texte contient de l'arabe"""
    return bool(re.search(r'[\u0600-\u06FF]', text))

def reorder_fields(fields: dict) -> dict:
    """Réordonne les champs : *_fr puis *_ar puis le reste"""
    fr_fields = {}
    ar_fields = {}
    other_fields = {}
    for k, v in fields.items():
        if k.endswith("_fr"):
            fr_fields[k] = v
        elif k.endswith("_ar"):
            ar_fields[k] = v
        else:
            other_fields[k] = v
    ordered = {}
    ordered.update(fr_fields)
    ordered.update(ar_fields)
    ordered.update(other_fields)
    return ordered

def filter_text_by_strictness(text: str, strict_content: bool) -> bool:
    """
    Retourne True si le texte est valide selon la règle strict_content.
    - strict_content=True : uniquement lettres (fr/ar), espaces autorisés
    - strict_content=False: lettres, chiffres, / . - autorisés, mais pas <, >, #, etc.
    """
    text = str(text).strip()
    if strict_content:
        # Autorise lettres arabes et latines + espace
        return bool(re.fullmatch(r"[A-Za-z\u0600-\u06FF\s]+", text))
    else:
        # Autorise lettres, chiffres, / . - et espaces
        return bool(re.fullmatch(r"[A-Za-z0-9\u0600-\u06FF\s\./-]+", text))

def clean_ocr_text(text: str) -> str:
    """
    Nettoie le texte OCR :
    - supprime les caractères invisibles (RTL, LTR, zero-width, etc.)
    - normalise les espaces
    """
    if not text:
        return ""

    text = str(text)

    # Caractères invisibles courants en OCR
    INVISIBLE_CHARS = [
        "\u200e",  # LTR mark
        "\u200f",  # RTL mark
        "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
        "\u2066", "\u2067", "\u2068", "\u2069",
        "\ufeff",  # BOM
    ]

    for ch in INVISIBLE_CHARS:
        text = text.replace(ch, "")

    # Normalisation espaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_lines(lines, min_confidence=70):
    """
    Nettoie les lignes OCR avant traitement mapping :
    1. Nettoie le texte OCR (caractères invisibles)
    2. Supprime les blocs avec confidence < min_confidence
    3. Supprime les lignes fantômes (caractères spéciaux seuls)
    """
    cleaned = []

    for line in lines:
        cleaned_blocks = []

        for b in line:
            conf = b.get("confidence", 0)
            text = clean_ocr_text(b.get("text", ""))

            if conf < min_confidence:
                continue

            if not text:
                continue

            # Met à jour le texte nettoyé
            b = b.copy()
            b["text"] = text
            cleaned_blocks.append(b)

        # Vérifie qu'il reste au moins un bloc utile
        has_valid_block = any(
            re.search(r"[A-Za-z0-9\u0600-\u06FF]", b["text"])
            for b in cleaned_blocks
        )

        if has_valid_block:
            cleaned.append(cleaned_blocks)

    return cleaned

def preprocessing(img):
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Appliquer un contraste élevé pour renforcer le texte noir
    alpha = 2.0  # facteur de contraste (augmenter pour renforcer le noir)
    beta = -150   # facteur de luminosité (baisser pour assombrir le fond)
    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Créer un masque pour les pixels foncés (texte)
    _, mask = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY)

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
    cv2.imwrite(PIVOT_IMG_PATH, final)
    return cv2.imread(PIVOT_IMG_PATH)

def main():
    img_original = cv2.imread(IMAGE_PATH)
    img = preprocessing(img_original)
    h, w = img.shape[:2]

    # Extraction et regroupement en lignes
    blocks = extract_text_tesseract_pos(img)
    lines = group_blocks_by_line(blocks)

    # Suppression header si besoin (réglé une seule fois)
    content_lines = lines[2:]
    content_lines = clean_lines(content_lines)

    # Mapping configurable avec marges
    mapping = [
        {"field": "prenom_ar", "rule": "value_ar", "margin_top": 0, "margin_bottom": 0, "strict_content": True},
        {"field": "prenom_fr", "rule": "value_fr", "margin_top": 0, "margin_bottom": 0, "strict_content": True},
        {"field": "nom_ar", "rule": "value_ar", "margin_top": 0, "margin_bottom": 0, "strict_content": True},
        {"field": "nom_fr", "rule": "value_fr", "margin_top": 0, "margin_bottom": 0, "strict_content": True},
        {"field": "date_naissance", "rule": "key_fr_value_ar", "margin_left": 150, "margin_right": 200, "margin_top": 5,
         "margin_bottom": 5, "strict_content": False},
        {"field": "lieu_naissance_ar", "rule": "value_ar", "margin_left": 0, "margin_right": 45, "margin_top": 12, "margin_bottom": 5, "strict_content": False},
        {"field": "lieu_naissance_fr", "rule": "value_fr", "margin_left": 15,"margin_top": 0, "margin_bottom": 0, "strict_content": False},
        {"field": "date_expiration", "rule": "key_fr_value_ar", "margin_left": 0, "margin_right": 230,
         "margin_top": 0, "margin_bottom": 0, "strict_content": False},
        {"field": "cin", "rule": "manual", "margin_left": 50, "margin_right": 50, "margin_top": 7, "margin_bottom": 50, "strict_content": False}
    ]


    template = {
        "document": "CIN_MAROC",
        "width": w,
        "height": h,
        "fields": {}
    }
    # Pré-calcul des bornes Y par ligne
    line_bounds = []
    for line in content_lines:
        ys = [b["y"] for b in line]
        ye = [b["y"] + b["height"] for b in line]
        line_bounds.append((min(ys), max(ye)))

    # Parcours mapping pour extraction automatique
    for i, item in enumerate(mapping[:-1]):  # dernier champ = CIN manuel
        strict_content = item.get("strict_content", True)
        line = content_lines[i]

        # Filtrer les blocs invalides selon strict_content
        filtered_line = [b for b in line if filter_text_by_strictness(b.get("text", ""), strict_content)]
        # 🔹 Sécurité : si aucun bloc valide, fallback sur la ligne originale
        content_lines[i] = filtered_line

        field = item["field"]
        rule = item["rule"]
        margin_left = item.get("margin_left", 0)
        margin_right = item.get("margin_right", 0)
        margin_top = item.get("margin_top", 0)
        margin_bottom = item.get("margin_bottom", 0)

        line = content_lines[i]

        full_text = " ".join(b["text"] for b in line)

        if rule == "value_ar":
            is_ar = True
        elif rule == "value_fr":
            is_ar = False
        elif rule == "key_fr_value_ar":
            is_ar = True
        else:
            is_ar = contains_arabic(full_text)

        # 📐 X : extension avec marges
        if rule == "key_fr_value_ar":
            x_min = line[0]["x"] - X_TOLERENCE + margin_left
            x_max = line[len(line) - 1]["x"] + line[len(line) - 1]["width"] + X_TOLERENCE - margin_right
        elif rule == "value_ar":
            x_min = int(w/3) + margin_left
            x_max = line[0]["x"] + line[0]["width"] + X_TOLERENCE - margin_right
        else:  # value_fr
            x_min = line[0]["x"] - X_TOLERENCE + margin_left
            x_max = int(w / 3) - margin_right

        # 📐 Y : extension avec marges
        y_min = line[0]["y"] - Y_TOLERENCE - margin_top
        y_max = line[0]["y"] + line[0]["height"] + Y_TOLERENCE + margin_bottom

        ww = x_max - x_min
        hh = y_max - y_min

        # 🟥 Dessin rectangle debug
        cv2.rectangle(img_original, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(img_original, field, (x_min + 5, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        template["fields"][field] = {
            "x": normalize(x_min, w),
            "y": normalize(y_min, h),
            "w": normalize(ww, w),
            "h": normalize(hh, h),
            "lang": "ar" if is_ar else "fr"
        }

    # ----------------------
    # Dernier champ : CIN manuel
    # ----------------------
    cin_item = mapping[-1]
    cin_margin_left = cin_item.get("margin_left", 0)
    cin_margin_right = cin_item.get("margin_right", 0)
    cin_margin_top = cin_item.get("margin_top", 0)
    cin_margin_bottom = cin_item.get("margin_bottom", 0)

    Y_START_PERCENT = 0.75
    y_min = int(h * Y_START_PERCENT) + cin_margin_top
    y_max = y_min + cin_margin_bottom

    cin_x_min = int(w *3/ 4) - cin_margin_left
    cin_x_max = w - 2 * X_TOLERENCE + - cin_margin_right # ajustable
    cin_width = cin_x_max - cin_x_min

    template["fields"]["cin"] = {
        "x": normalize(cin_x_min, w),
        "y": normalize(y_min, h),
        "w": normalize(cin_width, w),
        "h": normalize(y_max - y_min - 2 * Y_TOLERENCE, h),
        "lang": "fr"
    }

    cv2.rectangle(img_original, (cin_x_min, y_min), (cin_x_max, y_max), (255, 0, 0), 2)
    cv2.putText(img_original, "cin", (cin_x_min + 5, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Réordonner champs et écrire JSON
    template["fields"] = reorder_fields(template["fields"])

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    # Sauvegarde image debug
    cv2.imwrite(DEBUG_IMG, img_original)
    print("✅ Template généré avec marges personnalisables et rectangles :", DEBUG_IMG)

if __name__ == "__main__":
    main()
