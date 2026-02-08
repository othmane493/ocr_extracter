import cv2
import json
import re
from utils.ocr_utils import easyocr_full, group_blocks_by_line

IMAGE_PATH = "../images/cin_new.png"
DEBUG_IMG = "debug_cin_new_template.png"
OUTPUT = "cin_new_template.json"
X_TOLERENCE = 10
Y_TOLERENCE = 20


def normalize(val, maxv):
    return round(val / maxv, 4)


def contains_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def reorder_fields(fields: dict) -> dict:
    """
    Réordonne les champs :
    1️⃣ *_fr
    2️⃣ *_ar
    3️⃣ le reste
    """
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


def main():
    img = cv2.imread(IMAGE_PATH)
    h, w = img.shape[:2]

    blocks = easyocr_full(img)
    lines = group_blocks_by_line(blocks)

    # 🔴 Suppression header (réglé UNE seule fois)
    content_lines = lines[2:]

    mapping = [
        {"field": "prenom_ar", "rule": "value_ar"},
        {"field": "prenom_fr", "rule": "value_fr"},
        {"field": "nom_ar", "rule": "value_ar"},
        {"field": "nom_fr", "rule": "value_fr"},
        {"field": "date_naissance", "rule": "key_fr_value_ar", "margin_left": 200, "margin_right": 250},
        {"field": "lieu_naissance_ar", "rule": "value_ar", "margin_right": 47},
        {"field": "lieu_naissance_fr", "rule": "value_fr", "margin_left": 30},
        # dernière ligne construite manuellement
        {"field": "cin", "rule": "manual"},
        {"field": "date_expiration", "rule": "manual", "margin_left": 10}
    ]

    template = {
        "document": "CIN_MAROC",
        "width": w,
        "height": h,
        "fields": {}
    }

    # 🔵 Pré-calcul des bornes Y par ligne
    line_bounds = []
    for line in content_lines:
        ys = [b["y"] for b in line]
        ye = [b["y"] + b["height"] for b in line]
        line_bounds.append((min(ys), max(ye)))

    for i, item in enumerate(mapping[:-2]):
        field = item["field"]
        rule = item["rule"]
        margin_left = item.get("margin_left", 0)
        margin_right = item.get("margin_right", 0)

        line = content_lines[i]
        xs = [b["x"] for b in line]
        ys = [b["y"] for b in line]
        xe = [b["x"] + b["width"] for b in line]
        ye = [b["y"] + b["height"] for b in line]

        detected_x_min = min(xs)
        detected_x_max = max(xe)
        detected_y_min = min(ys)
        detected_y_max = max(ye)

        full_text = " ".join(b["text"] for b in line)

        if rule == "value_ar":
            is_ar = True
        elif rule == "value_fr":
            is_ar = False
        elif rule == "key_fr_value_ar":
            is_ar = True  # le champ final est AR
        else:
            is_ar = contains_arabic(full_text)

        # =====================
        # 📐 EXTENSION X
        # =====================
        if rule == "key_fr_value_ar":
            # on prend toute la ligne (clé + valeur)
            x_min = detected_x_min - X_TOLERENCE
            x_max = w - X_TOLERENCE

        elif rule == "value_ar":
            x_min = int(w * 3 / 5)
            x_max = w - X_TOLERENCE

        else:  # value_fr
            x_min = detected_x_min - X_TOLERENCE
            x_max = int(w * 3 / 4)

        # ✨ APPLICATION DES MARGES PERSONNALISÉES
        x_min = x_min + margin_left
        x_max = x_max - margin_right

        # =====================
        # 📐 EXTENSION Y
        # =====================
        if i > 0:
            y_min = line_bounds[i - 1][1]
        else:
            y_min = detected_y_min

        if is_ar:
            y_min = y_min - int(Y_TOLERENCE/4)
        else :
            y_min = y_min - Y_TOLERENCE

        if i < len(line_bounds) - 1:
            y_max = line_bounds[i + 1][0]
        else:
            y_max = detected_y_max

        if is_ar:
            y_max = y_max + Y_TOLERENCE
        # sécurité
        if y_max <= y_min:
            y_min, y_max = detected_y_min, detected_y_max

        ww = x_max - x_min
        hh = y_max - y_min

        # Sécurité : vérifier que la largeur reste positive
        if ww <= 0:
            print(f"⚠️ Warning: {field} - largeur négative ou nulle après marges. Marges ignorées.")
            x_min = detected_x_min - X_TOLERENCE
            x_max = detected_x_max + X_TOLERENCE
            ww = x_max - x_min

        # 🟥 DEBUG RECTANGLE
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(
            img,
            field,
            (x_min + 5, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        template["fields"][field] = {
            "x": normalize(x_min, w),
            "y": normalize(y_min, h),
            "w": normalize(ww, w),
            "h": normalize(hh, h),
            "lang": "ar" if is_ar else "fr"
        }

    # ============================
    # 🟣 DERNIÈRE LIGNE (MANUELLE)
    # ============================

    # Récupérer les marges pour CIN et date_expiration
    cin_item = mapping[-2]
    exp_item = mapping[-1]
    cin_margin_left = cin_item.get("margin_left", 0)
    cin_margin_right = cin_item.get("margin_right", 0)
    exp_margin_left = exp_item.get("margin_left", 0)
    exp_margin_right = exp_item.get("margin_right", 0)

    # 🔹 paramètres métier
    Y_START_PERCENT = 0.88  # ← ajuste si besoin (ex: 0.85 / 0.9)
    y_min = int(h * Y_START_PERCENT)
    y_max = h

    mid_x = int(w / 2)

    # --------
    # CIN (gauche)
    # --------
    cin_x_min = 3 * X_TOLERENCE + int(w / 10)
    cin_x_max = mid_x - X_TOLERENCE - int(w / 5)

    # ✨ APPLICATION DES MARGES POUR CIN
    cin_x_min = cin_x_min + cin_margin_left
    cin_x_max = cin_x_max - cin_margin_right

    cin_width = cin_x_max - cin_x_min
    if cin_width <= 0:
        print(f"⚠️ Warning: CIN - largeur négative après marges. Marges ignorées.")
        cin_x_min = 3 * X_TOLERENCE + int(w / 10)
        cin_x_max = mid_x - X_TOLERENCE - int(w / 5)
        cin_width = cin_x_max - cin_x_min

    template["fields"]["cin"] = {
        "x": normalize(cin_x_min, w),
        "y": normalize(y_min, h),
        "w": normalize(cin_width, w),
        "h": normalize(y_max - y_min - 2 * Y_TOLERENCE, h),
        "lang": "fr"
    }

    cv2.rectangle(img, (cin_x_min, y_min), (cin_x_max, y_max), (255, 0, 0), 2)
    cv2.putText(
        img,
        "cin",
        (cin_x_min + 5, y_min - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

    # --------
    # DATE EXPIRATION (droite)
    # --------
    exp_x_min = mid_x + X_TOLERENCE + int(w / 5)
    exp_x_max = w - X_TOLERENCE - int(w / 7)

    # ✨ APPLICATION DES MARGES POUR DATE_EXPIRATION
    exp_x_min = exp_x_min + exp_margin_left
    exp_x_max = exp_x_max - exp_margin_right

    exp_width = exp_x_max - exp_x_min
    if exp_width <= 0:
        print(f"⚠️ Warning: date_expiration - largeur négative après marges. Marges ignorées.")
        exp_x_min = mid_x + X_TOLERENCE + int(w / 5)
        exp_x_max = w - X_TOLERENCE - int(w / 7)
        exp_width = exp_x_max - exp_x_min

    template["fields"]["date_expiration"] = {
        "x": normalize(exp_x_min, w),
        "y": normalize(y_min, h),
        "w": normalize(exp_width, w),
        "h": normalize(y_max - y_min - 2 * Y_TOLERENCE, h),
        "lang": "fr"
    }

    cv2.rectangle(img, (exp_x_min, y_min), (exp_x_max, y_max), (255, 0, 0), 2)
    cv2.putText(
        img,
        "date_expiration",
        (exp_x_min + 5, y_min - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

    cv2.imwrite(DEBUG_IMG, img)

    # 🔁 Réordonner les champs avant écriture
    template["fields"] = reorder_fields(template["fields"])

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print("✅ Template généré avec extension intelligente + rectangles + marges personnalisées :", DEBUG_IMG)


if __name__ == "__main__":
    main()