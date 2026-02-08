import cv2
import pytesseract
import easyocr
from utils.geometry import y_center, compute_dynamic_y_tolerance
from typing import List, Dict
import numpy as np

def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY_INV)
    return thresh, img


def extract_tesseract_blocks(image):
    data = pytesseract.image_to_data(
        image,
        config="--oem 3 --psm 6 -l fra+ara",
        output_type=pytesseract.Output.DATAFRAME
    )

    data = data.dropna(subset=["text"])
    data = data[data.text.str.strip() != ""]

    blocks = []
    for _, r in data.iterrows():
        blocks.append({
            "text": r["text"],
            "x": int(r["left"]),
            "y": int(r["top"]),
            "width": int(r["width"]),
            "height": int(r["height"]),
            "confidence": int(r["conf"]) / 100
        })
    return blocks


def extract_text_tesseract(zone: np.ndarray) -> List[Dict]:
    """
    Extraction de texte avec Tesseract OCR

    Args:
        zone: Zone d'image prétraitée (numpy array)

    Returns:
        Liste de dictionnaires avec 'text' et 'confidence'
    """
    try:
        import pytesseract
        from PIL import Image

        # Convertir numpy array en PIL Image
        pil_image = Image.fromarray(zone)

        # Extraire avec données détaillées
        data = pytesseract.image_to_data(
            pil_image,
            lang='eng+ara',
            output_type=pytesseract.Output.DICT
        )

        blocks = []
        for i in range(len(data['text'])):
            text = str(data['text'][i]).strip()
            conf = data['conf'][i]

            if text and conf > 0:
                blocks.append({
                    'text': text,
                    'confidence': float(conf)
                })

        return blocks

    except ImportError:
        print("⚠️  pytesseract non installé. Installation: pip install pytesseract")
        print("    Tesseract OCR doit aussi être installé sur le système")
        return []
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction Tesseract: {e}")
        return []

def group_blocks_by_line(blocks):
    tol = compute_dynamic_y_tolerance(blocks)
    blocks = sorted(blocks, key=lambda b: y_center(b))

    lines = []
    for b in blocks:
        placed = False
        for line in lines:
            avg = sum(y_center(x) for x in line) / len(line)
            if abs(y_center(b) - avg) <= tol:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    for l in lines:
        l.sort(key=lambda x: x["x"])

    return lines


def easyocr_full(image):
    reader = easyocr.Reader(["en", "ar"], gpu=False)
    results = reader.readtext(image)

    blocks = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        blocks.append({
            "text": text,
            "x": int(min(xs)),
            "y": int(min(ys)),
            "width": int(max(xs) - min(xs)),
            "height": int(max(ys) - min(ys)),
            "confidence": conf
        })
    return blocks

def extract_text_tesseract(image):
    custom_config = r'--oem 3 --psm 6 -l fra+ara'
    data = pytesseract.image_to_data(
        image,
        config=custom_config,
        output_type=pytesseract.Output.DATAFRAME
    )
    #data["text"] = data["text"].astype(str)
    data = data.dropna(subset=["text"])

    words = []
    for _, row in data.iterrows():
        words.append({
            "text": row["text"],
            "confidence": int(row["conf"])
        })

    return words


def extract_text_tesseract_pos(image):
    custom_config = r'--oem 3 --psm 6 -l fra+ara'
    data = pytesseract.image_to_data(
        image,
        config=custom_config,
        output_type=pytesseract.Output.DATAFRAME
    )
    # On supprime les lignes sans texte
    data = data.dropna(subset=["text"])

    words = []
    for _, row in data.iterrows():
        # On ne garde que les mots avec un texte non vide
        if str(row["text"]).strip() != "":
            words.append({
                "text": row["text"],
                "confidence": int(row["conf"]),
                "x": int(row["left"]),
                "y": int(row["top"]),
                "width": int(row["width"]),
                "height": int(row["height"])
            })

    return words

