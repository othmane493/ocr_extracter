"""
Extracteur pour les cartes grises marocaines (Recto et Verso)
Intègre le code d'extraction existant
"""
import os
import sys
import json
import time
import re
import difflib
import unicodedata
import random
from difflib import SequenceMatcher

import pytesseract
import cv2
import easyocr

# Import du transformateur JSON
try:
    from json_transformer import transform_json
except ImportError:
    # Fallback si le module n'est pas trouvé
    def transform_json(data):
        return data


class CarteGriseExtractor:
    """Classe pour l'extraction de données des cartes grises"""

    # Configuration commune
    THRESHOLD_EASYOCR = 0.60
    MAX_ITEMS_EASY_OCR = 10

    # Configuration Recto
    FIELDS_KEY_RECTO = {
        "registration_number": {
            "fr": "Numéro d'immatriculation",
            "ar": "رقم التسجيل",
            "multi_line": False
        },
        "previous_registration": {
            "fr": "Immatriculation antérieure",
            "ar": "الترقيم السابق",
            "multi_line": False
        },
        "first_registration_date": {
            "fr": "Première mise en circulation",
            "ar": "أول شروع في الإستخدام",
            "multi_line": False
        },
        "first_usage_date": {
            "fr": "M.C au maroc",
            "ar": "أول استخدام بالمغرب",
            "multi_line": False
        },
        "date_mutation": {
            "fr": "Mutation le",
            "ar": "تحويل بتاريخ",
            "multi_line": False
        },
        "usage": {
            "fr": "Usage",
            "ar": "نوع الإستعمال",
            "multi_line": False
        },
        "owner": {
            "fr": "Propriétaire",
            "ar": "المالك",
            "multi_line": True
        },
        "address": {
            "fr": "Adresse",
            "ar": "العنوان",
            "multi_line": True
        },
        "expiry_date": {
            "fr": "Fin de validité",
            "ar": "نهاية الصلاحية",
            "multi_line": False
        }
    }

    CONF_RECTO = {
        "FIELDS_KEY": FIELDS_KEY_RECTO,
        "DELETE_LINES_BEFORE_Y": 200,
        "DELETE_LINES_AFTER_Y": 900,
        "SPACES_COUNT": 2
    }

    # Configuration Verso
    FIELDS_KEY_VERSO = {
        "Marque": {
            "fr": "Marque",
            "ar": "الاسم التجاري",
            "multi_line": False
        },
        "Type": {
            "fr": "Type",
            "ar": "الصنف",
            "multi_line": False,
            "enable_easy_ocr": False
        },
        "Genre": {
            "fr": "Genre",
            "ar": "النوع",
            "multi_line": False
        },
        "Modèle": {
            "fr": "Modèle",
            "ar": "النموذج",
            "multi_line": False
        },
        "Type_Carburant": {
            "fr": "Type carburant",
            "ar": "نوع الوقود",
            "multi_line": False
        },
        "Number_chassis": {
            "fr": "N° du chassis",
            "ar": "رقم الإطار الحديدي",
            "multi_line": False
        },
        "Number_Cylinders": {
            "fr": "Nombre de cylindres",
            "ar": "عدد الأسطوانات",
            "multi_line": False
        },
        "Puissance_Fiscale": {
            "fr": "Puissance fiscale",
            "ar": "القوة الجبائية",
            "multi_line": False
        },
        "Number_Places": {
            "fr": "Nombre de places",
            "ar": "عدد المقاعد",
            "multi_line": False
        },
        "P.T.A.C": {
            "fr": "P.T.A.C",
            "ar": "الوزن جمالي",
            "multi_line": False
        },
        "Poids_vide": {
            "fr": "Poids à vide",
            "ar": "الوزن الفارغ",
            "multi_line": False
        },
        "P.T.R.A": {
            "fr": "P.T.R.A",
            "ar": "الوزن الإجمالي مع المجرور",
            "multi_line": False
        },
        "Restrictions": {
            "fr": "Restrictions",
            "ar": "التقييدات",
            "multi_line": False
        }
    }

    CONF_VERSO = {
        "FIELDS_KEY": FIELDS_KEY_VERSO,
        "DELETE_LINES_BEFORE_Y": 1,
        "DELETE_LINES_AFTER_Y": 950,
        "SPACES_COUNT": 1.8
    }

    def __init__(self):
        """Initialise l'extracteur de carte grise"""
        self.reader = None  # Sera récupéré du singleton
        self.config = None

    def _get_easyocr_reader(self):
        """Récupère le reader EasyOCR depuis le singleton"""
        if self.reader is None:
            try:
                from ocr_manager import get_easyocr_reader
                self.reader = get_easyocr_reader()
            except ImportError:
                # Fallback si ocr_manager n'est pas disponible
                import easyocr
                self.reader = easyocr.Reader(["en", "ar"], gpu=False)
        return self.reader

    def detect_config_by_fr_fields(self, merged_lines, min_fields=3, threshold=0.6):
        """Détecte automatiquement si c'est un recto ou verso"""
        recto_found = set()
        verso_found = set()

        for line in merged_lines:
            for block in line:
                text = block.get("text", "")
                if not text:
                    continue

                # Check RECTO
                for field_name, field in self.CONF_RECTO["FIELDS_KEY"].items():
                    if field_name not in recto_found:
                        if SequenceMatcher(None, text, field["fr"]).ratio() >= threshold:
                            recto_found.add(field_name)
                            if len(recto_found) >= min_fields:
                                return self.CONF_RECTO

                # Check VERSO
                for field_name, field in self.CONF_VERSO["FIELDS_KEY"].items():
                    if field_name not in verso_found:
                        if SequenceMatcher(None, text, field["fr"]).ratio() >= threshold:
                            verso_found.add(field_name)
                            if len(verso_found) >= min_fields:
                                return self.CONF_VERSO

        # Par défaut
        return self.CONF_RECTO

    @staticmethod
    def preprocess_image(image_path):
        """Prétraite l'image pour améliorer la qualité OCR"""
        image_original = cv2.imread(image_path)
        if image_original is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")

        gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 148, 253, cv2.THRESH_BINARY_INV)

        return thresh, image_original

    @staticmethod
    def extract_text(image):
        """Utilise Tesseract pour extraire le texte"""
        custom_config = r'--oem 3 --psm 6 -l fra+ara'
        data = pytesseract.image_to_data(
            image,
            config=custom_config,
            output_type=pytesseract.Output.DATAFRAME
        )

        # Nettoyage
        data = data.dropna(subset=["text"])
        data = data[data.text.str.strip() != ""]

        words = []
        for _, row in data.iterrows():
            words.append({
                "text": row["text"],
                "x": int(row["left"]),
                "y": int(row["top"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "confidence": int(row["conf"])
            })

        return words

    @staticmethod
    def clean_invisible_chars(text):
        """Nettoie les caractères invisibles"""
        if not text:
            return text
        return "".join(
            ch for ch in text
            if unicodedata.category(ch) != "Cf"
        ).strip()

    @staticmethod
    def compute_dynamic_y_tolerance(blocks, ratio=0.5, min_tol=6, max_tol=30):
        """Calcule la tolérance Y dynamique pour le groupement de lignes"""
        heights = [b["height"] for b in blocks if b["height"] > 0]
        if not heights:
            return 10
        avg_height = sum(heights) / len(heights)
        y_tol = int(avg_height * ratio)
        return max(min_tol, min(y_tol, max_tol))

    @staticmethod
    def y_center(b):
        """Retourne le centre vertical d'un bloc"""
        return b["y"] + b["height"] / 2

    def group_blocks_by_line(self, blocks, y_tolerance):
        """Groupe les blocs par ligne"""
        blocks = sorted(blocks, key=lambda b: self.y_center(b))

        lines = []
        for block in blocks:
            placed = False
            block["text"] = self.clean_invisible_chars(block["text"])
            block["confidence"] = block["confidence"] / 100
            b_center = self.y_center(block)

            for line in lines:
                line_centers = [self.y_center(b) for b in line]
                avg_center = sum(line_centers) / len(line_centers)

                if abs(b_center - avg_center) <= y_tolerance:
                    line.append(block)
                    placed = True
                    break

            if not placed:
                lines.append([block])

        # Trier chaque ligne horizontalement
        for line in lines:
            line.sort(key=lambda b: b["x"])

        return lines

    @staticmethod
    def is_arabic(text):
        """Vérifie si le texte contient de l'arabe"""
        if not isinstance(text, str):
            return False
        return bool(re.search(r'[\u0600-\u06FF]', text))

    def merge_blocks_line_by_gap(self, line):
        """Fusionne les blocs d'une ligne selon les espaces"""
        if not line:
            return []

        spaces_count = self.config["SPACES_COUNT"]
        line = sorted(line, key=lambda b: b["x"])

        total_width = sum(b["width"] for b in line)
        total_chars = sum(len(b["text"].strip()) for b in line if len(b["text"].strip()) > 0)
        avg_char_width = total_width / total_chars if total_chars > 0 else 5
        x_merge_gap = spaces_count * avg_char_width

        merged_line = []
        current = line[0].copy()

        for b in line[1:]:
            gap = b["x"] - (current["x"] + current["width"])
            if gap <= x_merge_gap:
                if self.is_arabic(current["text"]) or self.is_arabic(b["text"]):
                    current["text"] = b["text"] + " " + current["text"]
                else:
                    current["text"] += " " + b["text"]
                current["width"] = (b["x"] + b["width"]) - current["x"]
                current["height"] = max(current["height"], b["height"])
                current["confidence"] = (current["confidence"] + b["confidence"]) / 2
            else:
                merged_line.append(current)
                current = b.copy()

        merged_line.append(current)
        return merged_line

    def get_best_field_match(self, text, fields, lang, threshold=0.6):
        """Retourne le champ qui correspond le mieux au texte"""
        best_match = None
        best_ratio = 0.0

        for field_name, labels in fields.items():
            label = labels.get(lang, "")
            ratio = SequenceMatcher(None, text, label).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = field_name

        if best_ratio >= threshold:
            return best_match
        return None

    def mark_fields_in_blocks(self, merged_lines, fields, threshold=0.6):
        """Marque les blocs qui correspondent à des champs"""
        for line in merged_lines:
            cut_from = None
            cut_to = None

            for idy, block in enumerate(line):
                lang = "fr"
                field_name = self.get_best_field_match(block["text"], fields, lang, threshold)

                if field_name is None:
                    lang = "ar"
                    field_name = self.get_best_field_match(block["text"], fields, lang, threshold)

                if field_name:
                    block["text"] = fields[field_name][lang]
                    block["is_key"] = True
                    block["field_name"] = field_name
                    block["langage"] = lang
                    block["confidence"] = 0.9

                    if self.is_arabic(block["text"]):
                        cut_from = idy + 1
                    else:
                        cut_to = idy

                    break
                else:
                    block["is_key"] = False

            # Nettoyage après la boucle
            if cut_from is not None:
                del line[cut_from:]

            if cut_to is not None:
                del line[:cut_to]

        return merged_lines

    @staticmethod
    def has_meaningful_char(text):
        """Vérifie si le texte contient des caractères significatifs"""
        return bool(re.search(r"[A-Za-z0-9\u0600-\u06FF]", text))

    def filter_lines(self, merged_lines):
        """Filtre les lignes selon la configuration"""
        y_min = self.config["DELETE_LINES_BEFORE_Y"]
        y_max = self.config["DELETE_LINES_AFTER_Y"]

        filtered_lines = []

        for line in merged_lines:
            if not line:
                continue

            if not (y_min <= line[0]["y"] <= y_max):
                continue

            cleaned_line = []
            for block in line:
                text = block.get("text", "").strip()

                if block.get("is_key", False):
                    cleaned_line.append(block)
                    continue

                if not self.has_meaningful_char(text):
                    continue

                cleaned_line.append(block)

            if cleaned_line:
                filtered_lines.append(cleaned_line)

        return filtered_lines

    def is_field_key_similar(self, text, ratio=0.65):
        """Vérifie si le texte est similaire à une clé de champ"""
        for field_key in self.config["FIELDS_KEY"].values():
            if difflib.SequenceMatcher(None, field_key["fr"], text).ratio() >= ratio or \
               difflib.SequenceMatcher(None, field_key["ar"], text).ratio() >= ratio:
                return True
        return False

    def get_safe_block_xy(self, line_idx, block_idx, lines, img_w, img_h):
        """Étend le bloc pour améliorer l'OCR"""
        block = lines[line_idx][block_idx]

        # Safe Y (vertical)
        block_y = int(block["y"])
        block_h = int(block["height"])
        block_y_max = block_y + block_h

        if line_idx > 0:
            prev_line = lines[line_idx - 1]
            y_prev_max = max(int(b["y"] + b["height"]) for b in prev_line)
        else:
            y_prev_max = 0

        if line_idx < len(lines) - 1:
            next_line = lines[line_idx + 1]
            y_next_min = min(int(b["y"]) for b in next_line)
        else:
            y_next_min = img_h - 10

        y_safe = min(block_y, y_prev_max)
        y_bottom = max(block_y_max, y_next_min)
        y_safe = max(0, y_safe)
        y_bottom = min(img_h, y_bottom)
        h_safe = max(1, y_bottom - y_safe)

        # Safe X (horizontal)
        line = lines[line_idx]
        block_x = int(block["x"])
        block_w = int(block["width"])
        block_x_max = block_x + block_w

        if block_idx > 0:
            prev_block = line[block_idx - 1]
            x_prev_max = int(prev_block["x"] + prev_block["width"])
        else:
            x_prev_max = 0

        if block_idx < len(line) - 1:
            next_block = line[block_idx + 1]
            x_next_min = int(next_block["x"])
        else:
            x_next_min = img_w

        x_safe = min(block_x, x_prev_max)
        x_right = max(block_x_max, x_next_min)
        x_safe = max(0, x_safe)
        x_right = min(img_w, x_right)
        w_safe = max(1, x_right - x_safe)

        return x_safe, y_safe, w_safe, h_safe

    def correct_suspicious_blocks(self, img, lines, max_item):
        """Corrige les blocs suspects avec EasyOCR"""
        reader = self._get_easyocr_reader()

        img_h, img_w = img.shape[:2]
        y_min = self.config["DELETE_LINES_BEFORE_Y"]
        y_max = self.config["DELETE_LINES_AFTER_Y"]
        counter_item = 0

        for line_idx, line in enumerate(lines):
            if y_min < line[0]["y"] < y_max:
                for block_idx, block in enumerate(line):
                    if counter_item >= max_item:
                        break

                    conf = block.get("confidence", 1)
                    if conf >= self.THRESHOLD_EASYOCR:
                        continue

                    if ("is_key" in block.keys() and block["is_key"]) or self.is_field_key_similar(block["text"]):
                        continue

                    x_safe, y_safe, w_safe, h_safe = self.get_safe_block_xy(
                        line_idx, block_idx, lines, img_w, img_h
                    )

                    x1 = max(0, x_safe)
                    y1 = max(0, y_safe)
                    x2 = min(img_w, x_safe + w_safe)
                    y2 = min(img_h, y_safe + h_safe)

                    cropped = img[y1:y2, x1:x2]

                    if cropped.size == 0:
                        continue

                    # OCR avec EasyOCR partagé
                    results = reader.readtext(cropped)
                    if not results:
                        continue

                    bbox, text, new_conf = results[0]

                    # Reprojection des coordonnées
                    new_x = int(bbox[0][0]) + x1
                    new_y = int(bbox[0][1]) + y1
                    new_w = int(bbox[1][0] - bbox[0][0])
                    new_h = int(bbox[2][1] - bbox[1][1])

                    block.update({
                        "text": text.strip(),
                        "x": new_x,
                        "y": new_y,
                        "width": new_w,
                        "height": new_h,
                        "confidence": new_conf
                    })
                    counter_item += 1

        return lines

    def get_block_pos_from_line(self, line, key, ratio=0.65):
        """Trouve la position d'un bloc dans une ligne"""
        if line is None or key is None:
            return -1
        for idx, block in enumerate(line):
            if difflib.SequenceMatcher(None, block["text"].lower(), key.lower()).ratio() >= ratio:
                return idx
        return -1

    def has_field_key(self, line):
        """Vérifie si la ligne contient une clé de champ"""
        for block in line:
            if self.is_field_key_similar(block["text"]):
                return True
        return False

    def next_value_after_x_key(self, pos_x_current_line, next_line, image_width, y_tolerance=20):
        """Trouve la prochaine valeur après une clé (pour FR)"""
        mid_x = image_width / 2

        for block in next_line:
            if block["x"] > (pos_x_current_line + y_tolerance):
                if block["x"] < mid_x:
                    return block["text"]
        return ""

    def next_value_before_x_key(self, pos_x_current_line, width, next_line, image_width, y_tolerance=20):
        """Trouve la prochaine valeur avant une clé (pour AR)"""
        mid_x = image_width / 2

        for block in next_line[::-1]:
            if block["x"] < (pos_x_current_line + width - y_tolerance):
                if (block["x"] + block["width"]) > mid_x:
                    return block["text"]
        return ""

    def extract_value(self, merge_lines, idx, pos_fr, pos_ar, multiline, lang, image_width):
        """Extrait la valeur d'un champ"""
        value = ""
        current_line = merge_lines[idx]
        confidence = 0
        x = y = width = height = 0

        block_result = {
            "text": value,
            "confidence": confidence,
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }

        if lang == "fr":
            if pos_fr != -1 and (pos_fr + 1) < len(current_line):
                if not self.is_field_key_similar(current_line[pos_fr + 1]["text"]):
                    value = current_line[pos_fr + 1]["text"]
                    x = current_line[pos_fr + 1]["x"]
                    y = current_line[pos_fr + 1]["y"]
                    width = current_line[pos_fr + 1]["width"]
                    height = current_line[pos_fr + 1]["height"]
                    confidence = current_line[pos_fr + 1]["confidence"]
                else:
                    value = ""
            else:
                return block_result
        else:
            if pos_ar != -1 and (pos_ar - 1) < len(current_line):
                if not self.is_field_key_similar(current_line[pos_ar - 1]["text"]):
                    value = current_line[pos_ar - 1]["text"]
                    x = current_line[pos_ar - 1]["x"]
                    y = current_line[pos_ar - 1]["y"]
                    width = current_line[pos_ar - 1]["width"]
                    height = current_line[pos_ar - 1]["height"]
                    confidence = current_line[pos_ar - 1]["confidence"]
                else:
                    value = ""
            else:
                return block_result

        # Gestion multiline
        i = 1
        while multiline and (idx + i) < len(merge_lines) and not self.has_field_key(merge_lines[idx + i]):
            if lang == "fr" and pos_fr != -1:
                value += " " + self.next_value_after_x_key(current_line[pos_fr]["x"], merge_lines[idx + i], image_width)
            elif lang == "ar" and pos_ar != -1:
                value += " " + self.next_value_before_x_key(
                    current_line[pos_ar]["x"],
                    current_line[pos_ar]["width"],
                    merge_lines[idx + i],
                    image_width
                )
            i += 1

        block_result = {
            "text": value,
            "confidence": confidence,
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }
        return block_result

    def parse_text(self, merge_lines, image_width):
        """Parse le texte pour extraire les champs"""
        result = []
        fields_key = self.config["FIELDS_KEY"]

        for idx, line in enumerate(merge_lines):
            for key_data in fields_key.keys():
                pos_fr = self.get_block_pos_from_line(line, fields_key[key_data]["fr"])
                pos_ar = self.get_block_pos_from_line(line, fields_key[key_data]["ar"])

                if pos_fr != -1 or pos_ar != -1:
                    block_value_fr = self.extract_value(
                        merge_lines, idx, pos_fr, pos_ar,
                        fields_key[key_data]["multi_line"], "fr", image_width
                    )
                    block_value_ar = self.extract_value(
                        merge_lines, idx, pos_fr, pos_ar,
                        fields_key[key_data]["multi_line"], "ar", image_width
                    )

                    item_result = {
                        "fr": {
                            "key": fields_key[key_data]["fr"] if pos_fr != -1 else None,
                            "value": block_value_fr["text"],
                            "confidence": int(block_value_fr["confidence"] * 100),
                            "x": int(block_value_fr["x"]),
                            "y": int(block_value_fr["y"]),
                            "width": int(block_value_fr["width"]),
                            "height": int(block_value_fr["height"])
                        },
                        "ar": {
                            "key": fields_key[key_data]["ar"] if pos_ar != -1 else None,
                            "value": block_value_ar["text"],
                            "confidence": int(block_value_ar["confidence"] * 100),
                            "x": int(block_value_ar["x"]),
                            "y": int(block_value_ar["y"]),
                            "width": int(block_value_ar["width"]),
                            "height": int(block_value_ar["height"])
                        }
                    }

                    result.append(item_result)
                    break

        return result

    def extract(self, image_path, document_type):
        """
        Méthode principale d'extraction

        Args:
            image_path: Chemin vers l'image
            document_type: Type de document (carte_grise_recto ou carte_grise_verso)

        Returns:
            Dictionnaire contenant les données extraites au format unifié
        """
        start_time = time.time()

        # Configuration selon le type
        if document_type == 'carte_grise_recto':
            self.config = self.CONF_RECTO
        elif document_type == 'carte_grise_verso':
            self.config = self.CONF_VERSO
        else:
            raise ValueError(f"Type de document invalide: {document_type}")

        # Prétraitement
        preprocessed_image, image_original = self.preprocess_image(image_path)
        _, image_width = image_original.shape[:2]

        # Extraction Tesseract
        blocks = self.extract_text(preprocessed_image)

        # Groupement par lignes
        dynamic_y_tol = self.compute_dynamic_y_tolerance(blocks)
        blocks = self.group_blocks_by_line(blocks, y_tolerance=dynamic_y_tol)

        # Fusion des blocs
        merged_lines = [self.merge_blocks_line_by_gap(line) for line in blocks]

        # Marquage des champs
        fields_key = self.config["FIELDS_KEY"]
        line_marked_fields = self.mark_fields_in_blocks(merged_lines, fields_key, threshold=0.6)

        # Correction EasyOCR
        lines_with_easy_ocr = self.correct_suspicious_blocks(
            image_original,
            line_marked_fields,
            self.MAX_ITEMS_EASY_OCR
        )

        # Filtrage
        lines_with_easy_ocr = self.filter_lines(lines_with_easy_ocr)
        lines_with_easy_ocr = self.mark_fields_in_blocks(lines_with_easy_ocr, fields_key, threshold=0.6)

        # Parsing
        parsed_data = self.parse_text(lines_with_easy_ocr, image_width)

        # Transformation au format unifié
        transformed_data = transform_json(parsed_data)

        elapsed = time.time() - start_time

        return transformed_data