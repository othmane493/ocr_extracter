import cv2
import json
import re
import unicodedata
from difflib import SequenceMatcher

import easyocr


FIELDS_KEY_RECTO = {
    "registration_number_matriculate": {
        "fr": "Numéro d'immatriculation",
        "ar": "رقم التسجيل",
        "normalise": "0-9أبدهوهـط-",
        "multi_line": False,
        "extract_ar": True,
        "type": "center_value"
    },
    "previous_registration": {
        "fr": "Immatriculation antérieure",
        "ar": "الترقيم السابق",
        "normalise": "A-Z0-9",
        "multi_line": False,
        "type": "center_value"
    },
    "first_registration_date": {
        "fr": "Première mise en circulation",
        "ar": "أول شروع في الإستخدام",
        "normalise": "0-9/",
        "multi_line": False,
        "type": "center_value"
    },
    "first_usage_date": {
        "fr": "M.C au maroc",
        "ar": "أول استخدام بالمغرب",
        "normalise": "0-9/",
        "multi_line": False,
        "type": "center_value"
    },
    "mutation_date": {
        "fr": "Mutation le",
        "ar": "تحويل بتاريخ",
        "normalise": "0-9/",
        "multi_line": False,
        "type": "center_value"
    },
    "usage": {
        "fr": "Usage",
        "ar": "نوع الإستعمال",
        "normalise": "A-Za-zÀ-ÿ0-9\\s_-",
        "multi_line": False,
        "type": "center_value"
    },
    "owner": {
        "fr": "Propriétaire",
        "ar": "المالك",
        "normalise": "A-Za-z\u0600-\u06FF\\s",
        "multi_line": True,
        "type": "owner_split"
    },
    "address": {
        "fr": "Adresse",
        "ar": "العنوان",
        "normalise": "A-Za-z0-9\u0600-\u06FF\\s",
        "multi_line": True,
        "type": "address_one_zone"
    },
    "expiry_date": {
        "fr": "Fin de validité",
        "ar": "نهاية الصلاحية",
        "normalise": "0-9/",
        "multi_line": False,
        "type": "center_value"
    }
}


FIELDS_KEY_VERSO = {
    "Marque": {
        "fr": "Marque",
        "ar": "الاسم التجاري",
        "normalise": "A-Za-z\\s_-",
        "multi_line": False,
        "type": "center_value"
    },
    "Type": {
        "fr": "Type",
        "ar": "الصنف",
        "normalise": "A-Z0-9",
        "multi_line": False,
        "enable_easy_ocr": False,
        "type": "center_value"
    },
    "Genre": {
        "fr": "Genre",
        "ar": "النوع",
        "normalise": "A-Za-z_\\s-",
        "multi_line": False,
        "type": "center_value"
    },
    "Modèle": {
        "fr": "Modèle",
        "ar": "النموذج",
        "normalise": "A-Za-z_\\s-",
        "multi_line": False,
        "type": "center_value"
    },
    "Type_Carburant": {
        "fr": "Type carburant",
        "ar": "نوع الوقود",
        "normalise": "A-Za-z",
        "multi_line": False,
        "type": "center_value"
    },
    "Number_chassis": {
        "fr": "N° du chassis",
        "ar": "رقم الإطار الحديدي",
        "normalise": "A-Z0-9",
        "multi_line": False,
        "type": "center_value"
    },
    "Number_Cylinders": {
        "fr": "Nombre de cylindres",
        "ar": "عدد الأسطوانات",
        "normalise": "0-9",
        "multi_line": False,
        "type": "center_value"
    },
    "Puissance_Fiscale": {
        "fr": "Puissance fiscale",
        "ar": "القوة الجبائية",
        "normalise": "0-9",
        "multi_line": False,
        "type": "center_value"
    },
    "Number_Places": {
        "fr": "Nombre de places",
        "ar": "عدد المقاعد",
        "normalise": "0-9",
        "multi_line": False,
        "type": "center_value"
    },
    "PTAC": {
        "fr": "P.T.A.C",
        "ar": "الوزن جمالي",
        "normalise": "0-9kKgG\\s",
        "multi_line": False,
        "type": "center_value"
    },
    "Poids_vide": {
        "fr": "Poids à vide",
        "ar": "الوزن الفارغ",
        "normalise": "0-9kKgG\\s",
        "multi_line": False,
        "type": "center_value"
    },
    "PTRA": {
        "fr": "P.T.R.A",
        "ar": "الوزن الإجمالي مع المجرور",
        "normalise": "0-9kKgG\\s-",
        "multi_line": False,
        "type": "center_value"
    },
    "Restrictions": {
        "fr": "Restrictions",
        "ar": "التقييدات",
        "normalise": "A-Za-z\u0600-\u06FF\\s\\-",
        "multi_line": False,
        "type": "center_value"
    }
}


class CarteGriseZoneGenerator:
    def __init__(
        self,
        image_path: str,
        fields_config: dict,
        document_name: str,
        gpu: bool = False,
        reader_langs=None,

    ):
        self.image_path = image_path
        self.fields_config = fields_config
        self.document_name = document_name
        self.reader_langs = reader_langs or ['ar', 'en']
        self.reader = easyocr.Reader(self.reader_langs, gpu=gpu)
        self.ref_w = 996
        self.ref_h = 627
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")

        self.h, self.w = self.img.shape[:2]
        self.ocr_results = []

    # =========================================================
    # Utils
    # =========================================================
    @staticmethod
    def strip_accents(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        return "".join(c for c in text if unicodedata.category(c) != "Mn")

    @staticmethod
    def normalize_arabic(text: str) -> str:
        if not text:
            return ""
        replacements = {
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ٱ": "ا",
            "ى": "ي",
            "ة": "ه",
            "ؤ": "و",
            "ئ": "ي",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()
        text = self.strip_accents(text)
        text = self.normalize_arabic(text)
        text = text.lower()
        text = re.sub(r"[^a-z0-9\u0600-\u06FF\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def clamp(v: int, min_v: int, max_v: int) -> int:
        return max(min_v, min(v, max_v))

    @staticmethod
    def bbox_from_easyocr(points):
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": x2 - x1,
            "h": y2 - y1,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2
        }

    def to_norm(self, box: dict) -> dict:
        return {
            "x": round(box["x1"] / self.w, 4),
            "y": round(box["y1"] / self.h, 4),
            "w": round((box["x2"] - box["x1"]) / self.w, 4),
            "h": round((box["y2"] - box["y1"]) / self.h, 4),
        }

    def resize_to_reference(self):
        """
        Force toutes les cartes sur la taille de référence.
        """
        if self.w != self.ref_w or self.h != self.ref_h:
            self.img = cv2.resize(self.img, (self.ref_w, self.ref_h), interpolation=cv2.INTER_CUBIC)
            self.h, self.w = self.img.shape[:2]

    def add_zone(self, zones: dict, name: str, x1: int, y1: int, x2: int, y2: int, lang: str = "mixed"):
        x1 = self.clamp(x1, 0, self.w - 1)
        y1 = self.clamp(y1, 0, self.h - 1)
        x2 = self.clamp(x2, x1 + 10, self.w - 1)
        y2 = self.clamp(y2, y1 + 10, self.h - 1)

        if (x2 - x1) >= 25 and (y2 - y1) >= 12:
            zones[name] = {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "lang": lang
            }

    # =========================================================
    # OCR
    # =========================================================
    def run_ocr(self):
        raw = self.reader.readtext(self.img, detail=1, paragraph=False)

        results = []
        for points, text, conf in raw:
            box = self.bbox_from_easyocr(points)
            results.append({
                "text": text,
                "norm_text": self.normalize_text(text),
                "conf": conf,
                **box
            })

        results.sort(key=lambda b: (b["cy"], b["x1"]))
        self.ocr_results = results
        return results

    # =========================================================
    # Matching ancres
    # =========================================================
    def find_best_anchor(self, label: str, prefer_side: str = None):
        target = self.normalize_text(label)
        if not target:
            return None

        best = None
        best_score = 0.0

        for box in self.ocr_results:
            candidate = box["norm_text"]
            if not candidate:
                continue

            score = self.similarity(target, candidate)

            if target in candidate or candidate in target:
                score += 0.15

            if prefer_side == "left" and box["cx"] < self.w * 0.45:
                score += 0.05
            elif prefer_side == "right" and box["cx"] > self.w * 0.55:
                score += 0.05

            if score > best_score:
                best_score = score
                best = box

        if best_score < 0.48:
            return None

        return best

    def build_anchor_map(self):
        anchors = {}

        for field_name, meta in self.fields_config.items():
            fr_box = self.find_best_anchor(meta.get("fr", ""), prefer_side="left")
            ar_box = self.find_best_anchor(meta.get("ar", ""), prefer_side="right")

            anchors[field_name] = {
                "fr": fr_box,
                "ar": ar_box,
                "type": meta.get("type", "center_value"),
                "multi_line": meta.get("multi_line", False)
            }

        # ancres de référence pour recentrage
        royaume_box = self.find_best_anchor("ROYAUME DU MAROC", prefer_side="left")
        if royaume_box is not None:
            anchors["royaume_du_maroc"] = {
                "fr": royaume_box,
                "ar": None,
                "type": "reference_anchor",
                "multi_line": False
            }

        return anchors

    # =========================================================
    # Calcul zones
    # =========================================================
    def compute_center_zone(self, field: str, fr_box, ar_box, next_row_top, zones: dict):
        if fr_box is None and ar_box is None:
            return

        boxes = [b for b in [fr_box, ar_box] if b is not None]
        if not boxes:
            return

        raw_top = min(b["y1"] for b in boxes)
        raw_bottom = max(b["y2"] for b in boxes)

        y1 = raw_top - 2

        if next_row_top is not None:
            y2 = min(raw_bottom + 2, next_row_top - 6)
        else:
            y2 = raw_bottom + 2

        if y2 <= y1 + 8:
            y2 = raw_bottom + 2

        if fr_box:
            x1 = fr_box["x2"] + 8
        else:
            x1 = int(self.w * 0.34)

        if ar_box:
            x2 = ar_box["x1"] - 8
        else:
            right_label_box = self.find_right_label_block_on_same_row(fr_box, y_tol=8)
            if right_label_box is not None:
                x2 = right_label_box["x1"] - 8
            else:
                x2 = int(self.w * 0.80)

        # patch manuel final
        x1, y1, x2, y2 = self.apply_manual_zone_fixes(field, x1, y1, x2, y2 + 10)

        zones[field] = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "lang": "mixed"
        }

    def compute_owner_split_zones(self, fr_box, ar_box, address_fr_box, address_ar_box, zones: dict):
        if address_fr_box and address_ar_box:
            address_mid_y = int((address_fr_box["cy"] + address_ar_box["cy"]) / 2)
        elif address_fr_box:
            address_mid_y = int(address_fr_box["cy"])
        elif address_ar_box:
            address_mid_y = int(address_ar_box["cy"])
        else:
            address_mid_y = int(self.h * 0.62)

        if fr_box:
            x1_fr = fr_box["x2"] + 12
            x2_fr = int(self.w * 0.48)
            y1_fr = fr_box["y1"] - 8
            y2_fr = max(fr_box["y2"] + 8, address_mid_y)
            self.add_zone(zones, "owner_fr", x1_fr, y1_fr, x2_fr, y2_fr, "mixed")

        if ar_box:
            x1_ar = int(self.w * 0.54)
            x2_ar = ar_box["x1"] - 10
            y1_ar = ar_box["y1"] - 8
            y2_ar = max(ar_box["y2"] + 8, address_mid_y)
            self.add_zone(zones, "owner_ar", x1_ar, y1_ar, x2_ar, y2_ar, "mixed")

    def compute_address_one_zone(self, fr_box, ar_box, expiry_fr_box, expiry_ar_box, zones: dict):
        if not fr_box and not ar_box:
            return

        if fr_box:
            x1 = fr_box["x1"]
        else:
            x1 = int(self.w * 0.10)

        if ar_box:
            x2 = ar_box["x1"] + 50
        else:
            x2 = int(self.w * 0.86)

        if fr_box and ar_box:
            y1 = min(fr_box["y2"], ar_box["y2"]) - 12
        elif fr_box:
            y1 = fr_box["y2"] - 12
        else:
            y1 = ar_box["y2"] - 12

        expiry_top_candidates = []
        if expiry_fr_box:
            expiry_top_candidates.append(expiry_fr_box["y1"])
        if expiry_ar_box:
            expiry_top_candidates.append(expiry_ar_box["y1"])

        if expiry_top_candidates:
            y2 = min(expiry_top_candidates) + 5
        else:
            if ar_box:
                y2 = ar_box["y2"] + int(self.h * 0.12)
            elif fr_box:
                y2 = fr_box["y2"] + int(self.h * 0.12)
            else:
                y2 = int(self.h * 0.78)

        y2 -= 20
        self.add_zone(zones, "address", x1, y1, x2, y2, "mixed")

    def compute_value_zones(self, anchors: dict):
        zones = {}

        # -------- champs spéciaux recto --------
        owner_fr_box = anchors.get("owner", {}).get("fr")
        owner_ar_box = anchors.get("owner", {}).get("ar")
        address_fr_box = anchors.get("address", {}).get("fr")
        address_ar_box = anchors.get("address", {}).get("ar")
        expiry_fr_box = anchors.get("expiry_date", {}).get("fr")
        expiry_ar_box = anchors.get("expiry_date", {}).get("ar")

        # -------- lignes center_value triées verticalement --------
        center_rows = []
        for field, a in anchors.items():
            if a.get("type") != "center_value":
                continue

            fr_box = a.get("fr")
            ar_box = a.get("ar")
            boxes = [b for b in [fr_box, ar_box] if b is not None]
            if not boxes:
                continue

            row_top = min(b["y1"] for b in boxes)
            row_bottom = max(b["y2"] for b in boxes)
            row_cy = sum(b["cy"] for b in boxes) / len(boxes)

            center_rows.append({
                "field": field,
                "fr": fr_box,
                "ar": ar_box,
                "top": row_top,
                "bottom": row_bottom,
                "cy": row_cy
            })

        center_rows.sort(key=lambda r: r["cy"])

        # -------- zones dynamiques ligne par ligne --------
        for i, row in enumerate(center_rows):
            next_top = None
            if i + 1 < len(center_rows):
                next_top = center_rows[i + 1]["top"]

            self.compute_center_zone(
                field=row["field"],
                fr_box=row["fr"],
                ar_box=row["ar"],
                next_row_top=next_top,
                zones=zones
            )

        # -------- recto : owner --------
        if "owner" in anchors and anchors["owner"].get("type") == "owner_split":
            self.compute_owner_split_zones(
                owner_fr_box,
                owner_ar_box,
                address_fr_box,
                address_ar_box,
                zones
            )

        # -------- recto : address --------
        if "address" in anchors and anchors["address"].get("type") == "address_one_zone":
            self.compute_address_one_zone(
                address_fr_box,
                address_ar_box,
                expiry_fr_box,
                expiry_ar_box,
                zones
            )

        return zones

    def find_right_label_block_on_same_row(self, fr_box, y_tol=8):
        """
        Cherche le premier bloc OCR gris dans la colonne arabe sur la même ligne.
        Sert de borne droite quand ar_box est absente ou mauvaise.
        """
        if fr_box is None:
            return None

        ref_cy = fr_box["cy"]
        ref_x2 = fr_box["x2"]

        candidates = []
        for box in self.ocr_results:
            if abs(box["cy"] - ref_cy) > y_tol:
                continue

            if box["x1"] <= ref_x2 + 25:
                continue

            # seulement la colonne droite
            if box["x1"] < self.w * 0.70:
                continue

            text = box.get("text", "") or ""
            has_ar = re.search(r"[\u0600-\u06FF]", text) is not None

            # éviter les grosses boîtes
            if box["w"] > self.w * 0.18:
                continue

            score = 0
            if has_ar:
                score += 10

            # plus proche = mieux
            score -= (box["x1"] - ref_x2) / 20.0

            candidates.append((score, box))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def expand_anchor_horizontally(self, anchor_box, side="left", y_tol=10, gap_tol=8):
        if anchor_box is None:
            return None

        line_y1 = anchor_box["y1"] - y_tol
        line_y2 = anchor_box["y2"] + y_tol

        x1 = anchor_box["x1"]
        y1 = anchor_box["y1"]
        x2 = anchor_box["x2"]
        y2 = anchor_box["y2"]

        changed = True
        while changed:
            changed = False

            for box in self.ocr_results:
                overlap_y = not (box["y2"] < line_y1 or box["y1"] > line_y2)
                if not overlap_y:
                    continue

                if abs(box["cy"] - ((y1 + y2) / 2)) > y_tol:
                    continue

                if box["w"] > self.w * 0.14:
                    continue

                if side == "left":
                    dist = box["x1"] - x2
                    if -3 <= dist <= gap_tol:
                        new_x1 = min(x1, box["x1"])
                        new_y1 = min(y1, box["y1"])
                        new_x2 = max(x2, box["x2"])
                        new_y2 = max(y2, box["y2"])
                        if (new_x1, new_y1, new_x2, new_y2) != (x1, y1, x2, y2):
                            x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
                            changed = True

                elif side == "right":
                    dist = x1 - box["x2"]
                    if -3 <= dist <= gap_tol:
                        new_x1 = min(x1, box["x1"])
                        new_y1 = min(y1, box["y1"])
                        new_x2 = max(x2, box["x2"])
                        new_y2 = max(y2, box["y2"])
                        if (new_x1, new_y1, new_x2, new_y2) != (x1, y1, x2, y2):
                            x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
                            changed = True

        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": x2 - x1,
            "h": y2 - y1,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2
        }

    def apply_manual_zone_fixes(self, field: str, x1: int, y1: int, x2: int, y2: int):
        """
        Corrections manuelles sur image normalisée 1680x1058.
        """
        if field == "expiry_date":
            x2 = x2 - 110
        # Number_chassis : réduire la largeur à droite
        elif field == "Number_chassis":
            x2 = 760

        # Poids_vide : réduire la largeur à droite
        elif field == "Poids_vide":
            x2 = 750
            y1 = int(y1 + ((y2 - y1) / 2))

        # PTRA : réduire la largeur à droite
        elif field == "PTRA":
            x2 = 675

        # PTAC : réduire un peu à droite + réduire hauteur basse
        elif field == "PTAC":
            x2 = 750
            y2 = int(y2 - ((y2 - y1)/2))

        x1 = self.clamp(x1, 0, self.w - 1)
        y1 = self.clamp(y1, 0, self.h - 1)
        x2 = self.clamp(x2, x1 + 20, self.w - 1)
        y2 = self.clamp(y2, y1 + 10, self.h - 1)

        return x1, y1, x2, y2

    # =========================================================
    # Export JSON
    # =========================================================
    def export_json(self, zones: dict, anchors: dict, out_json_path: str):
        data = {
            "document": self.document_name,
            "width": self.w,
            "height": self.h,
            "fields": {},
            "anchors": {}
        }

        for field_name, box in zones.items():
            norm = self.to_norm(box)
            data["fields"][field_name] = {
                **norm,
                "lang": box.get("lang", "mixed")
            }

        # ancres importantes à garder
        keep_anchor_keys = [
            #recto
            ("usage_fr", "usage", "fr"),
            ("owner_fr", "owner", "fr"),
            ("address_fr", "address", "fr"),
            ("royaume_du_maroc_fr", "royaume_du_maroc", "fr"),
            #verso
            ("marque_fr", "Marque", "fr"),
            ("type_fr", "Type", "fr"),
            ("number_chassis_fr", "Number_chassis", "fr")
        ]

        for export_name, field_name, side in keep_anchor_keys:
            box = anchors.get(field_name, {}).get(side)
            if box:
                norm_box = self.to_norm(box)

                # nom simple sans suffixe _fr / _ar
                simple_name = export_name.replace("_fr", "").replace("_ar", "")

                data["anchors"][export_name] = {
                    "name": simple_name,
                    **norm_box
                }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return data

    # =========================================================
    # Debug
    # =========================================================
    def draw_debug(self, anchors: dict, zones: dict, out_img_path: str):
        dbg = self.img.copy()

        for item in self.ocr_results:
            cv2.rectangle(dbg, (item["x1"], item["y1"]), (item["x2"], item["y2"]), (180, 180, 180), 1)

        for field_name, anchor in anchors.items():
            if anchor.get("fr"):
                b = self.expand_anchor_horizontally(anchor["fr"], side="left")
                cv2.rectangle(dbg, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (255, 0, 0), 2)
                cv2.putText(
                    dbg,
                    f"{field_name}_fr",
                    (b["x1"], max(15, b["y1"] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )

            if anchor.get("ar"):
                b = self.expand_anchor_horizontally(anchor["ar"], side="right")
                cv2.rectangle(dbg, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 140, 255), 2)
                cv2.putText(
                    dbg,
                    f"{field_name}_ar",
                    (b["x1"], max(15, b["y1"] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 140, 255),
                    1,
                    cv2.LINE_AA
                )
        for field_name, z in zones.items():
            color = (0, 200, 0)
            if field_name == "owner_fr":
                color = (255, 0, 255)
            elif field_name == "owner_ar":
                color = (0, 255, 255)
            elif field_name == "address":
                color = (255, 255, 0)

            cv2.rectangle(dbg, (z["x1"], z["y1"]), (z["x2"], z["y2"]), color, 3)
            cv2.putText(
                dbg,
                field_name,
                (z["x1"], max(20, z["y1"] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        cv2.imwrite(out_img_path, dbg)

    # =========================================================
    # Pipeline principal
    # =========================================================
    def generate(self, out_json_path: str, out_debug_path: str):
        self.resize_to_reference()
        self.run_ocr()
        anchors = self.build_anchor_map()
        zones = self.compute_value_zones(anchors)
        data = self.export_json(zones, anchors, out_json_path)
        self.draw_debug(anchors, zones, out_debug_path)
        return data, anchors, zones

if __name__ == "__main__":
    # -----------------------------
    # RECTO
    # -----------------------------
    recto_generator = CarteGriseZoneGenerator(
        image_path="../images/carte-grise-recto.jpg",
        fields_config=FIELDS_KEY_RECTO,
        document_name="CARTE_GRISE_MAROC_RECTO",
        gpu=False,
        reader_langs=['ar', 'en']
    )

    recto_data, recto_anchors, recto_zones = recto_generator.generate(
        out_json_path="carte_grise_recto_template.json",
        out_debug_path="carte_grise_recto_debug.jpg"
    )

    print("=== RECTO ===")
    print(json.dumps(recto_data, ensure_ascii=False, indent=2))

    # -----------------------------
    # VERSO
    # -----------------------------
    verso_generator = CarteGriseZoneGenerator(
        image_path="../images/carte_grise_verso.jpg",
        fields_config=FIELDS_KEY_VERSO,
        document_name="CARTE_GRISE_MAROC_VERSO",
        gpu=False,
        reader_langs=['ar', 'en']
    )

    verso_data, verso_anchors, verso_zones = verso_generator.generate(
        out_json_path="carte_grise_verso_template.json",
        out_debug_path="carte_grise_verso_debug.jpg"
    )

    print("=== VERSO ===")
    print(json.dumps(verso_data, ensure_ascii=False, indent=2))