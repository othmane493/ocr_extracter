"""
Nouvel extracteur de cartes grises marocaines (Recto / Verso)
- Recentrage ORB avant extraction (optionnel)
- OCR par zones via template JSON
- Tesseract en parallèle
- EasyOCR en fallback
- Conserve Normalize + transform_json + extract_ar
"""

import re
import json
import unicodedata
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import pytesseract
from pytesseract import Output

from utils.ProcessImage import ProcessImage
from utils.normalize import Normalize
from config.CarteGriseRecenter import CarteGriseORBAligner

try:
    from json_transformer import transform_json, is_arabic
except ImportError:
    def transform_json(data):
        return data

    def is_arabic(text):
        if not isinstance(text, str):
            return False
        return bool(re.search(r'[\u0600-\u06FF]', text))


class CarteGriseExtractor:
    """
    Extracteur nouvelle génération :
    1. Recentrage ORB (optionnel)
    2. Extraction par zones template
    3. Tesseract en parallèle
    4. EasyOCR en fallback
    """

    THRESHOLD_TESSERACT_GOOD = 70
    THRESHOLD_EASYOCR_GOOD = 0.69
    MAX_WORKERS = 4

    # =========================
    # CONFIG RECTO
    # =========================
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

    # =========================
    # CONFIG VERSO
    # =========================
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

    CONF_RECTO = {"FIELDS_KEY": FIELDS_KEY_RECTO}
    CONF_VERSO = {"FIELDS_KEY": FIELDS_KEY_VERSO}

    def __init__(
        self,
        recto_template_json: str,
        verso_template_json: str,
        recto_reference_image: Optional[str] = None,
        verso_reference_image: Optional[str] = None,
        gpu: bool = False
    ):
        self.reader_ar = None
        self.reader_en = None
        self.gpu = gpu
        self.config = None

        with open(recto_template_json, "r", encoding="utf-8") as f:
            self.recto_template = json.load(f)

        with open(verso_template_json, "r", encoding="utf-8") as f:
            self.verso_template = json.load(f)

        self.aligner_recto = None
        self.aligner_verso = None

        if recto_reference_image:
            self.aligner_recto = CarteGriseORBAligner(
                reference_image_path=recto_reference_image,
                template_json_path=recto_template_json
            )

        if verso_reference_image:
            self.aligner_verso = CarteGriseORBAligner(
                reference_image_path=verso_reference_image,
                template_json_path=verso_template_json
            )

    # =========================================================
    # EasyOCR singleton
    # =========================================================
    def _get_easyocr_reader(self):
        if self.reader_en is None or self.reader_ar is None:
            try:
                from ocr_manager import get_easyocr_reader
                self.reader_ar, self.reader_en = get_easyocr_reader()
            except ImportError:
                import easyocr
                self.reader_ar = easyocr.Reader(["ar"], gpu=self.gpu)
                self.reader_en = easyocr.Reader(["en"], gpu=self.gpu)
        return self.reader_ar, self.reader_en

    # =========================================================
    # Utils
    # =========================================================
    @staticmethod
    def clean_invisible_chars(text):
        if not text:
            return text
        return "".join(ch for ch in text if unicodedata.category(ch) != "Cf").strip()

    @staticmethod
    def _avg_conf(conf_list: List[float]) -> int:
        vals = [c for c in conf_list if c >= 0]
        if not vals:
            return 0
        return int(sum(vals) / len(vals))

    @staticmethod
    def _safe_text_join(words: List[str]) -> str:
        words = [w.strip() for w in words if w and w.strip()]
        return " ".join(words).strip()

    @staticmethod
    def _box_from_template_field(template_field: Dict, ref_w: int, ref_h: int) -> Dict:
        x = int(template_field["x"] * ref_w)
        y = int(template_field["y"] * ref_h)
        w = int(template_field["w"] * ref_w)
        h = int(template_field["h"] * ref_h)
        return {
            "x": x,
            "y": y,
            "width": w,
            "height": h
        }

    def _normalize_final_value(self, field_name: str, value: str, allowed_chars: str) -> str:
        if value is None:
            return value

        cleaned = Normalize.normalize_value(value, allowed_chars)

        if field_name.endswith("_date"):
            normalized_date = Normalize.normalize_date(cleaned)
            return normalized_date if normalized_date is not None else cleaned

        if field_name.endswith("_matriculate"):
            normalized_mat = Normalize.normalize_matricule(cleaned)
            return normalized_mat if normalized_mat is not None else cleaned

        return cleaned

    # =========================================================
    # OCR Tesseract
    # =========================================================
    def _ocr_tesseract_zone(self, zone_img, lang: str = "fra+ara", psm: int = 6) -> Dict[str, Any]:
        processed, _ = ProcessImage(image=zone_img).process("mode_cg_pytesseract")

        custom_config = rf'--oem 3 --psm {psm} -l {lang}'
        data = pytesseract.image_to_data(
            processed,
            config=custom_config,
            output_type=Output.DICT
        )

        words = []
        confs = []

        n = len(data["text"])
        for i in range(n):
            txt = self.clean_invisible_chars(data["text"][i])
            if not txt:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1
            if conf < 0:
                continue

            words.append(txt)
            confs.append(conf)

        return {
            "text": self._safe_text_join(words),
            "confidence": self._avg_conf(confs),
            "engine": "tesseract"
        }

    # =========================================================
    # OCR EasyOCR
    # =========================================================
    def _ocr_easyocr_zone(self, zone_img, lang: str = "en") -> Dict[str, Any]:
        reader_ar, reader_en = self._get_easyocr_reader()
        processed = ProcessImage(image=zone_img).process("mode_cg_easyocr")

        reader = reader_ar if lang == "ar" else reader_en
        results = reader.readtext(processed, detail=1, paragraph=True)

        if not results:
            return {
                "text": "",
                "confidence": 0,
                "engine": "easyocr"
            }

        words = []
        confs = []

        for item in results:
            if len(item) >= 3:
                _, text, conf = item
                text = self.clean_invisible_chars(text)
                if text:
                    words.append(text)
                    confs.append(conf)

        avg_conf = 0
        if confs:
            avg_conf = int((sum(confs) / len(confs)) * 100)

        return {
            "text": self._safe_text_join(words),
            "confidence": avg_conf,
            "engine": "easyocr"
        }

    # =========================================================
    # OCR selection
    # =========================================================
    def _best_text_result(
        self,
        zone_img,
        field_name: str,
        field_conf: Dict,
        lang_mode: str = "mixed"
    ) -> Dict[str, Any]:
        allow_easyocr = field_conf.get("enable_easy_ocr", True)

        if lang_mode == "ar":
            tess_lang = "ara"
            easy_lang = "ar"
        elif lang_mode == "latin":
            tess_lang = "fra"
            easy_lang = "en"
        else:
            tess_lang = "fra+ara"
            easy_lang = "en"

        tess = self._ocr_tesseract_zone(zone_img, lang=tess_lang, psm=6)

        if tess["text"] and tess["confidence"] >= self.THRESHOLD_TESSERACT_GOOD:
            return tess

        if not allow_easyocr:
            return tess

        easy = self._ocr_easyocr_zone(zone_img, lang=easy_lang)

        if easy["text"] and easy["confidence"] > tess["confidence"]:
            return easy

        if not tess["text"] and easy["text"]:
            return easy

        return tess

    # =========================================================
    # Extraction d'un champ logique
    # =========================================================
    def _extract_logical_field(
        self,
        field_name: str,
        field_conf: Dict,
        field_crops: Dict[str, Any],
        template_fields: Dict[str, Dict],
        ref_w: int,
        ref_h: int
    ) -> Dict[str, Any]:
        pattern = field_conf["normalise"]

        def make_empty_item():
            return {
                "fr": {
                    "key": field_conf["fr"],
                    "value": "",
                    "confidence": 0,
                    "x": 0,
                    "y": 0,
                    "width": 0,
                    "height": 0
                },
                "ar": {
                    "key": field_conf["ar"],
                    "value": "",
                    "confidence": 0,
                    "x": 0,
                    "y": 0,
                    "width": 0,
                    "height": 0
                }
            }

        result = make_empty_item()

        # owner => 2 zones
        if field_name == "owner":
            fr_zone_name = "owner_fr"
            ar_zone_name = "owner_ar"

            fr_zone = field_crops.get(fr_zone_name)
            ar_zone = field_crops.get(ar_zone_name)

            fr_box = self._box_from_template_field(template_fields[fr_zone_name], ref_w, ref_h)
            ar_box = self._box_from_template_field(template_fields[ar_zone_name], ref_w, ref_h)

            if fr_zone is not None:
                fr_res = self._best_text_result(fr_zone, field_name, field_conf, lang_mode="latin")
                fr_val = self._normalize_final_value(field_name, fr_res["text"], pattern)
                result["fr"].update({
                    "value": fr_val,
                    "confidence": int(fr_res["confidence"]),
                    **fr_box
                })

            if ar_zone is not None:
                ar_res = self._best_text_result(ar_zone, field_name, field_conf, lang_mode="ar")
                ar_val = self._normalize_final_value(field_name, ar_res["text"], pattern)
                result["ar"].update({
                    "value": ar_val,
                    "confidence": int(ar_res["confidence"]),
                    **ar_box
                })

            return result

        # address => 1 zone unique, OCR double
        if field_name == "address":
            zone_name = "address"
            zone = field_crops.get(zone_name)
            box = self._box_from_template_field(template_fields[zone_name], ref_w, ref_h)

            if zone is not None:
                fr_res = self._best_text_result(zone, field_name, field_conf, lang_mode="latin")
                ar_res = self._best_text_result(zone, field_name, field_conf, lang_mode="ar")

                fr_val = self._normalize_final_value(field_name, fr_res["text"], pattern)
                ar_val = self._normalize_final_value(field_name, ar_res["text"], pattern)

                result["fr"].update({
                    "value": fr_val,
                    "confidence": int(fr_res["confidence"]),
                    **box
                })
                result["ar"].update({
                    "value": ar_val,
                    "confidence": int(ar_res["confidence"]),
                    **box
                })

            return result

        # cas standard => 1 zone unique
        zone_name = field_name
        zone = field_crops.get(zone_name)
        if zone is None or zone_name not in template_fields:
            return result

        box = self._box_from_template_field(template_fields[zone_name], ref_w, ref_h)

        fr_res = self._best_text_result(zone, field_name, field_conf, lang_mode="mixed")
        fr_val = self._normalize_final_value(field_name, fr_res["text"], pattern)

        if field_conf.get("extract_ar", False):
            ar_res = self._best_text_result(zone, field_name, field_conf, lang_mode="ar")
            ar_val = self._normalize_final_value(field_name, ar_res["text"], pattern)
        else:
            ar_res = fr_res
            ar_val = fr_val

        result["fr"].update({
            "value": fr_val,
            "confidence": int(fr_res["confidence"]),
            **box
        })

        result["ar"].update({
            "value": ar_val,
            "confidence": int(ar_res["confidence"]),
            **box
        })

        return result

    # =========================================================
    # Main extraction
    # =========================================================
    def extract(self, image_path: str, document_type: str, debug: bool = False):
        """
        Nouveau pipeline :
        1. Recentrage ORB (optionnel)
        2. Crop des zones via template
        3. OCR par champ en parallèle
        4. Fallback EasyOCR intégré
        5. transform_json final
        """

        # Choix config + template + aligner
        if document_type == "carte_grise_recto":
            self.config = self.CONF_RECTO
            aligner = self.aligner_recto
            template = self.recto_template
        elif document_type == "carte_grise_verso":
            self.config = self.CONF_VERSO
            aligner = self.aligner_verso
            template = self.verso_template
        else:
            raise ValueError(f"Type de document invalide: {document_type}")

        fields_key = self.config["FIELDS_KEY"]

        # Champs template nécessaires
        template_field_names = []
        for field_name in fields_key.keys():
            if field_name == "owner":
                template_field_names.extend(["owner_fr", "owner_ar"])
            else:
                template_field_names.append(field_name)

        # AVEC ORB
        if aligner is not None:
            process_result = aligner.process_card(
                input_image_path=image_path,
                field_names=template_field_names,
                save_aligned_path=None,
                save_debug_matches_path=f"{document_type}_orb_matches.jpg" if debug else None,
                save_debug_polygon_path=f"{document_type}_orb_polygon.jpg" if debug else None,
                save_debug_fields_path=f"{document_type}_fields_debug.jpg" if debug else None,
                ransac_thresh=5.0,
                min_matches=30,
                debug=debug
            )

            aligned_image = process_result["aligned_image"]
            field_crops = process_result["field_crops"]
            template = process_result["template"]
            meta = process_result["meta"]
            debug_matches_image = process_result["debug_matches_image"]
            debug_polygon_image = process_result["debug_polygon_image"]
            debug_fields_image = process_result["debug_fields_image"]

        # SANS ORB
        else:
            aligned_image = cv2.imread(image_path)
            if aligned_image is None:
                raise ValueError(f"Impossible de lire l'image: {image_path}")

            ref_h, ref_w = aligned_image.shape[:2]
            template_fields = template["fields"]

            field_crops = {}
            for field_name in template_field_names:
                field = template_fields.get(field_name)
                if not field:
                    continue

                x1 = int(field["x"] * ref_w)
                y1 = int(field["y"] * ref_h)
                x2 = int((field["x"] + field["w"]) * ref_w)
                y2 = int((field["y"] + field["h"]) * ref_h)

                field_crops[field_name] = aligned_image[y1:y2, x1:x2]

            meta = {
                "reference_width": ref_w,
                "reference_height": ref_h,
                "input_image": image_path,
                "aligned_image_path": None,
                "debug_enabled": False,
                "debug_matches_image_path": None,
                "debug_polygon_image_path": None,
                "debug_fields_image_path": None,
                "total_good_matches": None,
                "inliers": None,
                "homography": None,
                "template_document": template.get("document"),
                "template_json_path": None,
                "orb_used": False
            }
            debug_matches_image = None
            debug_polygon_image = None
            debug_fields_image = None

        template_fields = template["fields"]
        ref_w = meta["reference_width"]
        ref_h = meta["reference_height"]

        # Extraction parallèle par champ
        parsed_data = []
        futures = []

        with ThreadPoolExecutor(max_workers=min(self.MAX_WORKERS, max(1, len(fields_key)))) as executor:
            for field_name, field_conf in fields_key.items():
                futures.append(
                    executor.submit(
                        self._extract_logical_field,
                        field_name,
                        field_conf,
                        field_crops,
                        template_fields,
                        ref_w,
                        ref_h
                    )
                )

            for future in as_completed(futures):
                parsed_data.append(future.result())

        # Garder l'ordre du config initial
        ordered_data = []
        parsed_map = {}

        for item in parsed_data:
            fr_key = item["fr"]["key"]
            matched_field = None
            for k, v in fields_key.items():
                if v["fr"] == fr_key:
                    matched_field = k
                    break
            if matched_field:
                parsed_map[matched_field] = item

        for field_name in fields_key.keys():
            if field_name in parsed_map:
                ordered_data.append(parsed_map[field_name])

        transformed_data = transform_json(ordered_data)

        return transformed_data