"""
Extracteur de cartes grises marocaines (Recto / Verso)
- Recentrage ORB avant extraction (optionnel)
- OCR par zones via template JSON
- PaddleOCR uniquement
- Conserve Normalize + transform_json + extract_ar
"""

import re
import json
import unicodedata
from typing import Dict, Any, List, Optional

import cv2
from paddleocr import PaddleOCR

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
    3. PaddleOCR par zone
    """

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
        self.gpu = gpu
        self.config = None
        if not hasattr(CarteGriseExtractor, "_reader_fr"):
            CarteGriseExtractor._reader_fr = None
        if not hasattr(CarteGriseExtractor, "_reader_ar"):
            CarteGriseExtractor._reader_ar = None

        if CarteGriseExtractor._reader_fr is None:
            CarteGriseExtractor._reader_fr = PaddleOCR(
                lang="fr",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )

        if CarteGriseExtractor._reader_ar is None:
            CarteGriseExtractor._reader_ar = PaddleOCR(
                lang="ar",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
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
    # PaddleOCR singleton
    # =========================================================
    def _get_paddle_reader(self, lang: str) -> PaddleOCR:
        return (
            CarteGriseExtractor._reader_ar
            if lang == "ar"
            else CarteGriseExtractor._reader_fr
        )

    def _ocr_paddle_zone(self, zone_img, lang: str = "fr") -> Dict[str, Any]:
        if zone_img is None or zone_img.size == 0:
            return {"text": "", "confidence": 0, "engine": "paddleocr"}

        processed = zone_img

        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        elif len(processed.shape) == 3 and processed.shape[2] == 4:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)

        reader = self._get_paddle_reader(lang)

        # pour debug
        results = reader.predict(processed)

        texts = []
        confs = []

        for page in results:
            # cas dict
            if isinstance(page, dict):
                for t in page.get("rec_texts", []):
                    t = self.clean_invisible_chars(str(t))
                    if t:
                        texts.append(t)

                for s in page.get("rec_scores", []):
                    try:
                        confs.append(float(s))
                    except Exception:
                        pass

            # cas objet avec attributs
            elif hasattr(page, "rec_texts") or hasattr(page, "rec_scores"):
                for t in getattr(page, "rec_texts", []):
                    t = self.clean_invisible_chars(str(t))
                    if t:
                        texts.append(t)

                for s in getattr(page, "rec_scores", []):
                    try:
                        confs.append(float(s))
                    except Exception:
                        pass

            # cas ancien format liste/tuple
            elif isinstance(page, (list, tuple)):
                for item in page:
                    try:
                        text = item[1][0]
                        score = item[1][1]
                        text = self.clean_invisible_chars(str(text))
                        if text:
                            texts.append(text)
                            confs.append(float(score))
                    except Exception:
                        pass

        conf = int(sum(confs) / len(confs) * 100) if confs else 0

        return {
            "text": self._safe_text_join(texts),
            "confidence": conf,
            "engine": "paddleocr"
        }
    # =========================================================
    # Utils
    # =========================================================

    @staticmethod
    def contains_arabic(text: str) -> bool:
        return bool(re.search(r'[\u0600-\u06FF]', text or ""))

    @staticmethod
    def has_required_matricule_arabic_letter(text: str) -> bool:
        return bool(re.search(r'[أبدهوهـط]', text or ""))

    @staticmethod
    def _ensure_bgr(img):
        if img is None or img.size == 0:
            return img
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if len(img.shape) == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    @staticmethod
    def _resize_zone_fast(img, max_width=640):
        if img is None or img.size == 0:
            return img
        h, w = img.shape[:2]
        if w <= max_width:
            return img
        ratio = max_width / float(w)
        new_h = max(1, int(h * ratio))
        return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)

    def _prepare_zone_for_ocr(self, zone_img):
        img = self._ensure_bgr(zone_img)
        img = self._resize_zone_fast(img, max_width=640)
        return img

    @staticmethod
    def _shrink_box_vertical(box: Dict[str, int], top_ratio=0.0, bottom_ratio=0.0) -> Dict[str, int]:
        h = box["height"]
        dy_top = int(h * top_ratio)
        dy_bottom = int(h * bottom_ratio)

        y = box["y"] + dy_top
        new_h = h - dy_top - dy_bottom
        if new_h <= 0:
            return box.copy()

        return {
            "x": box["x"],
            "y": y,
            "width": box["width"],
            "height": new_h
        }

    @staticmethod
    def _crop_from_box(image, box: Dict[str, int]):
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        return image[y:y + h, x:x + w]

    def _choose_best_matricule(self, candidates: List[Dict[str, Any]], field_name: str, pattern: str) -> Dict[str, Any]:
        cleaned_candidates = []

        for c in candidates:
            raw = c.get("text", "")
            conf = int(c.get("confidence", 0))
            value = self._normalize_final_value(field_name, raw, pattern)
            value = re.sub(r"\s+", "", value or "").strip()

            if value:
                cleaned_candidates.append({
                    "text": value,
                    "confidence": conf,
                    "engine": c.get("engine", "paddleocr")
                })

        if not cleaned_candidates:
            return {"text": "", "confidence": 0, "engine": "none"}

        with_ar = [c for c in cleaned_candidates if self.has_required_matricule_arabic_letter(c["text"])]
        if with_ar:
            with_ar.sort(key=lambda c: (c["confidence"], len(c["text"])), reverse=True)
            return with_ar[0]

        cleaned_candidates.sort(key=lambda c: (c["confidence"], len(c["text"])), reverse=True)
        return cleaned_candidates[0]

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

    def _extract_texts_from_predict_result(self, results) -> List[Dict[str, Any]]:
        extracted = []

        for page in results:
            texts = []
            confs = []

            if isinstance(page, dict):
                for t in page.get("rec_texts", []):
                    t = self.clean_invisible_chars(str(t))
                    if t:
                        texts.append(t)

                for s in page.get("rec_scores", []):
                    try:
                        confs.append(float(s))
                    except Exception:
                        pass

            elif hasattr(page, "rec_texts") or hasattr(page, "rec_scores"):
                for t in getattr(page, "rec_texts", []):
                    t = self.clean_invisible_chars(str(t))
                    if t:
                        texts.append(t)

                for s in getattr(page, "rec_scores", []):
                    try:
                        confs.append(float(s))
                    except Exception:
                        pass

            elif isinstance(page, (list, tuple)):
                for item in page:
                    try:
                        text = self.clean_invisible_chars(str(item[1][0]))
                        score = float(item[1][1])
                        if text:
                            texts.append(text)
                            confs.append(score)
                    except Exception:
                        pass

            conf = int(sum(confs) / len(confs) * 100) if confs else 0

            extracted.append({
                "text": self._safe_text_join(texts),
                "confidence": conf,
                "engine": "paddleocr"
            })

        return extracted

    def _ocr_paddle_batch(self, images: List, lang: str = "fr") -> List[Dict[str, Any]]:
        if not images:
            return []

        prepared = [self._prepare_zone_for_ocr(img) for img in images]
        reader = self._get_paddle_reader(lang)
        results = reader.predict(prepared)
        return self._extract_texts_from_predict_result(results)

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
    # OCR selection
    # =========================================================
    def _best_text_result(self, zone_img, lang_mode: str = "fr") -> Dict[str, Any]:
        lang = "ar" if lang_mode == "ar" else "fr"
        return self._ocr_paddle_zone(zone_img, lang=lang)

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

        if field_name == "owner":
            fr_zone_name = "owner_fr"
            ar_zone_name = "owner_ar"

            fr_zone = field_crops.get(fr_zone_name)
            ar_zone = field_crops.get(ar_zone_name)
            fr_box = self._box_from_template_field(template_fields[fr_zone_name], ref_w, ref_h)
            ar_box = self._box_from_template_field(template_fields[ar_zone_name], ref_w, ref_h)

            if fr_zone is not None:
                fr_res = self._best_text_result(fr_zone, lang_mode="fr")
                fr_val = self._normalize_final_value(field_name, fr_res["text"], pattern)
                result["fr"].update({
                    "value": fr_val,
                    "confidence": int(fr_res["confidence"]),
                    **fr_box
                })

            if ar_zone is not None:
                ar_res = self._best_text_result(ar_zone, lang_mode="ar")
                ar_val = self._normalize_final_value(field_name, ar_res["text"], pattern)
                result["ar"].update({
                    "value": ar_val,
                    "confidence": int(ar_res["confidence"]),
                    **ar_box
                })

            return result

        if field_name == "address":
            zone_name = "address"
            zone = field_crops.get(zone_name)

            if zone_name not in template_fields:
                return result

            full_box = self._box_from_template_field(template_fields[zone_name], ref_w, ref_h)

            if zone is not None:
                full_box = self._shrink_box_vertical(full_box, top_ratio=0.18, bottom_ratio=0.03)

                zone = self._crop_from_box(zone, {
                    "x": 0,
                    "y": int(zone.shape[0] * 0.18),
                    "width": zone.shape[1],
                    "height": max(1, int(zone.shape[0] * (1 - 0.18 - 0.03)))
                })

                h, w = zone.shape[:2]
                split_x = int(w * 0.55)

                fr_zone = zone[:, :split_x]
                ar_zone = zone[:, split_x:]

                fr_box = {
                    "x": full_box["x"],
                    "y": full_box["y"],
                    "width": split_x,
                    "height": full_box["height"]
                }

                ar_box = {
                    "x": full_box["x"] + split_x,
                    "y": full_box["y"],
                    "width": full_box["width"] - split_x,
                    "height": full_box["height"]
                }

                fr_res = self._best_text_result(fr_zone, lang_mode="fr")
                ar_res = self._best_text_result(ar_zone, lang_mode="ar")

                fr_val = self._normalize_final_value(field_name, fr_res["text"], pattern)
                ar_val = self._normalize_final_value(field_name, ar_res["text"], pattern)

                result["fr"].update({
                    "value": fr_val,
                    "confidence": int(fr_res["confidence"]),
                    **fr_box
                })
                result["ar"].update({
                    "value": ar_val,
                    "confidence": int(ar_res["confidence"]),
                    **ar_box
                })

            return result

        zone_name = field_name
        zone = field_crops.get(zone_name)
        if zone is None or zone_name not in template_fields:
            return result

        box = self._box_from_template_field(template_fields[zone_name], ref_w, ref_h)

        if field_name == "registration_number_matriculate" and field_conf.get("extract_ar", False):
            fr_res = self._best_text_result(zone, lang_mode="fr")
            ar_res = self._best_text_result(zone, lang_mode="ar")

            best_res = self._choose_best_matricule(
                candidates=[fr_res, ar_res],
                field_name=field_name,
                pattern=pattern
            )
            best_val = self._normalize_final_value(field_name, best_res["text"], pattern)

            result["fr"].update({
                "value": best_val,
                "confidence": int(best_res["confidence"]),
                **box
            })
            result["ar"].update({
                "value": best_val,
                "confidence": int(best_res["confidence"]),
                **box
            })
            return result

        fr_res = self._best_text_result(zone, lang_mode="fr")
        fr_val = self._normalize_final_value(field_name, fr_res["text"], pattern)

        if field_conf.get("extract_ar", False):
            ar_res = self._best_text_result(zone, lang_mode="ar")
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

        template_field_names = []
        for field_name in fields_key.keys():
            if field_name == "owner":
                template_field_names.extend(["owner_fr", "owner_ar"])
            else:
                template_field_names.append(field_name)

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

        template_fields = template["fields"]
        ref_w = meta["reference_width"]
        ref_h = meta["reference_height"]

        fr_jobs = []
        ar_jobs = []

        def make_empty_item(field_conf):
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

        result_map = {field_name: make_empty_item(field_conf) for field_name, field_conf in fields_key.items()}

        for field_name, field_conf in fields_key.items():
            pattern = field_conf["normalise"]

            if field_name == "owner":
                fr_zone_name = "owner_fr"
                ar_zone_name = "owner_ar"

                fr_zone = field_crops.get(fr_zone_name)
                ar_zone = field_crops.get(ar_zone_name)

                if fr_zone is not None:
                    fr_jobs.append({
                        "field_name": field_name,
                        "slot": "fr",
                        "pattern": pattern,
                        "box": self._box_from_template_field(template_fields[fr_zone_name], ref_w, ref_h),
                        "image": fr_zone
                    })

                if ar_zone is not None:
                    ar_jobs.append({
                        "field_name": field_name,
                        "slot": "ar",
                        "pattern": pattern,
                        "box": self._box_from_template_field(template_fields[ar_zone_name], ref_w, ref_h),
                        "image": ar_zone
                    })
                continue

            if field_name == "address":
                zone_name = "address"
                zone = field_crops.get(zone_name)
                if zone is None or zone_name not in template_fields:
                    continue

                full_box = self._box_from_template_field(template_fields[zone_name], ref_w, ref_h)
                full_box = self._shrink_box_vertical(full_box, top_ratio=0.18, bottom_ratio=0.03)

                zone = self._crop_from_box(zone, {
                    "x": 0,
                    "y": int(zone.shape[0] * 0.18),
                    "width": zone.shape[1],
                    "height": max(1, int(zone.shape[0] * (1 - 0.18 - 0.03)))
                })

                h, w = zone.shape[:2]
                split_x = int(w * 0.50)

                fr_zone = zone[:, :split_x]
                ar_zone = zone[:, split_x:]

                fr_box = {
                    "x": full_box["x"],
                    "y": full_box["y"],
                    "width": split_x,
                    "height": full_box["height"]
                }
                ar_box = {
                    "x": full_box["x"] + split_x,
                    "y": full_box["y"],
                    "width": full_box["width"] - split_x,
                    "height": full_box["height"]
                }

                fr_jobs.append({
                    "field_name": field_name,
                    "slot": "fr",
                    "pattern": pattern,
                    "box": fr_box,
                    "image": fr_zone
                })
                ar_jobs.append({
                    "field_name": field_name,
                    "slot": "ar",
                    "pattern": pattern,
                    "box": ar_box,
                    "image": ar_zone
                })
                continue

            zone_name = field_name
            zone = field_crops.get(zone_name)
            if zone is None or zone_name not in template_fields:
                continue

            box = self._box_from_template_field(template_fields[zone_name], ref_w, ref_h)

            if field_name == "registration_number_matriculate":
                ar_jobs.append({
                    "field_name": field_name,
                    "slot": "both",
                    "pattern": pattern,
                    "box": box,
                    "image": zone
                })
                continue

            fr_jobs.append({
                "field_name": field_name,
                "slot": "fr",
                "pattern": pattern,
                "box": box,
                "image": zone
            })

        fr_results = self._ocr_paddle_batch([job["image"] for job in fr_jobs], lang="fr")
        ar_results = self._ocr_paddle_batch([job["image"] for job in ar_jobs], lang="ar")

        for job, ocr_res in zip(fr_jobs, fr_results):
            field_name = job["field_name"]
            value = self._normalize_final_value(field_name, ocr_res["text"], job["pattern"])

            result_map[field_name]["fr"].update({
                "value": value,
                "confidence": int(ocr_res["confidence"]),
                **job["box"]
            })

            # par défaut, copier FR vers AR pour les champs non arabes
            if field_name not in ("owner", "address", "registration_number_matriculate"):
                result_map[field_name]["ar"].update({
                    "value": value,
                    "confidence": int(ocr_res["confidence"]),
                    **job["box"]
                })

        for job, ocr_res in zip(ar_jobs, ar_results):
            field_name = job["field_name"]
            value = self._normalize_final_value(field_name, ocr_res["text"], job["pattern"])

            if job["slot"] == "ar":
                result_map[field_name]["ar"].update({
                    "value": value,
                    "confidence": int(ocr_res["confidence"]),
                    **job["box"]
                })
            elif job["slot"] == "both":
                result_map[field_name]["fr"].update({
                    "value": value,
                    "confidence": int(ocr_res["confidence"]),
                    **job["box"]
                })
                result_map[field_name]["ar"].update({
                    "value": value,
                    "confidence": int(ocr_res["confidence"]),
                    **job["box"]
                })

        ordered_data = [result_map[field_name] for field_name in fields_key.keys()]
        transformed_data = transform_json(ordered_data)
        return transformed_data