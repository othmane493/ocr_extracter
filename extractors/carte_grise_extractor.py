"""
Extracteur de cartes grises marocaines (Recto / Verso)
- Recentrage ORB avant extraction (optionnel)
- OCR par zones via template JSON
- PaddleOCR uniquement pour les champs configures avec ocr="fr" ou ocr="ar"
- Tesseract uniquement pour les champs configures avec ocr="tesseract"
- Detection double dash pour certains champs verso
- Batch OCR FR / AR
- Réduction des marges blanches autour du texte avant OCR
- Séparation intelligente address_fr / address_ar
- Conserve Normalize + transform_json + extract_ar
"""

import re
import json
import unicodedata
import time
from contextlib import contextmanager
from typing import Dict, Any, List, Optional

import cv2
import pytesseract
from pytesseract import Output
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
    Fusion des deux versions :
    - Pipeline PaddleOCR = version 1
    - Pipeline Tesseract + detect_double_dash = version 2
    """

    FIELDS_KEY_RECTO = {
        "registration_number_matriculate": {
            "fr": "Numéro d'immatriculation",
            "ar": "رقم التسجيل",
            "normalise": "0-9أبدهوهـط-",
            "multi_line": False,
            "extract_ar": True,
            "type": "center_value",
            "ocr": "ar"
        },
        "previous_registration": {
            "fr": "Immatriculation antérieure",
            "ar": "الترقيم السابق",
            "normalise": "A-Z0-9",
            "multi_line": False,
            "type": "center_value",
            "ocr": "tesseract"
        },
        "first_registration_date": {
            "fr": "Première mise en circulation",
            "ar": "أول شروع في الإستخدام",
            "normalise": "0-9/",
            "multi_line": False,
            "type": "center_value",
            "ocr": "tesseract"
        },
        "first_usage_date": {
            "fr": "M.C au maroc",
            "ar": "أول استخدام بالمغرب",
            "normalise": "0-9/",
            "multi_line": False,
            "type": "center_value",
            "ocr": "tesseract"
        },
        "mutation_date": {
            "fr": "Mutation le",
            "ar": "تحويل بتاريخ",
            "normalise": "0-9/",
            "multi_line": False,
            "type": "center_value",
            "ocr": "tesseract"
        },
        "usage": {
            "fr": "Usage",
            "ar": "نوع الإستعمال",
            "normalise": "A-Za-zÀ-ÿ0-9\\s_-",
            "multi_line": False,
            "type": "center_value",
            "ocr": "fr"
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
            "type": "center_value",
            "ocr": "tesseract"
        }
    }

    FIELDS_KEY_VERSO = {
        "Marque": {
            "fr": "Marque",
            "ar": "الاسم التجاري",
            "normalise": "A-Za-z\\s_-",
            "multi_line": False,
            "type": "center_value",
            "ocr": "fr",
            "detect_double_dash": False
        },
        "Type": {
            "fr": "Type",
            "ar": "الصنف",
            "normalise": "A-Z0-9-",
            "multi_line": False,
            "type": "center_value",
            "ocr": "tesseract",
            "detect_double_dash": True
        },
        "Genre": {
            "fr": "Genre",
            "ar": "النوع",
            "normalise": "A-Za-z_\\s-",
            "multi_line": False,
            "type": "center_value",
            "ocr": "fr",
            "detect_double_dash": False
        },
        "Modèle": {
            "fr": "Modèle",
            "ar": "النموذج",
            "normalise": "A-Za-z_\\s-",
            "multi_line": False,
            "type": "center_value",
            "ocr": "fr",
            "detect_double_dash": True
        },
        "Type_Carburant": {
            "fr": "Type carburant",
            "ar": "نوع الوقود",
            "normalise": "A-Za-z-",
            "multi_line": False,
            "type": "center_value",
            "ocr": "fr",
            "detect_double_dash": False
        },
        "Number_chassis": {
            "fr": "N° du chassis",
            "ar": "رقم الإطار الحديدي",
            "normalise": "A-Z0-9",
            "multi_line": False,
            "type": "center_value",
            "ocr": "tesseract",
            "detect_double_dash": False
        },
        "Number_Cylinders": {
            "fr": "Nombre de cylindres",
            "ar": "عدد الأسطوانات",
            "normalise": "0-9",
            "multi_line": False,
            "type": "center_value",
            "ocr": "tesseract",
            "detect_double_dash": True
        },
        "Puissance_Fiscale": {
            "fr": "Puissance fiscale",
            "ar": "القوة الجبائية",
            "normalise": "0-9",
            "multi_line": False,
            "type": "center_value",
            "ocr": "tesseract",
            "detect_double_dash": False
        },
        "Number_Places": {
            "fr": "Nombre de places",
            "ar": "عدد المقاعد",
            "normalise": "0-9",
            "multi_line": False,
            "type": "center_value",
            "ocr": "tesseract",
            "detect_double_dash": True
        },
        "PTAC": {
            "fr": "P.T.A.C",
            "ar": "الوزن جمالي",
            "normalise": "0-9kKgG\\s-",
            "multi_line": False,
            "type": "center_value",
            "ocr": "fr",
            "detect_double_dash": True
        },
        "Poids_vide": {
            "fr": "Poids à vide",
            "ar": "الوزن الفارغ",
            "normalise": "0-9kKgG\\s",
            "multi_line": False,
            "type": "center_value",
            "ocr": "fr",
            "detect_double_dash": True
        },
        "PTRA": {
            "fr": "P.T.R.A",
            "ar": "الوزن الإجمالي مع المجرور",
            "normalise": "0-9kKgG\\s",
            "multi_line": False,
            "type": "center_value",
            "ocr": "fr",
            "detect_double_dash": True
        },
        "Restrictions": {
            "fr": "Restrictions",
            "ar": "التقييدات",
            "normalise": "A-Za-z\u0600-\u06FF\\s\\-",
            "multi_line": False,
            "type": "center_value",
            "ocr": "tesseract",
            "detect_double_dash": True
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
        self._current_timings = None

        if not hasattr(CarteGriseExtractor, "_reader_fr"):
            CarteGriseExtractor._reader_fr = None
        if not hasattr(CarteGriseExtractor, "_reader_ar"):
            CarteGriseExtractor._reader_ar = None

        init_start = time.perf_counter()

        if CarteGriseExtractor._reader_fr is None:
            t0 = time.perf_counter()
            CarteGriseExtractor._reader_fr = PaddleOCR(
                lang="fr",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
            print(f"[PERF] init Paddle FR: {time.perf_counter() - t0:.4f}s")

        if CarteGriseExtractor._reader_ar is None:
            t0 = time.perf_counter()
            CarteGriseExtractor._reader_ar = PaddleOCR(
                lang="ar",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
            print(f"[PERF] init Paddle AR: {time.perf_counter() - t0:.4f}s")

        t0 = time.perf_counter()
        with open(recto_template_json, "r", encoding="utf-8") as f:
            self.recto_template = json.load(f)
        print(f"[PERF] load recto template: {time.perf_counter() - t0:.4f}s")

        t0 = time.perf_counter()
        with open(verso_template_json, "r", encoding="utf-8") as f:
            self.verso_template = json.load(f)
        print(f"[PERF] load verso template: {time.perf_counter() - t0:.4f}s")

        self.aligner_recto = None
        self.aligner_verso = None

        if recto_reference_image:
            t0 = time.perf_counter()
            self.aligner_recto = CarteGriseORBAligner(
                reference_image_path=recto_reference_image,
                template_json_path=recto_template_json
            )
            print(f"[PERF] init aligner recto: {time.perf_counter() - t0:.4f}s")

        if verso_reference_image:
            t0 = time.perf_counter()
            self.aligner_verso = CarteGriseORBAligner(
                reference_image_path=verso_reference_image,
                template_json_path=verso_template_json
            )
            print(f"[PERF] init aligner verso: {time.perf_counter() - t0:.4f}s")

        print(f"[PERF] __init__ total: {time.perf_counter() - init_start:.4f}s")

    @contextmanager
    def _timer(self, name: str):
        if self._current_timings is None:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._current_timings[name] = self._current_timings.get(name, 0.0) + elapsed

    def _add_timing(self, name: str, elapsed: float):
        if self._current_timings is not None:
            self._current_timings[name] = self._current_timings.get(name, 0.0) + elapsed

    def _get_paddle_reader(self, lang: str) -> PaddleOCR:
        return (
            CarteGriseExtractor._reader_ar
            if lang == "ar"
            else CarteGriseExtractor._reader_fr
        )

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

    def _tight_crop_text(self, img):
        if img is None or img.size == 0:
            return img

        src = self._ensure_bgr(img)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

        coords = cv2.findNonZero(bw)
        if coords is None:
            return src

        x, y, w, h = cv2.boundingRect(coords)

        pad_x = max(4, int(w * 0.03))
        pad_y = max(4, int(h * 0.20))

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(src.shape[1], x + w + pad_x)
        y2 = min(src.shape[0], y + h + pad_y)

        cropped = src[y1:y2, x1:x2]
        return cropped if cropped.size > 0 else src

    def _prepare_zone_for_ocr(self, zone_img, lang: str = "fr"):
        t0 = time.perf_counter()

        img = self._ensure_bgr(zone_img)
        if img is None or img.size == 0:
            self._add_timing(f"prepare_zone_{lang}", time.perf_counter() - t0)
            return img

        img = self._tight_crop_text(img)

        h, w = img.shape[:2]

        if h < 80 or w < 250:
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        h, w = img.shape[:2]
        if w > 640:
            img = self._resize_zone_fast(img, max_width=350)

        self._add_timing(f"prepare_zone_{lang}", time.perf_counter() - t0)
        return img

    def _detect_vertical_split(self, zone_img, default_ratio: float = 0.50) -> int:
        t0 = time.perf_counter()

        if zone_img is None or zone_img.size == 0:
            val = int(zone_img.shape[1] * default_ratio) if zone_img is not None else 0
            self._add_timing("detect_vertical_split", time.perf_counter() - t0)
            return val

        img = self._ensure_bgr(zone_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

        h, w = bw.shape
        col_sum = bw.sum(axis=0).astype("float32")

        k = max(9, (w // 30) | 1)
        col_sum_smooth = cv2.GaussianBlur(col_sum.reshape(1, -1), (k, 1), 0).reshape(-1)

        x_min = int(w * 0.42)
        x_max = int(w * 0.62)
        if x_max <= x_min:
            split_x = int(w * default_ratio)
            self._add_timing("detect_vertical_split", time.perf_counter() - t0)
            return split_x

        band = col_sum_smooth[x_min:x_max]
        split_x = x_min + int(band.argmin())

        if split_x < int(w * 0.35) or split_x > int(w * 0.65):
            split_x = int(w * default_ratio)

        self._add_timing("detect_vertical_split", time.perf_counter() - t0)
        return split_x

    @staticmethod
    def contains_arabic(text: str) -> bool:
        return bool(re.search(r'[\u0600-\u06FF]', text or ""))

    @staticmethod
    def has_required_matricule_arabic_letter(text: str) -> bool:
        return bool(re.search(r'[أبدهوهـط]', text or ""))

    @staticmethod
    def clean_invisible_chars(text):
        if not text:
            return text
        return "".join(ch for ch in text if unicodedata.category(ch) != "Cf").strip()

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
        return {"x": x, "y": y, "width": w, "height": h}

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

    def _normalize_final_value(self, field_name: str, value: str, allowed_chars: str) -> str:
        t0 = time.perf_counter()

        if value is None:
            self._add_timing("normalize_final_value", time.perf_counter() - t0)
            return value

        cleaned = Normalize.normalize_value(value, allowed_chars)

        if field_name.endswith("_date"):
            normalized_date = Normalize.normalize_date(cleaned)
            result = normalized_date if normalized_date is not None else cleaned
            self._add_timing("normalize_final_value", time.perf_counter() - t0)
            return result

        if field_name.endswith("_matriculate"):
            normalized_mat = Normalize.normalize_matricule(cleaned)
            result = normalized_mat if normalized_mat is not None else cleaned
            self._add_timing("normalize_final_value", time.perf_counter() - t0)
            return result

        self._add_timing("normalize_final_value", time.perf_counter() - t0)
        return cleaned

    def _choose_best_matricule(self, candidates: List[Dict[str, Any]], field_name: str, pattern: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

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
                    "engine": c.get("engine", "ocr")
                })

        if not cleaned_candidates:
            self._add_timing("choose_best_matricule", time.perf_counter() - t0)
            return {"text": "", "confidence": 0, "engine": "none"}

        with_ar = [c for c in cleaned_candidates if self.has_required_matricule_arabic_letter(c["text"])]
        if with_ar:
            with_ar.sort(key=lambda c: (c["confidence"], len(c["text"])), reverse=True)
            self._add_timing("choose_best_matricule", time.perf_counter() - t0)
            return with_ar[0]

        cleaned_candidates.sort(key=lambda c: (c["confidence"], len(c["text"])), reverse=True)
        self._add_timing("choose_best_matricule", time.perf_counter() - t0)
        return cleaned_candidates[0]

    def _extract_texts_from_predict_result(self, results) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()

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

        self._add_timing("extract_texts_from_predict_result", time.perf_counter() - t0)
        return extracted

    def _ocr_paddle_batch(self, images: List, lang: str = "fr") -> List[Dict[str, Any]]:
        if not images:
            return []

        with self._timer(f"paddle_prepare_batch_{lang}"):
            prepared = [self._prepare_zone_for_ocr(img, lang=lang) for img in images]

        reader = self._get_paddle_reader(lang)

        with self._timer(f"paddle_predict_{lang}"):
            results = reader.predict(prepared)

        return self._extract_texts_from_predict_result(results)

    def _ocr_tesseract_zone(self, zone_img, field_name: str = "") -> Dict[str, Any]:
        t0 = time.perf_counter()

        if zone_img is None or zone_img.size == 0:
            self._add_timing("tesseract_total", time.perf_counter() - t0)
            return {"text": "", "confidence": 0, "engine": "tesseract"}

        t1 = time.perf_counter()
        processed, _ = ProcessImage(image=zone_img).process("mode_cg_pytesseract")
        self._add_timing("tesseract_preprocess", time.perf_counter() - t1)

        whitelist = ""

        if field_name.endswith("_date"):
            whitelist = r" -c tessedit_char_whitelist=0123456789/"
        elif "registration" in field_name or "chassis" in field_name:
            whitelist = r" -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        custom_config = rf'--oem 3 --psm 7{whitelist}'

        t1 = time.perf_counter()
        data = pytesseract.image_to_data(
            processed,
            config=custom_config,
            output_type=Output.DICT
        )
        self._add_timing("tesseract_image_to_data", time.perf_counter() - t1)

        t1 = time.perf_counter()
        words = []
        confs = []

        for txt, conf in zip(data["text"], data["conf"]):
            txt = self.clean_invisible_chars(txt)
            if not txt:
                continue

            try:
                conf = float(conf)
            except Exception:
                conf = -1

            if conf < 0:
                continue

            words.append(txt)
            confs.append(conf)

        text = self._safe_text_join(words)
        avg_conf = int(sum(confs) / len(confs)) if confs else 0
        self._add_timing("tesseract_postprocess", time.perf_counter() - t1)
        self._add_timing("tesseract_total", time.perf_counter() - t0)

        return {
            "text": text,
            "confidence": avg_conf,
            "engine": "tesseract"
        }

    def _is_double_dash_zone(self, zone_img) -> bool:
        t0 = time.perf_counter()
        try:
            txt = ProcessImage(image=zone_img).process("detect_double_dash")
            txt = (txt or "").strip().replace(" ", "")
            result = txt in {"-", "--", "---", "—", "––"}
            self._add_timing("detect_double_dash", time.perf_counter() - t0)
            return result
        except Exception:
            self._add_timing("detect_double_dash", time.perf_counter() - t0)
            return False

    def _save_debug_zone(self, name: str, img):
        if img is None or img.size == 0:
            return
        cv2.imwrite(f"debug_{name}.jpg", img)

    def extract(self, image_path: str, document_type: str, debug: bool = False, return_profile: bool = False):
        timings = {}
        self._current_timings = timings
        global_start = time.perf_counter()

        try:
            with self._timer("select_config"):
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
                with self._timer("aligner_process_card"):
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

                with self._timer("extract_process_result"):
                    field_crops = process_result["field_crops"]
                    template = process_result["template"]
                    meta = process_result["meta"]

            else:
                with self._timer("read_image"):
                    aligned_image = cv2.imread(image_path)
                    if aligned_image is None:
                        raise ValueError(f"Impossible de lire l'image: {image_path}")

                with self._timer("manual_crop_fields"):
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

            with self._timer("meta_prepare"):
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

                result_map = {
                    field_name: make_empty_item(field_conf)
                    for field_name, field_conf in fields_key.items()
                }

            with self._timer("prepare_jobs"):
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

                        if zone is None or zone.size == 0:
                            continue

                        h, w = zone.shape[:2]
                        split_x = self._detect_vertical_split(zone, default_ratio=0.50)

                        margin = max(6, int(w * 0.01))
                        fr_end = max(1, split_x - margin)
                        ar_start = min(w - 1, split_x + margin)

                        fr_zone = zone[:, :fr_end]
                        ar_zone = zone[:, ar_start:]

                        if debug:
                            debug_zone = self._ensure_bgr(zone.copy())
                            cv2.line(debug_zone, (split_x, 0), (split_x, debug_zone.shape[0] - 1), (0, 0, 255), 2)
                            self._save_debug_zone("address_full", zone)
                            self._save_debug_zone("address_fr", fr_zone)
                            self._save_debug_zone("address_ar", ar_zone)
                            self._save_debug_zone("address_split_visual", debug_zone)

                        fr_box = {
                            "x": full_box["x"],
                            "y": full_box["y"],
                            "width": fr_end,
                            "height": full_box["height"]
                        }
                        ar_box = {
                            "x": full_box["x"] + ar_start,
                            "y": full_box["y"],
                            "width": full_box["width"] - ar_start,
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
                    ocr_mode = field_conf.get("ocr", "fr")

                    if field_conf.get("detect_double_dash", False):
                        t0 = time.perf_counter()
                        is_dash = self._is_double_dash_zone(zone)
                        self._add_timing("detect_double_dash_total_loop", time.perf_counter() - t0)

                        if is_dash:
                            result_map[field_name]["fr"].update({
                                "value": "--",
                                "confidence": 100,
                                **box
                            })
                            result_map[field_name]["ar"].update({
                                "value": "--",
                                "confidence": 100,
                                **box
                            })
                            continue

                    if ocr_mode == "tesseract":
                        t0 = time.perf_counter()
                        tess_res = self._ocr_tesseract_zone(zone, field_name=field_name)
                        self._add_timing("tesseract_loop_total", time.perf_counter() - t0)

                        value = self._normalize_final_value(field_name, tess_res["text"], pattern)

                        result_map[field_name]["fr"].update({
                            "value": value,
                            "confidence": int(tess_res["confidence"]),
                            **box
                        })
                        result_map[field_name]["ar"].update({
                            "value": value,
                            "confidence": int(tess_res["confidence"]),
                            **box
                        })
                        continue

                    if field_name == "registration_number_matriculate":
                        ar_jobs.append({
                            "field_name": field_name,
                            "slot": "both",
                            "pattern": pattern,
                            "box": box,
                            "image": zone
                        })
                        continue

                    if ocr_mode == "ar":
                        ar_jobs.append({
                            "field_name": field_name,
                            "slot": "both",
                            "pattern": pattern,
                            "box": box,
                            "image": zone
                        })
                    else:
                        fr_jobs.append({
                            "field_name": field_name,
                            "slot": "both",
                            "pattern": pattern,
                            "box": box,
                            "image": zone
                        })

            with self._timer("paddle_fr_batch_total"):
                fr_results = self._ocr_paddle_batch([job["image"] for job in fr_jobs], lang="fr")

            with self._timer("paddle_ar_batch_total"):
                ar_results = self._ocr_paddle_batch([job["image"] for job in ar_jobs], lang="ar")

            with self._timer("merge_fr_results"):
                for job, ocr_res in zip(fr_jobs, fr_results):
                    field_name = job["field_name"]
                    value = self._normalize_final_value(field_name, ocr_res["text"], job["pattern"])

                    if job["slot"] == "fr":
                        result_map[field_name]["fr"].update({
                            "value": value,
                            "confidence": int(ocr_res["confidence"]),
                            **job["box"]
                        })
                    else:
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

            with self._timer("merge_ar_results"):
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
                        if field_name == "registration_number_matriculate":
                            best_res = self._choose_best_matricule(
                                candidates=[ocr_res],
                                field_name=field_name,
                                pattern=job["pattern"]
                            )
                            value = self._normalize_final_value(field_name, best_res["text"], job["pattern"])
                            conf = int(best_res["confidence"])
                        else:
                            conf = int(ocr_res["confidence"])

                        result_map[field_name]["fr"].update({
                            "value": value,
                            "confidence": conf,
                            **job["box"]
                        })
                        result_map[field_name]["ar"].update({
                            "value": value,
                            "confidence": conf,
                            **job["box"]
                        })

            with self._timer("final_transform"):
                ordered_data = [result_map[field_name] for field_name in fields_key.keys()]
                transformed_data = transform_json(ordered_data)

            timings["extract_total"] = time.perf_counter() - global_start

            print("\n===== PERF REPORT =====")
            for k, v in sorted(timings.items(), key=lambda x: x[1], reverse=True):
                print(f"{k:35s}: {v:.4f}s")
            print("=======================\n")

            if return_profile:
                return {
                    "data": transformed_data,
                    "profile": dict(sorted(timings.items(), key=lambda x: x[1], reverse=True))
                }

            return transformed_data

        finally:
            self._current_timings = None