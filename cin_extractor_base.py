import json
import os
import re
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any, Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR

from json_transformer import is_arabic


class CINExtractor(ABC):
    def __init__(
        self,
        template_path: str,
        image_path: str,
        debug: bool = True,
        recenter_handler=None
    ):
        self.template_path = template_path
        self.image_path = image_path
        self.debug = debug
        self.debug_image_path = "debug_zones.png"
        self.recenter_handler = recenter_handler

        self.template = None
        self.img = None
        self._temp_aligned_path = None
        if (
                not hasattr(CINExtractor, "_reader_ar")
                or not hasattr(CINExtractor, "_reader_fr")
                or CINExtractor._reader_ar is None
                or CINExtractor._reader_fr is None
        ):
            try:
                import sys
                import os

                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)

                from ocr_manager import get_paddle_reader
                CINExtractor._reader_ar, CINExtractor._reader_fr = get_paddle_reader()
                print("[INFO] Paddle readers récupérés depuis ocr_manager.")

            except (ImportError, Exception) as e:
                print(f"[WARN] Impossible d'utiliser ocr_manager, fallback local PaddleOCR: {e}")

                CINExtractor._reader_ar = PaddleOCR(
                    lang="ar",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False
                )

                CINExtractor._reader_fr = PaddleOCR(
                    lang="fr",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False
                )

        self.reader_ar = CINExtractor._reader_ar
        self.reader_fr = CINExtractor._reader_fr

    def load_template(self) -> Dict:
        with open(self.template_path, encoding="utf-8") as f:
            self.template = json.load(f)
        return self.template

    def maybe_recenter_image(self) -> str:
        """
        Si un handler de recentrage est fourni, on génère une image alignée
        temporaire puis on travaille dessus.
        """
        if self.recenter_handler is None:
            return self.image_path

        os.makedirs("debug", exist_ok=True)

        fd, temp_path = tempfile.mkstemp(prefix="cin_aligned_", suffix=".jpg")
        os.close(fd)
        self._temp_aligned_path = temp_path

        try:
            self.recenter_handler.process_card(
                input_image_path=self.image_path,
                save_aligned_path=temp_path,
                save_debug_matches_path="debug/cin_orb_matches.jpg" if self.debug else None,
                save_debug_polygon_path="debug/cin_orb_polygon.jpg" if self.debug else None,
                save_debug_fields_path="debug/cin_orb_fields.jpg" if self.debug else None,
                debug=self.debug
            )
            return temp_path
        except Exception as e:
            if self.debug:
                print(f"⚠️ Recentrage ignoré, fallback image brute: {e}")
            return self.image_path

    def load_image(self) -> np.ndarray:
        image_to_read = self.maybe_recenter_image()

        self.img = cv2.imread(image_to_read)
        if self.img is None:
            raise ValueError(f"Image non trouvée: {image_to_read}")

        if self.template and "width" in self.template and "height" in self.template:
            target_size = (self.template["width"], self.template["height"])
            if (self.img.shape[1], self.img.shape[0]) != target_size:
                self.img = cv2.resize(self.img, target_size)

        return self.img

    @staticmethod
    def safe_crop(img: np.ndarray, rule: Dict) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = img.shape[:2]

        x = int(rule["x"] * w)
        y = int(rule["y"] * h)
        ww = int(rule["w"] * w)
        hh = int(rule["h"] * h)

        x = max(0, x)
        y = max(0, y)
        ww = min(ww, w - x)
        hh = min(hh, h - y)

        return img[y:y + hh, x:x + ww], (x, y, ww, hh)

    @staticmethod
    def normalize_date(text: str) -> str:
        digits = re.findall(r"\d", text)
        if len(digits) != 8:
            return text
        return "{}{}.{}{}.{}{}{}{}".format(*digits)

    @staticmethod
    def filter_text_by_strictness(text: str) -> bool:
        text = str(text).strip()
        return bool(re.fullmatch(r"[A-Za-z0-9\u0600-\u06FF\s\./-]+", text))

    def _extract_texts_from_predict_result(self, results) -> List[Dict[str, Any]]:
        extracted = []

        for page in results:
            texts = []
            confs = []

            if isinstance(page, dict):
                for t in page.get("rec_texts", []):
                    t = str(t).strip()
                    if t:
                        texts.append(t)

                for s in page.get("rec_scores", []):
                    try:
                        confs.append(float(s))
                    except Exception:
                        pass

            elif hasattr(page, "rec_texts") or hasattr(page, "rec_scores"):
                for t in getattr(page, "rec_texts", []):
                    t = str(t).strip()
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
                        text = str(item[1][0]).strip()
                        score = float(item[1][1])
                        if text:
                            texts.append(text)
                            confs.append(score)
                    except Exception:
                        pass

            conf = int(sum(confs) / len(confs) * 100) if confs else 0

            extracted.append({
                "text": " ".join(texts).strip(),
                "confidence": conf,
                "engine": "paddleocr"
            })
        return extracted

    def paddle_text(self, zone: np.ndarray, lang: str) -> str:
        try:
            reader = self.reader_ar if lang == "ar" else self.reader_fr

            if zone is None or zone.size == 0:
                print("PADDLE DEBUG -> zone vide")
                return ""

            print(f"PADDLE DEBUG -> lang={lang}, shape={zone.shape}, dtype={zone.dtype}")

            raw_result = reader.predict(zone)

            print("PADDLE DEBUG -> raw_result type =", type(raw_result))
            print("PADDLE DEBUG -> raw_result =", raw_result)

            extracted = self._extract_texts_from_predict_result(raw_result)
            print("PADDLE DEBUG -> extracted =", extracted)

            if not extracted:
                return ""

            return extracted[0].get("text", "").strip()

        except Exception as e:
            print("PADDLE ERROR =", repr(e))
            return ""

    @staticmethod
    def is_wrong_lang(field: str, text: str) -> bool:
        if field.endswith("_ar") and not is_arabic(text):
            return True
        if field.endswith("_fr") and is_arabic(text):
            return True
        return False

    @staticmethod
    def cmp_phonetic_lang(
        field: str,
        text: str,
        results: Dict,
        compare_func=None,
        confidence: float = 0
    ) -> Any:
        if not field.endswith("_ar") or compare_func is None:
            return confidence

        fr_field = field.replace("_ar", "_fr")
        if fr_field in results and results[fr_field]:
            try:
                cmp = compare_func(text, results[fr_field])
                return cmp.get("score", 0)
            except Exception:
                return confidence

        return confidence

    @staticmethod
    def reorder_identity_fields(data: Dict) -> Dict:
        ordered = {}
        groups = ["prenom", "nom", "lieu_naissance"]

        for g in groups:
            ar_key = f"{g}_ar"
            fr_key = f"{g}_fr"

            if ar_key in data:
                ordered[ar_key] = data[ar_key]
            if fr_key in data:
                ordered[fr_key] = data[fr_key]

        for k, v in data.items():
            if k not in ordered:
                ordered[k] = v

        return ordered

    def create_debug_image(self, debug_img: np.ndarray) -> None:
        if self.debug:
            cv2.imwrite(self.debug_image_path, debug_img)
            print(f"🟢 Image debug générée : {self.debug_image_path}")

    @abstractmethod
    def preprocess_zone(self, zone: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def preprocess_zone_ocr(self, zone: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def extract_text_tesseract(self, zone: np.ndarray) -> List[Dict]:
        pass

    @abstractmethod
    def get_confidence_threshold(self) -> int:
        pass

    def extract(self, compare_name_func=None) -> Dict:
        self.load_template()
        self.load_image()

        debug_img = self.img.copy()
        results = {}

        for field, rule in self.template["fields"].items():
            zone, (x, y, w, h) = self.safe_crop(self.img, rule)

            if self.debug:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    debug_img,
                    field,
                    (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1
                )

            zone_preprocessed = self.preprocess_zone(zone)
            blocks = self.extract_text_tesseract(zone_preprocessed)
            blocks = [b for b in blocks if self.filter_text_by_strictness(b.get("text", ""))]

            lang_field = "fr" if field.endswith("_fr") or field == "cin" or "date" in field else "ar"

            if blocks:
                ocr_text = " ".join(str(b.get("text", "")).strip() for b in blocks).strip()

                conf_values = [float(b.get("confidence", 0)) for b in blocks if str(b.get("confidence", "")).strip() != ""]
                confidence = sum(conf_values) / len(conf_values) if conf_values else 0

                if self.is_wrong_lang(field, ocr_text):
                    zone_retry = self.preprocess_zone_ocr(zone)
                    cv2.imwrite(f"debug_{field}_paddle1.jpg", zone_retry)
                    ocr_text = self.paddle_text(zone_retry, lang_field)

                elif field.endswith("_ar") or confidence < self.get_confidence_threshold():
                    cmp_tesseract = self.cmp_phonetic_lang(
                        field,
                        ocr_text,
                        results,
                        compare_name_func,
                        int(confidence / 100) if confidence > 1 else confidence
                    )
                    zone_retry = self.preprocess_zone_ocr(zone)
                    cv2.imwrite(f"debug_{field}_paddle2.jpg", zone_retry)
                    paddle_retry_text = self.paddle_text(zone_retry, lang_field)
                    print(paddle_retry_text)
                    cmp_paddle = self.cmp_phonetic_lang(
                        field,
                        paddle_retry_text,
                        results,
                        compare_name_func
                    )

                    if paddle_retry_text and cmp_tesseract < cmp_paddle:
                        ocr_text = paddle_retry_text
            else:
                zone_retry = self.preprocess_zone_ocr(zone)
                cv2.imwrite(f"debug_{field}_paddle3.jpg", zone_retry)
                ocr_text = self.paddle_text(zone_retry, lang_field)

            if "date" in field:
                ocr_text = self.normalize_date(ocr_text)

            results[field] = ocr_text.strip()

        if self.debug:
            cv2.imwrite("debug_all_zones.jpg", debug_img)

        self.create_debug_image(debug_img)
        results = self.reorder_identity_fields(results)
        return results