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
        debug: bool = False,
        recenter_handler=None
    ):
        self.template_path = template_path
        self.image_path = image_path
        self.debug = debug
        self.debug_image_path = os.path.join(
            "debug",
            f"debug_zones_{os.path.basename(image_path)}.jpg"
        )
        self.recenter_handler = recenter_handler
        self.runtime_field_boxes = {}

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

    @staticmethod
    def crop_with_pixel_box(
        img: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        return img[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    def get_field_zone(
        self,
        field: str,
        rule: Dict
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if field in self.runtime_field_boxes:
            return self.crop_with_pixel_box(self.img, self.runtime_field_boxes[field])
        return self.safe_crop(self.img, rule)

    def maybe_recenter_image(self) -> Dict[str, Any]:
        if self.recenter_handler is None:
            return {
                "image_path": self.image_path,
                "runtime_field_boxes": {}
            }

        os.makedirs("debug", exist_ok=True)

        fd, temp_path = tempfile.mkstemp(prefix="cin_aligned_", suffix=".jpg")
        os.close(fd)
        self._temp_aligned_path = temp_path

        try:
            recenter_result = self.recenter_handler.process_card(
                input_image_path=self.image_path,
                save_aligned_path=temp_path,
                save_debug_matches_path="debug/cin_orb_matches.jpg" if self.debug else None,
                save_debug_polygon_path="debug/cin_orb_polygon.jpg" if self.debug else None,
                save_debug_fields_path="debug/cin_orb_fields_initial.jpg" if self.debug else None,
                save_refined_fields_path="debug/cin_orb_fields_refined.jpg" if self.debug else None,
                debug=self.debug
            )

            return {
                "image_path": temp_path,
                "runtime_field_boxes": recenter_result.get("refined_boxes", {}) or {}
            }

        except Exception as e:
            if self.debug:
                print(f"⚠️ Recentrage ignoré, fallback image brute: {e}")
            return {
                "image_path": self.image_path,
                "runtime_field_boxes": {}
            }

    def load_image(self) -> np.ndarray:
        recenter_payload = self.maybe_recenter_image()
        image_to_read = recenter_payload["image_path"]
        self.runtime_field_boxes = recenter_payload.get("runtime_field_boxes", {}) or {}

        self.img = cv2.imread(image_to_read)
        if self.img is None:
            raise ValueError(f"Image non trouvée: {image_to_read}")

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

    @staticmethod
    def _poly_to_rect(poly) -> Optional[Dict[str, float]]:
        try:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            return {
                "x1": min(xs),
                "y1": min(ys),
                "x2": max(xs),
                "y2": max(ys),
                "cx": (min(xs) + max(xs)) / 2.0,
                "cy": (min(ys) + max(ys)) / 2.0,
                "h": max(1.0, max(ys) - min(ys))
            }
        except Exception:
            return None

    def _join_paddle_tokens_in_reading_order(self, tokens: List[Dict[str, Any]], lang: str) -> str:
        if not tokens:
            return ""

        tokens = [t for t in tokens if t.get("text")]
        if not tokens:
            return ""

        avg_h = sum(float(t["rect"]["h"]) for t in tokens if t.get("rect")) / max(1, len(tokens))
        line_tol = max(10.0, avg_h * 0.6)

        tokens.sort(key=lambda t: (t["rect"]["cy"], t["rect"]["cx"]))

        lines: List[List[Dict[str, Any]]] = []
        for token in tokens:
            placed = False
            for line in lines:
                line_cy = sum(float(x["rect"]["cy"]) for x in line) / len(line)
                if abs(float(token["rect"]["cy"]) - line_cy) <= line_tol:
                    line.append(token)
                    placed = True
                    break
            if not placed:
                lines.append([token])

        lines.sort(key=lambda line: min(float(t["rect"]["y1"]) for t in line))

        line_texts = []
        for line in lines:
            reverse_x = (lang == "ar")
            line.sort(key=lambda t: float(t["rect"]["cx"]), reverse=reverse_x)
            text = " ".join(str(t["text"]).strip() for t in line if str(t["text"]).strip()).strip()
            if text:
                line_texts.append(text)

        return " ".join(line_texts).strip()

    def _extract_texts_from_predict_result(self, results, lang: str = "fr") -> List[Dict[str, Any]]:
        extracted = []

        for page in results:
            tokens = []
            confs = []

            if isinstance(page, dict):
                rec_texts = page.get("rec_texts", []) or []
                rec_scores = page.get("rec_scores", []) or []
                dt_polys = page.get("dt_polys", []) or []

                for idx, raw_text in enumerate(rec_texts):
                    text = str(raw_text).strip()
                    if not text:
                        continue

                    rect = None
                    if idx < len(dt_polys):
                        rect = self._poly_to_rect(dt_polys[idx])

                    tokens.append({
                        "text": text,
                        "rect": rect or {
                            "x1": float(idx),
                            "y1": 0.0,
                            "x2": float(idx) + 1.0,
                            "y2": 1.0,
                            "cx": float(idx) + 0.5,
                            "cy": 0.5,
                            "h": 1.0
                        }
                    })

                for s in rec_scores:
                    try:
                        confs.append(float(s))
                    except Exception:
                        pass

            elif hasattr(page, "rec_texts") or hasattr(page, "rec_scores"):
                rec_texts = getattr(page, "rec_texts", []) or []
                rec_scores = getattr(page, "rec_scores", []) or []
                dt_polys = getattr(page, "dt_polys", []) or []

                for idx, raw_text in enumerate(rec_texts):
                    text = str(raw_text).strip()
                    if not text:
                        continue

                    rect = None
                    if idx < len(dt_polys):
                        rect = self._poly_to_rect(dt_polys[idx])

                    tokens.append({
                        "text": text,
                        "rect": rect or {
                            "x1": float(idx),
                            "y1": 0.0,
                            "x2": float(idx) + 1.0,
                            "y2": 1.0,
                            "cx": float(idx) + 0.5,
                            "cy": 0.5,
                            "h": 1.0
                        }
                    })

                for s in rec_scores:
                    try:
                        confs.append(float(s))
                    except Exception:
                        pass

            elif isinstance(page, (list, tuple)):
                for idx, item in enumerate(page):
                    try:
                        text = str(item[1][0]).strip()
                        score = float(item[1][1])
                        if not text:
                            continue

                        rect = None
                        if item and item[0]:
                            rect = self._poly_to_rect(item[0])

                        tokens.append({
                            "text": text,
                            "rect": rect or {
                                "x1": float(idx),
                                "y1": 0.0,
                                "x2": float(idx) + 1.0,
                                "y2": 1.0,
                                "cx": float(idx) + 0.5,
                                "cy": 0.5,
                                "h": 1.0
                            }
                        })
                        confs.append(score)
                    except Exception:
                        pass

            conf = int(sum(confs) / len(confs) * 100) if confs else 0

            extracted.append({
                "text": self._join_paddle_tokens_in_reading_order(tokens, lang=lang),
                "confidence": conf,
                "engine": "paddleocr"
            })

        return extracted

    def paddle_text(self, zone: np.ndarray, lang: str) -> Dict[str, Any]:
        try:
            reader = self.reader_ar if lang == "ar" else self.reader_fr

            if zone is None or zone.size == 0:
                return {"text": "", "confidence": 0, "engine": "paddleocr"}

            raw_result = reader.predict(zone)
            extracted = self._extract_texts_from_predict_result(raw_result, lang=lang)

            if not extracted:
                return {"text": "", "confidence": 0, "engine": "paddleocr"}

            return extracted[0]

        except Exception as e:
            print("PADDLE ERROR =", repr(e))
            return {"text": "", "confidence": 0, "engine": "paddleocr"}

    @staticmethod
    def is_wrong_lang(field: str, text: str) -> bool:
        text = str(text).strip()
        if not text:
            return False

        if field.endswith("_ar") and not is_arabic(text):
            return True
        if field.endswith("_fr") and is_arabic(text):
            return True

        return False

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
            os.makedirs(os.path.dirname(self.debug_image_path), exist_ok=True)
            cv2.imwrite(self.debug_image_path, debug_img)
            print(f"🟢 Image debug générée : {self.debug_image_path}")

    def _get_field_lang(self, field: str) -> str:
        if field.endswith("_fr") or field == "cin" or "date" in field:
            return "fr"
        return "ar"

    def _blocks_to_result(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not blocks:
            return {"text": "", "confidence": 0, "engine": "tesseract"}

        text = " ".join(str(b.get("text", "")).strip() for b in blocks).strip()

        conf_values = []
        for b in blocks:
            try:
                c = float(b.get("confidence", 0))
                conf_values.append(c)
            except Exception:
                pass

        confidence = sum(conf_values) / len(conf_values) if conf_values else 0

        return {
            "text": text,
            "confidence": confidence,
            "engine": "tesseract"
        }

    def _should_try_paddle(
        self,
        field: str,
        tesseract_result: Dict[str, Any]
    ) -> bool:
        text = str(tesseract_result.get("text", "")).strip()
        confidence = float(tesseract_result.get("confidence", 0) or 0)

        if not text:
            return True

        if self.is_wrong_lang(field, text):
            return True

        if confidence < self.get_confidence_threshold():
            return True

        return False

    @abstractmethod
    def preprocess_zone(self, zone: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def preprocess_zone_ocr(self, zone: np.ndarray, lang : str) -> np.ndarray:
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
            zone, (x, y, w, h) = self.get_field_zone(field, rule)

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

            zone_tesseract = self.preprocess_zone(zone)
            tesseract_blocks = self.extract_text_tesseract(zone_tesseract)
            tesseract_blocks = [
                b for b in tesseract_blocks
                if self.filter_text_by_strictness(b.get("text", ""))
            ]

            tesseract_result = self._blocks_to_result(tesseract_blocks)
            final_text = str(tesseract_result.get("text", "")).strip()

            if self._should_try_paddle(field, tesseract_result):
                lang_field = self._get_field_lang(field)
                zone_paddle = self.preprocess_zone_ocr(zone, lang_field)
                paddle_result = self.paddle_text(zone_paddle, lang_field)

                paddle_text = str(paddle_result.get("text", "")).strip()

                if paddle_text:
                    final_text = paddle_text

                if self.debug:
                    print(
                        f"[OCR FALLBACK] field={field} | "
                        f"Tesseract='{tesseract_result.get('text', '')}' "
                        f"({tesseract_result.get('confidence', 0)}) -> "
                        f"Paddle='{paddle_text}' "
                        f"({paddle_result.get('confidence', 0)})"
                    )

            if "date" in field:
                final_text = self.normalize_date(final_text)

            results[field] = final_text.strip()

        self.create_debug_image(debug_img)
        results = self.reorder_identity_fields(results)
        return results