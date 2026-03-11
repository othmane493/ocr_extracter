import cv2
import numpy as np
from typing import Optional, Dict
import os
from cin_new_extractor import CINNewExtractor
from cin_old_extractor import CINOldExtractor
from utils.similarity import compare_name_ar_fr
from config.CinRecenter import CINORBAligner


class CINTypeDetector:
    @staticmethod
    def load_image(path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image impossible à charger: {path}")
        return img

    @staticmethod
    def skin_ratio(img):
        if img is None or img.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 135))
        mask2 = cv2.inRange(hsv, (0, 15, 40), (35, 255, 255))
        mask = cv2.bitwise_and(mask1, mask2)
        return np.count_nonzero(mask) / mask.size

    @classmethod
    def score_region(cls, region):
        if region is None or region.size == 0:
            return -1

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        std = np.std(gray)
        mean = np.mean(gray)
        edges = cv2.Canny(gray, 60, 140)
        edge_density = np.count_nonzero(edges) / edges.size
        skin = cls.skin_ratio(region)

        score = 0
        score += min(std / 50, 1) * 4
        score += min(edge_density / 0.12, 1) * 2
        score += min(skin / 0.2, 1) * 5
        if 60 < mean < 210:
            score += 1

        return score

    @classmethod
    def sliding_search(cls, img, x1, x2, y1, y2):
        region = img[y1:y2, x1:x2]
        rh, rw = region.shape[:2]
        if rh <= 0 or rw <= 0:
            return None

        best = None
        widths = [int(rw * 0.35), int(rw * 0.42), int(rw * 0.50)]
        heights = [int(rh * 0.50), int(rh * 0.60), int(rh * 0.70)]

        for w in widths:
            for h in heights:
                if w < 40 or h < 60:
                    continue

                ratio = w / h
                if ratio < 0.45 or ratio > 0.9:
                    continue

                step_x = max(10, w // 6)
                step_y = max(10, h // 6)

                for y in range(0, rh - h, step_y):
                    for x in range(0, rw - w, step_x):
                        roi = region[y:y + h, x:x + w]
                        score = cls.score_region(roi)

                        candidate = {
                            "x": x1 + x,
                            "y": y1 + y,
                            "w": w,
                            "h": h,
                            "score": score
                        }

                        if best is None or score > best["score"]:
                            best = candidate

        return best

    @classmethod
    def detect_big_photo(cls, image_path):
        img = cls.load_image(image_path)
        h, w = img.shape[:2]

        left_zone = (
            int(w * 0.03), int(w * 0.48),
            int(h * 0.10), int(h * 0.82)
        )
        right_zone = (
            int(w * 0.52), int(w * 0.97),
            int(h * 0.10), int(h * 0.82)
        )

        left_best = cls.sliding_search(img, left_zone[0], left_zone[1], left_zone[2], left_zone[3])
        right_best = cls.sliding_search(img, right_zone[0], right_zone[1], right_zone[2], right_zone[3])

        left_score = left_best["score"] if left_best else -1
        right_score = right_best["score"] if right_best else -1

        if left_score < 5 and right_score < 5:
            return None

        if left_score > right_score:
            left_best["side"] = "left"
            return left_best

        right_best["side"] = "right"
        return right_best

    @classmethod
    def detect_big_photo_side(cls, image_path):
        result = cls.detect_big_photo(image_path)
        if result is None:
            return None
        return result["side"]

    @classmethod
    def detect_cin_type(cls, image_path):
        side = cls.detect_big_photo_side(image_path)
        if side == "left":
            return "NEW"
        if side == "right":
            return "OLD"
        raise ValueError("Photo CIN non détectée")


class UnifiedCINExtractor:
    DEFAULT_TEMPLATES = {
        "NEW": "config/cin_new_template.json",
        "OLD": "config/cin_old_template.json"
    }

    def __init__(self, image_path, cin_type=None, template_path=None, debug=True, recenter_handler=None):
        self.image_path = image_path
        self.debug = debug
        self.recenter_handler = recenter_handler

        if cin_type is None:
            self.cin_type = CINTypeDetector.detect_cin_type(image_path)
        else:
            self.cin_type = cin_type.upper()

        if template_path is None:
            self.template_path = self.DEFAULT_TEMPLATES.get(self.cin_type)
            if self.template_path is None:
                raise ValueError(f"Type de CIN inconnu: {self.cin_type}")
        else:
            self.template_path = template_path

        if self.cin_type == "NEW":
            self.extractor = CINNewExtractor(
                self.template_path,
                self.image_path,
                self.debug,
                recenter_handler=self.recenter_handler
            )
        elif self.cin_type == "OLD":
            self.extractor = CINOldExtractor(
                self.template_path,
                self.image_path,
                self.debug,
                recenter_handler=self.recenter_handler
            )
        else:
            raise ValueError(f"Type de CIN invalide: {self.cin_type}")

    def extract(self, compare_name_func=compare_name_ar_fr):
        return self.extractor.extract(compare_name_func=compare_name_func)

def extract_cin(image_path, cin_type=None, template_path=None, debug=True, recenter_handler=None):
    extractor = UnifiedCINExtractor(
        image_path=image_path,
        cin_type=cin_type,
        template_path=template_path,
        debug=debug,
        recenter_handler=recenter_handler
    )
    return extractor.extract(compare_name_func=compare_name_ar_fr)