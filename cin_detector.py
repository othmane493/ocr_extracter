# cin_detector.py

import cv2
import numpy as np


class CINTypeDetector:
    DETECT_MAX_WIDTH = 900
    MIN_VALID_SCORE = 5.0

    @staticmethod
    def load_image(path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image impossible à charger: {path}")
        return img

    @staticmethod
    def resize_for_detection(img, max_width=DETECT_MAX_WIDTH):
        h, w = img.shape[:2]
        if w <= max_width:
            return img, 1.0, 1.0

        scale = max_width / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        scale_x = w / float(new_w)
        scale_y = h / float(new_h)
        return resized, scale_x, scale_y

    @staticmethod
    def _rect_sum(ii, x, y, w, h):
        x2 = x + w
        y2 = y + h
        return ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x]

    @classmethod
    def _prepare_features(cls, img):
        det_img, scale_x, scale_y = cls.resize_for_detection(img)

        gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

        edges = cv2.Canny(gray_blur, 60, 140)

        ycrcb = cv2.cvtColor(det_img, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(det_img, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 135))
        mask2 = cv2.inRange(hsv, (0, 15, 40), (35, 255, 255))
        skin = cv2.bitwise_and(mask1, mask2)

        gray_f = gray.astype(np.float32)
        gray_sq = gray_f * gray_f
        edges_bin = (edges > 0).astype(np.uint8)
        skin_bin = (skin > 0).astype(np.uint8)

        return {
            "img": det_img,
            "gray": gray,
            "h": det_img.shape[0],
            "w": det_img.shape[1],
            "scale_x": scale_x,
            "scale_y": scale_y,
            "gray_ii": cv2.integral(gray_f),
            "gray_sq_ii": cv2.integral(gray_sq),
            "edge_ii": cv2.integral(edges_bin),
            "skin_ii": cv2.integral(skin_bin),
        }

    @classmethod
    def score_region_fast(cls, feat, x, y, w, h):
        area = float(w * h)
        if area <= 0:
            return -1.0

        gray_sum = cls._rect_sum(feat["gray_ii"], x, y, w, h)
        gray_sq_sum = cls._rect_sum(feat["gray_sq_ii"], x, y, w, h)
        edge_sum = cls._rect_sum(feat["edge_ii"], x, y, w, h)
        skin_sum = cls._rect_sum(feat["skin_ii"], x, y, w, h)

        mean = gray_sum / area
        var = max((gray_sq_sum / area) - (mean * mean), 0.0)
        std = np.sqrt(var)
        edge_density = edge_sum / area
        skin_ratio = skin_sum / area

        score = 0.0
        score += min(std / 50.0, 1.0) * 4.0
        score += min(edge_density / 0.12, 1.0) * 2.0
        score += min(skin_ratio / 0.20, 1.0) * 5.0

        if 60 < mean < 210:
            score += 1.0

        return score

    @classmethod
    def sliding_search_fast(cls, feat, x1, x2, y1, y2):
        rw = x2 - x1
        rh = y2 - y1

        if rw <= 0 or rh <= 0:
            return None

        best = None
        widths = [int(rw * 0.35), int(rw * 0.42), int(rw * 0.50)]
        heights = [int(rh * 0.50), int(rh * 0.60), int(rh * 0.70)]

        for w in widths:
            for h in heights:
                if w < 30 or h < 40:
                    continue

                ratio = w / float(h)
                if ratio < 0.45 or ratio > 0.90:
                    continue

                step_x = max(8, w // 5)
                step_y = max(8, h // 5)

                max_y = y2 - h
                max_x = x2 - w

                for y in range(y1, max_y + 1, step_y):
                    for x in range(x1, max_x + 1, step_x):
                        score = cls.score_region_fast(feat, x, y, w, h)

                        if best is None or score > best["score"]:
                            best = {
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                                "score": score,
                            }

        return best

    @classmethod
    def detect_big_photo(cls, image_path):
        img = cls.load_image(image_path)
        feat = cls._prepare_features(img)

        h = feat["h"]
        w = feat["w"]

        left_zone = (
            int(w * 0.03), int(w * 0.48),
            int(h * 0.10), int(h * 0.82)
        )
        right_zone = (
            int(w * 0.52), int(w * 0.97),
            int(h * 0.10), int(h * 0.82)
        )

        left_best = cls.sliding_search_fast(
            feat, left_zone[0], left_zone[1], left_zone[2], left_zone[3]
        )
        right_best = cls.sliding_search_fast(
            feat, right_zone[0], right_zone[1], right_zone[2], right_zone[3]
        )

        left_score = left_best["score"] if left_best else -1.0
        right_score = right_best["score"] if right_best else -1.0

        if left_score < cls.MIN_VALID_SCORE and right_score < cls.MIN_VALID_SCORE:
            return None

        if left_score > right_score:
            left_best["side"] = "left"
            left_best["x"] = int(left_best["x"] * feat["scale_x"])
            left_best["y"] = int(left_best["y"] * feat["scale_y"])
            left_best["w"] = int(left_best["w"] * feat["scale_x"])
            left_best["h"] = int(left_best["h"] * feat["scale_y"])
            return left_best

        right_best["side"] = "right"
        right_best["x"] = int(right_best["x"] * feat["scale_x"])
        right_best["y"] = int(right_best["y"] * feat["scale_y"])
        right_best["w"] = int(right_best["w"] * feat["scale_x"])
        right_best["h"] = int(right_best["h"] * feat["scale_y"])
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