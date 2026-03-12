import cv2
import numpy as np
from typing import List, Dict
from cin_extractor_base import CINExtractor


class CINOldExtractor(CINExtractor):
    def __init__(
        self,
        template_path: str,
        image_path: str,
        debug: bool = False,
        recenter_handler=None
    ):
        super().__init__(template_path, image_path, debug, recenter_handler=recenter_handler)
        self.cin_type = "CIN_OLD"
        self.pivot_img_path = "enhanced_cin_old.jpg"

    def preprocess_zone(self, zone: np.ndarray, scale : float = 2.0) -> np.ndarray:
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

        # Augmentation du contraste
        gray = cv2.addWeighted(gray, 2.0, gray, -0.5, 0)

        # Binarisation avec Otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.debug:
            cv2.imwrite(self.pivot_img_path, thresh)
        h, w = thresh.shape[:2]
        return cv2.resize(thresh, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    def preprocess_zone_ocr(self, zone: np.ndarray, lang: str, scale : float = 2.0) -> np.ndarray:

        img = zone.copy()

        if lang == "ar":
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return img

    def preprocessing_alternative(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        alpha = 2.0
        beta = -180
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        _, mask = cv2.threshold(enhanced, 70, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        text_only = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        background = cv2.bitwise_and(gray, gray, mask=mask_inv)
        background = cv2.GaussianBlur(background, (5, 5), 0)

        final = cv2.add(text_only, background)
        cv2.imwrite(self.pivot_img_path, final)
        return cv2.imread(self.pivot_img_path)

    def extract_text_tesseract(self, zone: np.ndarray) -> List[Dict]:
        try:
            from utils.ocr_utils import extract_text_tesseract
            return extract_text_tesseract(zone)
        except ImportError:
            return []

    def get_confidence_threshold(self) -> int:
        return 80


def create_cin_old_extractor(
    image_path: str,
    template_path: str = "config/cin_old_template.json",
    debug: bool = False,
    recenter_handler=None
) -> CINOldExtractor:
    return CINOldExtractor(
        template_path=template_path,
        image_path=image_path,
        debug=debug,
        recenter_handler=recenter_handler
    )