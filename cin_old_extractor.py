import cv2
import numpy as np
from typing import List, Dict
from cin_extractor_base import CINExtractor


class CINOldExtractor(CINExtractor):
    def __init__(
        self,
        template_path: str,
        image_path: str,
        debug: bool = True,
        recenter_handler=None
    ):
        super().__init__(template_path, image_path, debug, recenter_handler=recenter_handler)
        self.cin_type = "CIN_OLD"
        self.pivot_img_path = "enhanced_cin_old.jpg"

    def preprocess_zone(self, zone: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        gray = cv2.addWeighted(gray, 1.90, gray, -0.6, 0)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.debug:
            cv2.imwrite(self.pivot_img_path, thresh)

        return thresh

    def preprocess_zone_ocr(self, zone: np.ndarray) -> np.ndarray:
        if zone is None or zone.size == 0:
            return zone

        img = zone.copy()
        h, w = img.shape[:2]

        if h < 70 or w < 180:
            img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

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
        return 60


def create_cin_old_extractor(
    image_path: str,
    template_path: str = "config/cin_old_template.json",
    debug: bool = True,
    recenter_handler=None
) -> CINOldExtractor:
    return CINOldExtractor(
        template_path=template_path,
        image_path=image_path,
        debug=debug,
        recenter_handler=recenter_handler
    )