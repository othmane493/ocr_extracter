from typing import Optional
import cv2
import numpy as np
import pytesseract
import time


class ProcessImage:
    def __init__(self, image_path: Optional[str] = None, image: Optional[np.ndarray] = None):
        self.image_path = image_path
        self.image = image

        if self.image is None and self.image_path:
            self.image = cv2.imread(self.image_path)

        if self.image is None:
            raise ValueError("Aucune image valide fournie.")

    # -------------------------
    # Méthodes utilitaires
    # -------------------------
    def to_gray(self, img: Optional[np.ndarray] = None) -> np.ndarray:
        img = img if img is not None else self.image
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def threshold(self, img: Optional[np.ndarray] = None) -> np.ndarray:
        img = img if img is not None else self.image
        gray = self.to_gray(img)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def denoise(self, img: Optional[np.ndarray] = None) -> np.ndarray:
        img = img if img is not None else self.image
        if len(img.shape) == 2:
            return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    def resize(self, scale: float = 2.0, img: Optional[np.ndarray] = None) -> np.ndarray:
        img = img if img is not None else self.image
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    def invert(self, img: Optional[np.ndarray] = None) -> np.ndarray:
        img = img if img is not None else self.image
        return 255 - img

    def binarize_keep_black_fast(self, black_threshold: int = 140, img: Optional[np.ndarray] = None) -> np.ndarray:
        img = img if img is not None else self.image
        gray = self.to_gray(img)
        result = np.where(gray <= black_threshold, 0, 255).astype(np.uint8)
        return result

    def preprocess_date(self, thresh: float, scale: float = 3.0, img: Optional[np.ndarray] = None) -> np.ndarray:
        img = img if img is not None else self.image
        gray = self.to_gray(img)
        gray = self.resize(scale, gray)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return binary

    # -------------------------
    # Modes de traitement
    # -------------------------
    def mode_ocr(self, black_threshold=145, scale=1.0) -> np.ndarray:
        """
        Prétraitement générique pour OCR.
        """
        processed = self.preprocess_date(black_threshold, scale)
        return processed

    def mode_tesseract(self, scale: float = 2.5):
        """
        Prétraitement optimisé pour Tesseract carte grise
        """

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 148, 253, cv2.THRESH_BINARY_INV)
        # agrandir pour Tesseract
        thresh = self.resize(scale, thresh)
        #cv2.imwrite(f"debug_{time.time()}.jpg", thresh)
        return thresh, self.image

    def mode_paddle(self) -> np.ndarray:
        """
        Prétraitement léger pour PaddleOCR.
        """
        gray = self.to_gray(self.image)
        gray = self.resize(2.0, gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    def detect_double_dash(self) -> str:
        gray = self.to_gray(self.image)
        config = r'--psm 10 -c tessedit_char_whitelist=-'
        return pytesseract.image_to_string(gray, config=config)

    # -------------------------
    # Méthode principale
    # -------------------------
    def process(self, mode: str):
        modes = {
            "mode_cg_pytesseract": self.mode_tesseract,
            "mode_cg_ocr": lambda: self.mode_ocr(black_threshold=135, scale=1.0),
            "mode_cg_paddle": self.mode_paddle,
            "detect_double_dash": self.detect_double_dash,
        }

        if mode not in modes:
            raise ValueError(f"Mode inconnu : {mode}. Modes disponibles : {list(modes.keys())}")

        return modes[mode]()