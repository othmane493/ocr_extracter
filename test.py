import cv2
from paddleocr import PaddleOCR


class PaddlePreprocessTester:
    def __init__(self):
        self.reader_ar = PaddleOCR(
            lang="ar",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )

    def paddle_text(self, zone):
        try:
            raw_result = self.reader_ar.predict(zone)

            if not raw_result:
                return "", 0, []

            item = raw_result[0] if isinstance(raw_result, list) and raw_result else raw_result

            dt_polys = item.get("dt_polys", []) if isinstance(item, dict) else []
            rec_texts = item.get("rec_texts", []) if isinstance(item, dict) else []
            rec_scores = item.get("rec_scores", []) if isinstance(item, dict) else []

            text = " | ".join(str(t).strip() for t in rec_texts if str(t).strip())
            return text, len(dt_polys), rec_scores

        except Exception as e:
            return f"ERROR: {repr(e)}", 0, []

    def preprocess_raw(self, zone):
        return zone.copy()

    def preprocess_resize_x2(self, zone):
        img = zone.copy()
        h, w = img.shape[:2]
        if h < 60 or w < 180:
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return img

    def preprocess_resize_x3(self, zone):
        img = zone.copy()
        h, w = img.shape[:2]
        if h < 60 or w < 180:
            img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        return img

    def preprocess_resize_x4(self, zone):
        img = zone.copy()
        h, w = img.shape[:2]
        if h < 60 or w < 180:
            img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        return img

    def preprocess_resize_x3_contrast(self, zone):
        img = self.preprocess_resize_x3(zone)
        img = cv2.convertScaleAbs(img, alpha=1.15, beta=5)
        return img

    def preprocess_resize_x3_blur(self, zone):
        img = self.preprocess_resize_x3(zone)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def preprocess_resize_x3_gray_bgr(self, zone):
        img = self.preprocess_resize_x3(zone)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_zone(self, image_path, crop=None):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")

        zone = img.copy()
        if crop is not None:
            x, y, w, h = crop
            zone = img[y:y + h, x:x + w]

        variants = {
            "raw": self.preprocess_raw(zone),
            "resize_x2": self.preprocess_resize_x2(zone),
            "resize_x3": self.preprocess_resize_x3(zone),
            "resize_x4": self.preprocess_resize_x4(zone),
            "resize_x3_contrast": self.preprocess_resize_x3_contrast(zone),
            "resize_x3_blur": self.preprocess_resize_x3_blur(zone),
            "resize_x3_gray_bgr": self.preprocess_resize_x3_gray_bgr(zone),
        }

        print("\n===== TEST PREPROCESS =====\n")
        for name, processed in variants.items():
            text, nb_boxes, scores = self.paddle_text(processed)

            print(f"[{name}]")
            print(f"shape   = {processed.shape}")
            print(f"boxes   = {nb_boxes}")
            print(f"text    = {text}")
            print(f"scores  = {scores}")
            print("-" * 40)

            cv2.imshow(name, processed)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tester = PaddlePreprocessTester()

    image_path = "debug_chaimae.jpg"
    crop = None

    tester.test_zone(image_path, crop)