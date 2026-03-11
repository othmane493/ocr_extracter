import json
import cv2
from typing import Dict, Optional


class CINOldTemplateGenerator:
    def __init__(self, width: int = 996, height: int = 623):
        self.width = width
        self.height = height

        self.fields = {
            "prenom_fr": {
                "x": -0.001,
                "y": 0.318,
                "w": 0.3343,
                "h": 0.0851,
                "lang": "fr"
            },
            "nom_fr": {
                "x": -0.001,
                "y": 0.455,
                "w": 0.3343,
                "h": 0.0803,
                "lang": "fr"
            },
            "lieu_naissance_fr": {
                "x": 0.020,
                "y": 0.660,
                "w": 0.550,
                "h": 0.058,
                "lang": "fr"
            },
            "prenom_ar": {
                "x": 0.3333,
                "y": 0.248,
                "w": 0.3454,
                "h": 0.0883,
                "lang": "ar"
            },
            "nom_ar": {
                "x": 0.3333,
                "y": 0.396,
                "w": 0.3424,
                "h": 0.0835,
                "lang": "ar"
            },
            "lieu_naissance_ar": {
                "x": 0.070,
                "y": 0.590,
                "w": 0.550,
                "h": 0.066,
                "lang": "ar"
            },
            "date_naissance": {
                "x": 0.2008,
                "y": 0.527,
                "w": 0.2731,
                "h": 0.0610,
                "lang": "ar"
            },
            "cin": {
                "x": 0.6998,
                "y": 0.7608,
                "w": 0.2098,
                "h": 0.0680,
                "lang": "fr"
            },
            "date_expiration": {
                "x": 0.2671,
                "y": 0.720,
                "w": 0.1787,
                "h": 0.0514,
                "lang": "ar"
            }
        }

    def build_template(self) -> Dict:
        return {
            "document": "CIN_MAROC",
            "width": self.width,
            "height": self.height,
            "fields": self.fields
        }

    def save_json(self, output_json_path: str) -> None:
        template = self.build_template()
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _clamp(v: int, min_v: int, max_v: int) -> int:
        return max(min_v, min(v, max_v))

    def load_size_from_image(self, image_path: str) -> None:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")

        self.height, self.width = img.shape[:2]

    def draw_debug(self, image_path: str, output_debug_path: str) -> None:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")

        img_h, img_w = img.shape[:2]

        for field_name, field in self.fields.items():
            x1 = int(round(field["x"] * img_w))
            y1 = int(round(field["y"] * img_h))
            x2 = int(round((field["x"] + field["w"]) * img_w))
            y2 = int(round((field["y"] + field["h"]) * img_h))

            x1 = self._clamp(x1, 0, img_w - 1)
            y1 = self._clamp(y1, 0, img_h - 1)
            x2 = self._clamp(x2, 1, img_w)
            y2 = self._clamp(y2, 1, img_h)

            color = (0, 0, 255)
            if field_name == "cin":
                color = (255, 0, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                field_name,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        cv2.imwrite(output_debug_path, img)


if __name__ == "__main__":
    image_path = "../images/cin_recto_1.jpeg"

    generator = CINOldTemplateGenerator()
    generator.load_size_from_image(image_path)

    generator.save_json("cin_old_template.json")
    generator.draw_debug(
        image_path=image_path,
        output_debug_path="cin_old_template_debug.png"
    )

    print("✅ JSON généré : cin_old_template.json")
    print("✅ Debug image générée : cin_old_template_debug.png")