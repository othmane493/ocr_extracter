import cv2
import json
import numpy as np
from typing import Dict, Optional, List


class CarteGriseORBAligner:
    def __init__(
        self,
        reference_image_path: str,
        template_json_path: Optional[str] = None
    ):
        self.reference_image_path = reference_image_path
        self.template_json_path = template_json_path

        self.reference = cv2.imread(reference_image_path)
        if self.reference is None:
            raise ValueError(f"Impossible de lire l'image de référence: {reference_image_path}")

        self.ref_h, self.ref_w = self.reference.shape[:2]

        self.template = None
        if template_json_path:
            with open(template_json_path, "r", encoding="utf-8") as f:
                self.template = json.load(f)

        self.orb = cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=15
        )

    # =========================================================
    # Utils
    # =========================================================
    @staticmethod
    def to_gray(img):
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def resize_keep_ratio(img, max_dim=1800):
        h, w = img.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        if scale == 1.0:
            return img, 1.0

        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized, scale

    @staticmethod
    def draw_polygon(img, pts, color=(0, 255, 0), thickness=3):
        out = img.copy()
        pts = np.int32(pts).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
        return out

    @staticmethod
    def _safe_write(path: Optional[str], img) -> None:
        if path:
            cv2.imwrite(path, img)

    # =========================================================
    # ORB matching
    # =========================================================
    def compute_keypoints(self, img):
        gray = self.to_gray(img)
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    # =========================================================
    # Debug template boxes
    # =========================================================
    def draw_field_boxes(self, aligned_img, field_names: Optional[List[str]] = None):
        """
        Dessine les rectangles des champs du template sur l'image alignée.
        """
        if self.template is None:
            return None

        dbg = aligned_img.copy()
        fields = self.template.get("fields", {})

        if field_names is None:
            field_names = list(fields.keys())

        for field_name in field_names:
            field = fields.get(field_name)
            if not field:
                continue

            x1 = int(field["x"] * self.ref_w)
            y1 = int(field["y"] * self.ref_h)
            x2 = int((field["x"] + field["w"]) * self.ref_w)
            y2 = int((field["y"] + field["h"]) * self.ref_h)

            cv2.rectangle(
                dbg,
                (x1, y1),
                (x2, y2),
                (0, 200, 0),
                2
            )

            cv2.putText(
                dbg,
                field_name,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 0),
                1,
                cv2.LINE_AA
            )

        return dbg

    # =========================================================
    # Internal align
    # =========================================================
    def _align_image_internal(
        self,
        input_image_path: str,
        out_aligned_path: Optional[str] = None,
        out_debug_matches_path: Optional[str] = None,
        out_debug_polygon_path: Optional[str] = None,
        out_debug_fields_path: Optional[str] = None,
        field_names_for_debug: Optional[List[str]] = None,
        ransac_thresh: float = 5.0,
        min_matches: int = 30,
        debug: bool = False
    ) -> Dict:
        input_img = cv2.imread(input_image_path)
        if input_img is None:
            raise ValueError(f"Impossible de lire l'image input: {input_image_path}")

        input_small, scale_input = self.resize_keep_ratio(input_img, max_dim=1800)
        ref_small, scale_ref = self.resize_keep_ratio(self.reference, max_dim=1800)

        kp1, des1 = self.compute_keypoints(ref_small)
        kp2, des2 = self.compute_keypoints(input_small)

        if des1 is None or des2 is None:
            raise ValueError("Impossible de calculer les descripteurs ORB.")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn_matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < min_matches:
            raise ValueError(
                f"Pas assez de bons matchs ORB: {len(good)} trouvés, minimum requis = {min_matches}"
            )

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # dst_pts -> src_pts
        H_small, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
        if H_small is None:
            raise ValueError("Homography introuvable.")

        matches_mask = mask.ravel().tolist() if mask is not None else None
        inliers = int(sum(matches_mask)) if matches_mask is not None else 0

        # input_small -> ref_small  => input_original -> ref_original
        S_in = np.array([
            [scale_input, 0, 0],
            [0, scale_input, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        S_ref = np.array([
            [scale_ref, 0, 0],
            [0, scale_ref, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        H = np.linalg.inv(S_ref) @ H_small @ S_in

        aligned = cv2.warpPerspective(
            input_img,
            H,
            (self.ref_w, self.ref_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        self._safe_write(out_aligned_path, aligned)

        debug_matches = None
        debug_polygon = None
        debug_fields = None

        if debug:
            debug_matches = cv2.drawMatches(
                ref_small, kp1,
                input_small, kp2,
                good[:80],
                None,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                matchesMask=matches_mask[:80] if matches_mask else None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            self._safe_write(out_debug_matches_path, debug_matches)

            ref_corners = np.float32([
                [0, 0],
                [self.ref_w - 1, 0],
                [self.ref_w - 1, self.ref_h - 1],
                [0, self.ref_h - 1]
            ]).reshape(-1, 1, 2)

            H_inv = np.linalg.inv(H)
            projected_on_input = cv2.perspectiveTransform(ref_corners, H_inv)
            debug_polygon = self.draw_polygon(input_img, projected_on_input, color=(0, 255, 0), thickness=3)
            self._safe_write(out_debug_polygon_path, debug_polygon)

            if self.template is not None:
                debug_fields = self.draw_field_boxes(aligned, field_names=field_names_for_debug)
                self._safe_write(out_debug_fields_path, debug_fields)

        result = {
            "aligned_image": aligned,
            "meta": {
                "reference_width": self.ref_w,
                "reference_height": self.ref_h,
                "input_image": input_image_path,
                "aligned_image_path": out_aligned_path,
                "debug_enabled": debug,
                "debug_matches_image_path": out_debug_matches_path if debug else None,
                "debug_polygon_image_path": out_debug_polygon_path if debug else None,
                "debug_fields_image_path": out_debug_fields_path if debug else None,
                "total_good_matches": len(good),
                "inliers": inliers,
                "homography": H.tolist()
            },
            "debug_matches_image": debug_matches,
            "debug_polygon_image": debug_polygon,
            "debug_fields_image": debug_fields
        }

        if self.template:
            result["meta"]["template_document"] = self.template.get("document")
            result["meta"]["template_json_path"] = self.template_json_path
            result["template"] = self.template
        else:
            result["template"] = None

        return result

    # =========================================================
    # Field extraction from aligned image
    # =========================================================
    def _extract_field_crops(self, aligned_image, field_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        if self.template is None:
            raise ValueError("Aucun template JSON fourni.")

        fields = self.template.get("fields", {})
        if field_names is None:
            field_names = list(fields.keys())

        crops = {}
        for field_name in field_names:
            field = fields.get(field_name)
            if not field:
                continue

            x1 = int(field["x"] * self.ref_w)
            y1 = int(field["y"] * self.ref_h)
            x2 = int((field["x"] + field["w"]) * self.ref_w)
            y2 = int((field["y"] + field["h"]) * self.ref_h)

            crops[field_name] = aligned_image[y1:y2, x1:x2]

        return crops

    def save_field_crops(self, field_crops: Dict[str, np.ndarray], output_dir: str, prefix: str = ""):
        for field_name, crop in field_crops.items():
            filename = f"{prefix}{field_name}.jpg" if prefix else f"{field_name}.jpg"
            path = f"{output_dir.rstrip('/')}/{filename}"
            cv2.imwrite(path, crop)

    # =========================================================
    # Public main method
    # =========================================================
    def process_card(
        self,
        input_image_path: str,
        field_names: Optional[List[str]] = None,
        save_aligned_path: Optional[str] = None,
        save_debug_matches_path: Optional[str] = None,
        save_debug_polygon_path: Optional[str] = None,
        save_debug_fields_path: Optional[str] = None,
        ransac_thresh: float = 5.0,
        min_matches: int = 30,
        debug: bool = False
    ) -> Dict:
        result = self._align_image_internal(
            input_image_path=input_image_path,
            out_aligned_path=save_aligned_path,
            out_debug_matches_path=save_debug_matches_path,
            out_debug_polygon_path=save_debug_polygon_path,
            out_debug_fields_path=save_debug_fields_path,
            field_names_for_debug=field_names,
            ransac_thresh=ransac_thresh,
            min_matches=min_matches,
            debug=debug
        )

        aligned_image = result["aligned_image"]

        field_crops = {}
        if self.template is not None:
            field_crops = self._extract_field_crops(aligned_image, field_names=field_names)

        return {
            "aligned_image": aligned_image,
            "template": result["template"],
            "meta": result["meta"],
            "field_crops": field_crops,
            "debug_matches_image": result["debug_matches_image"],
            "debug_polygon_image": result["debug_polygon_image"],
            "debug_fields_image": result["debug_fields_image"]
        }


if __name__ == "__main__":
    # -----------------------------
    # RECTO
    # -----------------------------
    recto_aligner = CarteGriseORBAligner(
        reference_image_path="../images/carte-grise-recto.jpg",
        template_json_path="carte_grise_recto_template.json"
    )

    recto_fields = [
        "registration_number_matriculate",
        "previous_registration",
        "first_registration_date",
        "first_usage_date",
        "mutation_date",
        "usage",
        "owner_fr",
        "owner_ar",
        "address",
        "expiry_date"
    ]

    recto_result = recto_aligner.process_card(
        input_image_path="../images/carte_grise_recto_1.jpeg",
        field_names=recto_fields,
        save_aligned_path="recto_aligned_orb.jpg",
        save_debug_matches_path="recto_orb_matches.jpg",
        save_debug_polygon_path="recto_orb_polygon.jpg",
        save_debug_fields_path="recto_fields_debug.jpg",
        ransac_thresh=5.0,
        min_matches=30,
        debug=False
    )

    print("=== RECTO ORB ===")
    print(json.dumps(recto_result["meta"], ensure_ascii=False, indent=2))

    # Exemple : sauvegarder un crop précis
    if "registration_number_matriculate" in recto_result["field_crops"]:
        cv2.imwrite(
            "recto_registration_crop.jpg",
            recto_result["field_crops"]["registration_number_matriculate"]
        )

    # -----------------------------
    # VERSO
    # -----------------------------
    verso_aligner = CarteGriseORBAligner(
        reference_image_path="../images/carte_grise_verso.jpg",
        template_json_path="carte_grise_verso_template.json"
    )

    verso_fields = [
        "Marque",
        "Type",
        "Genre",
        "Modèle",
        "Type_Carburant",
        "Number_chassis",
        "Number_Cylinders",
        "Puissance_Fiscale",
        "Number_Places",
        "PTAC",
        "Poids_vide",
        "PTRA",
        "Restrictions"
    ]

    verso_result = verso_aligner.process_card(
        input_image_path="../images/carte_grise_verso_1.jpeg",
        field_names=verso_fields,
        save_aligned_path="verso_aligned_orb.jpg",
        save_debug_matches_path="verso_orb_matches.jpg",
        save_debug_polygon_path="verso_orb_polygon.jpg",
        save_debug_fields_path="verso_fields_debug.jpg",
        ransac_thresh=5.0,
        min_matches=30,
        debug=True
    )

    print("=== VERSO ORB ===")
    print(json.dumps(verso_result["meta"], ensure_ascii=False, indent=2))