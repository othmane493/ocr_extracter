import os
import cv2
import json
import numpy as np
from typing import Dict, Optional, List


class CINORBAligner:
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
        if path and img is not None:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            cv2.imwrite(path, img)

    @staticmethod
    def _clamp(v, min_v, max_v):
        return max(min_v, min(v, max_v))

    # =========================================================
    # ORB matching
    # =========================================================
    def compute_keypoints(self, img, mask=None):
        gray = self.to_gray(img)
        kp, des = self.orb.detectAndCompute(gray, mask)
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

            x1, y1, x2, y2 = self._field_to_pixels(field)

            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 200, 0), 2)
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

    def _field_to_pixels(self, field: Dict):
        """
        Convertit un champ template (x,y,w,h en ratio) vers des pixels
        dans le repère de l'image de référence.
        """
        x = float(field["x"])
        y = float(field["y"])
        w = float(field["w"])
        h = float(field["h"])

        x1 = int(round(x * self.ref_w))
        y1 = int(round(y * self.ref_h))
        x2 = int(round((x + w) * self.ref_w))
        y2 = int(round((y + h) * self.ref_h))

        x1 = self._clamp(x1, 0, self.ref_w - 1)
        y1 = self._clamp(y1, 0, self.ref_h - 1)
        x2 = self._clamp(x2, 1, self.ref_w)
        y2 = self._clamp(y2, 1, self.ref_h)

        if x2 <= x1:
            x2 = min(self.ref_w, x1 + 1)
        if y2 <= y1:
            y2 = min(self.ref_h, y1 + 1)

        return x1, y1, x2, y2

    def _field_to_pixels_on_size(self, field: Dict, img_w: int, img_h: int):
        """
        Convertit un champ template (x,y,w,h en ratio) vers des pixels
        pour une taille arbitraire (img_w, img_h).
        """
        x = float(field["x"])
        y = float(field["y"])
        w = float(field["w"])
        h = float(field["h"])

        x1 = int(round(x * img_w))
        y1 = int(round(y * img_h))
        x2 = int(round((x + w) * img_w))
        y2 = int(round((y + h) * img_h))

        x1 = self._clamp(x1, 0, img_w - 1)
        y1 = self._clamp(y1, 0, img_h - 1)
        x2 = self._clamp(x2, 1, img_w)
        y2 = self._clamp(y2, 1, img_h)

        if x2 <= x1:
            x2 = min(img_w, x1 + 1)
        if y2 <= y1:
            y2 = min(img_h, y1 + 1)

        return x1, y1, x2, y2

    # =========================================================
    # ORB exclusion mask
    # =========================================================
    def build_orb_exclusion_mask(
        self,
        img_shape,
        margin_ratio: float = 0.015
    ):
        """
        Masque ORB :
        - blanc (255) = zone autorisée pour ORB
        - noir (0)   = zone exclue

        Ici on exclut uniquement les champs variables du JSON.
        On n'utilise PAS de rectangles approximatifs photo/signature,
        car ils ne sont pas robustes si le cadrage change.
        """
        h, w = img_shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        if self.template is not None:
            for _, field in self.template.get("fields", {}).items():
                x1, y1, x2, y2 = self._field_to_pixels_on_size(field, w, h)

                mx = int((x2 - x1) * margin_ratio)
                my = int((y2 - y1) * margin_ratio)

                x1 = self._clamp(x1 - mx, 0, w - 1)
                y1 = self._clamp(y1 - my, 0, h - 1)
                x2 = self._clamp(x2 + mx, 1, w)
                y2 = self._clamp(y2 + my, 1, h)

                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        return mask

    def visualize_mask(self, img, mask):
        dbg = img.copy()
        overlay = dbg.copy()
        overlay[mask == 0] = (0, 0, 255)
        return cv2.addWeighted(overlay, 0.28, dbg, 0.72, 0)

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
        raw_input = cv2.imread(input_image_path)
        if raw_input is None:
            raise ValueError(f"Impossible de lire l'image input: {input_image_path}")

        # Etape 1 : redressement de la carte
        rectified = self.detect_and_rectify_card(
            input_image_path=input_image_path,
            out_rectified_path="debug/rectified_input.jpg" if debug else None,
            out_debug_contour_path="debug/card_contour.jpg" if debug else None
        )

        input_img = rectified
        if input_img is None:
            raise ValueError(f"Impossible de lire l'image input: {input_image_path}")

        input_small, scale_input = self.resize_keep_ratio(input_img, max_dim=1800)
        ref_small, scale_ref = self.resize_keep_ratio(self.reference, max_dim=1800)

        # =========================================================
        # Construction des masques ORB
        # On exclut uniquement les champs variables du JSON
        # =========================================================
        ref_mask = self.build_orb_exclusion_mask(
            ref_small.shape,
            margin_ratio=0.05
        )

        input_mask = self.build_orb_exclusion_mask(
            input_small.shape,
            margin_ratio=0.05
        )

        # =========================================================
        # Keypoints ORB uniquement sur zones autorisées
        # =========================================================
        kp1, des1 = self.compute_keypoints(ref_small, ref_mask)
        kp2, des2 = self.compute_keypoints(input_small, input_mask)

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            raise ValueError("Impossible de calculer les descripteurs ORB sur les zones autorisées.")

        # =========================================================
        # Matching KNN + ratio test
        # =========================================================
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

        # =========================================================
        # Homography : input_small -> ref_small
        # =========================================================
        H_small, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
        if H_small is None:
            raise ValueError("Homography introuvable.")

        matches_mask = mask.ravel().tolist() if mask is not None else None
        inliers = int(sum(matches_mask)) if matches_mask is not None else 0

        # input_small -> ref_small => input_original -> ref_original
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

        # =========================================================
        # Warp final
        # =========================================================
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

        # =========================================================
        # Debug
        # =========================================================
        if debug:
            base_dir = ""
            if out_debug_matches_path:
                base_dir = os.path.dirname(out_debug_matches_path)

            # 1) Debug masque référence / input
            ref_mask_dbg = self.visualize_mask(ref_small, ref_mask)
            input_mask_dbg = self.visualize_mask(input_small, input_mask)

            ref_mask_path = os.path.join(base_dir, "ref_mask.jpg") if base_dir else "ref_mask.jpg"
            input_mask_path = os.path.join(base_dir, "input_mask.jpg") if base_dir else "input_mask.jpg"

            self._safe_write(ref_mask_path, ref_mask_dbg)
            self._safe_write(input_mask_path, input_mask_dbg)

            # 2) Debug matches
            max_debug_matches = min(80, len(good))
            debug_matches = cv2.drawMatches(
                ref_small,
                kp1,
                input_small,
                kp2,
                good[:max_debug_matches],
                None,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                matchesMask=matches_mask[:max_debug_matches] if matches_mask else None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            self._safe_write(out_debug_matches_path, debug_matches)

            # 3) Debug polygone projeté sur l'image input
            ref_corners = np.float32([
                [0, 0],
                [self.ref_w - 1, 0],
                [self.ref_w - 1, self.ref_h - 1],
                [0, self.ref_h - 1]
            ]).reshape(-1, 1, 2)

            H_inv = np.linalg.inv(H)
            projected_on_input = cv2.perspectiveTransform(ref_corners, H_inv)

            debug_polygon = self.draw_polygon(
                input_img,
                projected_on_input,
                color=(0, 255, 0),
                thickness=3
            )
            self._safe_write(out_debug_polygon_path, debug_polygon)

            # 4) Debug zones de champs sur image alignée
            if self.template is not None:
                debug_fields = self.draw_field_boxes(
                    aligned,
                    field_names=field_names_for_debug
                )
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

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Ordonne 4 points dans l'ordre :
        top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Ordonne 4 points dans l'ordre :
        top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        return warped

    def detect_card_contour(
            self,
            image: np.ndarray,
            debug_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Détecte le contour principal de la carte.
        Retourne 4 points si trouvé, sinon None.
        """
        dbg = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Détection de bords
        edges = cv2.Canny(blur, 50, 150)

        # Fermer les petits trous
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        image_area = image.shape[0] * image.shape[1]

        best_quad = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < image_area * 0.15:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > best_area:
                best_quad = approx
                best_area = area

        if best_quad is None:
            return None

        pts = best_quad.reshape(4, 2).astype(np.float32)

        if debug_path:
            cv2.drawContours(dbg, [best_quad], -1, (0, 255, 0), 3)
            self._safe_write(debug_path, dbg)

        return pts

    def detect_and_rectify_card(
            self,
            input_image_path: str,
            out_rectified_path: Optional[str] = None,
            out_debug_contour_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Détecte la carte dans une photo puis la remet à plat.
        Si aucun contour correct n'est trouvé, retourne l'image d'origine.
        """
        img = cv2.imread(input_image_path)
        if img is None:
            raise ValueError(f"Impossible de lire l'image input: {input_image_path}")

        pts = self.detect_card_contour(img, debug_path=out_debug_contour_path)

        if pts is None:
            # fallback : image d'origine
            rectified = img.copy()
        else:
            rectified = self.four_point_transform(img, pts)

        self._safe_write(out_rectified_path, rectified)
        return rectified
    # =========================================================
    # Field extraction from aligned image
    # =========================================================
    def _extract_field_crops(
        self,
        aligned_image,
        field_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
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

            x1, y1, x2, y2 = self._field_to_pixels(field)
            crop = aligned_image[y1:y2, x1:x2]

            if crop is not None and crop.size > 0:
                crops[field_name] = crop

        return crops

    def save_field_crops(self, field_crops: Dict[str, np.ndarray], output_dir: str, prefix: str = ""):
        os.makedirs(output_dir, exist_ok=True)
        for field_name, crop in field_crops.items():
            filename = f"{prefix}{field_name}.jpg" if prefix else f"{field_name}.jpg"
            path = os.path.join(output_dir, filename)
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
    # =========================================================
    # CIN OLD
    # =========================================================
    cin_old_aligner = CINORBAligner(
        reference_image_path="../images/cin_recto_1.jpeg",
        template_json_path="cin_old_template.json"
    )

    cin_old_fields = [
        "prenom_fr",
        "nom_fr",
        "lieu_naissance_fr",
        "prenom_ar",
        "nom_ar",
        "lieu_naissance_ar",
        "date_naissance",
        "cin",
        "date_expiration"
    ]

    cin_old_result = cin_old_aligner.process_card(
        input_image_path="../images/old-cin-recto.jpg",
        field_names=cin_old_fields,
        save_aligned_path="debug/cin_old_aligned.jpg",
        save_debug_matches_path="debug/cin_old_orb_matches.jpg",
        save_debug_polygon_path="debug/cin_old_orb_polygon.jpg",
        save_debug_fields_path="debug/cin_old_fields_debug.jpg",
        ransac_thresh=5.0,
        min_matches=30,
        debug=True
    )

    print("=== CIN OLD ORB ===")
    print(json.dumps(cin_old_result["meta"], ensure_ascii=False, indent=2))

    cin_old_aligner.save_field_crops(
        cin_old_result["field_crops"],
        output_dir="debug/crops_cin_old",
        prefix="cin_old_"
    )

    # =========================================================
    # CIN NEW
    # =========================================================
    cin_new_aligner = CINORBAligner(
        reference_image_path="../images/cin_new.png",
        template_json_path="cin_new_template.json"
    )

    cin_new_fields = [
        "prenom_fr",
        "nom_fr",
        "lieu_naissance_fr",
        "prenom_ar",
        "nom_ar",
        "lieu_naissance_ar",
        "date_naissance",
        "date_expiration",
        "cin"
    ]

    cin_new_result = cin_new_aligner.process_card(
        input_image_path="../images/cin_new1.jpg",
        field_names=cin_new_fields,
        save_aligned_path="debug/cin_new_aligned.jpg",
        save_debug_matches_path="debug/cin_new_orb_matches.jpg",
        save_debug_polygon_path="debug/cin_new_orb_polygon.jpg",
        save_debug_fields_path="debug/cin_new_fields_debug.jpg",
        ransac_thresh=5.0,
        min_matches=30,
        debug=True
    )

    print("=== CIN NEW ORB ===")
    print(json.dumps(cin_new_result["meta"], ensure_ascii=False, indent=2))

    cin_new_aligner.save_field_crops(
        cin_new_result["field_crops"],
        output_dir="debug/crops_cin_new",
        prefix="cin_new_"
    )