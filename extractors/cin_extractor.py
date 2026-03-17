"""
Extracteur unifié pour les CIN (Old et New)
"""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from unified_cin_extractor import extract_cin
from config.CinRecenter import CINORBAligner


class CINExtractor:
    def __init__(self):
        pass

    def extract(self, image_path, cin_type):
        if cin_type == "cin_old":
            detected_type = "OLD"
            template_path = "config/cin_old_template.json"
            reference_path = "images/cin_recto_1.jpeg"

        elif cin_type == "cin_new":
            detected_type = "NEW"
            template_path = "config/cin_new_template.json"
            reference_path = "images/cin_new.png"

        else:
            raise ValueError(f"Type de CIN invalide: {cin_type}")

        recenter_handler = None

        if os.path.exists(reference_path):
            recenter_handler = CINORBAligner(
                reference_image_path=reference_path,
                template_json_path=template_path
            )

        return extract_cin(
            image_path=image_path,
            cin_type=detected_type,
            template_path=template_path,
            debug=False,
            recenter_handler=recenter_handler
        )