"""
Extracteur unifié pour les CIN (Old et New)
Utilise la fonction extract_cin
"""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cin_detector import extract_cin


class CINExtractor:
    def __init__(self):
        pass

    def extract(self, image_path, cin_type):
        if cin_type == "cin_old":
            detected_type = "OLD"
        elif cin_type == "cin_new":
            detected_type = "NEW"
        else:
            raise ValueError(f"Type de CIN invalide: {cin_type}")

        return extract_cin(
            image_path=image_path,
            cin_type=detected_type,
            debug=True
        )