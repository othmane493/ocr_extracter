# unified_cin_extractor.py

from cin_new_extractor import CINNewExtractor
from cin_old_extractor import CINOldExtractor
from cin_detector import CINTypeDetector
from utils.similarity import compare_name_ar_fr


class UnifiedCINExtractor:
    DEFAULT_TEMPLATES = {
        "NEW": "config/cin_new_template.json",
        "OLD": "config/cin_old_template.json"
    }

    def __init__(self, image_path, cin_type=None, template_path=None, debug=False, recenter_handler=None):
        self.image_path = image_path
        self.debug = debug
        self.recenter_handler = recenter_handler

        if cin_type is None:
            self.cin_type = CINTypeDetector.detect_cin_type(image_path)
        else:
            self.cin_type = cin_type.upper()

        if template_path is None:
            self.template_path = self.DEFAULT_TEMPLATES.get(self.cin_type)
            if self.template_path is None:
                raise ValueError(f"Type de CIN inconnu: {self.cin_type}")
        else:
            self.template_path = template_path

        if self.cin_type == "NEW":
            self.extractor = CINNewExtractor(
                self.template_path,
                self.image_path,
                self.debug,
                recenter_handler=self.recenter_handler
            )
        elif self.cin_type == "OLD":
            self.extractor = CINOldExtractor(
                self.template_path,
                self.image_path,
                self.debug,
                recenter_handler=self.recenter_handler
            )
        else:
            raise ValueError(f"Type de CIN invalide: {self.cin_type}")

    def extract(self, compare_name_func=compare_name_ar_fr):
        return self.extractor.extract(compare_name_func=compare_name_func)


def extract_cin(image_path, cin_type=None, template_path=None, debug=False, recenter_handler=None):
    extractor = UnifiedCINExtractor(
        image_path=image_path,
        cin_type=cin_type,
        template_path=template_path,
        debug=debug,
        recenter_handler=recenter_handler
    )
    return extractor.extract(compare_name_func=compare_name_ar_fr)