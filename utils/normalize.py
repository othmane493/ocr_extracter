import re
from typing import Optional


class Normalize:
    OCR_ARABIC_FIXES = {
        "9": "و",  # confusion OCR fréquente
    }

    @staticmethod
    def normalize_date(text: str) -> Optional[str]:
        """
        Normalise une date OCR vers le format dd/MM/yyyy.

        Exemples :
        - 16406/2022   -> 16/06/2022
        - 1606/2022    -> 16/06/2022
        - 164062022    -> 16/06/2022
        - 1630642022   -> 16/06/2022
        """
        if not text:
            return None

        # garder uniquement les chiffres
        digits = re.sub(r"\D", "", text)

        # il faut au moins jour/mois/année
        if len(digits) < 8:
            return None

        # année = 4 derniers chiffres
        year = digits[-4:]
        dm = digits[:-4]

        # il faut au moins 2 chiffres pour le jour et 2 pour le mois
        if len(dm) < 4:
            return None

        # jour = 2 premiers chiffres
        day = dm[:2]

        # mois = 2 derniers chiffres
        month = dm[-2:]

        try:
            day = max(1, min(int(day), 31))
            month = max(1, min(int(month), 12))
        except ValueError:
            return None

        return f"{day:02d}/{month:02d}/{year}"

    @staticmethod
    def normalize_value(value: str, allowed_chars: str) -> str:
        """
        Supprime tous les caractères non autorisés.
        allowed_chars ex: A-Za-z0-9-_
        """
        if not value:
            return value

        pattern = rf"[^{allowed_chars}]"
        return re.sub(pattern, "", value)

    @staticmethod
    def normalize_matricule(value: str) -> Optional[str]:
        """
        Normalise un matricule vers le format :
        nombre-lettre_arabe-nombre

        Exemples :
        - 40313-9-6     -> 40313-و-6
        - 40313-و-9     -> 40313-و-9
        - 1-ط37777      -> 1-ط-37777
        - 1ط37777       -> 1-ط-37777
        - 1/ط/37777     -> 1-ط-37777
        """

        if not value:
            return None

        value = value.strip()

        # uniformiser les séparateurs
        value = value.replace("/", "-").replace("_", "-").replace("—", "-")
        value = re.sub(r"\s+", "", value)

        # corriger les confusions OCR les plus fréquentes
        for wrong, correct in Normalize.OCR_ARABIC_FIXES.items():
            value = value.replace(wrong, correct)

        # chercher : chiffres + lettre arabe + chiffres
        # avec ou sans tirets
        match = re.match(r"^(\d+)-?([\u0600-\u06FF])-?(\d+)$", value)
        if match:
            part1, middle, part3 = match.groups()
            return f"{part1}-{middle}-{part3}"

        # cas où les tirets sont présents mais mal placés
        parts = [p for p in re.split(r"-+", value) if p]

        if len(parts) == 3:
            part1, middle, part3 = parts
        elif len(parts) == 2:
            left, right = parts

            # cas: 1-ط37777
            m = re.match(r"^([\u0600-\u06FF])(\d+)$", right)
            if m and left.isdigit():
                middle, part3 = m.groups()
                part1 = left
                return f"{part1}-{middle}-{part3}"

            # cas: 37777ط-1
            m = re.match(r"^(\d+)([\u0600-\u06FF])$", left)
            if m and right.isdigit():
                part1, middle = m.groups()
                part3 = right
                return f"{part1}-{middle}-{part3}"

            return value
        else:
            return value

        # nettoyage final
        part1 = re.sub(r"\D", "", part1)
        part3 = re.sub(r"\D", "", part3)
        middle = middle.strip()

        if not part1 or not part3:
            return value

        if not re.fullmatch(r"[\u0600-\u06FF]", middle):
            return value

        return f"{part1}-{middle}-{part3}"