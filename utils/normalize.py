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

        Cas gérés :
        - 40313-9-6     -> 40313-و-6
        - 73138-1-7     -> 73138-ط-7
        - 40313-و-9     -> 40313-و-9
        - 1-ط37777      -> 1-ط-37777
        - 1ط37777       -> 1-ط-37777
        - 1/ط/37777     -> 1-ط-37777
        - 37777ط1       -> 37777-ط-1
        - 37777-ط1      -> 37777-ط-1
        - 377771-ط      -> 37777-ط-1
        - 377771ط       -> 37777-ط-1
        - 73138-7-1     -> 73138-ط-1   (si 7 est confondu avec la lettre)
        """

        if not value:
            return None

        value = value.strip()
        if not value:
            return None

        # 1) uniformiser les séparateurs
        value = (
            value.replace("/", "-")
            .replace("_", "-")
            .replace("—", "-")
            .replace("–", "-")
        )
        value = re.sub(r"\s+", "", value)
        value = re.sub(r"-{2,}", "-", value).strip("-")

        # 2) garder seulement chiffres / lettres arabes / tirets
        value = re.sub(r"[^\d\u0600-\u06FF-]", "", value)
        if not value:
            return None

        arabic_letter_pattern = r"[\u0600-\u06FF]"

        def is_arabic_letter(s: str) -> bool:
            return bool(re.fullmatch(arabic_letter_pattern, s or ""))

        def normalize_middle_token(token: str) -> str:
            """
            Corrige le token central si OCR l'a lu comme chiffre
            alors qu'il représente une lettre arabe.
            """
            if not token:
                return token

            # si déjà lettre arabe, on garde
            if is_arabic_letter(token):
                return token

            # mapping OCR ambigu uniquement pour la position centrale
            middle_fixes = {
                "1": "ا",
                "9": "و",
                "7": "ط",
            }

            return middle_fixes.get(token, token)

        def rebuild(a: str, b: str, c: str) -> str:
            a = re.sub(r"\D", "", a or "")
            c = re.sub(r"\D", "", c or "")
            b = normalize_middle_token(b)

            if not a or not c:
                return value

            return f"{a}-{b}-{c}"

        # ------------------------------------------------------------------
        # Cas déjà bien séparé : a-b-c
        # ------------------------------------------------------------------
        parts = [p for p in re.split(r"-+", value) if p]

        if len(parts) == 3:
            a, b, c = parts
            return rebuild(a, b, c)

        # ------------------------------------------------------------------
        # Cas compact : chiffres + lettre/chiffre central + chiffres
        # ex: 37777ط1, 7313817, 377771ط (traité plus bas)
        # ------------------------------------------------------------------
        m = re.fullmatch(rf"(\d+)({arabic_letter_pattern}|\d)(\d+)", value)
        if m:
            a, b, c = m.groups()
            return rebuild(a, b, c)

        # ------------------------------------------------------------------
        # Cas à 2 blocs
        # ------------------------------------------------------------------
        if len(parts) == 2:
            left, right = parts

            # cas: 1-ط37777  -> 1-ط-37777
            m = re.fullmatch(rf"({arabic_letter_pattern}|\d)(\d+)", right)
            if left.isdigit() and m:
                b, c = m.groups()
                return rebuild(left, b, c)

            # cas: 37777ط-1  -> 37777-ط-1
            m = re.fullmatch(rf"(\d+)({arabic_letter_pattern}|\d)", left)
            if right.isdigit() and m:
                a, b = m.groups()
                return rebuild(a, b, right)

            # cas: 377771-ط  -> 37777-ط-1
            if left.isdigit() and re.fullmatch(arabic_letter_pattern, right):
                if len(left) >= 2:
                    return rebuild(left[:-1], right, left[-1])

            # cas: 377771-1  -> 37777-ط-1
            if left.isdigit() and right.isdigit():
                if len(right) == 1 and len(left) >= 2:
                    # on suppose que le dernier chiffre du bloc gauche est la partie 3
                    # et le bloc droit est une lettre mal lue
                    return rebuild(left[:-1], right, left[-1])

        # ------------------------------------------------------------------
        # Cas sans tirets : chiffres + lettre/chiffre final
        # ex: 377771ط -> 37777-ط-1
        # ex: 7313817 -> peut devenir 73138-1-7 puis 73138-ط-7
        # ------------------------------------------------------------------
        m = re.fullmatch(rf"(\d+)({arabic_letter_pattern}|\d)", value)
        if m:
            digits, middle = m.groups()
            if len(digits) >= 2:
                return rebuild(digits[:-1], middle, digits[-1])

        # ------------------------------------------------------------------
        # Cas tout en chiffres, longueur suffisante
        # ex: 7313817 -> 73138-1-7 -> 73138-ط-7
        # on essaie une reconstruction heuristique
        # ------------------------------------------------------------------
        if value.isdigit() and len(value) >= 3:
            # dernier chiffre = partie 3
            # avant-dernier chiffre = milieu OCR
            # reste = partie 1
            a = value[:-2]
            b = value[-2]
            c = value[-1]
            if a:
                return rebuild(a, b, c)

        # ------------------------------------------------------------------
        # Extraction générique
        # ------------------------------------------------------------------
        tokens = re.findall(rf"\d+|{arabic_letter_pattern}", value)

        if len(tokens) == 3 and tokens[0].isdigit() and tokens[2].isdigit():
            return rebuild(tokens[0], tokens[1], tokens[2])

        if len(tokens) == 2 and tokens[0].isdigit():
            if is_arabic_letter(tokens[1]) and len(tokens[0]) >= 2:
                return rebuild(tokens[0][:-1], tokens[1], tokens[0][-1])

            if tokens[1].isdigit() and len(tokens[0]) >= 2 and len(tokens[1]) == 1:
                return rebuild(tokens[0][:-1], tokens[1], tokens[0][-1])

        return value