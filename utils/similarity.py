import re
import unicodedata
from difflib import SequenceMatcher

# =========================
# NORMALISATION
# =========================

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z\u0600-\u06FF]", "", text)
    return text


# =========================
# ARABIC
# =========================

ARABIC_NORMALIZATION_MAP = {
    "أ": "ا", "إ": "ا", "آ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و",
    "ة": "ه",
}

def normalize_arabic(text: str) -> str:
    for k, v in ARABIC_NORMALIZATION_MAP.items():
        text = text.replace(k, v)
    return text


AR_TO_PHONETIC = {
    "ا": "a",
    "ب": "b",
    "ت": "t",
    "ث": "s",
    "ج": "j",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ذ": "d",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "d",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "k",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "u",
    "ي": "i",
}

def arabic_to_phonetic(text: str) -> str:
    return "".join(AR_TO_PHONETIC.get(c, c) for c in text)


# =========================
# LATIN → PHONETIC
# =========================

def latin_to_phonetic(text: str) -> str:
    rules = [
        ("you", "u"),
        ("yous", "us"),
        ("ou", "u"),
        ("oo", "u"),
        ("o", "u"),
        ("u", "u"),

        ("ee", "i"),
        ("ie", "i"),
        ("y", "i"),
        ("i", "i"),

        ("ph", "f"),
        ("ck", "k"),
        ("qu", "k"),
        ("c", "k"),

        ("ch", "sh"),
        ("sh", "sh"),

        ("e", "a"),
        ("a", "a"),
    ]

    for src, tgt in rules:
        text = text.replace(src, tgt)

    return text


# =========================
# CONSONANT LOGIC (KEY)
# =========================

VOWELS = set("aeiou")

def consonant_skeleton(text: str) -> str:
    """
    Supprime les voyelles → garde l'identité du nom
    """
    return "".join(c for c in text if c not in VOWELS)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# =========================
# MAIN COMPARATOR
# =========================

def compare_name_ar_fr(name_ar: str, name_fr: str,
                       min_consonant_score=0.75,
                       global_threshold=0.7) -> dict:

    ar = normalize_arabic(normalize_text(name_ar))
    fr = normalize_text(name_fr)

    ar_ph = arabic_to_phonetic(ar)
    fr_ph = fr

    ar_cons = consonant_skeleton(ar_ph)
    fr_cons = consonant_skeleton(fr_ph)

    score_cons = similarity(ar_cons, fr_cons)
    score_global = similarity(ar_ph, fr_ph)

    # 🔥 règle métier : consonnes = voyelles
    final_score = round((score_cons * 0.5) + (score_global * 0.5), 3)

    return {
        "ar": name_ar,
        "fr": name_fr,
        "ar_phonetic": ar_ph,
        "fr_phonetic": fr_ph,
        "ar_consonants": ar_cons,
        "fr_consonants": fr_cons,
        "score_consonants": round(score_cons, 3),
        "score_global": round(score_global, 3),
        "score": final_score,
        "is_match": score_cons >= min_consonant_score and final_score >= global_threshold
    }
