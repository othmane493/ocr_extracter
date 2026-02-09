#!/usr/bin/env python3
"""
Transformateur de format JSON - CIN/Carte Grise
Convertit le format avec clés fr/ar en format plat avec suffixes _fr/_ar
Détecte automatiquement la langue quand les valeurs sont identiques
"""

import json
import re
from typing import Dict, List, Any


def is_arabic(text: str) -> bool:
    """
    Détecte si un texte contient de l'arabe

    Args:
        text: Texte à analyser

    Returns:
        True si le texte contient des caractères arabes
    """
    if not text or not isinstance(text, str):
        return False

    # Plage de caractères arabes Unicode
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9]')

    return bool(arabic_pattern.search(text))


def normalize_key(key_fr: str, key_ar: str) -> str:
    """
    Normaliser les noms de clés en format snake_case

    Priorité à la clé française pour éviter les clés en arabe.

    Mapping des clés connues:
    - Marque → marque
    - Type → type
    - Genre → genre
    - Type carburant → type_carburant
    - N° du chassis → numero_chassis
    - etc.
    """
    # Mapping personnalisé pour les clés courantes (français)
    key_mappings_fr = {
        # CIN
        'prénom': 'prenom',
        'nom': 'nom',
        'date de naissance': 'date_naissance',
        'lieu de naissance': 'lieu_naissance',
        'cin': 'cin',
        'numéro cin': 'cin',
        'date d\'expiration': 'date_expiration',
        'date expiration': 'date_expiration',
        'sexe': 'sexe',

        # Carte Grise RECTO
        'numéro d\'immatriculation': 'numero_immatriculation',
        'numero d\'immatriculation': 'numero_immatriculation',
        'immatriculation': 'numero_immatriculation',
        'immatriculation antérieure': 'immatriculation_anterieure',
        'immatriculation anterieure': 'immatriculation_anterieure',
        'première mise en circulation': 'premiere_mise_circulation',
        'premiere mise en circulation': 'premiere_mise_circulation',
        'mise en circulation': 'premiere_mise_circulation',
        'm.c au maroc': 'mc_maroc',
        'mc au maroc': 'mc_maroc',
        'mutation le': 'date_mutation',
        'mutation': 'date_mutation',
        'usage': 'usage',
        'propriétaire': 'proprietaire',
        'proprietaire': 'proprietaire',
        'adresse': 'adresse',
        'fin de validité': 'fin_validite',
        'fin de validite': 'fin_validite',
        'validité': 'fin_validite',
        'validite': 'fin_validite',

        # Carte Grise VERSO
        'marque': 'marque',
        'type': 'type',
        'genre': 'genre',
        'modèle': 'modele',
        'modele': 'modele',
        'type carburant': 'type_carburant',
        'carburant': 'type_carburant',
        'n° du chassis': 'numero_chassis',
        'n du chassis': 'numero_chassis',
        'numero chassis': 'numero_chassis',
        'chassis': 'numero_chassis',
        'nombre de cylindres': 'nombre_cylindres',
        'cylindres': 'nombre_cylindres',
        'puissance fiscale': 'puissance_fiscale',
        'nombre de places': 'nombre_places',
        'places': 'nombre_places',
        'p.t.a.c': 'ptac',
        'p.t.a.c.': 'ptac',
        'ptac': 'ptac',
        'poids total': 'ptac',
        'poids à vide': 'poids_vide',
        'poids a vide': 'poids_vide',
        'poids vide': 'poids_vide',
        'p.t.r.a': 'ptra',
        'p.t.r.a.': 'ptra',
        'ptra': 'ptra',
        'restrictions': 'restrictions',
    }

    # Mapping arabe vers français (pour convertir les clés arabes)
    key_mappings_ar = {
        # CIN
        'الاسم الشخصي': 'prenom',
        'الاسم العائلي': 'nom',
        'تاريخ الازدياد': 'date_naissance',
        'مكان الازدياد': 'lieu_naissance',
        'رقم البطاقة': 'cin',
        'تاريخ انتهاء الصلاحية': 'date_expiration',

        # Carte Grise RECTO
        'رقم التسجيل': 'numero_immatriculation',
        'الترقيم السابق': 'immatriculation_anterieure',
        'أول شروع في الإستخدام': 'premiere_mise_circulation',
        'أول استخدام بالمغرب': 'mc_maroc',
        'تحويل بتاريخ': 'date_mutation',
        'نوع الإستعمال': 'usage',
        'المالك': 'proprietaire',
        'العنوان': 'adresse',
        'نهاية الصلاحية': 'fin_validite',

        # Carte Grise VERSO
        'الاسم التجاري': 'marque',
        'الصنف': 'type',
        'النوع': 'genre',
        'النموذج': 'modele',
        'نوع الوقود': 'type_carburant',
        'رقم الإطار الحديدي': 'numero_chassis',
        'عدد الأسطوانات': 'nombre_cylindres',
        'القوة الجبائية': 'puissance_fiscale',
        'عدد المقاعد': 'nombre_places',
        'الوزن جمالي': 'ptac',
        'الوزن الفارغ': 'poids_vide',
        'الوزن الإجمالي مع المجرور': 'ptra',
        'التقييدات': 'restrictions',
    }

    # PRIORITÉ 1: Clé française
    if key_fr and key_fr.strip():
        key_clean = key_fr.lower().strip()

        # Normaliser les caractères accentués
        import unicodedata
        key_clean = ''.join(
            c for c in unicodedata.normalize('NFD', key_clean)
            if unicodedata.category(c) != 'Mn'
        )

        key_clean = re.sub(r'[°\'"]', '', key_clean)  # Enlever °, ', "
        key_clean = re.sub(r'\s+', ' ', key_clean)  # Normaliser espaces

        # Chercher dans le mapping français
        if key_clean in key_mappings_fr:
            return key_mappings_fr[key_clean]

        # Sinon, convertir en snake_case
        key_snake = key_clean.replace(' ', '_')
        key_snake = re.sub(r'[^\w_]', '', key_snake)
        return key_snake

    # PRIORITÉ 2: Clé arabe (seulement si pas de clé française)
    if key_ar and key_ar.strip():
        # Chercher dans le mapping arabe
        key_ar_clean = key_ar.strip()
        if key_ar_clean in key_mappings_ar:
            return key_mappings_ar[key_ar_clean]

        # Sinon, translitérer ou utiliser un nom générique
        # (Éviter les clés en caractères arabes)
        return 'field_ar_' + str(hash(key_ar_clean) % 10000)

    return None


def transform_entry(entry: Dict) -> Dict[str, Any]:
    """
    Transformer une entrée du format {fr: {...}, ar: {...}}
    vers le format plat avec suffixes _fr et _ar

    Args:
        entry: Dictionnaire avec clés 'fr' et 'ar'

    Returns:
        Dictionnaire transformé
    """
    fr_data = entry.get('fr', {})
    ar_data = entry.get('ar', {})

    # Extraire les clés et valeurs
    key_fr = fr_data.get('key')
    key_ar = ar_data.get('key')
    value_fr = fr_data.get('value', '').strip()
    value_ar = ar_data.get('value', '').strip()

    # Si pas de clé valide, ignorer
    base_key = normalize_key(key_fr, key_ar)
    if not base_key:
        return {}

    result = {}

    # CAS 1: Les deux valeurs sont vides
    if not value_fr and not value_ar:
        return {}

    # CAS 2: Les valeurs sont identiques
    if value_fr == value_ar and value_fr:
        # Détecter automatiquement la langue
        if is_arabic(value_fr):
            # C'est de l'arabe
            result[f"{base_key}_ar"] = value_ar
        else:
            # C'est du français/latin
            result[f"{base_key}_fr"] = value_fr

    # CAS 3: Les valeurs sont différentes
    else:
        # Ajouter la valeur française si présente
        if value_fr:
            result[f"{base_key}_fr"] = value_fr

        # Ajouter la valeur arabe si présente
        if value_ar:
            result[f"{base_key}_ar"] = value_ar

    # Ajouter les métadonnées si nécessaire (optionnel)
    # result[f"{base_key}_confidence"] = fr_data.get('confidence', 0)
    # result[f"{base_key}_bbox"] = {
    #     'x': fr_data.get('x', 0),
    #     'y': fr_data.get('y', 0),
    #     'width': fr_data.get('width', 0),
    #     'height': fr_data.get('height', 0)
    # }

    return result


def transform_json(input_data: List[Dict]) -> Dict[str, Any]:
    """
    Transformer le JSON complet

    Args:
        input_data: Liste d'entrées au format original

    Returns:
        Dictionnaire plat transformé
    """
    result = {}

    for entry in input_data:
        transformed = transform_entry(entry)
        result.update(transformed)

    return result


def clean_date_format(date_str: str) -> str:
    """
    Nettoyer et formater les dates

    Convertit "11.02.1994" ou "11/02/1994" en "11 02 1994"
    """
    if not date_str:
        return date_str

    # Remplacer les séparateurs par des espaces
    date_clean = re.sub(r'[./\-]', ' ', date_str)
    # Normaliser les espaces multiples
    date_clean = re.sub(r'\s+', ' ', date_clean).strip()

    return date_clean


def post_process(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-traitement pour nettoyer et formater les données
    """
    # Nettoyer les dates
    for key in list(data.keys()):
        if 'date' in key.lower() and data[key]:
            data[key] = clean_date_format(data[key])

    return data


def transform_file(input_path: str, output_path: str = None, pretty: bool = True):
    """
    Transformer un fichier JSON

    Args:
        input_path: Chemin du fichier d'entrée
        output_path: Chemin du fichier de sortie (optionnel)
        pretty: Formatter le JSON en sortie
    """
    # Charger le JSON d'entrée
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # Transformer
    output_data = transform_json(input_data)

    # Post-traitement
    output_data = post_process(output_data)

    # Sauvegarder ou afficher
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            else:
                json.dump(output_data, f, ensure_ascii=False)
        print(f"✅ Transformé: {output_path}")
    else:
        # Afficher sur stdout
        if pretty:
            print(json.dumps(output_data, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(output_data, ensure_ascii=False))

    return output_data
