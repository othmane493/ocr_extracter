# API d'Extraction de Documents Marocains

API Flask pour l'extraction automatique de données depuis les documents marocains (CIN et Cartes Grises).

## Documents Supportés

- **cin_old**: Anciennes cartes d'identité nationales (fond vert/jaune)
- **cin_new**: Nouvelles cartes d'identité nationales (fond rose)
- **carte_grise_recto**: Cartes grises (face avant)
- **carte_grise_verso**: Cartes grises (face arrière)

## Installation

### 1. Prérequis Système

**Tesseract OCR** doit être installé:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fra

# macOS
brew install tesseract tesseract-lang

# Windows
# Télécharger depuis: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Dépendances Python

```bash
pip install -r requirements.txt
```

### 3. Structure des Fichiers

```
.
├── app.py                          # Application Flask principale
├── extractors/
│   ├── __init__.py
│   ├── cin_extractor.py           # Extracteur CIN
│   └── carte_grise_extractor.py   # Extracteur Carte Grise
├── cin_detector.py                 # Détecteur de type CIN
├── cin_extractor_base.py           # Classe de base CIN
├── cin_new_extractor.py            # Extracteur CIN New
├── cin_old_extractor.py            # Extracteur CIN Old
├── utils/
│   └── ocr_utils.py               # Utilitaires OCR
├── requirements_flask.txt          # Dépendances
└── uploads/                        # Dossier temporaire (créé auto)
```

## Démarrage

```bash
python app.py
```

L'API démarre sur `http://0.0.0.0:5000`

## Utilisation

### 1. Health Check

Vérifier l'état de l'API:

```bash
curl http://localhost:5000/health
```

**Réponse:**
```json
{
  "status": "ok",
  "service": "Document Extraction API",
  "version": "1.0.0",
  "supported_documents": [
    "cin_old",
    "cin_new",
    "carte_grise_recto",
    "carte_grise_verso"
  ]
}
```

### 2. Extraction de Document

**Endpoint:** `POST /extract`

**Paramètres:**
- `file`: Fichier image (PNG, JPG, JPEG)
- `document_type`: Type de document

#### Exemple: Extraction CIN Old

```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@/path/to/cin_old.jpg" \
  -F "document_type=cin_old"
```

#### Exemple: Extraction CIN New

```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@/path/to/cin_new.png" \
  -F "document_type=cin_new"
```

#### Exemple: Extraction Carte Grise Recto

```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@/path/to/carte_grise_recto.jpg" \
  -F "document_type=carte_grise_recto"
```

#### Exemple: Extraction Carte Grise Verso

```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@/path/to/carte_grise_verso.jpg" \
  -F "document_type=carte_grise_verso"
```

### 3. Réponse Succès

```json
{
  "success": true,
  "document_type": "cin_new",
  "data": {
    "prenom_ar": "ابراهيم",
    "prenom_fr": "IBRAHIM",
    "nom_ar": "دمور",
    "nom_fr": "DAMOUR",
    "lieu_naissance_ar": "أكدال فاس",
    "lieu_naissance_fr": "AGDAL FES",
    "date_naissance": "29.06.1993",
    "cin": "CD426274",
    "date_expiration": "30.09.2030"
  },
  "processing_time": {
    "extraction_seconds": 3.45,
    "total_seconds": 3.52
  },
  "metadata": {
    "filename": "cin_new.png",
    "timestamp": 1707430123456
  }
}
```

### 4. Réponse Erreur

```json
{
  "error": "Type de document invalide",
  "message": "Type \"cin_ancien\" non supporté",
  "valid_types": [
    "cin_old",
    "cin_new",
    "carte_grise_recto",
    "carte_grise_verso"
  ]
}
```

## Exemples Python

### Avec requests

```python
import requests

# Extraction CIN
with open('cin_new.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/extract',
        files={'file': f},
        data={'document_type': 'cin_new'}
    )

if response.status_code == 200:
    result = response.json()
    print("Données extraites:")
    print(result['data'])
else:
    print("Erreur:", response.json())
```

### Avec curl (détaillé)

```bash
# Extraction avec affichage formaté
curl -X POST http://localhost:5000/extract \
  -F "file=@cin_new.png" \
  -F "document_type=cin_new" \
  | python -m json.tool
```

## Codes de Statut HTTP

| Code | Signification |
|------|---------------|
| 200  | Extraction réussie |
| 400  | Requête invalide (paramètres manquants ou invalides) |
| 413  | Fichier trop volumineux (max 16MB) |
| 500  | Erreur serveur lors de l'extraction |

## Limites

- **Taille maximale des fichiers**: 16MB
- **Extensions autorisées**: PNG, JPG, JPEG
- **Formats supportés**: Images couleur ou niveaux de gris
- **Qualité recommandée**: Minimum 300 DPI pour une extraction optimale

## Performance

### Temps d'Extraction Typiques

| Document | Temps moyen |
|----------|-------------|
| CIN Old | 2-4 secondes |
| CIN New | 2-4 secondes |
| Carte Grise Recto | 4-8 secondes |
| Carte Grise Verso | 4-8 secondes |

**Note:** Le premier appel peut être plus lent (chargement des modèles EasyOCR).

## Architecture

### CIN (Cartes d'Identité)

L'extraction des CIN utilise une architecture POO avec:
- Détection automatique du type (Old vs New)
- Prétraitement adapté à chaque type
- Double OCR (Tesseract + EasyOCR en fallback)
- Normalisation des dates automatique
- Réorganisation des champs par ordre logique

### Carte Grise

L'extraction des cartes grises utilise:
- Détection automatique recto/verso
- Groupement intelligent des lignes
- Correction des blocs suspects avec EasyOCR
- Support des champs multilignes
- Extraction bilingue (français + arabe)

## Logs et Debug

L'application affiche des logs détaillés dans la console:

```
Initialisation de l'extracteur CIN
Extraction de cin_new depuis: /tmp/file.png
Détection automatique: CIN NEW
Chargement du template
Extraction des zones
Prétraitement Tesseract
...
Extraction cin_new terminée en 3.12s
```

## Sécurité

- Les fichiers temporaires sont automatiquement supprimés après traitement
- Les noms de fichiers sont sécurisés avec `secure_filename()`
- Validation stricte des extensions de fichiers
- Limite de taille de fichier pour éviter les abus

## Déploiement Production

### Avec Gunicorn

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Avec Docker

```dockerfile
FROM python:3.9

# Install Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-fra

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements_flask.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build et run:
```bash
docker build -t doc-extraction-api .
docker run -p 5000:5000 doc-extraction-api
```

## Dépannage

### Erreur: "Tesseract not found"

Vérifier l'installation:
```bash
tesseract --version
```

Si non installé, voir la section Installation.

### Erreur: "No module named 'easyocr'"

```bash
pip install easyocr
```

### Performance lente

- Premier appel toujours plus lent (chargement modèles)
- Utiliser un GPU si disponible (modifier `gpu=False` à `gpu=True`)
- Augmenter la RAM disponible
- Réduire `MAX_ITEMS_EASY_OCR` dans carte_grise_extractor.py

## Support

Pour les questions ou problèmes:
1. Vérifier les logs de la console
2. Tester avec `/health` endpoint
3. Vérifier les dépendances système (Tesseract)
4. Consulter la documentation des extracteurs

## Licence

Code fourni à titre d'exemple. Adaptez selon vos besoins.
