# Système Complet d'Extraction de Documents Marocains

Application Flask complète pour l'extraction automatique de données depuis les documents d'identité et cartes grises marocains.

## Vue d'Ensemble

Ce projet combine:
- **Architecture POO** pour les CIN (Cartes d'Identité Nationales)
- **Code d'extraction** optimisé pour les Cartes Grises
- **API REST Flask** unifiée pour tous les documents

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────┐
│                     API Flask (app.py)                      │
│                    POST /extract                            │
└────────────────────────┬────────────────────────────────────┘
                         │
           ┌─────────────┴──────────────┐
           │                            │
    ┌──────▼──────┐            ┌────────▼────────┐
    │ CINExtractor│            │CarteGriseExtractor│
    └──────┬──────┘            └────────┬────────┘
           │                            │
    ┌──────▼──────────┐         ┌──────▼─────────┐
    │ CIN Architecture│         │ Carte Grise    │
    │ POO (détecteur +│         │ Code Original  │
    │ extracteurs)    │         │ (intégré)      │
    └─────────────────┘         └────────────────┘
```

## Structure des Fichiers

```
.
├── app.py                          # Application Flask principale
│
├── extractors/                     # Package des extracteurs
│   ├── __init__.py
│   ├── cin_extractor.py           # Wrapper CIN
│   └── carte_grise_extractor.py   # Extracteur Carte Grise
│
├── Architecture CIN (POO)
│   ├── cin_detector.py            # Détecteur + point d'entrée unifié
│   ├── cin_extractor_base.py      # Classe de base abstraite
│   ├── cin_new_extractor.py       # Extracteur CIN nouvelles
│   └── cin_old_extractor.py       # Extracteur CIN anciennes
│
├── utils/                          # Utilitaires
│   ├── __init__.py
│   └── ocr_utils.py               # Fonctions OCR Tesseract
│
├── config/                         # Configuration
│   ├── cin_new_template.json      # Template zones CIN New
│   └── cin_old_template.json      # Template zones CIN Old
│
├── uploads/                        # Fichiers temporaires (auto-nettoyé)
│
├── Documentation
│   ├── README_API.md              # Guide API Flask
│   ├── README.md                  # Guide architecture CIN
│   ├── MIGRATION_GUIDE.md         # Guide de migration
│   ├── DETECTION_IMPROVEMENTS.md  # Améliorations détection
│   └── INDEX.md                   # Index des fichiers
│
├── Tests
│   ├── test_api.py                # Tests API Flask
│   ├── test_structure.py          # Tests structure POO
│   └── test_detection_standalone.py # Tests détection
│
└── Configuration
    ├── requirements_flask.txt     # Dépendances Flask
    ├── requirements.txt           # Dépendances CIN
    └── .gitignore                 # Fichiers à ignorer
```

## Installation Rapide

### 1. Prérequis Système

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fra

# macOS
brew install tesseract tesseract-lang
```

### 2. Installation Python

```bash
# Installer toutes les dépendances
pip install -r requirements.txt
```

### 3. Démarrage de l'API

```bash
python app.py
```

L'API démarre sur `http://0.0.0.0:5000`

## Utilisation de l'API

### Extraction CIN Old

```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@images/cin-old.jpg" \
  -F "document_type=cin_old"
```

### Extraction CIN New

```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@images/cin_new.png" \
  -F "document_type=cin_new"
```

### Extraction Carte Grise Recto

```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@images/carte-grise-recto.jpg" \
  -F "document_type=carte_grise_recto"
```

### Extraction Carte Grise Verso

```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@images/carte-grise-verso.jpg" \
  -F "document_type=carte_grise_verso"
```

## Format de Réponse

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

## Architecture Technique

### CIN (Cartes d'Identité)

#### Détection Automatique

Le système détecte automatiquement si c'est une CIN OLD ou NEW basé sur:
- **Couleur de la bande supérieure** (20% du haut)
- **Ratio Rouge/Vert** (critère principal)
- **Teinte HSV**
- **Saturation**

Score de détection:
- R>G+B → NEW (5 points)
- G≥R → OLD (5 points)
- Plus critères secondaires (HSV, saturation, plages)

#### Extraction POO

```
CINExtractor (classe de base)
    ├── Méthodes communes
    │   ├── load_template()
    │   ├── load_image()
    │   ├── safe_crop()
    │   ├── normalize_date()
    │   └── extract()
    │
    └── Méthodes abstraites
        ├── preprocess_zone()
        ├── preprocess_zone_easyocr()
        ├── extract_text_tesseract()
        └── get_confidence_threshold()

CINNewExtractor (hérite de CINExtractor)
    ├── Seuil: 80%
    ├── Prétraitement: Contraste élevé + Otsu
    └── EasyOCR: Zone originale

CINOldExtractor (hérite de CINExtractor)
    ├── Seuil: 60%
    ├── Prétraitement: Contraste adapté
    └── EasyOCR: Agrandissement 3x + CLAHE
```

### Carte Grise

#### Détection Recto/Verso

Détection automatique basée sur les champs présents:
- **Recto**: "Propriétaire", "Adresse", "Usage"
- **Verso**: "Marque", "Type", "Genre", "Modèle"

Seuil: 3 champs minimum pour valider

#### Pipeline d'Extraction

```
1. Prétraitement Image
   └── Binarisation (seuil 148)

2. Extraction Tesseract
   └── OCR français + arabe

3. Groupement par Lignes
   └── Tolérance Y dynamique

4. Fusion des Blocs
   └── Selon espaces (2.0 pour recto, 1.8 pour verso)

5. Marquage des Champs
   └── Similarité ≥ 0.6

6. Correction EasyOCR
   └── Blocs avec confiance < 60%

7. Filtrage
   └── Selon zones Y configurées

8. Parsing Final
   └── Extraction FR + AR
```

## Logs et Debug

### Logs CIN

```
Initialisation de l'extracteur CIN
Extraction de cin_new depuis: /tmp/1234_file.png
Détection automatique: CIN NEW (score: 8 vs 5)
Chargement du template: config/cin_new_template.json
Prétraitement des zones
...
Extraction cin_new terminée en 3.12s
```

### Logs Carte Grise

```
Initialisation de l'extracteur de carte grise
Configuration: RECTO (forcé)
Chargement du modèle EasyOCR en 2.45s
Extraction du texte avec Tesseract
Groupement des blocs par ligne
Fusion des blocs par ligne
Marquage des champs détectés
Correction des blocs suspects (max 10 blocs)
OCR corrigé: 123456 (confiance: 0.89)
...
Extraction carte grise terminée en 6.78s
```

## Performance

### Temps d'Exécution Typiques

| Document | Premier appel | Appels suivants |
|----------|---------------|-----------------|
| CIN Old | ~5s | ~3s |
| CIN New | ~5s | ~3s |
| CG Recto | ~8s | ~6s |
| CG Verso | ~8s | ~6s |

**Note**: Premier appel plus lent à cause du chargement des modèles EasyOCR.

### Optimisations

- **GPU**: Activer `gpu=True` dans les extracteurs (gain ~40%)
- **Cache**: Modèles EasyOCR mis en cache après premier chargement
- **Parallélisme**: Utiliser Gunicorn avec workers multiples
- **RAM**: Minimum 4GB recommandé, 8GB optimal

## Tests

### Test de l'API

```bash
python test_api.py
```

Tests effectués:
- Health check
- Extraction CIN Old
- Extraction CIN New
- Extraction CG Recto
- Extraction CG Verso
- Type invalide (erreur attendue)
- Fichier manquant (erreur attendue)

### Test de la Structure POO

```bash
python test_structure.py
```

Valide:
- Présence de tous les fichiers
- Héritage des classes
- Méthodes implémentées
- Documentation

### Test de Détection

```bash
python test_detection_standalone.py
```

Valide la détection automatique CIN OLD vs NEW.

## Déploiement Production

### Avec Gunicorn

```bash
pip install gunicorn

# 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app

# Avec logs
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 \
  --access-logfile access.log \
  --error-logfile error.log \
  app:app
```

### Avec Docker

```bash
docker build -t doc-extraction-api .
docker run -p 5000:5000 doc-extraction-api
```

### Avec Nginx (reverse proxy)

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
        
        client_max_body_size 16M;
    }
}
```

## Sécurité

### Mesures Implémentées

1. **Validation des fichiers**
   - Extensions autorisées: PNG, JPG, JPEG
   - Taille max: 16MB
   - Noms sécurisés avec `secure_filename()`

2. **Nettoyage automatique**
   - Suppression fichiers temporaires après traitement
   - Même en cas d'erreur

3. **Validation des types**
   - Liste blanche de types de documents
   - Rejet des types inconnus

4. **Isolation**
   - Dossier uploads séparé
   - Pas d'accès direct aux fichiers uploadés

### Recommandations Additionnelles

- Utiliser HTTPS en production
- Implémenter rate limiting (ex: Flask-Limiter)
- Ajouter authentification API (tokens, OAuth)
- Logger les tentatives suspectes
- Scanner antivirus sur uploads (optionnel)

## Dépannage

### Problème: Tesseract not found

```bash
# Vérifier installation
tesseract --version

# Installer si nécessaire
sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fra
```

### Problème: Extraction lente

Solutions:
1. Activer GPU (si disponible)
2. Réduire MAX_ITEMS_EASY_OCR (actuellement 10)
3. Augmenter RAM serveur
4. Utiliser cache Redis pour résultats

### Problème: Erreur 500

Vérifier:
1. Logs dans la console
2. Format de l'image (couleur, dimensions)
3. Dépendances installées
4. Espace disque disponible

### Problème: Résultats incorrects

Actions:
1. Vérifier qualité image (min 300 DPI)
2. Tester avec mode debug=True
3. Ajuster seuils de confiance
4. Vérifier templates de zones

## Roadmap / Améliorations Futures

- [ ] Support d'autres formats (PDF multi-pages)
- [ ] Cache Redis pour performances
- [ ] API asynchrone (Celery + RabbitMQ)
- [ ] Batch processing (plusieurs documents)
- [ ] Interface web de test
- [ ] Métriques et monitoring (Prometheus)
- [ ] Support GPU automatique
- [ ] Compression des réponses (gzip)
- [ ] Webhooks pour résultats asynchrones
- [ ] Support multi-langues API

## Contributeurs

Ce projet combine:
- Architecture POO CIN (refactorisation et optimisation)
- Code d'extraction Carte Grise (intégration)
- API Flask unifiée (développement)

## Licence

Code fourni à titre d'exemple. Adaptez selon vos besoins.

## Support

Pour questions ou problèmes:
1. Consulter README_API.md pour détails API
2. Consulter README.md pour architecture CIN
3. Vérifier les logs dans la console
4. Tester avec les scripts de test fournis

---

**Version**: 1.0.0  
**Dernière mise à jour**: Février 2026
