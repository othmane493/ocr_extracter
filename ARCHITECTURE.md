# Architecture — OCR Extracter

Vue d’ensemble du dépôt et du flux d’une requête d’extraction.

## Structure des dossiers (logique)

```mermaid
flowchart TB
  subgraph api ["Entrée HTTP"]
    APP["app.py — Flask"]
  end
  subgraph extractors ["Extracteurs"]
    CIN["extractors/cin_extractor.py"]
    CG["extractors/carte_grise_extractor.py"]
    DET["extractors/document_detector.py"]
  end
  subgraph core ["Cœur métier"]
    OCR["ocr_manager.py — PaddleOCR singleton"]
    CINNEW["cin_new_extractor.py / cin_old_extractor.py"]
    ALIGN_CIN["config/CinRecenter.py"]
    ALIGN_CG["config/CarteGriseRecenter.py"]
  end
  subgraph assets ["Données & utilitaires"]
    CFG["config/*.json gabarits"]
    UTILS["utils/*"]
  end
  APP --> CIN
  APP --> CG
  APP --> DET
  CIN --> CINNEW
  CIN --> ALIGN_CIN
  CG --> ALIGN_CG
  CG --> OCR
  CINNEW --> OCR
  DET --> TESS["Tesseract — indices carte grise"]
  DET --> CINDET["cin_detector.py — CIN old vs new"]
  ALIGN_CIN --> CFG
  ALIGN_CG --> CFG
```

## Flux d’une requête `POST /extract`

```mermaid
sequenceDiagram
  participant Client
  participant Flask as app.py
  participant Det as document_detector
  participant Ext as CIN ou CarteGrise
  participant OCR as OCRManager / Tesseract

  Client->>Flask: multipart file (+ document_type optionnel)
  Flask->>Flask: sauvegarde uploads/, valide extension
  alt document_type valide fourni
    Flask->>Ext: extraction directe (sans détection)
  else document_type vide / absent
    Flask->>Det: detect_document_type(path)
    Det->>OCR: heuristiques carte grise (Tesseract)
    Det->>Det: sinon CIN old/new (géométrie photo)
    Det-->>Flask: cin_* ou carte_grise_*
    Flask->>Ext: extraction selon type
  end
  Ext->>OCR: PaddleOCR + pipelines champs
  Ext-->>Flask: dict structuré
  Flask-->>Client: JSON success + document_type_source
```

## Décision automatique du type de document

```mermaid
flowchart TD
  A["Image reçue"] --> B{"Carte grise ?\n(mots-clés OCR recto/verso)"}
  B -->|oui recto| R["carte_grise_recto"]
  B -->|oui verso| V["carte_grise_verso"]
  B -->|non| C{"Grande photo CIN\nà gauche ou droite ?"}
  C -->|gauche| N["cin_new"]
  C -->|droite| O["cin_old"]
  C -->|indéterminé| E["Erreur: document inconnu"]
```

Ordre implémenté dans `extractors/document_detector.py` : **carte grise d’abord**, puis **CIN** (via `cin_detector.py`).

## Composants clés

| Fichier / module | Rôle |
|------------------|------|
| `app.py` | Routes, upload, choix auto vs `document_type` client, orchestration extracteurs |
| `ocr_manager.py` | Instance unique PaddleOCR `ar` + `fr`, pool de threads, warmup |
| `extractors/document_detector.py` | Détection auto (Tesseract + règles mots-clés, fallback) |
| `cin_detector.py` | Distinction CIN ancien / nouveau selon la position de la photo |
| `extractors/cin_extractor.py` | Branche vers alignement ORB + extracteurs CIN |
| `extractors/carte_grise_extractor.py` | Alignement carte grise, champs, Tesseract/Paddle selon zones |
| `config/*.json` | Gabarits de zones / métadonnées document |

## Dépendances Python (résumé)

- **Flask / Werkzeug** : API HTTP  
- **OpenCV, NumPy, Pillow** : images et géométrie  
- **pytesseract** : OCR léger pour la **détection** (carte grise)  
- **paddlepaddle / paddleocr** : OCR principal à l’**extraction**  
- **requests** : `test_api.py`  
- **easyocr** : scripts outils sous `config/` et `utils/ocr_utils.py` (génération de templates)

Le détail des versions est dans `requirements.txt`.

## Prérequis système : Tesseract (fra + ara)

Le module `extractors/document_detector.py` appelle Tesseract avec **`lang="fra+ara"`**. Il faut donc les **données de langue** Tesseract pour le **français** (`fra`) et l’**arabe** (`ara`), pas seulement le binaire.

```mermaid
flowchart LR
  subgraph req ["Requis au runtime"]
    BIN["binaire tesseract"]
    FRA["traineddata fra"]
    ARA["traineddata ara"]
  end
  BIN --> LIST["tesseract --list-langs"]
  FRA --> LIST
  ARA --> LIST
  LIST --> OK["Détection carte grise OK"]
```

- **Linux / macOS** : `start.sh` peut installer **en CLI** le moteur et, si besoin, les paquets **fra/ara** (`AUTO_INSTALL_TESSERACT_LANGS=1` ou invite interactive).  
- **Windows** : pas d’installation auto des langues dans le script ; l’installateur graphique doit inclure **French** et **Arabic** ; `start.ps1` vérifie `tesseract --list-langs` avant de lancer l’API.

Détails et commandes manuelles : [`README.md`](README.md) (section Tesseract).
