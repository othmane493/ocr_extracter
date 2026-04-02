# OCR Extracter — API d’extraction (documents marocains)

Service **Flask** qui reçoit une image et renvoie les champs structurés pour :

- **CIN** ancien / nouveau (`cin_old`, `cin_new`)
- **Carte grise** recto / verso (`carte_grise_recto`, `carte_grise_verso`)

La pile combine **OpenCV** (prétraitement / alignement), **Tesseract** (indices rapides pour la détection carte grise), **PaddleOCR** (extraction principale), et des gabarits JSON sous `config/`.

## Prérequis système

- **Python 3.10+** recommandé  
- **Tesseract** installé et dans le `PATH`, avec les **données de langue français (`fra`) et arabe (`ara`)** — indispensables pour la détection de carte grise et les appels `fra+ara` dans le code  
- **pip ≥ 23** (les scripts de démarrage mettent pip à jour si besoin)

### Tesseract : choisir les dictionnaires français et arabe

L’API s’appuie sur Tesseract avec la chaîne de langues **`fra+ara`**. Sans les fichiers de données correspondants (`.traineddata`), la détection et l’OCR léger échouent ou sont dégradés.

| Plateforme | Ce qu’il faut faire |
|------------|---------------------|
| **Linux** | Installer les paquets qui fournissent `fra` et `ara` (voir ci‑dessous). **`start.sh`** peut les installer **en ligne de commande** via `apt`, `dnf`, `pacman`, `zypper` ou `brew` : d’abord le binaire Tesseract si besoin, puis une étape dédiée qui vérifie `tesseract --list-langs` et propose (ou force avec `AUTO_INSTALL_TESSERACT_LANGS=1`) l’installation des paquets **fra** et **ara**. |
| **macOS** | `brew install tesseract tesseract-lang` — le formule `tesseract-lang` apporte les données multilingues, dont **fra** et **ara**. |
| **Windows** | Aucun équivalent fiable « tout en CLI » dans ce dépôt : utilisez l’**installateur** (ex. [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)) et, à l’étape **Additional language data**, cochez **French** et **Arabic** (codes **fra**, **ara**). **`start.ps1`** vérifie la présence de ces langues avec `tesseract --list-langs` et refuse de continuer si elles manquent. |

Commandes manuelles (si vous n’utilisez pas les scripts ou si l’auto‑installation échoue) :

- **Debian/Ubuntu** : `sudo apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara`  
- **Fedora** : `sudo dnf install tesseract tesseract-langpack-fra tesseract-langpack-ara`  
- **Arch** : `sudo pacman -S tesseract tesseract-data-fra tesseract-data-ara`  
- **openSUSE** : `sudo zypper install tesseract-ocr tesseract-traineddata-fra tesseract-traineddata-ara` (noms de paquets pouvant varier selon la version)

Variables d’environnement utiles avec **`start.sh`** :

- `AUTO_INSTALL_TESSERACT=1` — installation du binaire Tesseract sans invite (si absent).  
- `AUTO_INSTALL_TESSERACT_LANGS=1` — installation des paquets **fra/ara** sans invite si `tesseract --list-langs` signale qu’il en manque une.

### PaddlePaddle

`paddlepaddle` est listé dans `requirements.txt`. Si l’installation échoue selon l’OS, suivre la [documentation officielle Paddle](https://www.paddlepaddle.org.cn/install/quick) (index de paquets CPU/GPU).

## Démarrage rapide

### Linux / macOS (Bash)

```bash
chmod +x start.sh
./start.sh
```

Si le **binaire** Tesseract est absent, le script propose une installation automatique (`y`) via **apt**, **dnf**, **pacman**, **zypper** ou **brew** (paquets incluant **fra** et **ara** lorsque le dépôt les sépare). Ensuite, **`start.sh`** exécute `tesseract --list-langs` : si **fra** ou **ara** manque, une **deuxième invite** propose d’installer uniquement les paquets de langue (ou utilisez `AUTO_INSTALL_TESSERACT_LANGS=1`). Sans terminal interactif, sans ces variables, le script affiche les commandes à lancer à la main.

### Windows

- **Double-clic** sur `start.bat`, ou dans PowerShell :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # une seule fois si besoin
cd c:\chemin\vers\ocr_extracter
.\start.ps1
```

Le script cherche `tesseract` dans le **PATH**, puis sous  
`%ProgramFiles%\Tesseract-OCR\tesseract.exe` et **Program Files (x86)**.  
S’il est installé mais pas dans le PATH, le dossier est ajouté au PATH **pour la session en cours**.

Si Tesseract est **introuvable**, un message indique d’installer l’exécutable Windows (lien UB-Mannheim) **avant** de relancer — pas d’installation silencieuse sous Windows. Si le binaire est présent mais que **fra** ou **ara** manque (`tesseract --list-langs`), le script **s’arrête** avec des instructions pour ajouter les langues via l’installateur ou en copiant les `.traineddata` dans `tessdata`.

Ou manuellement :

```bash
python -m pip install -r requirements.txt
python app.py
```

L’API écoute par défaut sur `http://0.0.0.0:5000`.

## Endpoints

| Méthode | Chemin     | Rôle |
|---------|------------|------|
| `GET`   | `/health`  | Santé du service, types supportés, OCR initialisé |
| `POST`  | `/extract` | Envoi d’une image + extraction |

### `POST /extract`

- **Corps** : `multipart/form-data`  
- **Champ fichier** : `file` (extensions : `png`, `jpg`, `jpeg`)  
- **Optionnel** : `document_type` (même nom en champ formulaire ou en query `?document_type=…`)

Valeurs possibles pour `document_type` : `cin_old`, `cin_new`, `carte_grise_recto`, `carte_grise_verso`.

- Si `document_type` est **absent**, **vide**, ou littéralement `null` / `none` → **détection automatique** (plus lent, passe par l’OCR/heuristiques du détecteur).  
- Si `document_type` est **renseigné et valide** → **pas de détection** : le pipeline d’extraction correspondant est utilisé directement (gain de temps).  
- Si la valeur est **inconnue** → réponse **400** avec la liste `valid_types`.

Réponse JSON en succès (extraits) :

- `document_type` : type utilisé pour l’extraction  
- `document_type_source` : `"request"` ou `"auto"`  
- `data` : champs extraits  
- `processing_time` : durées approximatives  

Exemple avec **curl** (détection auto) :

```bash
curl -s -X POST http://localhost:5000/extract -F "file=@chemin/vers/image.jpg"
```

Exemple avec type imposé :

```bash
curl -s -X POST "http://localhost:5000/extract?document_type=cin_new" -F "file=@image.png"
```

Tests d’intégration locaux (API déjà démarrée) :

```bash
python test_api.py
```

## Documentation d’architecture

Voir [`ARCHITECTURE.md`](ARCHITECTURE.md) : flux requête → détection → extracteurs, et schémas (Mermaid).

## Limites

- Taille max upload : **16 Mo** (configurable dans `app.py`).  
- Qualité des scans, éclairage et résolution influencent fortement l’OCR.
