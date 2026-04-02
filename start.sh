#!/usr/bin/env bash
# Démarrage du projet : vérifie Python, pip (version minimale), Tesseract, installe les deps, lance l'API.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MIN_PIP_MAJOR="${MIN_PIP_MAJOR:-23}"
MIN_PIP_MINOR="${MIN_PIP_MINOR:-0}"

echo "=========================================="
echo "API d'extraction de documents marocains"
echo "=========================================="
echo ""

if ! command -v python3 &>/dev/null; then
  echo "Erreur: python3 introuvable."
  exit 1
fi

echo "1. Python"
python3 --version
echo ""

PIP_CMD=(python3 -m pip)

pip_meets_minimum() {
  python3 <<PY
import re
import subprocess
import sys

need = (${MIN_PIP_MAJOR}, ${MIN_PIP_MINOR})
r = subprocess.run(
    [sys.executable, "-m", "pip", "--version"],
    capture_output=True,
    text=True,
)
out = (r.stdout or "") + (r.stderr or "")
m = re.search(r"pip (\d+)\.(\d+)", out)
if not m:
    print("Impossible de lire la version de pip.", file=sys.stderr)
    sys.exit(1)
maj, minv = int(m.group(1)), int(m.group(2))
if (maj, minv) < need:
    print(f"pip {maj}.{minv} < requis {need[0]}.{need[1]}", file=sys.stderr)
    sys.exit(2)
sys.exit(0)
PY
}

echo "2. pip (minimum ${MIN_PIP_MAJOR}.${MIN_PIP_MINOR})"
if ! pip_meets_minimum; then
  echo "   Mise à jour de pip..."
  "${PIP_CMD[@]}" install --upgrade "pip>=${MIN_PIP_MAJOR}.${MIN_PIP_MINOR}"
  if ! pip_meets_minimum; then
    echo "Erreur: pip est toujours trop ancien après mise à jour."
    exit 1
  fi
fi
python3 -m pip --version | head -1
echo ""

if ! command -v tesseract &>/dev/null; then
  echo "Erreur: Tesseract OCR introuvable (commande: tesseract)."
  echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fra"
  echo "  macOS: brew install tesseract tesseract-lang"
  exit 1
fi

echo "3. Tesseract"
tesseract --version | head -1
echo ""

echo "4. Dépendances Python (requirements.txt)"
echo "   (peut prendre plusieurs minutes la première fois, notamment PaddleOCR)"
"${PIP_CMD[@]}" install -r requirements.txt
echo ""

echo "5. Dossiers locaux"
mkdir -p uploads images
echo ""

echo "6. Lancement de l'API sur http://0.0.0.0:5000"
echo "   Ctrl+C pour arrêter"
echo "=========================================="
exec python3 app.py
