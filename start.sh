#!/usr/bin/env bash
# Démarrage du projet : vérifie Python, pip (version minimale), Tesseract (+ langues fra/ara), deps, lance l'API.
#
# Variables optionnelles :
#   AUTO_INSTALL_TESSERACT=1       — installer le binaire Tesseract sans question (si absent)
#   AUTO_INSTALL_TESSERACT_LANGS=1 — installer les données fra/ara via le gestionnaire de paquets sans question

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

install_tesseract_interactive() {
  local os
  os="$(uname -s 2>/dev/null || echo unknown)"
  case "$os" in
    Darwin)
      if command -v brew &>/dev/null; then
        echo "   Installation via Homebrew (moteur + données multilingues, dont fra/ara)..."
        brew install tesseract tesseract-lang
      else
        echo "Erreur: Homebrew introuvable. Installez https://brew.sh puis relancez."
        return 1
      fi
      ;;
    Linux)
      if command -v apt-get &>/dev/null; then
        echo "   Installation via apt (sudo requis) — tesseract + dictionnaires fra et ara..."
        sudo apt-get update -qq
        sudo apt-get install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara
      elif command -v dnf &>/dev/null; then
        echo "   Installation via dnf (sudo requis) — tesseract + langpacks fra et ara..."
        sudo dnf install -y tesseract tesseract-langpack-fra tesseract-langpack-ara
      elif command -v pacman &>/dev/null; then
        echo "   Installation via pacman (sudo requis) — tesseract + traineddata fra et ara..."
        sudo pacman -S --noconfirm tesseract tesseract-data-fra tesseract-data-ara
      elif command -v zypper &>/dev/null; then
        echo "   Installation via zypper (sudo requis) — tesseract-ocr + traineddata fra/ara..."
        sudo zypper install -y tesseract-ocr tesseract-traineddata-fra tesseract-traineddata-ara
      else
        echo "Gestionnaire de paquets non reconnu. Installez tesseract manuellement."
        return 1
      fi
      ;;
    *)
      echo "OS non géré pour l'installation auto ($os). Installez Tesseract manuellement."
      return 1
      ;;
  esac
}

tesseract_lang_lines() {
  tesseract --list-langs 2>/dev/null | tail -n +2 | tr -d '\r' | sed '/^$/d'
}

tesseract_has_lang() {
  local want="$1"
  tesseract_lang_lines | grep -qx "$want"
}

install_tesseract_lang_packs_only() {
  local os
  os="$(uname -s 2>/dev/null || echo unknown)"
  case "$os" in
    Darwin)
      if command -v brew &>/dev/null; then
        echo "   Homebrew : paquets tesseract-lang (données fra, ara, etc.)..."
        brew install tesseract-lang
      else
        return 1
      fi
      ;;
    Linux)
      if command -v apt-get &>/dev/null; then
        echo "   apt : tesseract-ocr-fra et tesseract-ocr-ara..."
        sudo apt-get update -qq
        sudo apt-get install -y tesseract-ocr-fra tesseract-ocr-ara
      elif command -v dnf &>/dev/null; then
        echo "   dnf : langpacks fra et ara..."
        sudo dnf install -y tesseract-langpack-fra tesseract-langpack-ara
      elif command -v pacman &>/dev/null; then
        echo "   pacman : tesseract-data-fra et tesseract-data-ara..."
        sudo pacman -S --noconfirm tesseract-data-fra tesseract-data-ara
      elif command -v zypper &>/dev/null; then
        echo "   zypper : traineddata fra et ara..."
        sudo zypper install -y tesseract-traineddata-fra tesseract-traineddata-ara || return 1
      else
        return 1
      fi
      ;;
    *)
      return 1
      ;;
  esac
}

print_tesseract_lang_manual() {
  echo "Installez les données de langue **français (fra)** et **arabe (ara)** pour Tesseract."
  echo "  Debian/Ubuntu: sudo apt-get install tesseract-ocr-fra tesseract-ocr-ara"
  echo "  Fedora:        sudo dnf install tesseract-langpack-fra tesseract-langpack-ara"
  echo "  Arch:          sudo pacman -S tesseract-data-fra tesseract-data-ara"
  echo "  macOS:         brew install tesseract-lang"
  echo "  openSUSE:      sudo zypper install tesseract-traineddata-fra tesseract-traineddata-ara"
}

ensure_tesseract_langs() {
  if ! tesseract_has_lang fra || ! tesseract_has_lang ara; then
    echo ""
    echo "Les langues Tesseract **fra** (français) et **ara** (arabe) sont requises (détection carte grise, OCR fra+ara)."
    if ! tesseract_has_lang fra; then echo "  — manquant : fra"; fi
    if ! tesseract_has_lang ara; then echo "  — manquant : ara"; fi
    echo ""

    local do_install=0
    if [[ "${AUTO_INSTALL_TESSERACT_LANGS:-}" == "1" ]]; then
      do_install=1
    elif [[ -t 0 ]]; then
      read -r -p "Installer automatiquement les paquets fra/ara via votre gestionnaire de paquets ? [y/N] " _ans
      case "${_ans:-}" in
        y|Y|yes|oui|Oui) do_install=1 ;;
      esac
    else
      print_tesseract_lang_manual
      echo "Ou définissez AUTO_INSTALL_TESSERACT_LANGS=1 puis relancez."
      return 1
    fi

    if [[ "$do_install" -ne 1 ]]; then
      print_tesseract_lang_manual
      return 1
    fi

    if ! install_tesseract_lang_packs_only; then
      echo "Installation automatique des langues impossible sur ce système."
      print_tesseract_lang_manual
      return 1
    fi

    hash -r 2>/dev/null || true
    if ! tesseract_has_lang fra || ! tesseract_has_lang ara; then
      echo "Après installation, **fra** ou **ara** est toujours absent. Vérifiez tessdata ou installez à la main."
      print_tesseract_lang_manual
      return 1
    fi
  fi
  return 0
}

ensure_tesseract() {
  if command -v tesseract &>/dev/null; then
    return 0
  fi

  echo "Tesseract OCR introuvable."
  local do_install=0

  if [[ "${AUTO_INSTALL_TESSERACT:-}" == "1" ]]; then
    do_install=1
  elif [[ -t 0 ]]; then
    read -r -p "Tenter l'installation automatique de Tesseract ? [y/N] " _ans
    case "${_ans:-}" in
      y|Y|yes|oui|Oui) do_install=1 ;;
    esac
  else
    echo "STDIN non interactif : définissez AUTO_INSTALL_TESSERACT=1 ou installez Tesseract à la main (+ fra/ara)."
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara"
    echo "  macOS: brew install tesseract tesseract-lang"
    return 1
  fi

  if [[ "$do_install" -ne 1 ]]; then
    echo "Installez Tesseract puis relancez ce script (avec dictionnaires fra et ara)."
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara"
    echo "  macOS: brew install tesseract tesseract-lang"
    return 1
  fi

  if ! install_tesseract_interactive; then
    return 1
  fi

  hash -r 2>/dev/null || true
  if ! command -v tesseract &>/dev/null; then
    echo "Erreur: tesseract toujours introuvable après installation."
    return 1
  fi
  return 0
}

if ! ensure_tesseract; then
  exit 1
fi

echo "3. Tesseract"
tesseract --version | head -1
if ! ensure_tesseract_langs; then
  exit 1
fi
echo "   Langues détectées (requis fra + ara) : OK"
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
