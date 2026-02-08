#!/bin/bash
# Script de démarrage rapide pour l'API d'extraction

echo "=========================================="
echo "API d'Extraction de Documents Marocains"
echo "Démarrage Rapide"
echo "=========================================="
echo ""

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "Erreur: Python 3 n'est pas installé"
    exit 1
fi

echo "1. Vérification de Python"
python3 --version
echo ""

# Vérifier Tesseract
if ! command -v tesseract &> /dev/null; then
    echo "Erreur: Tesseract OCR n'est pas installé"
    echo ""
    echo "Installation:"
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fra"
    echo "  macOS: brew install tesseract tesseract-lang"
    exit 1
fi

echo "2. Vérification de Tesseract OCR"
tesseract --version | head -1
echo ""

# Installer les dépendances
echo "3. Installation des dépendances Python"
echo "Cela peut prendre quelques minutes..."
pip install -q -r requirements_flask.txt
echo "Dépendances installées"
echo ""

# Créer les dossiers nécessaires
echo "4. Création des dossiers"
mkdir -p uploads
mkdir -p images
echo "Dossiers créés"
echo ""

# Lancer l'API
echo "5. Démarrage de l'API"
echo ""
echo "=========================================="
echo "L'API démarre sur http://0.0.0.0:5000"
echo ""
echo "Endpoints disponibles:"
echo "  GET  /health  - Vérification de l'état"
echo "  POST /extract - Extraction de documents"
echo ""
echo "Documents supportés:"
echo "  - cin_old"
echo "  - cin_new"
echo "  - carte_grise_recto"
echo "  - carte_grise_verso"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter"
echo "=========================================="
echo ""

python3 app.py
