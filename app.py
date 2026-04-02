"""
Application Flask pour l'extraction de documents marocains
Supporte: CIN Old, CIN New, Carte Grise Recto, Carte Grise Verso
"""
import os
import sys
import time
from pathlib import Path

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractors.cin_extractor import CINExtractor
from extractors.carte_grise_extractor import CarteGriseExtractor
from ocr_manager import OCRManager

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

SUPPORTED_DOCUMENT_TYPES = frozenset({
    'cin_old',
    'cin_new',
    'carte_grise_recto',
    'carte_grise_verso',
})

Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def cleanup_file(filepath):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Fichier temporaire supprimé: {filepath}")
    except Exception as e:
        print(f"Erreur lors de la suppression du fichier {filepath}: {e}")


def _normalize_optional_document_type(raw):
    """
    None / vide / 'null' / 'none' (insensible à la casse) => détection auto.
    Sinon retourne la chaîne stripped.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    if s.lower() in ('null', 'none'):
        return None
    return s


def parse_document_type_from_request():
    """
    Priorité : champ formulaire multipart, puis query string (?document_type=).
    """
    raw = request.form.get('document_type')
    if raw is None:
        raw = request.args.get('document_type')
    return _normalize_optional_document_type(raw)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'service': 'Document Extraction API',
        'version': '1.0.0',
        'supported_documents': sorted(SUPPORTED_DOCUMENT_TYPES),
        'ocr_ready': OCRManager.is_ready()
    }), 200


@app.route('/extract', methods=['POST'])
def extract_document():
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({
            'error': 'Aucun fichier fourni',
            'message': 'Le paramètre "file" est requis'
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'error': 'Nom de fichier vide',
            'message': 'Le fichier doit avoir un nom valide'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Extension non autorisée',
            'message': f'Extensions autorisées: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
        }), 400

    filename = secure_filename(file.filename)
    timestamp = int(time.time() * 1000)
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(filepath)

        requested = parse_document_type_from_request()
        document_type_source = 'auto'

        if requested is not None:
            if requested not in SUPPORTED_DOCUMENT_TYPES:
                cleanup_file(filepath)
                return jsonify({
                    'error': 'Type de document invalide',
                    'message': (
                        f'Valeur reçue: {requested!r}. '
                        'Omettre le paramètre pour la détection automatique.'
                    ),
                    'valid_types': sorted(SUPPORTED_DOCUMENT_TYPES),
                }), 400
            document_type = requested
            document_type_source = 'request'
            print(f"Type fourni par le client: {document_type}")
        else:
            from extractors.document_detector import detect_document_type
            document_type = detect_document_type(filepath)
            print(f"Type détecté automatiquement: {document_type}")

        extraction_start = time.time()

        if document_type in ['cin_old', 'cin_new']:
            extractor = CINExtractor()
            result = extractor.extract(filepath, document_type)
        else:
            extractor = CarteGriseExtractor(
                recto_template_json="config/carte_grise_recto_template.json",
                verso_template_json="config/carte_grise_verso_template.json"
            )
            result = extractor.extract(filepath, document_type, debug=False)

        extraction_time = time.time() - extraction_start
        total_time = time.time() - start_time

        cleanup_file(filepath)

        return jsonify({
            'success': True,
            'document_type': document_type,
            'document_type_source': document_type_source,
            'data': result,
            'processing_time': {
                'extraction_seconds': round(extraction_time, 2),
                'total_seconds': round(total_time, 2)
            },
            'metadata': {
                'filename': filename,
                'timestamp': timestamp
            }
        }), 200

    except Exception as e:
        cleanup_file(filepath)

        print(f"Erreur lors de l'extraction: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'error': 'Erreur lors de l\'extraction',
            'message': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'Fichier trop volumineux',
        'message': 'La taille maximale autorisée est de 16MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Route non trouvée',
        'message': 'Endpoint non disponible. Utilisez /health ou /extract'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Erreur interne du serveur',
        'message': 'Une erreur inattendue s\'est produite'
    }), 500

def init_ocr():
    print("=" * 70)
    print("Initialisation des modèles OCR au démarrage...")
    print("=" * 70)

    OCRManager()
    OCRManager.warmup()

    print("Modèles OCR prêts !")
    print("=" * 70)

if __name__ == '__main__':
    init_ocr()
    print("=" * 70)
    print("Démarrage de l'API d'extraction de documents")
    print("=" * 70)
    print(f"Dossier des uploads: {app.config['UPLOAD_FOLDER']}")
    print(f"Extensions autorisées: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")
    print(f"Taille max des fichiers: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    print("\nDocuments supportés:")
    print("  - cin_old: Anciennes cartes d'identité marocaines")
    print("  - cin_new: Nouvelles cartes d'identité marocaines")
    print("  - carte_grise_recto: Cartes grises (recto)")
    print("  - carte_grise_verso: Cartes grises (verso)")
    print("\nEndpoints disponibles:")
    print("  GET  /health  - Vérification de l'état du service")
    print("  POST /extract - Extraction de document")
    print("    (optionnel) document_type en form ou ?document_type= ; vide => auto")
    print("=" * 70)
    print("\nServeur démarré sur http://0.0.0.0:5000")
    print("Appuyez sur Ctrl+C pour arrêter\n")

    app.run(host='0.0.0.0', port=5000, debug=False)