"""
Application Flask pour l'extraction de documents marocains
Supporte: CIN Old, CIN New, Carte Grise Recto, Carte Grise Verso
"""
import os
import sys
import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import des extracteurs
from extractors.cin_extractor import CINExtractor
from extractors.carte_grise_extractor import CarteGriseExtractor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Créer le dossier uploads
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)


def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def cleanup_file(filepath):
    """Supprime un fichier temporaire de manière sécurisée"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Fichier temporaire supprimé: {filepath}")
    except Exception as e:
        print(f"Erreur lors de la suppression du fichier {filepath}: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé pour vérifier que l'API fonctionne"""
    return jsonify({
        'status': 'ok',
        'service': 'Document Extraction API',
        'version': '1.0.0',
        'supported_documents': [
            'cin_old',
            'cin_new',
            'carte_grise_recto',
            'carte_grise_verso'
        ]
    }), 200


@app.route('/extract', methods=['POST'])
def extract_document():
    """
    Endpoint principal pour l'extraction de documents
    
    Paramètres:
        - file: Fichier image du document
        - document_type: Type de document (cin_old, cin_new, carte_grise_recto, carte_grise_verso)
        
    Retour:
        JSON contenant les données extraites
    """
    start_time = time.time()
    
    # Validation du fichier
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
    
    # Récupération du type de document
    document_type = request.form.get('document_type', '').lower()
    
    if not document_type:
        return jsonify({
            'error': 'Type de document manquant',
            'message': 'Le paramètre "document_type" est requis',
            'valid_types': ['cin_old', 'cin_new', 'carte_grise_recto', 'carte_grise_verso']
        }), 400
    
    valid_types = ['cin_old', 'cin_new', 'carte_grise_recto', 'carte_grise_verso']
    if document_type not in valid_types:
        return jsonify({
            'error': 'Type de document invalide',
            'message': f'Type "{document_type}" non supporté',
            'valid_types': valid_types
        }), 400
    
    # Sauvegarde temporaire du fichier
    filename = secure_filename(file.filename)
    timestamp = int(time.time() * 1000)
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        file.save(filepath)
        print(f"Fichier sauvegardé: {filepath}")
        
        # Extraction selon le type de document
        print(f"Début de l'extraction pour type: {document_type}")
        extraction_start = time.time()
        
        if document_type in ['cin_old', 'cin_new']:
            extractor = CINExtractor()
            result = extractor.extract(filepath, document_type)
        else:  # carte_grise_recto ou carte_grise_verso
            extractor = CarteGriseExtractor()
            result = extractor.extract(filepath, document_type)
        
        extraction_time = time.time() - extraction_start
        total_time = time.time() - start_time
        
        print(f"Extraction terminée en {extraction_time:.2f}s")
        print(f"Temps total de traitement: {total_time:.2f}s")
        
        # Nettoyage du fichier temporaire
        cleanup_file(filepath)
        
        # Retour du résultat
        return jsonify({
            'success': True,
            'document_type': document_type,
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
        # Nettoyage en cas d'erreur
        cleanup_file(filepath)
        
        print(f"Erreur lors de l'extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': 'Erreur lors de l\'extraction',
            'message': str(e),
            'document_type': document_type
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Gestion des fichiers trop volumineux"""
    return jsonify({
        'error': 'Fichier trop volumineux',
        'message': 'La taille maximale autorisée est de 16MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Gestion des routes non trouvées"""
    return jsonify({
        'error': 'Route non trouvée',
        'message': 'Endpoint non disponible. Utilisez /health ou /extract'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestion des erreurs internes"""
    return jsonify({
        'error': 'Erreur interne du serveur',
        'message': 'Une erreur inattendue s\'est produite'
    }), 500


if __name__ == '__main__':
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
    print("=" * 70)
    print("\nServeur démarré sur http://0.0.0.0:5000")
    print("Appuyez sur Ctrl+C pour arrêter\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
