"""
Script de test pour l'API d'extraction de documents
Teste tous les endpoints et types de documents
"""
import requests
import json
import time
from pathlib import Path


API_BASE_URL = "http://localhost:5000"


def test_health():
    """Test du endpoint health"""
    print("=" * 70)
    print("Test 1: Health Check")
    print("-" * 70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print("Statut:", data.get('status'))
            print("Service:", data.get('service'))
            print("Version:", data.get('version'))
            print("Documents supportés:", ', '.join(data.get('supported_documents', [])))
            print("Test RÉUSSI")
            return True
        else:
            print(f"Erreur HTTP {response.status_code}")
            print("Test ÉCHOUÉ")
            return False
            
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        print("Test ÉCHOUÉ")
        return False


def test_extraction(image_path, document_type):
    """Test d'extraction pour un type de document"""
    print("\n" + "=" * 70)
    print(f"Test: Extraction {document_type}")
    print("-" * 70)
    
    if not Path(image_path).exists():
        print(f"Fichier non trouvé: {image_path}")
        print("Test IGNORÉ")
        return None
    
    try:
        start_time = time.time()
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'document_type': document_type}
            
            print(f"Envoi de la requête pour {document_type}...")
            response = requests.post(
                f"{API_BASE_URL}/extract",
                files=files,
                data=data
            )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Statut: {response.status_code} OK")
            print(f"Temps de réponse: {elapsed:.2f}s")
            print(f"Temps d'extraction: {result.get('processing_time', {}).get('extraction_seconds')}s")
            print("\nDonnées extraites:")
            print(json.dumps(result.get('data', {}), indent=2, ensure_ascii=False))
            print("\nTest RÉUSSI")
            return True
        else:
            error_data = response.json()
            print(f"Erreur HTTP {response.status_code}")
            print("Erreur:", error_data.get('error'))
            print("Message:", error_data.get('message'))
            print("Test ÉCHOUÉ")
            return False
            
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        print("Test ÉCHOUÉ")
        return False


def test_invalid_document_type():
    """Test avec un type de document invalide"""
    print("\n" + "=" * 70)
    print("Test: Type de document invalide")
    print("-" * 70)
    
    # Créer un fichier temporaire bidon
    temp_file = Path("temp_test.jpg")
    temp_file.write_bytes(b"dummy")
    
    try:
        with open(temp_file, 'rb') as f:
            files = {'file': f}
            data = {'document_type': 'document_invalide'}
            
            response = requests.post(
                f"{API_BASE_URL}/extract",
                files=files,
                data=data
            )
        
        if response.status_code == 400:
            error_data = response.json()
            print(f"Statut: {response.status_code} (attendu)")
            print("Erreur:", error_data.get('error'))
            print("Message:", error_data.get('message'))
            print("Types valides:", ', '.join(error_data.get('valid_types', [])))
            print("Test RÉUSSI (erreur détectée correctement)")
            return True
        else:
            print(f"Statut inattendu: {response.status_code}")
            print("Test ÉCHOUÉ")
            return False
            
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        print("Test ÉCHOUÉ")
        return False
    finally:
        # Nettoyage
        if temp_file.exists():
            temp_file.unlink()


def test_missing_file():
    """Test sans fichier"""
    print("\n" + "=" * 70)
    print("Test: Requête sans fichier")
    print("-" * 70)
    
    try:
        data = {'document_type': 'cin_new'}
        response = requests.post(
            f"{API_BASE_URL}/extract",
            data=data
        )
        
        if response.status_code == 400:
            error_data = response.json()
            print(f"Statut: {response.status_code} (attendu)")
            print("Erreur:", error_data.get('error'))
            print("Message:", error_data.get('message'))
            print("Test RÉUSSI (erreur détectée correctement)")
            return True
        else:
            print(f"Statut inattendu: {response.status_code}")
            print("Test ÉCHOUÉ")
            return False
            
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        print("Test ÉCHOUÉ")
        return False


def main():
    """Lance tous les tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "TESTS DE L'API D'EXTRACTION" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    results = {}
    
    # Test 1: Health check
    results['health'] = test_health()
    
    # Test 2: CIN Old
    results['cin_old'] = test_extraction(
        'images/cin-old.jpg',
        'cin_old'
    )
    
    # Test 3: CIN New
    results['cin_new'] = test_extraction(
        'images/cin_new.png',
        'cin_new'
    )
    
    # Test 4: Carte Grise Recto
    results['carte_grise_recto'] = test_extraction(
        'images/carte-grise-recto.jpg',
        'carte_grise_recto'
    )
    
    # Test 5: Carte Grise Verso
    results['carte_grise_verso'] = test_extraction(
        'images/carte-grise-verso.jpg',
        'carte_grise_verso'
    )
    
    # Test 6: Type invalide
    results['invalid_type'] = test_invalid_document_type()
    
    # Test 7: Fichier manquant
    results['missing_file'] = test_missing_file()
    
    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES TESTS")
    print("=" * 70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        status = "RÉUSSI" if result is True else "ÉCHOUÉ" if result is False else "IGNORÉ"
        symbol = "✓" if result is True else "✗" if result is False else "○"
        print(f"{symbol} {test_name}: {status}")
    
    print("-" * 70)
    print(f"Total: {total} tests")
    print(f"Réussis: {passed}")
    print(f"Échoués: {failed}")
    print(f"Ignorés: {skipped}")
    
    if failed == 0 and passed > 0:
        print("\nTous les tests ont réussi!")
    else:
        print(f"\n{failed} test(s) ont échoué.")
    
    print("=" * 70)


if __name__ == "__main__":
    print("Assurez-vous que l'API est démarrée sur http://localhost:5000")
    print("Pour démarrer l'API: python app.py")
    print()
    input("Appuyez sur Entrée pour commencer les tests...")
    
    main()
