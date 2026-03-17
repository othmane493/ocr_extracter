"""
Gestionnaire singleton pour les modèles OCR
Initialise PaddleOCR et le pool de workers une seule fois au démarrage
"""
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
from paddleocr import PaddleOCR


class OCRManager:
    """Singleton pour gérer les modèles OCR et le pool de threads"""

    _instance = None
    _reader_ar = None
    _reader_fr = None
    _executor = None
    _paddle_lock = Lock()
    _tesseract_lock = Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCRManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialise le manager une seule fois"""
        if not OCRManager._initialized:
            print("Initialisation des modèles OCR et du pool de workers...")
            start = time.time()

            OCRManager._reader_ar = PaddleOCR(
                lang="ar",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )

            OCRManager._reader_fr = PaddleOCR(
                lang="fr",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )

            # Pool global partagé par toute l'application
            OCRManager._executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="ocr-worker"
            )

            elapsed = time.time() - start
            print(f"Modèles OCR + executor chargés en {elapsed:.2f}s")
            OCRManager._initialized = True

        # ... existing code ...

    @classmethod
    def warmup(cls):
        """Warmup OCR pour éviter le coût du premier predict en pleine requête"""
        if not cls._initialized:
            cls()

        dummy = np.ones((64, 256, 3), dtype=np.uint8) * 255

        try:
            with cls._paddle_lock:
                cls._reader_fr.predict(dummy)
            print("[WARMUP] Paddle FR OK")
        except Exception as e:
            print(f"[WARMUP] Paddle FR échec: {e}")

        try:
            with cls._paddle_lock:
                cls._reader_ar.predict(dummy)
            print("[WARMUP] Paddle AR OK")
        except Exception as e:
            print(f"[WARMUP] Paddle AR échec: {e}")

        try:
            future = cls._executor.submit(lambda: None)
            future.result(timeout=5)
            print("[WARMUP] Executor OK")
        except Exception as e:
            print(f"[WARMUP] Executor échec: {e}")

    @classmethod
    def get_reader(cls):
        """Retourne les readers Paddle partagés"""
        if cls._reader_ar is None or cls._reader_fr is None:
            cls()
        return cls._reader_ar, cls._reader_fr

    @classmethod
    def get_executor(cls):
        """Retourne le pool de threads partagé"""
        if cls._executor is None:
            cls()
        return cls._executor

    @classmethod
    def get_locks(cls):
        """Retourne les locks partagés"""
        if not cls._initialized:
            cls()
        return cls._paddle_lock, cls._tesseract_lock

    @classmethod
    def is_ready(cls):
        return cls._initialized

    @classmethod
    def shutdown(cls):
        """Fermeture propre du pool si besoin"""
        if cls._executor is not None:
            cls._executor.shutdown(wait=True)
            cls._executor = None


# Instance globale
_ocr_manager = OCRManager()


def get_paddle_reader():
    return OCRManager.get_reader()


def get_ocr_executor():
    return OCRManager.get_executor()


def get_ocr_locks():
    return OCRManager.get_locks()