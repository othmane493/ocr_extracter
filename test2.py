from paddleocr import PaddleOCR
import cv2

ocr_fr = PaddleOCR(lang="fr", use_doc_orientation_classify=False, use_doc_unwarping=False)
ocr_ar = PaddleOCR(lang="ar", use_doc_orientation_classify=False, use_doc_unwarping=False)

def read_text(img_path, lang="fr"):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {img_path}")

    ocr = ocr_ar if lang == "ar" else ocr_fr
    results = ocr.predict(img)

    texts = []
    for page in results:
        if "rec_texts" in page:
            texts.extend(page["rec_texts"])

    return " ".join(texts)

print(read_text("images/carte-grise-recto.jpg", lang="ar"))