import cv2
import os

def pipeline_tratamento(img_bgra, nome_arquivo):
    """Aplica o tratamento final à imagem já segmentada."""
    if img_bgra is None or img_bgra.size == 0: return None
    
    print(f"  -> Executando tratamento na imagem '{os.path.basename(nome_arquivo)}'...")

    b, g, r, a = cv2.split(img_bgra)
    
    img_bgr = cv2.merge([b, g, r])
    
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_cinza)
    
    img_normalizada = cv2.normalize(img_clahe, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    target_size = 400
    h, w = img_normalizada.shape
    
    if max(h, w) == 0: return None
    
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    img_resized = cv2.resize(img_normalizada, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    alpha_resized = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    img_resized[alpha_resized == 0] = 0
    
    delta_w, delta_h = target_size - new_w, target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    img_final = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
    
    return img_final