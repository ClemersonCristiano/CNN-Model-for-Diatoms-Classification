import numpy as np
from rembg import remove
from PIL import Image
import os
import cv2

def segmentacao_com_ia(imagem_roi, nome_arquivo):
    """Remove o fundo usando IA e suaviza as bordas."""
    
    print(f"  -> Executando segmentação com IA na imagem '{os.path.basename(nome_arquivo)}'...")
    
    imagem_rgb = cv2.cvtColor(imagem_roi, cv2.COLOR_BGR2RGB)
    
    imagem_pil = Image.fromarray(imagem_rgb)
    
    resultado_pil_ia = remove(imagem_pil, alpha_matting=True)
    
    resultado_array_ia = np.array(resultado_pil_ia)
    
    print(f"  -> Suavizando as bordas do recorte da imagem '{os.path.basename(nome_arquivo)}'...")
    
    b, g, r, a = cv2.split(resultado_array_ia)
    
    kernel_size = (7, 7)
    
    mascara_suavizada = cv2.GaussianBlur(a, kernel_size, 0)
    
    resultado_final_bgra = cv2.merge([b, g, r, mascara_suavizada])
    
    return resultado_final_bgra