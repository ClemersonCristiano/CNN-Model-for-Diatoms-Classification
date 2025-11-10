import cv2

def salva_img(caminho_completo, imagem):
    """
    Salva uma imagem de forma segura, evitando problemas com caracteres especiais no caminho.
    """
    try:
        # Codifica a imagem para o formato PNG na memória
        sucesso, buffer = cv2.imencode('.png', imagem)
        if not sucesso:
            print(f"[ERRO] Falha ao codificar a imagem para salvamento em '{caminho_completo}'")
            return False
        
        # Salva o buffer da memória para o disco usando o file handler do Python
        with open(caminho_completo, 'wb') as f:
            f.write(buffer)
        return True
    
    except Exception as e:
        print(f"[ERRO] Falha ao salvar a imagem em '{caminho_completo}': {e}")
        return False
