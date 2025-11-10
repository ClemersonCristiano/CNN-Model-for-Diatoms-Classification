import cv2
import numpy as np

def carregar_imagem(caminho_do_arquivo):
    # Carrega uma imagem de um caminho de arquivo.
    try:
        with open(caminho_do_arquivo, 'rb') as f:
            encoded_img = np.frombuffer(f.read(), dtype=np.uint8)
        
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"[ERRO] Falha ao decodificar a imagem. O arquivo pode estar corrompido.")
            return None
        
        print("\n[SUCESSO] Imagem carregada corretamente.")
        return img

    except FileNotFoundError:
        print(f"[ERRO] O arquivo não foi encontrado no caminho: {caminho_do_arquivo}")
        return None
    
    except Exception as e:
        print(f"[ERRO] Uma exceção ocorreu ao tentar ler o arquivo: {e}")
        return None