from time import sleep
from carregar_imagem import carregar_imagem
from obter_caminhos import obter_caminhos
import os
import shutil
import cv2

def mover_img_original(imgSalva, CAMINHO_IMAGEM_ENTRADA, nome_arquivo):

    caminho_ultima_recorte_salvo = carregar_imagem(imgSalva)
                
    print("\nAbrindo imagem recortada para visualização...")
    sleep(1)
    cv2.imshow("Pressione Enter para fechar...", caminho_ultima_recorte_salvo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    while True:
            
        resposta_pasta = input("\n >>> Mova a imagem original para uma pasta de imagens usadas! Digite o número da opção e pressione Enter:\n >>> [1] para escolher uma pasta ou [2] para continuar sem mover: ")
        
        if resposta_pasta == "1":
            
            pasta_destino_imgUsadas = obter_caminhos(2)
            
            if not pasta_destino_imgUsadas:
                print("[AVISO] Seleção cancelada. Voltando a seleção de pasta.")
                continue
            
            caminho_destino_completo = os.path.join(pasta_destino_imgUsadas, nome_arquivo + ".png")
            
            if os.path.exists(caminho_destino_completo):
                print("[ERRO] Já existe uma imagem com esse nome na pasta de destino. Operação cancelada.")
            else:
                try:
                    shutil.move(CAMINHO_IMAGEM_ENTRADA, caminho_destino_completo)
                    print(f"[SUCESSO] Imagem original movida para: '{caminho_destino_completo}'")
                    break
                except Exception as e:
                    print(f"[ERRO] Falha ao mover a imagem: {e}")          
                    continue
                
        elif resposta_pasta == "2":
            print("Continuando sem mover a imagem original...")
            break
        
        else:
            print("[AVISO] Opção inválida. Por favor, digite [1] ou [2].")
            sleep(1)
            continue