from menu_genero import menu_genero
from processa_img import processa_img
from mover_img_original import mover_img_original
from obter_caminhos import obter_caminhos
from carregar_imagem import carregar_imagem
import os

def tratar_img_cortada():
    
    modo = "Individual"
    
    CAMINHO_IMAGEM_ENTRADA, PASTA_SAIDA_DATASET = obter_caminhos(1)
          
    if not CAMINHO_IMAGEM_ENTRADA or not PASTA_SAIDA_DATASET:
        print("[AVISO] Seleção cancelada. Voltando ao menu principal.")
        return
    
    nome_arquivo = os.path.splitext(os.path.basename(CAMINHO_IMAGEM_ENTRADA))[0]

    img_original = carregar_imagem(CAMINHO_IMAGEM_ENTRADA)

    if img_original is None:
        print("[AVISO] Falha ao carregar a imagem. Voltando ao menu principal.")
        return
          
    genero = menu_genero()

    if not genero:
        print("[AVISO] Gênero nao selecionado. Voltando...")
        return

    imgSalva = processa_img(img_original, nome_arquivo, PASTA_SAIDA_DATASET, modo, genero)

    if imgSalva:
        mover_img_original(imgSalva, CAMINHO_IMAGEM_ENTRADA, nome_arquivo)
        
    input("\n >>> Pressione Enter para voltar ao menu principal...")