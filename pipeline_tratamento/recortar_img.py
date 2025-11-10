from obter_caminhos import obter_caminhos
from carregar_imagem import carregar_imagem
from ferramenta_selecao import ferramenta_selecao
from estado_selecao import EstadoSelecao
import os

estado = EstadoSelecao()

def recortar_img():
    
    while True:
        
        CAMINHO_IMAGEM_ENTRADA, PASTA_SAIDA_DATASET = obter_caminhos(1)
                        
        if not CAMINHO_IMAGEM_ENTRADA or not PASTA_SAIDA_DATASET:
            print("[AVISO] Seleção cancelada. Voltando ao menu principal.")
            break

        nome_arquivo = os.path.splitext(os.path.basename(CAMINHO_IMAGEM_ENTRADA))[0]

        img_original = carregar_imagem(CAMINHO_IMAGEM_ENTRADA)

        if img_original is None:
            print("[AVISO] Falha ao carregar a imagem. Voltando ao menu principal.")
            break

        largura_max_janela = 1000

        ferramenta_selecao(CAMINHO_IMAGEM_ENTRADA , PASTA_SAIDA_DATASET, img_original, largura_max_janela, nome_arquivo, estado)
        break
