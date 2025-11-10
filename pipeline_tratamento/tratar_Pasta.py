from obter_caminhos import obter_caminhos
from menu_genero import menu_genero
from aplicar_tratamento_imgs_pasta import aplicar_tratamento_imgs_pasta
from time import sleep

def tratar_pasta():
    
    while True:
        
        modo = "Pasta"
            
        print("\n")
        print("=" * 200)
        
        print("\n\n >>> Iniciando o processo de tratamento de imagens em uma pasta...")
        print("\n-> Por favor, selecione as pastas de entrada e de saída...")
                            
        PASTA_ENTRADA, PASTA_SAIDA = obter_caminhos(3)
        
        if not PASTA_ENTRADA or not PASTA_SAIDA:
            print("[AVISO] Seleção cancelada. Voltando ao menu principal.")
            break
        
        genero = menu_genero()
        
        if not genero:
            print("[AVISO] Gênero nao selecionado. Voltando ao menu principal.")
            break
            
        aplicar_tratamento_imgs_pasta(PASTA_ENTRADA, PASTA_SAIDA, modo, genero)
        
        input("\nPressione qualquer tecla para voltar ao menu principal...")
        
        print("\nVoltando ao menu principal...")
        sleep(1)
        
        break        