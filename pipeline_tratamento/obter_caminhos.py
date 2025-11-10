from tkinter import filedialog
from time import sleep
import os
import tkinter as tk

"""
    Tipos de seleção: 
        1 - Modo Individual - Selecionar imagem de entrada e pasta de saida
        2 - Modo Individual - Selecionar pasta de imagens usadas
        3 - Modo Pasta - Selecionar pasta de entrada e pasta de saida
"""

def obter_caminhos(tipoSeleção):
    # obtem os caminhos da imagem de entrada e da pasta de saida
    root = None

    # Cria uma janela temporária para obter os caminhos
    root = tk.Tk()
    # root.withdraw() # se for rodar no terminal, comenta esta linha
    
    if tipoSeleção in [1, 2, 3]:
        
        # Esse tipo é para selecionar a imagem de entrada e a pasta de saída
        if tipoSeleção == 1:
            try:

                print("\n+-----------------------------------------------------+")
                print("-> Por favor, selecione o arquivo de imagem de entrada...")
                sleep(1)
                caminho_img = filedialog.askopenfilename(
                    title="Selecione o arquivo de imagem",
                    filetypes=[("Arquivos de Imagem", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("Todos os arquivos", "*.*")]
                )

                if not caminho_img:
                    print("[AVISO] Nenhuma imagem selecionada. voltando ao menu principal.")
                    sleep(1)
                    return None, None

                print(f"-> Imagem selecionada: {os.path.basename(caminho_img)}")
                
                print("\n+-----------------------------------------------------------------+")
                print("\n-> Por favor, selecione a pasta de destino para salvar a imagem...")
                sleep(1)
                caminho_pasta = filedialog.askdirectory(title="Selecione a pasta de destino da imagem")

                if not caminho_pasta:
                    print("[AVISO] Nenhuma pasta de destino selecionada. voltando ao menu principal.")
                    sleep(1)
                    return None, None

                print(f"-> Pasta de destino: {caminho_pasta}")
                
                return caminho_img, caminho_pasta
            
            finally:
                if root:
                    root.destroy()
                    
        # Esse tipo é para selecionar a pasta onde será movida a imagem original após o recorte
        elif tipoSeleção == 2:
            try:
                
                print("\n+-----------------------------------------------------------------+")
                print("\n-> Por favor, selecione a pasta de destino para mover a imagem original...")
                sleep(1)
                pasta_destino_imgUsadas = filedialog.askdirectory(title="Selecione a pasta para mover a imagem original")

                if not pasta_destino_imgUsadas:
                    print("[AVISO] Nenhuma pasta de entrada selecionada. voltando ao menu principal.")
                    sleep(1)
                    return None

                print(f"-> Pasta que guardará a imagem original movida: {pasta_destino_imgUsadas}")
                
                return pasta_destino_imgUsadas
            
            finally:
                if root:
                    root.destroy()
                    
        # Esse tipo é para selecionar a pasta de entrada e a pasta de saida quando for tratar uma pasta inteira
        elif tipoSeleção == 3:
            try:
                
                print("\n+-----------------------------------------------------------------+")
                print("\n-> Por favor, selecione a pasta de entrada das imagens...")
                sleep(1)
                caminho_pasta_entrada = filedialog.askdirectory(title="Selecione a pasta de entrada das imagens")

                if not caminho_pasta_entrada:
                    print("[AVISO] Nenhuma pasta de entrada selecionada. voltando ao menu principal.")
                    sleep(1)
                    return None, None

                print(f"-> Pasta de entrada: {caminho_pasta_entrada}")
                
                print("\n+-----------------------------------------------------------------+")
                print("\n-> Por favor, selecione a pasta de destino das imagens...")
                sleep(1)
                caminho_pasta_destino = filedialog.askdirectory(title="Selecione a pasta de destino das imagens")

                if not caminho_pasta_destino:
                    print("[AVISO] Nenhuma pasta de destino selecionada. voltando ao menu principal.")
                    sleep(1)
                    return None, None

                print(f"-> Pasta de destino: {caminho_pasta_destino}")
                
                return caminho_pasta_entrada, caminho_pasta_destino
            
            finally:
                if root:
                    root.destroy()
                    
    else:
        print("[ERRO] Tipo inválido. Tente novamente.")
        sleep(1)