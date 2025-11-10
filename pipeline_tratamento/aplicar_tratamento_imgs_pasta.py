from carregar_imagem import carregar_imagem
from processa_img import processa_img
import os

def aplicar_tratamento_imgs_pasta(pasta_entrada, pasta_saida, modo, genero):

    total = 0
    print(f"\n >>> Aplicando tratamento na pasta: {pasta_saida}...")
    print("=" * 100)
    
    for arquivo in os.listdir(pasta_entrada):
        
        if arquivo is not None and arquivo.endswith('.png') or arquivo.endswith('.jpg') or arquivo.endswith('.jpeg') or arquivo.endswith('.tif'):
            
            arquivo_path = os.path.join(pasta_entrada, arquivo)
            
            img = carregar_imagem(arquivo_path)
            
            print(f"\n >>> IMG-N: {total+1} | Processando arquivo: {arquivo_path}")
            
            if img is not None:
                
                print("=" * 100)
                processa_img(img, arquivo, pasta_saida, modo, genero)
                
                print(f" >>> Arquivo: {arquivo} processado com sucesso.")
                
            total += 1
            
        else:
            print(f"Arquivo: {arquivo} é inválido.")      
            
    print("=" * 100)  
    print(f"\n >>>Processo concluido. Total de arquivos processados na pasta '{pasta_entrada}': {total}")
    print("=" * 100)  