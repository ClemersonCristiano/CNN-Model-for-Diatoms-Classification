from segmentacao_com_ia import segmentacao_com_ia
from pipeline_tratamento import pipeline_tratamento
from salva_img import salva_img
import os
import datetime

def processa_img(roi, nome_arquivo, PASTA_SAIDA_DATASET, modo, genero):
    """Função centralizada para processar e salvar uma imagem (ROI)."""
        
    if modo in ["Individual", "Pasta"]:
        
        genero_result = genero
            
        if genero_result:

            print(f"\nGênero selecionado: '{genero_result}'")
            print("+-------------------------------------------+")
        
            genero_pasta_nome = genero_result.strip().replace(" ", "_").capitalize()
                
            caminho_genero = os.path.join(PASTA_SAIDA_DATASET, genero_pasta_nome)

            if not os.path.exists(caminho_genero):
                os.makedirs(caminho_genero)
                print(f"  -> Criada nova pasta para o gênero: '{caminho_genero}'")
                
            if roi.size > 0:
                roi_segmentada = segmentacao_com_ia(roi, nome_arquivo)
                processed_roi = pipeline_tratamento(roi_segmentada, nome_arquivo)
                
                if processed_roi is not None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    genero_nome = os.path.basename(caminho_genero)
                    base_filename = f"{genero_nome}_{nome_arquivo}_{timestamp}"

                    print("\n  -> Processando e salvando imagem...")
                    print("+-------------------------------------------+")
                    
                    filename = f"{base_filename}.png"
                    output_path = os.path.join(caminho_genero, filename)
                    
                    if salva_img(output_path, processed_roi):
                        print(f"-> Imagem salva em: '{output_path}'")
                            
                    print("+-------------------------------------------+")
                    print("  -> Imagem processada e salva com sucesso!")
                            
                    if modo == "Individual":      
                        return output_path
                    elif modo == "Pasta":   
                        return True      
                    
        else: 
            print("\n[AVISO] Gênero nao informado")
            return False
        
    else:
        print("\n[ERRO] Modo inválido. Escolha 'Individual' ou 'Pasta'.\n")
        return False