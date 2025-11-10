import os
import re

# regex para localizar arquivo que tenha "vertical" ou "horizontal" no nome
regex = r"vertical|horizontal"

pasta = "./dataset_final/Dataset_Final_Puro_Tratad/Pinnularia"

total_arquivos = len(os.listdir(pasta))

for i, arquivo in enumerate(os.listdir(pasta)):
    if arquivo.endswith(".png") and re.search(regex, arquivo):
        try:
            # os.remove(os.path.join(pasta, arquivo))
            print(f"{i+1} Arquivo: {arquivo} removido com sucesso.")
        except OSError as e:
            print(f"Erro ao remover o arquivo {arquivo}: {e}")
            
print(f"Total de arquivos da pasta no inicio: {total_arquivos}")
# print(f"Total de arquivos da pasta no final: {len(os.listdir(pasta))}")