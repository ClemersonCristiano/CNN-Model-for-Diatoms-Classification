# Monta o Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define o caminho para o arquivo zip no seu Drive
# Substitua 'caminho/para/seu/arquivo.zip' pelo caminho real do seu arquivo
zip_file_path = '/content/drive/MyDrive/dataset/dataset.zip'

# Define o diretório de destino para extrair os arquivos
extract_path = '/content/'

# Cria o diretório de destino se ele não existir
import os
if not os.path.exists(extract_path):
    os.makedirs(extract_path)

# Descompacta o arquivo zip
# Usa um comando shell para descompactar. O '!' permite executar comandos shell no Colab.
print(f"Descompactando {zip_file_path} para {extract_path}...")
!unzip -q "{zip_file_path}" -d "{extract_path}"
print("Descompactação concluída.")

# Agora você pode acessar os arquivos descompactados no diretório especificado (por exemplo, '/content/dataset')
# Exemplo: listar o conteúdo do diretório extraído
print(f"\nConteúdo do diretório descompactado ({extract_path}):")
!ls "{extract_path}"

# Você pode agora usar a variável 'extract_path' nas células subsequentes
# para referenciar a localização dos seus dados. Por exemplo, para definir DATASET_DIR:
# DATASET_DIR = extract_path