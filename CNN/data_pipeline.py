import os
import re
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit
from config import DIATOMS_CLASSES, IMAGE_SIZE, BATCH_SIZE, AUTOTUNE

def get_file_lists_and_groups(DATASET_DIR, CLASSES, class_to_int):
    """
    Mapeia todos os arquivos, extrai seus labels e seus "grupos de família"(Para o datset com augmentação bruta, mas funciona os demais).
    """
    
    try:   
        filepaths = []  # Lista de todos os caminhos de arquivo
        labels = []     # Lista de labels (0, 1, 2, 3, 4)
        groups = []     # Lista de "IDs de família" (o base_filename)

        # Regex para extrair o "base_filename"
        # Ele captura (grupo 1) o nome base e (grupo 2) o sufixo de augmentação
        # Ex: "Gomphonema_..._timestamp_horizontal.png"
        # Grupo 1: "Gomphonema_..._timestamp"
        # Grupo 2: "_horizontal"
        # Ex: "Gomphonema_..._timestamp.png"
        # Grupo 1: "Gomphonema_..._timestamp"
        # Grupo 2: None (ou string vazia)
        pattern = re.compile(
            r'(.+?)(_horizontal|_vertical|_90_graus|_270_graus)?\.(png|jpg|jpeg|bmp|tiff)$',
            re.IGNORECASE
        )

        print(f"Mapeando arquivos em '{DATASET_DIR}'...")

        for class_name in CLASSES:
            class_dir = os.path.join(DATASET_DIR, class_name)
            if not os.path.isdir(class_dir):
                print(f"[AVISO] Pasta não encontrada: {class_dir}")
                continue

            label = class_to_int[class_name]

            for filename in os.listdir(class_dir):
                match = pattern.match(filename)

                if match:
                    # O "base_filename" é nosso ID de grupo (família)
                    base_filename = match.group(1)
                    filepath = os.path.join(class_dir, filename)

                    filepaths.append(filepath)
                    labels.append(label)
                    groups.append(base_filename)

        print(f"Mapeamento concluído. {len(filepaths)} arquivos encontrados.")
        
        return np.array(filepaths), np.array(labels), np.array(groups)
    
    except Exception as e:
        print(f"Erro durante o mapeamento dos arquivos: {e}")
        return None, None, None

def split_train_val(DATASET_DIR, CLASSES, class_to_int):
    
    try:
        # --- 2. Execução do Mapeamento ---
        
        all_filepaths, all_labels, all_groups = get_file_lists_and_groups(DATASET_DIR, CLASSES, class_to_int)

        # --- 3. Divisão por Grupo (Família) ---

        print("\nIniciando divisão Treino/Validação por 'Família'...")

        # Queremos 1 divisão (n_splits=1) com 20% dos *grupos* para teste/validação
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        # gss.split() nos dá os *índices* dos arrays para treino e validação
        # Usamos 'all_filepaths' como X (só para ter o tamanho), e 'all_groups' como 'groups'
        train_idx, val_idx = next(gss.split(all_filepaths, groups=all_groups))

        # --- 4. Criação das Listas Finais ---

        # Selecionamos os arquivos e labels com base nos índices
        train_files = all_filepaths[train_idx]
        train_labels = all_labels[train_idx]

        val_files = all_filepaths[val_idx]
        val_labels = all_labels[val_idx]

        print("Divisão concluída.")
        print(f"  Imagens de Treino:    {len(train_files)}")
        print(f"  Imagens de Validação: {len(val_files)}")
        
        return train_files, train_labels, val_files, val_labels, all_groups, train_idx, val_idx
    
    except Exception as e:
        print(f"Erro durante a divisão do dataset: {e}")
        return None, None, None, None, None, None, None

def check_data_leakage(all_groups, train_idx, val_idx):
    # --- 5. Verificação de Vazamento (Data Leakage) ---

    # Para confirmar, vamos verificar se algum 'base_filename'
    # existe em *ambos* os conjuntos. A interseção deve ser 0.

    try:
        print("\nVerificando vazamento de dados entre Treino e Validação...")
        train_groups_set = set(all_groups[train_idx])
        val_groups_set = set(all_groups[val_idx])

        leakage = train_groups_set.intersection(val_groups_set)

        if not leakage:
            print("\n[SUCESSO] Verificação de vazamento concluída. Nenhuma família de imagens vazou entre os conjuntos de Treino e Validação.")
            return True
        else:
            print(f"\n[ERRO] Verificação de vazamento falhou! {len(leakage)} famílias estão em ambos os conjuntos.")
            return False
        
    except Exception as e:
        print(f"Erro durante a verificação de vazamento: {e}")
        return False
    
def compute_class_weights(train_labels, int_to_class):
    # --- 6. Cálculo dos Pesos de Classe (Class Weights) ---

    try:
        
        print("\nCalculando pesos de classe para lidar com desbalanceamento...")

        # É importante calcular os pesos com base APENAS nos dados de TREINO,
        # pois é neles que o modelo será treinado.
        y_integers = train_labels

        # Obter as classes únicas presentes nos dados de treino
        classes_unicas = np.unique(y_integers)
        print(f"Classes únicas encontradas nos dados de treino: {classes_unicas}")

        # Calcular os pesos
        # O modo 'balanced' faz o cálculo automaticamente: N_amostras / (N_classes * N_amostras_por_classe)
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=classes_unicas,
            y=y_integers
        )

        # Criar o dicionário de pesos que o Keras espera (ex: {0: 1.5, 1: 0.8, ...})
        class_weights = dict(zip(classes_unicas, class_weights_array))

        print("Cálculo de pesos concluído.")
        print("Estes pesos serão usados para penalizar erros em classes minoritárias:")

        # Imprimir os pesos de forma legível
        # Lembre-se: int_to_class = {0: 'Encyonema', 1: 'Eunotia', ...}
        # (Se você não tiver o 'int_to_class', pode imprimir o dicionário 'class_weights' diretamente)
        for class_int, weight in class_weights.items():
            class_name = int_to_class.get(class_int, "Classe Desconhecida")
            print(f"  -> Classe {class_int} ({class_name}): Peso = {weight:.4f}")
            
        return class_weights
    
    except Exception as e:
        print(f"Erro durante o cálculo dos pesos de classe: {e}")
        return None

# --- 7. Definição do Pipeline de Dados (com Augmentação e Rotação) ---
def load_image(filepath, label):
    """
    Carrega, decodifica, converte (Grayscale->RGB) e redimensiona.
    Saída: pixels no intervalo [0, 255]
    """
    try:
        image = tf.io.read_file(filepath)
        image = tf.io.decode_image(image, channels=1)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
        
        return image, label
    
    except Exception as e:
        print(f"Erro ao carregar a imagem {filepath}: {e}")
        return None, label

def augment_image(image, label):
    """
    Aplica augmentações aleatórias em tempo real.
    Entrada/Saída: pixels no intervalo [0, 255]
    """
    try:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        k_rot = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k_rot)
        image = tf.image.random_brightness(image, max_delta=0.1 * 255.0)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0.0, 255.0)
    
        return image, label
    
    except Exception as e:
        print(f"Erro ao aplicar augmentação na imagem: {e}")
        return image, label

def normalize_image(image, label):
    """
    Normaliza a imagem para o formato que a ResNetV2 espera [-1, 1].
    """
    try:
        image = tf.keras.applications.resnet_v2.preprocess_input(image)
    
        return image, label
    
    except Exception as e:
        print(f"Erro ao normalizar a imagem: {e}")
        return image, label

def create_dataset(filepaths, labels, is_training):
    """
    Cria um objeto tf.data.Dataset completo.
    """
    try:
        dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

        if is_training:
            # 1. Embaralhar os FILEPATHS (strings), o que é leve para a RAM.
            dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)

        # 2. Carregar as imagens (agora em ordem aleatória)
        dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)

        # Para treino sem augmentação, basta passar is_training=False
        if is_training:
            # 3. Aplicar a augmentação
            dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)

        # 4. Agrupar em lotes (Batch)
        dataset = dataset.batch(BATCH_SIZE)

        # 5. Normalizar (depois do batch, é mais rápido na GPU)
        dataset = dataset.map(normalize_image, num_parallel_calls=AUTOTUNE)

        # 6. Otimização: Pré-carregar o próximo lote
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset
    
    except Exception as e:
        print(f"Erro ao criar o dataset: {e}")
        return None

def create_data_pipelines(train_files, train_labels, val_files, val_labels, CLASSES, is_training):
    
    try:
        print("\nCriando pipelines de dados com AUGMENTAÇÃO ONLINE...")
    
        # --- 8. Criar os Datasets Finais ---

        # Converter labels para o formato correto (one-hot encoding)
        train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=len(CLASSES))
        val_labels_one_hot = tf.keras.utils.to_categorical(val_labels, num_classes=len(CLASSES))

        # Criar os pipelines
        train_dataset = create_dataset(train_files, train_labels_one_hot, is_training)
        val_dataset = create_dataset(val_files, val_labels_one_hot, is_training=False) # Validação NUNCA é aumentada

        print("Pipelines criados com sucesso.")
        print(f"  -> train_dataset (com augmentação): {train_dataset}")
        print(f"  -> val_dataset (sem augmentação):   {val_dataset}")
        
        return train_dataset, val_dataset
    
    except Exception as e:
        print(f"Erro ao criar os pipelines de dados: {e}")
        return None, None

def dataset_preparation(DATASET_DIR, is_training):
    
    try:
        print("="*30)
        print("\n--- INICIANDO A PREPARAÇÃO DO DATASET ---")
        print("="*30)
        
        print("\n\n")
        
        print("="*30)
        print("\nCarregando Classes...")
        CLASSES = DIATOMS_CLASSES().Diatoms_Classes_names
        print(f"\nClasses carregadas: {CLASSES}")
        class_to_int = DIATOMS_CLASSES().class_to_int
        int_to_class = DIATOMS_CLASSES().int_to_class
        print("="*30)
        
        print("\n\n")
        
        print("="*30)
        print("\nDividindo Dataset em Treino e Validação ...")
        train_files, train_labels, val_files, val_labels, all_groups, train_idx, val_idx = split_train_val(DATASET_DIR, CLASSES, class_to_int)
        print("="*30)
        
        print("="*30)
        print("\nCalculando Pesos de Classe...")
        class_weights = compute_class_weights(train_labels, int_to_class)
        print("="*30)
        
        print("\n\n")
        
        print("="*30)
        print("\nVerificando Vazamento de Dados (Data Leakage)...")
        data_leakage = check_data_leakage(all_groups, train_idx, val_idx)
        
        if not data_leakage:
            print("="*30)
            print("\n[ERRO] Corrija o vazamento de dados antes de prosseguir.")
            print("="*30)
            return
        
        print("\n\n")
        
        print("="*30)
        print("\nCriando Pipelines de Dados...")
        train_dataset, val_dataset = create_data_pipelines(train_files, train_labels, val_files, val_labels, class_to_int, is_training)
        print("="*30)
        
        print("\n\n")
        
        return CLASSES, class_weights, train_dataset, val_dataset, train_files, val_files, train_labels, val_labels
    
    except Exception as e:
        print(f"Erro durante a preparação do dataset: {e}")
        
        
        
if __name__ == "__main__":
    
    DATASET_DIR = r'D:\facul\Disciplinas\VisãoComp\ProjetoFinal\dataset_final\Dataset_Final_Tratado\2augmentations\dataset'
    is_training = True
    
dataset_preparation(DATASET_DIR, is_training)