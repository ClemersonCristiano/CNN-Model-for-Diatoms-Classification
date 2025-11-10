import numpy as np
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
def plot_tsne_visualization(model, val_dataset, y_true_labels, CLASSES, MODEL_NAME):
    """
    Extrai features (embeddings) do modelo, aplica t-SNE e plota a visualização 2D.

    Argumentos:
        model (tf.keras.Model): O modelo treinado e carregado.
        val_dataset (tf.data.Dataset): O pipeline de dados de validação.
        y_true_labels (np.array): O array de labels verdadeiros (como índices, ex: [0, 1, 4...]).
        CLASSES (list): A lista de nomes das CLASSES (ex: ['Encyonema', ...]).
        model_name (str): O nome do modelo para o título do gráfico.
    """
    try:
        
        print("\n" + "="*50)
        print("--- PASSO 10: Gerando Visualização t-SNE ---")
        print("="*50)

        # 1. Criar o "extrator de features"
        # Usamos o nome 'global_average_pooling2d' que acabamos de adicionar
        try:
            feature_layer = model.get_layer('global_average_pooling2d')
        except ValueError:
            print("ERRO: Não foi possível encontrar a camada 'global_average_pooling2d'.")
            print("Certifique-se de que você nomeou a camada e re-treinou o modelo.")
            return

        # Cria um novo modelo que termina na camada de features
        feature_extractor = Model(inputs=model.input, outputs=feature_layer.output)

        # 2. Extrair as features (embeddings) de todo o dataset de validação
        print("Extraindo features (embeddings) do dataset de validação...")
        # A função .predict() irá iterar por todo o val_dataset
        features = feature_extractor.predict(val_dataset, verbose=1)
        # O resultado será (1468, 2048) - (N_imagens_val, N_features)
        print(f"Features extraídas. Shape: {features.shape}")

        # 3. Rodar o t-SNE
        # Isso pode levar alguns minutos.
        print("Calculando t-SNE (isso pode levar alguns minutos)...")
        tsne = TSNE(n_components=2,       # Reduzir para 2 dimensões (x, y)
                    perplexity=30.0,    # Valor padrão
                    max_iter=1000,        # Iterações
                    random_state=42,    # Reprodutibilidade
                    verbose=1)          # Mostrar progresso

        tsne_results = tsne.fit_transform(features)
        # O resultado será (1468, 2)
        print("Cálculo do t-SNE concluído.")

        # 4. Plotar os resultados
        plt.figure(figsize=(14, 10))
        for i, class_name in enumerate(CLASSES):
            # Encontra os índices (posições) de todas as imagens que pertencem a esta classe
            indices = np.where(y_true_labels == i)

            # Plota um scatter plot apenas para esses pontos
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                        label=class_name,
                        alpha=0.7,
                        s=15)

        plt.title(f'Visualização t-SNE das Features Extraídas - {MODEL_NAME}')
        plt.xlabel('Componente t-SNE 1')
        plt.ylabel('Componente t-SNE 2')
        plt.legend(markerscale=3) # Legendas com marcadores maiores
        plt.show()
        
    except Exception as e:
        print(f"ERRO ao gerar visualização t-SNE: {e}")