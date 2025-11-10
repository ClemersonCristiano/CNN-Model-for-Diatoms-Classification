import matplotlib.pyplot as plt

# --- Função de Plotagem das Curvas de Treinamento ---

def plot_training_curves(MODEL_NAME, history, title_suffix):
    """
    Plota as curvas de Acurácia e Perda do treinamento.

    Argumentos:
        history (tf.keras.callbacks.History): O objeto retornado por model.fit().
        title_suffix (str): O sufixo para o título (ex: "Extração de Features" ou "FineTuning").
    """
    
    try:
        print(f"\nGerando curvas de treinamento para: {title_suffix}...")

        # Pega as métricas do histórico
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        # Gráfico de Acurácia
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo-', label='Acurácia (Treino)')
        plt.plot(epochs, val_acc, 'ro-', label='Acurácia (Validação)')
        plt.title(f'Curva de Acurácia - {title_suffix}')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.grid(True)

        # Gráfico de Perda (Loss)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo-', label='Perda (Treino)')
        plt.plot(epochs, val_loss, 'ro-', label='Perda (Validação)')
        plt.title(f'Curva de Perda - {title_suffix}')
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f'Curvas de Treinamento - {MODEL_NAME}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'training_curves_{MODEL_NAME}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Erro ao plotar curvas de treinamento: {e}")
    