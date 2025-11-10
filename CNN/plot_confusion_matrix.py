import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(results, y_true_labels, y_pred_labels, MODEL_NAME, CLASSES):
    
    try:
        # --- 22. Matriz de Confusão ---
        print("\nGerando Matriz de Confusão...")
        cm = confusion_matrix(y_true_labels, y_pred_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title(f'Matriz de Confusão - Acurácia: {results[1]*100:.2f}% - {MODEL_NAME}')
        plt.ylabel('Classe Verdadeira (True Label)')
        plt.xlabel('Classe Prevista (Predicted Label)')
        plt.savefig(f'confusion_matrix_{MODEL_NAME}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Erro ao gerar a Matriz de Confusão: {e}")