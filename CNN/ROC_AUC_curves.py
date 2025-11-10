import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def roc_auc_curves(y_true_one_hot, y_pred_proba, CLASSES, MODEL_NAME):
    
    try:
        # --- Curvas ROC e AUC ---
        print("\n" + "="*50)
        print("--- PASSO 9: Gerando Curvas ROC e AUC (One-vs-Rest) ---")
        print("="*50)

        # Dicionários para armazenar as taxas
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(CLASSES)

        plt.figure(figsize=(10, 8))

        # Binariza os labels verdadeiros (necessário para One-vs-Rest)
        # y_true_one_hot já foi carregado na célula anterior

        # Calcula a curva ROC e AUC para cada classe
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2,
                    label=f'Classe {CLASSES[i]} (AUC = {roc_auc[i]:.4f})')

        # Plota a linha de "chute aleatório"
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.50)')

        # Formatação do gráfico
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
        plt.ylabel('Taxa de Verdadeiros Positivos (Recall)')
        plt.title(f'Curvas ROC (One-vs-Rest) - {MODEL_NAME}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'roc_auc_curves_{MODEL_NAME}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"ERRO ao gerar curvas ROC e AUC: {e}")