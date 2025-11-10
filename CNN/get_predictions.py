import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from data_pipeline import create_dataset
import matplotlib.pyplot as plt

def plot_classification_report(y_true_labels, y_pred_labels, CLASSES, MODEL_NAME):

    # Gera o relatório
    report_dict = classification_report(y_true_labels, y_pred_labels, target_names=CLASSES, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose().round(3)

    # Divide o relatório
    df_classes = df_report.iloc[:len(CLASSES)]
    df_summary = df_report.iloc[len(CLASSES):]

    # Cria figura
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax in axes:
        ax.axis('off')

    # ---- Tabela 1 ----
    axes[0].set_title("Métricas por Classe", fontsize=12, pad=10, weight='bold')

    tabela1 = axes[0].table(
        cellText=df_classes.values,
        colLabels=df_classes.columns,
        rowLabels=df_classes.index,
        cellLoc='center',
        loc='center'
    )
    tabela1.auto_set_font_size(False)
    tabela1.set_fontsize(10)
    tabela1.scale(1.2, 1.2)

    for (i, j), cell in tabela1.get_celld().items():
        cell.set_edgecolor('black')
        if i == 0:
            cell.set_facecolor('#d9d9d9')
            cell.set_text_props(weight='bold')

    # ---- Tabela 2 ----
    axes[1].set_title("Resumo Geral", fontsize=12, pad=10, weight='bold')

    tabela2 = axes[1].table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        rowLabels=df_summary.index,
        cellLoc='center',
        loc='center'
    )
    tabela2.auto_set_font_size(False)
    tabela2.set_fontsize(10)
    tabela2.scale(1.2, 1.2)

    for (i, j), cell in tabela2.get_celld().items():
        cell.set_edgecolor('black')
        if i == 0:
            cell.set_facecolor('#d9d9d9')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#f5f5f5')

    # ---- Título geral e layout ----
    fig.suptitle('Relatório de Classificação', fontsize=14, y=0.98, weight='bold')
    plt.subplots_adjust(hspace=0.5)

    plt.savefig(f'classification_report_{MODEL_NAME}.png', dpi=300, bbox_inches='tight')
    plt.show()


def get_predictions(model, val_dataset, val_files, val_labels, CLASSES, BATCH_SIZE, MODEL_NAME):
    
    try:
        
        # --- 19. Avaliação Final Simples ---
        print("\nAvaliando o modelo final no dataset de validação...")
        # (val_dataset, val_files, BATCH_SIZE vêm das células anteriores)
        validation_steps = math.ceil(len(val_files) / BATCH_SIZE)
        results = model.evaluate(val_dataset, steps=validation_steps, verbose=1)

        print("\nResultados da Avaliação Final:")
        print(f"  Perda (Loss):    {results[0]:.4f}")
        print(f"  Acurácia (Acc): {results[1]*100:.2f}%")

        # --- 20. Obter Predições vs. Labels Reais (para as próximas células) ---
        print("\nGerando predições em todo o dataset de validação...")

        # Recria os labels one-hot (caso o notebook tenha sido reiniciado)
        val_labels_one_hot = tf.keras.utils.to_categorical(val_labels, num_classes=len(CLASSES))
        # Recria o dataset de validação (sem shuffle)
        val_dataset_eval = create_dataset(val_files, val_labels_one_hot, is_training=False)

        # Listas para armazenar os resultados
        y_pred_proba_list = []
        y_true_labels_list = []

        # Itera pelo dataset para pegar predições e labels
        for images, labels in val_dataset_eval:
            y_pred_proba_list.append(model.predict(images, verbose=0))
            y_true_labels_list.extend(np.argmax(labels.numpy(), axis=1))

        # Converte as listas em arrays NumPy
        y_pred_proba = np.concatenate(y_pred_proba_list, axis=0) # Probabilidades (ex: [0.1, 0.9, ...])
        y_true_labels = np.array(y_true_labels_list)           # Labels de índice (ex: [1, 1, ...])
        y_true_one_hot = val_labels_one_hot                      # Labels One-Hot (para ROC)
        y_pred_labels = np.argmax(y_pred_proba, axis=1)        # Labels de índice (ex: [1, 2, ...])

        # --- 21. Relatório de Classificação ---
        print("\n--- Relatório de Classificação (Precision, Recall, F1-Score) ---")
        print(classification_report(y_true_labels, y_pred_labels, target_names=CLASSES, digits=4))
        
        plot_classification_report(y_true_labels, y_pred_labels, CLASSES, MODEL_NAME=MODEL_NAME)
        
        return results, y_true_labels, y_pred_labels, y_pred_proba, y_true_one_hot
    
    except Exception as e:
        print(f"Erro ao obter predições: {e}")
        return None, None, None, None, None
    
    
if __name__ == "__main__":
    
    MODEL_NAME = 'teste AAAAAA'
    MODEL_PATH = r'D:\facul\Github\CNN-model-for-diatom-classification\CNN\models\teste_model_feature_extraction.keras'
    DATASER_DIR = r'D:\facul\Github\CNN-model-for-diatom-classification\dataset_final\teste_dataset'
    
    from data_pipeline import dataset_preparation
    from tensorflow.keras.models import load_model
    from config import BATCH_SIZE
        
    CLASSES, class_weights, train_dataset, val_dataset, train_files, val_files, train_labels, val_labels = dataset_preparation(DATASER_DIR)
    
    model = load_model(MODEL_PATH)
    
    results, y_true_labels, y_pred_labels, y_pred_proba, y_true_one_hot = get_predictions(model, val_dataset, val_files, val_labels, CLASSES, BATCH_SIZE, MODEL_NAME)