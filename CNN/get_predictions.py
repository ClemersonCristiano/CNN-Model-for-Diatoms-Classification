import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from data_pipeline import create_dataset
import matplotlib.pyplot as plt

def plot_classification_report(y_true_labels, y_pred_labels, CLASSES, MODEL_NAME="modelo"):
    
    class_report = classification_report(
        y_true_labels,
        y_pred_labels,
        target_names=CLASSES,
        digits=4,
        output_dict=True
    )

    # Classes individuais
    class_rows = []
    for c in CLASSES:
        row = [
            c,
            round(class_report[c]["precision"], 4),
            round(class_report[c]["recall"], 4),
            round(class_report[c]["f1-score"], 4),
            int(class_report[c]["support"])
        ]
        class_rows.append(row)

    # Métricas globais
    macro = class_report["macro avg"]
    weighted = class_report["weighted avg"]
    accuracy_val = round(class_report["accuracy"], 4)
    total_support = sum([class_report[c]["support"] for c in CLASSES])
    # Accuracy na coluna F1-Score
    global_rows = [
        ["accuracy", "-", "-", accuracy_val, total_support],
        [
            "macro avg",
            round(macro["precision"], 4),
            round(macro["recall"], 4),
            round(macro["f1-score"], 4),
            int(macro["support"])
        ],
        [
            "weighted avg",
            round(weighted["precision"], 4),
            round(weighted["recall"], 4),
            round(weighted["f1-score"], 4),
            int(weighted["support"])
        ]
    ]

    # Cabeçalhos
    column_labels = ["Classe", "Precision", "Recall", "F1-Score", "Support"]
    global_labels = ["", "Precision", "Recall", "F1-Score", "Support"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    # Tabela das classes
    table_classes = ax.table(
        cellText=class_rows,
        colLabels=column_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.26, 0.5, 0.48, 0.35]
    )
    # Tabela global separada, com célula a esquerda vazia
    table_global = ax.table(
        cellText=global_rows,
        colLabels=global_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.26, 0.25, 0.48, 0.18]
    )

    table_classes.auto_set_font_size(False)
    table_classes.set_fontsize(10)
    table_classes.auto_set_column_width(col=list(range(len(column_labels))))
    table_global.auto_set_font_size(False)
    table_global.set_fontsize(10)
    table_global.auto_set_column_width(col=list(range(len(global_labels))))
    fig.tight_layout()

    plt.title(f"Relatório de Classificação ({MODEL_NAME})", fontsize=15, pad=20)
    plt.savefig(f"classification_report_{MODEL_NAME}.png", bbox_inches='tight', dpi=200)
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