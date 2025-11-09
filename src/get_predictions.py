import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from data_pipeline import create_dataset
import matplotlib.pyplot as plt

def get_predictions(model, val_dataset, val_files, val_labels, CLASSES, BATCH_SIZE):
    
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

        print("Predições concluídas e armazenadas em variáveis.")

        # --- 21. Relatório de Classificação ---
        print("\n--- Relatório de Classificação (Precision, Recall, F1-Score) ---")
        print(classification_report(y_true_labels, y_pred_labels, target_names=CLASSES))
        
        # salva o relatório em uma imgemm
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, classification_report(y_true_labels, y_pred_labels, target_names=CLASSES), fontsize=12, ha='center', va='center')
        plt.axis('off')
        plt.title('Relatório de Classificação')
        
        return results, y_true_labels, y_pred_labels, y_pred_proba, y_true_one_hot
    
    except Exception as e:
        print(f"Erro ao obter predições: {e}")
        return None, None, None, None, None