import tensorflow as tf
from tensorflow.keras.models import load_model
from config import BATCH_SIZE
from data_pipeline import dataset_preparation
from ROC_AUC_curves import roc_auc_curves
from get_predictions import get_predictions
from plot_confusion_matrix import plot_confusion_matrix
from plot_tsne_visualization import plot_tsne_visualization

def GPU_check():
    
    try:
        print("\nVerificando disponibilidade de GPU...")
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"GPUs disponíveis: {[gpu.name for gpu in gpus]}")
            return True
        else:
            print("Nenhuma GPU disponível.")
            return False
        
    except Exception as e:
        print(f"Erro ao verificar GPUs: {e}")
        return False
        
def evaluate_model(DATASER_DIR, MODEL_PATH, MODEL_NAME):

    try:

        CLASSES, _, _, val_dataset, _, val_files, _, val_labels = dataset_preparation(DATASER_DIR)
        
        # --- 18. Carregar o Modelo ---
        
        print("\n--- INICIANDO AVALIAÇÃO DO MODELO ---")

        print(f"Carregando o modelo final de: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        
        if model is not None:
            print("\nModelo carregado com sucesso.")

            print("\nObtendo predições e labels...")
            results, y_true_labels, y_pred_labels, y_pred_proba, y_true_one_hot = get_predictions(model, val_dataset, val_files, val_labels, CLASSES, BATCH_SIZE, MODEL_NAME)
            
            print("\nPlotando Matriz de Confusão...")
            plot_confusion_matrix(results, y_true_labels, y_pred_labels, MODEL_NAME, CLASSES)
            
            print("\nPlotando Curvas ROC e AUC...")
            roc_auc_curves(y_true_one_hot, y_pred_proba, CLASSES, MODEL_NAME)
            
            print("\nPlotando Visualização t-SNE...")
            plot_tsne_visualization(model, val_dataset, y_true_labels, CLASSES, MODEL_NAME)
            
        else:
            print("Erro ao carregar o modelo.")
            
    except Exception as e:
        print(f"Erro durante a avaliação do modelo: {e}")
        
if __name__ == "__main__":
    
    
    DATASER_DIR = r'D:\facul\Disciplinas\VisãoComp\ProjetoFinal\dataset_final\Dataset_Final_Tratado\2augmentations\dataset'
    MODEL_PATH = r'D:\facul\Disciplinas\VisãoComp\ProjetoFinal\src/CNN/models/diatom_classifier_best_model_finetuned.keras'
    MODEL_NAME = 'teste AAAAAA'
    
    gpu = GPU_check()
    
    if gpu:
        
        input(f"\n GPU disponível para uso. Pressione Enter para iniciar a avaliação do modelo:")
        evaluate_model(DATASER_DIR, MODEL_PATH, MODEL_NAME)
        
    else:
        escolha = input("\nNenhuma GPU disponível. Pressione a tecla [C] para continuar a avaliação do modelo na CPU ou qualquer outra tecla para sair: ").lower()
        
        if escolha == 'c':
            evaluate_model(DATASER_DIR, MODEL_PATH, MODEL_NAME)
            
        else:
            print("\nAvaliação do modelo cancelada pelo usuário.")