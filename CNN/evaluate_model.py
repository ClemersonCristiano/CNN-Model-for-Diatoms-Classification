import tensorflow as tf
from tensorflow.keras.models import load_model
from config import BATCH_SIZE
from data_pipeline import dataset_preparation
from ROC_AUC_curves import roc_auc_curves
from get_predictions import get_predictions
from plot_confusion_matrix import plot_confusion_matrix
from plot_tsne_visualization import plot_tsne_visualization
from plot_gradcam import plot_gradcam
        
def evaluate_model(DATASER_DIR, MODEL_PATH, MODEL_NAME, is_training):

    try:

        CLASSES, _, _, val_dataset, _, val_files, _, val_labels = dataset_preparation(DATASER_DIR, is_training)
        
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
            
            print("Plotando GradCam...")
            IMAGE_PATH = r'D:\facul\Github\CNN-model-for-diatom-classification\dataset_final\validação\Pinnularia_modificada\tratadas\Pinnularia\Pinnularia_image1015.tif_20251105_161224_396856.png'
            plot_gradcam(MODEL_PATH, IMAGE_PATH, CLASSES)
            
        else:
            print("Erro ao carregar o modelo.")
            
    except Exception as e:
        print(f"Erro durante a avaliação do modelo: {e}")
        
if __name__ == "__main__":
    
    DATASER_DIR = r"D:\facul\Github\CNN-model-for-diatom-classification\dataset_final\Dataset_Final_Puro_Tratad"
    MODEL_PATH = r"D:\facul\Github\CNN-model-for-diatom-classification\CNN\models\modelo_7k\fineTuned_model_7k\Diatom_Classifier_FineTuned_Model_7k.keras"
    MODEL_NAME = "Model_7k_Final"
    is_training = False
    
    evaluate_model(DATASER_DIR, MODEL_PATH, MODEL_NAME, is_training)