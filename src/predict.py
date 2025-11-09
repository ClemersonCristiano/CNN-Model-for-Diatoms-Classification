import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import resnet_v2
from config import IMAGE_SIZE
from config import DIATOMS_CLASSES

CLASSES = DIATOMS_CLASSES().Diatoms_Classes_names

def load_trained_model(MODEL_PATH):

    try:
        print(f"\nCarregando modelo de: {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        
        if model is None:
            print("\nERRO: O modelo não pôde ser carregado.")
            return None, None
        
        print("\nModelo carregado com sucesso.")
        return model
    
    except FileNotFoundError as e:
        print(f"\nERRO AO CARREGAR O MODELO: {e}")
        sys.exit()
    

def preprocess_inference_image(filepath, IMAGE_SIZE):
    
    try:
        image = tf.io.read_file(filepath)
        
        if image is None:
            print(f"ERRO: Não foi possível ler o arquivo: {filepath}")
            return None
            
        image = tf.io.decode_image(image, channels=1)        
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
        image = resnet_v2.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        
        return image

    except FileNotFoundError as e:
        print(f"ERRO: Não foi possível ler o arquivo: {filepath}")
        print(f"Detalhe: {e}")
        return None

def predict_diatom_class(model, CLASSES, image_path):
    
    try:
        processed_image = preprocess_inference_image(image_path, IMAGE_SIZE)
        
        if processed_image is None:
            return "Erro", 0.0, []

        # 'prediction' será um array de arrays, ex: [[0.01, 0.02, 0.95, 0.01, 0.01]]
        prediction = model.predict(processed_image, verbose=0)
        
        # Pega a lista de probabilidades
        prediction_list = prediction[0]
        
        # Pega o índice da classe com maior probabilidade
        predicted_index = np.argmax(prediction_list)
        
        # Pega o valor (confiança) dessa probabilidade
        confidence = np.max(prediction_list)
        
        # Encontra o nome da classe
        class_name = CLASSES[predicted_index]
        
        return class_name, confidence, prediction_list
    
    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return "Erro", 0.0, []
    
def prediction_results(model, CLASSES, IMAGE_PATH):
    
    try:
        classe_prevista, confianca, todas_predicoes = predict_diatom_class(model, CLASSES, IMAGE_PATH)
            
        if classe_prevista == "Erro":
            print("\nNão foi possível fazer a predição.")
            
        else:
            print("\n--- Resultado da Predição ---")
            print(f"  Classe Prevista: {classe_prevista}")
            print(f"  Confiança:       {confianca * 100:.2f}%")
            
            print("\n--- Detalhes da Confiança (Todas as Classes) ---")
            
            # Combina os nomes das classes com suas probabilidades
            # e ordena da mais provável para a menos provável
            results = list(zip(CLASSES, todas_predicoes))
            results.sort(key=lambda x: x[1], reverse=True)
            
            for classe, prob in results:
                print(f"  {classe:12}: {prob * 100:>6.2f}%")
    
    except Exception as e:
        print(f"Erro ao exibir os resultados da predição: {e}")
    
def predict():
    
    # Caminho para o arquivo do seu modelo
    MODEL_PATH = r'D:\facul\Disciplinas\VisãoComp\ProjetoFinal\src\CNN\models\diatom_classifier_best_model_finetuned.keras'
    
    model = load_trained_model(MODEL_PATH)
            
    # Caminho para a imagem
    IMAGE_PATH = r'D:\facul\Disciplinas\VisãoComp\ProjetoFinal\dataset_final\Dataset_Final_Tratado\2augmentations\dataset\Pinnularia\Pinnularia_PAML_006.png_20251016_143924_215381.png'
    print(f"\nFazendo predição para a imagem: {IMAGE_PATH}...")
    
    try:
        prediction_results(model, CLASSES, IMAGE_PATH)
        
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        
if __name__ == "__main__":
    
    predict()