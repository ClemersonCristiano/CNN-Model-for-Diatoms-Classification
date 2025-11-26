import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from config import DIATOMS_CLASSES, IMAGE_SIZE, LAST_CONV_LAYER

CLASSES = DIATOMS_CLASSES().Diatoms_Classes_names

def get_img_array(img_path, size=(IMAGE_SIZE, IMAGE_SIZE)):
    """
    Carrega e pré-processa a imagem mantendo a consistência com o pipeline de treino.
    """
    try:
        # 1. Carregar arquivo bruto
        img = tf.io.read_file(img_path)
        
        # 2. Decodificar (força 1 canal inicial para garantir compatibilidade)
        img = tf.io.decode_image(img, channels=1, expand_animations=False)
        
        # 3. Converter Grayscale -> RGB (o modelo espera 3 canais)
        img = tf.image.grayscale_to_rgb(img)
        
        # 4. Resize com PAD (CRUCIAL: adiciona barras pretas para manter proporção)
        img = tf.image.resize_with_pad(img, size[0], size[1])
        
        # 5. Adiciona a dimensão do batch: (1, 400, 400, 3)
        array = np.expand_dims(img.numpy(), axis=0)
        
        # 6. Pré-processamento da ResNetV2 (normaliza para -1 a 1)
        return tf.keras.applications.resnet_v2.preprocess_input(array)
        
    except Exception as e:
        print(f"[ERRO] Falha ao processar {img_path}: {e}")
        return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Gera o mapa de calor e retorna também o índice da classe predita.
    """
    # 1. Encontrar o índice da camada alvo
    layers = model.layers
    try:
        target_layer_idx = next(i for i, layer in enumerate(layers) if layer.name == last_conv_layer_name)
    except StopIteration:
        raise ValueError(f"Camada '{last_conv_layer_name}' não encontrada no modelo.")

    # 2. Converter input para Tensor
    x = tf.convert_to_tensor(img_array)

    # 3. Forward Pass: Da entrada até a camada de convolução
    conv_output = x
    for layer in layers[:target_layer_idx + 1]:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        conv_output = layer(conv_output)
    
    # 4. Gradient Pass: Da convolução até a saída
    with tf.GradientTape() as tape:
        tape.watch(conv_output)
        
        preds = conv_output
        for layer in layers[target_layer_idx + 1:]:
            preds = layer(preds)
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0]) # Encontra a classe vencedora
        
        class_channel = preds[:, pred_index]

    # 5. Calcular gradientes e pooling
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Gerar heatmap
    conv_output_val = conv_output[0]
    heatmap = conv_output_val @ pooled_grads[..., tf.newaxis]
    
    # 7. Limpeza (ReLU e Normalização)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy(), int(pred_index)

def save_composite_image(img_path, heatmap, save_path, predicted_name, alpha):
    """
    Gera e salva a imagem composta, exibindo o nome da classe predita no título.
    """
    # Carregar imagem original visualmente
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE)
    img = img.numpy().astype("uint8")
    
    # Processar Heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Superposição
    superimposed_img = cv2.addWeighted(img, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)

    h, w, c = img.shape
    border = 10
    panel = np.zeros((h, (w * 3) + (border * 2), c), dtype=np.uint8)
    
    panel[:, :w] = img
    panel[:, w+border : (w*2)+border] = heatmap_colored_rgb
    panel[:, (w*2)+(border*2) :] = superimposed_img
    
    plt.figure(figsize=(15, 5))
    plt.imshow(panel)
    plt.title(f"Predição: {predicted_name} | Arquivo: {os.path.basename(img_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Resultado salvo em: {save_path} (Classe Predita: {predicted_name})")

# --- EXEMPLO DE USO ---
if __name__ == "__main__":
    
    # 1. Defina os caminhos
    MODEL_PATH = r'D:\facul\Github\CNN-model-for-diatom-classification\CNN\models\modelo_7k\fineTuned_model_7k\Diatom_Classifier_FineTuned_Model_7k.keras'
    IMAGE_PATH = r'D:\facul\Github\CNN-model-for-diatom-classification\dataset_final\validação\Pinnularia_modificada\tratadas\Pinnularia\Pinnularia_image1015.tif_20251105_161224_396856.png'
    OUTPUT_PATH = 'gradcam_analise.png'

    if os.path.exists(MODEL_PATH) and os.path.exists(IMAGE_PATH):
        print("Carregando modelo...")
        model = tf.keras.models.load_model(MODEL_PATH)
        
        print(f"Processando {IMAGE_PATH}...")
        img_array = get_img_array(IMAGE_PATH)
        
        if img_array is not None:
            # Gera o heatmap e recupera o índice da classe
            heatmap, pred_idx = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
            
            # Obtém o nome da classe
            class_name = CLASSES.get(pred_idx, f"Classe {pred_idx}")
            
            # Salva e exibe com o nome correto
            save_composite_image(IMAGE_PATH, heatmap, OUTPUT_PATH, class_name, alpha=0.6)
            
    else:
        print("Por favor, configure MODEL_PATH e IMAGE_PATH no script.")