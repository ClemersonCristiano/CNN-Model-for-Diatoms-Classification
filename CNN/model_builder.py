import math
import sys
from time import sleep
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from plot_training_curves import plot_training_curves
from config import IMAGE_SIZE, NUM_CHANNELS, BATCH_SIZE
from data_pipeline import dataset_preparation
from clean_terminal import clean_terminal

def model_builder_callbacks(path_to_save__model):
    
    # --- Configuração de Callbacks ---

    print("\nConfigurando callbacks (ModelCheckpoint e EarlyStopping)...")

    # 1. ModelCheckpoint: Salva o modelo com a melhor acurácia de validação
    model_checkpoint = ModelCheckpoint(
        filepath=path_to_save__model,
        save_best_only=True,       # Salva apenas se for melhor que o anterior
        monitor='val_accuracy',    # Monitora a acurácia da validação
        mode='max',                # Queremos maximizar a acurácia
        verbose=1                  # Imprime uma mensagem quando salva
    )

    # 2. EarlyStopping: Para o treinamento se não houver melhoria
    early_stopping = EarlyStopping(
        monitor='val_loss',        # Monitora a perda da validação
        patience=5,                # Número de épocas sem melhoria antes de parar
        mode='min',                # Queremos minimizar a perda
        verbose=1,
        restore_best_weights=True  # Restaura os pesos da melhor época ao final
    )
    
    return model_checkpoint, early_stopping

def feature_extraction_model_build(CLASSES, IMAGE_SIZE, NUM_CHANNELS, path_to_save_feature_extraction_model):

    try:
        # --- 9. Definição da Arquitetura do Modelo ---

        print("\nConstruindo o modelo de Transfer Learning (ResNet50V2)...")

        INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS) # (400, 400, 3)
        NUM_CLASSES = len(CLASSES)

        # --- Parte 1: Carregar a Base Pré-Treinada ---

        # Carregamos a ResNet50V2, treinada no ImageNet
        # include_top=False: remove a camada final original (que classificava 1000 classes)
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=INPUT_SHAPE
        )

        # --- Parte 2: Congelar a Base ---

        # "Congelar" os pesos da base.
        # Não queremos re-treiná-los na primeira fase (Feature Extraction).
        base_model.trainable = False

        # --- Parte 3: Construir o "Head" (Nosso Classificador) ---

        # 1. Definir a entrada do nosso modelo
        inputs = Input(shape=INPUT_SHAPE)

        # 2. Passar a entrada pela base (em modo "inferência")
        # training=False garante que as camadas de BatchNormalization da base
        # usem suas estatísticas salvas e não tentem se atualizar.
        x = base_model(inputs, training=False)

        # 3. Adicionar nossas camadas no topo
        # GlobalAveragePooling2D achata a saída 4D da ResNet em um vetor 1D
        x = GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        # Adicionamos uma camada densa para aprender os padrões específicos das diatomáceas
        x = Dense(256, activation='relu', name='dense_head')(x)
        # Dropout é uma técnica de regularização crucial para evitar overfitting
        x = Dropout(0.5)(x)
        # Camada de saída final. 'softmax' para classificação multiclasse
        outputs = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

        # --- Parte 4: Criar o Modelo Final ---
        model = Model(inputs, outputs)

        # --- Parte 5: Compilar o Modelo ---
        model.compile(
            # Adam é um otimizador robusto. 0.001 é um bom learning rate inicial
            optimizer=Adam(learning_rate=0.001),
            # 'categorical_crossentropy' é a loss function correta
            # porque usamos to_categorical (one-hot) em nossos labels.
            loss='categorical_crossentropy',
            metrics=['accuracy'] # Vamos monitorar a acurácia
        )

        print("Modelo construído e compilado com sucesso.")

        # --- 10. Exibir Resumo do Modelo ---
        print("\nResumo do modelo:")
        model.summary()

        model_checkpoint, early_stopping = model_builder_callbacks(path_to_save_feature_extraction_model)
        
        return model, model_checkpoint, early_stopping, path_to_save_feature_extraction_model
    
    except Exception as e:
        print(f"Erro ao construir o modelo: {e}")
        return None, None, None, None

def train_feature_extraction_model(model, model_checkpoint, early_stopping, path_to_save_feature_extraction_model, MODEL_NAME, train_dataset, val_dataset, class_weights, train_files, val_files, NUM_EPOCHS, BATCH_SIZE):
    # --- 12. Início do Treinamento (Extração de Features) ---

    try:
        print("\n--- INICIANDO TREINAMENTO (Extração de Features: FEATURE EXTRACTION) ---")

        # Calcular os "steps" (passos) por época.
        # Isso é necessário ao usar tf.data
        steps_per_epoch = math.ceil(len(train_files) / BATCH_SIZE)
        validation_steps = math.ceil(len(val_files) / BATCH_SIZE)

        history = model.fit(
            train_dataset,
            epochs=NUM_EPOCHS,
            validation_data=val_dataset,
            class_weight=class_weights,  # <-- Aplicando os pesos contra o desbalanceamento!
            callbacks=[model_checkpoint, early_stopping],
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )

        print("\n--- TREINAMENTO (Extração de Features) CONCLUÍDO ---")
        print(f"O melhor modelo foi salvo em: {path_to_save_feature_extraction_model}")

        try:
            plot_training_curves(MODEL_NAME, history, f"Extração de Features - {MODEL_NAME}")
        except NameError:
            plot_training_curves(MODEL_NAME, history, "Extração de Features: Extração de Features")
                        
    except Exception as e:
        print(f"Erro ao treinar o modelo: {e}")
        
def fine_tune_model_build(load_model_to_finetuning, path_to_save_finetuning_model):
    # --- 13. Início da FineTuning: Ajuste Fino (Fine-Tuning) ---

    try:
        print("\n--- INICIANDO FineTuning: AJUSTE FINO (FINE-TUNING) ---")
        print("Carregando o melhor modelo da Extração de Features...")

        # Carrega o modelo que atingiu melhor acurácia
        model = load_model(load_model_to_finetuning)
        
        if model is None:
            print("\nERRO: O modelo não foi carregado.")
            sys.exit()

        # --- 14. "Descongelar" a Base ---

        # Precisamos acessar a "base_model" (a ResNet) dentro do nosso modelo salvo
        # O nome 'resnet50v2' é o nome padrão que Keras deu a ela (vimos no model.summary())
        try:
            base_model = model.get_layer('resnet50v2')
            base_model.trainable = True # <-- A MÁGICA ACONTECE AQUI
            print(f"Camada '{base_model.name}' foi descongelada e está pronta para o ajuste fino.")
            
        except ValueError:
            print("ERRO: Não foi possível encontrar a camada 'resnet50v2'. Verifique o model.summary() da Extração de Features.")
            # Se o nome for diferente, ajuste-o.

        # --- 15. Recompilar com Taxa de Aprendizado Baixíssima ---

        # Este é o passo MAIS CRÍTICO do ajuste fino.
        # Usamos uma taxa de aprendizado 100x a 1000x menor que antes.
        # Queremos "ajustar" os pesos, não "destruí-los".
        LEARNING_RATE_FINETUNING_2 = 0.00001 # (1e-5)

        model.compile(
            # Adam com um learning rate muito baixo
            optimizer=Adam(learning_rate=LEARNING_RATE_FINETUNING_2),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"Modelo recompilado para ajuste fino com learning rate de {LEARNING_RATE_FINETUNING_2}.")

        # Vamos verificar o resumo. Agora, TODOS os parâmetros devem ser "Trainable".
        print("\nResumo do modelo para FineTuning:")
        model.summary()

        model_checkpoint_finetuning, early_stopping_finetuning = model_builder_callbacks(path_to_save_finetuning_model)
        
        return model, model_checkpoint_finetuning, early_stopping_finetuning, path_to_save_finetuning_model
    
    except Exception as e:
        print(f"Erro ao preparar o modelo para ajuste fino: {e}")
        return None, None, None, None
    
def train_fine_tune_model(model, model_checkpoint_finetuning, early_stopping_finetuning, path_to_save_finetuning_model, MODEL_NAME, train_dataset, val_dataset, class_weights, train_files, val_files, NUM_EPOCHS, BATCH_SIZE):
    
    # --- 17. Início do Treinamento (FineTuning) Fine-Tuning ---

    try:
        print("\n--- CONTINUANDO TREINAMENTO (FineTuning: AJUSTE FINO) ---")

        # Reutilizar os steps calculados na Extração de Features
        try:
            steps_per_epoch = math.ceil(len(train_files) / BATCH_SIZE)
            validation_steps = math.ceil(len(val_files) / BATCH_SIZE)
        except NameError:
            print("ERRO: Os arquivos de treinamento e validação ainda estão na memória....")
            return

        history_finetuning = model.fit(
            train_dataset,
            epochs=NUM_EPOCHS,
            validation_data=val_dataset,
            class_weight=class_weights,  # <-- Ainda usamos os pesos de classe!
            callbacks=[model_checkpoint_finetuning, early_stopping_finetuning],
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
            # O modelo continua porque carregamos seus pesos.
        )

        print("\n--- TREINAMENTO (FineTuning) CONCLUÍDO ---")
        print(f"O melhor modelo de ajuste fino foi salvo em: {path_to_save_finetuning_model}")

        try:
            plot_training_curves(MODEL_NAME, history_finetuning, f"FineTuning - {MODEL_NAME}")
        except NameError:
            plot_training_curves(MODEL_NAME, history_finetuning, "FineTuning: Ajuste Fino")
            
    except Exception as e:
        print(f"Erro ao treinar o modelo na fase de ajuste fino: {e}")
        
def main():
    
    path_to_save_feature_extraction_model = r'D:\facul\Disciplinas\VisãoComp\ProjetoFinal\src\CNN\models\teste.keras'
    load_model_to_finetuning = r'D:\facul\Disciplinas\VisãoComp\ProjetoFinal\src\CNN\models\diatom_classifier_best_model_finetuned.keras'
    path_to_save_finetuning_model = r'D:\facul\Disciplinas\VisãoComp\ProjetoFinal\src\CNN\models\testeFine.keras'
    model_name = r'Teste'
    DATASER_DIR = r'D:\facul\Disciplinas\VisãoComp\ProjetoFinal\dataset_final\Dataset_Final_Tratado\2augmentations\dataset'
    NUM_EPOCHS = 20
    
    while True:
        
        print("\n--- INICIANDO CONSTRUÇÃO E TREINAMENTO DO MODELO CNN ---")
        
        print("\n+ --- Selecione a fase de treinamento --- +")
        print("[1] Extração de Features (Feature Extraction)")
        print("[2] Ajuste Fino (Fine-Tuning)")
        print("[3] Sair")
        
        op = input("\nOpção: ")
        
        if op == '1':
            try:
                CLASSES, class_weights, train_dataset, val_dataset, train_files, val_files, _, _ = dataset_preparation(DATASER_DIR)
                
                model, model_checkpoint, early_stopping, save_feature_extraction_model = feature_extraction_model_build(CLASSES, IMAGE_SIZE, NUM_CHANNELS, path_to_save_feature_extraction_model)
                
                train_feature_extraction_model(model, model_checkpoint, early_stopping, save_feature_extraction_model, model_name, train_dataset, val_dataset, class_weights, train_files, val_files, NUM_EPOCHS, BATCH_SIZE)
            
            except Exception as e:
                print(f"\nErro durante a fase de Extração de Features: {e}")
            
        elif op == '2':
            
            try:
            
                CLASSES, class_weights, train_dataset, val_dataset, train_files, val_files, _, _ = dataset_preparation(DATASER_DIR)
                
                model, model_checkpoint_finetuning, early_stopping_finetuning, path_to_save_finetuning_model = fine_tune_model_build(load_model_to_finetuning)
                
                train_fine_tune_model(model, model_checkpoint_finetuning, early_stopping_finetuning, path_to_save_finetuning_model, model_name, train_dataset, val_dataset, class_weights, train_files, val_files, NUM_EPOCHS, BATCH_SIZE)
            
            except Exception as e:
                print(f"\nErro durante a fase de Fine-Tuning: {e}")
                
        elif op == '3':
            print("\nSaindo...")
            sleep(1)
            clean_terminal()
            break
            
        else:
            print("\nOpção inválida. Selecione uma opção válida.")
            sleep(1)
            clean_terminal()
            
    print("\nPrograma finalizado.")
            
if __name__ == "__main__":
    main()