
#Extração e Conversão para PNG
import zipfile
import os
from cv2 import imread, imwrite
import numpy as np
from time import sleep
import requests
from tqdm import tqdm

#Relacionado a Modelos
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

#Plot do Grafico e Imagens
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import sample

TAMANHO_IMG = (128, 128)
TAMANHO_BATCH = 32

## Funcoes Importantes

# Importar Modelo
def importar_modelo() -> tuple[tf.data.Dataset, tf.data.Dataset, Sequential]:
    """
    Função que importa modelo caso exista na mesma dirétoria e da display do gráfico do seu treino.<br><br>
    Caso não esteja criado, ele começa a treinar o modelo após 3s:
    * Da unzip;
    * Prepara a pasta;
    * Cria os datasets;
    * Cria modelo e treina-o
    * Guarda o modelo e o gráfico do seu treino. 
    """

    #Confirmacao da Existência do Modelo
    if os.path.exists("./modelo.keras"):
        print("Modelo encontrado!")
        #Gráfico do treino do modelo
        img = mpimg.imread("grafico_modelo.png")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        modelo = keras.models.load_model(
            "modelo.keras",
            custom_objects={"preprocess_input": preprocess_input}
        )

        train_ds, val_ds = criar_datasets()
    else:

        print("Modelo não encontrado, a treinar um em 3s")
        sleep(3)

        #Confirmar o download do dataset
        if not confirm_download_dataset():
            return
        
        #Unzip e Converter para PNG
        preparar_pasta()

        #Criar Datasets
        print("A criar datasets")
        train_ds, val_ds = criar_datasets()
        print("Datasets Criados")

        #Criar Modelo
        print("A criar modelo")
        modelo = model()
        print("Modelo Criado")

        #Treinar Modelo
        print("A treinar modelo")
        modelo = treinar_modelo(modelo, train_ds, val_ds)
        print("Modelo treinado")
    
    return (train_ds, val_ds, modelo)


# Mostrar Classes
def mostrar_classes(train_ds):
    """
    Esta função serve para ver imagens referentes às classes existentes, tirando imagens aleatórias do dataset de treino.
    """
    num_classes = 2

    examples_per_class = {}

    for images, labels in train_ds.unbatch():
        label = int(labels.numpy())
        if label not in examples_per_class:
            examples_per_class[label] = images
        if len(examples_per_class) == num_classes:
            break

    plt.figure(figsize=(15, 10))
    for i in range(num_classes):
        img = examples_per_class[i]
        plt.subplot(6, 8, i+1)
        plt.imshow(img.numpy().astype("uint8"))
        plt.title(f"Label: {i}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Testar o Modelo com Imagens
def testar_modelo(model, n_imagens=2):
    """
    Escolher x imagens random da pasta de teste para o modelo categorizar.
    """
    arquivos = [f for f in os.listdir("./teste") if f.lower().endswith(".png")]
    imagens_escolhidas = sample(arquivos, n_imagens)

    plt.figure(figsize=(20, 5))

    for i, nome in enumerate(imagens_escolhidas):
        caminho = os.path.join("./teste", nome)

        # Carregar imagem
        img = tf.keras.utils.load_img(caminho, target_size=TAMANHO_IMG)
        img_array = tf.keras.utils.img_to_array(img)

        # Obter predição
        classe, confianca = predict_image(model, img_array)

        # Mostrar imagem
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_array.astype("uint8"))
        plt.title(f"{classe}\nConf: {confianca:.2f}")
        plt.axis("off")

    plt.show()





## Funcoes Assistentes
#Confirmar download Dataset
def confirm_download_dataset() -> bool:
    if not os.path.exists("./cats_vs_dogs.zip"):
        r = input("Não tens o dataset, queres fazer download dele? (Y/N)\n>")
        if r != "Y":
            return False
        
        print("A fazer download...")
        url = "http://www.dropbox.com/scl/fi/kax0iwr8cw2u8j4dvemtu/cats_vs_dogs.zip?rlkey=sadu201u8ea2cjroh0ojy3l1s&st=1t3vqyti&dl=1"

        # Faz o request em modo streaming
        response = requests.get(url, stream=True)
        response.raise_for_status()  # levanta erro se houver problema

        # Obtém o tamanho total do ficheiro (em bytes)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB por bloco

        # Barra de progresso
        with open("cats_vs_dogs.zip", "wb") as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc="Downloading"
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))

        print("Download concluído e guardado como cats_vs_dogs.zip")
    
    return True
    


#Preparacao das Pastas
def preparar_pasta():
    ## Extraír o Zip
    if not os.path.exists("./cats_vs_dogs/"):
        print("A extraír Zip")
        with zipfile.ZipFile("./cats_vs_dogs.zip", "r") as zip_ref:
            zip_ref.extractall("./cats_vs_dogs/")

        print("Zip Extraído")

    ## Converter de jpg para png
    if not os.path.exists("./cats_vs_dogs_png/"):
        print("A converter imagens para PNG")
        for root, _, files in os.walk("./cats_vs_dogs/"):
            rel_path = os.path.relpath(root, "./cats_vs_dogs/")
            out_folder = os.path.join("./cats_vs_dogs_png/", rel_path)
            os.makedirs(out_folder, exist_ok=True)

            for file in files:
                if file.lower().endswith(".jpg"):
                    src_path = os.path.join(root, file)
                    dst_file = os.path.splitext(file)[0] + ".png"
                    dst_path = os.path.join(out_folder, dst_file)

                    # Conversão
                    img = imread(src_path)
                    if img is None:
                        continue

                    imwrite(dst_path, img)

        print("Pasta com PNG's criada")


#Criacao dos Datasets
def criar_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    #Criacao dos Datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "./cats_vs_dogs_png/cats_vs_dogs/",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=TAMANHO_IMG,
        batch_size=TAMANHO_BATCH,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "./cats_vs_dogs_png/cats_vs_dogs/",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=TAMANHO_IMG,
        batch_size=TAMANHO_BATCH,
        label_mode="binary"
    )

    train_ds = train_ds.shuffle(len(train_ds)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    
    for images, labels in train_ds.take(1):
        print("Shape das Imagens:", images.shape)
        print("Exemplo de Labels:", labels.numpy().T)

    print("Train:", len(train_ds)*TAMANHO_BATCH)
    print("Val:", len(val_ds)*TAMANHO_BATCH)
    return train_ds, val_ds


#Criacao do Modelo
def model(input_shape=(TAMANHO_IMG[0], TAMANHO_IMG[1], 3)) -> Sequential:
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    preprocess = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
        layers.Lambda(preprocess_input)
    ])

    model = Sequential([
        layers.Input(shape=input_shape),
        preprocess,
        base_model,
        layers.GlobalAveragePooling2D(),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Dense(1, activation="sigmoid")
    ])

    model.base_model = base_model

    model.summary()
    return model


#Grafico Final
def plot_combined_history(history_antes, history_depois):
    # --- Loss ---
    loss = history_antes.history['loss']
    val_loss = history_antes.history['val_loss']
    
    epocas_antes = len(history_antes.history['loss'])
    
    loss.extend(history_depois.history['loss'])
    val_loss.extend(history_depois.history['val_loss'])
    
    # --- Accuracy ---
    acc = history_antes.history['accuracy']
    val_acc = history_antes.history['val_accuracy']
    
    acc.extend(history_depois.history['accuracy'])
    val_acc.extend(history_depois.history['val_accuracy'])
    
    # --- Figura com 2 gráficos ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axs[0].plot(loss, label='Treino')
    axs[0].plot(val_loss, label='Validação')
    axs[0].axvline(epocas_antes, color='green', linestyle='--', label='Início do Fine-Tuning')
    axs[0].set_xlabel('Épocas')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss do Modelo')
    axs[0].legend()
    
    # Accuracy
    axs[1].plot(acc, label='Treino')
    axs[1].plot(val_acc, label='Validação')
    axs[1].axvline(epocas_antes, color='green', linestyle='--', label='Início do Fine-Tuning')
    
    # --- Destaque dos máximos ---
    # Máximo treino
    max_train_acc = max(acc)
    max_train_epoch = acc.index(max_train_acc)
    axs[1].scatter(max_train_epoch, max_train_acc, color='blue', zorder=5)
    axs[1].text(max_train_epoch, max_train_acc + 0.01, 
                f"{max_train_acc:.4f}", color='blue', fontsize=10, ha='center')
    
    # Máximo validação
    max_val_acc = max(val_acc)
    max_val_epoch = val_acc.index(max_val_acc)
    axs[1].scatter(max_val_epoch, max_val_acc, color='red', zorder=5)
    axs[1].text(max_val_epoch, max_val_acc + 0.01, 
                f"{max_val_acc:.4f}", color='red', fontsize=10, ha='center')
    
    axs[1].set_xlabel('Épocas')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Precisão do Modelo')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig("grafico_modelo.png")
    plt.show()
    
    # Imprimir máximos
    acc_train_antes = max(history_antes.history['accuracy'])
    acc_val_antes = max(history_antes.history['val_accuracy'])
    acc_train_depois = max(history_depois.history['accuracy'])
    acc_val_depois = max(history_depois.history['val_accuracy'])

    print(f"Precisão antes do Fine-Tuning: Treino = {acc_train_antes:.4f}, Validação = {acc_val_antes:.4f}")
    print(f"Precisão depois do Fine-Tuning: Treino = {acc_train_depois:.4f}, Validação = {acc_val_depois:.4f}")



#Treinar Modelo
def treinar_modelo(model, train_ds, val_ds) -> Sequential:
    #Configuração do Treino
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )
    
    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    #Treino Normal
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[early_stop],
    )
    
    #Ativar o Fine-Tune
    for layer in model.base_model.layers[-30:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    
    #Treino com Fine-Tune
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[early_stop],
    )
    

    #Gráfico e Salvar Modelo
    plot_combined_history(history, history2)
    model.save("modelo.keras")

    return model


#Prever a Classe da Imagem
def predict_image(model, img_array):
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)[0][0]

    class_names = ['Cat', 'Dog']

    if pred < 0.5:
        return class_names[0], float(1 - pred)
    else:
        return class_names[1], float(pred)

