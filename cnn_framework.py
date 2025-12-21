
#Extração e Conversão para PNG
import zipfile
import os
from cv2 import imread, imwrite
import numpy as np
from time import sleep
import requests
from tqdm import tqdm

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Pre-processamento
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory

#Relacionado a CNN
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

#Plot do Grafico e Imagens
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import sample


## Funcoes Importantes

class Modelo_CNN:
    def __init__(self):
        #Config
        self.tamanho_img = (128, 128)
        self.tamanho_batch = 32

        #ETC
        self.modelo: Sequential = None
        self.train_ds: tf.data.Dataset = None
        self.val_ds: tf.data.Dataset = None
        self.test_ds: tf.data.Dataset = None

    def setup(self):
        """
        Função que importa modelo caso exista na mesma dirétoria.<br><br>
        Caso não esteja criado, ele começa a treinar o modelo após 3s:
        * Da unzip;
        * Prepara a pasta;
        * Cria os datasets;
        * Cria modelo e treina-o
        * Guarda o modelo e o gráfico do seu treino. 
        """
        # Caso Exista Modelo Ja Treinado
        if os.path.exists("./modelo.keras"):
            print("Modelo encontrado!")

            #Carregar Modelo
            self.modelo = keras.models.load_model(
                "modelo.keras",
                custom_objects={"preprocess_input": preprocess_input}
            )

            #Carregar Datasets
            self.train_ds, self.val_ds, self.test_ds = self._setup_datasets()



        # Caso Seja Preciso Treinar Modelo
        else:
            print("Modelo não encontrado, a treinar um em 3s")
            sleep(3)
    
            #Confirmar o download do dataset
            if not self._confirm_download_dataset():
                return

            self._extract_zip()

            self._convert_to_png()
    
            #Criar Datasets
            print("A criar datasets")
            self.train_ds, self.val_ds, self.test_ds = self._setup_datasets()
            print("Datasets Criados")

            print("A criar modelo")
            self.modelo = self._setup_model()
            
            #Treinar Modelo
            print("A treinar modelo")
            self.modelo = self._train_model()
            print("Modelo treinado")


    def display_model_info(self):
        """
        Função que da plot do gráfico de treino do modelo.
        """
        self.modelo.summary()

        plt.figure(figsize=(12, 8))
        img = mpimg.imread("grafico_modelo.png")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    
    def show_existing_classes(self):
        """
        Esta função serve para ver imagens referentes às classes existentes, tirando imagens aleatórias do dataset de treino.
        """    
        examples_per_class = {}
    
        for images, labels in self.train_ds.unbatch():
            label = int(labels.numpy())
            if label not in examples_per_class:
                examples_per_class[label] = images
    
        plt.figure(figsize=(15, 10))
        for i in range(len(examples_per_class)):
            img = examples_per_class[i]
            plt.subplot(6, 8, i+1)
            plt.imshow(img.numpy().astype("uint8"))
            plt.title(f"Label: {i}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


    def predict_n_images(self, qty_imgs: int = 5):
        """
        Escolher x imagens random da pasta de teste para o modelo categorizar.
        """
        # Convert test dataset to list and extract images
        test_images = []
        for batch in self.test_ds.unbatch():
            test_images.append(batch[0].numpy())
        
        # Randomly select images
        imagens_escolhidas = sample(test_images, qty_imgs)
        
        plt.figure(figsize=(20, 5))
        
        for i, img_array in enumerate(imagens_escolhidas):
            # Obter predição
            classe, confianca = self._predict_image(img_array)
            
            # Mostrar imagem
            plt.subplot(2, 5, i + 1)
            plt.imshow(img_array.astype("uint8"))
            plt.title(f"{classe}\nConf: {confianca:.2f}")
            plt.axis("off")
        
        plt.show()


    def model_accuracy(self):
        """
        Testa a eficácia do modelo e devolve a sua precisão perantes os dados de teste
        """
        _, test_accuracy = self.modelo.evaluate(self.test_ds)
        return test_accuracy




    ## Funcoes Assistentes
    def _confirm_download_dataset(_self) -> bool:
        """
        Função que confirma se o dataset ja existe, pedindo para fazer download caso não exista.
        """
        # Confirma se o dataset ja está presente
        if not os.path.exists("./cats_vs_dogs.zip"):
            
            #Confirmação do utilizador
            r = input("Não tens o dataset dos cães/gatos, queres fazer download dele? (Y/N)\n>")
            if r != "Y":
                return False

            # Download do ficheiro
            print("A fazer download...")
            url = "http://www.dropbox.com/scl/fi/kax0iwr8cw2u8j4dvemtu/cats_vs_dogs.zip?rlkey=sadu201u8ea2cjroh0ojy3l1s&st=1t3vqyti&dl=1"
    
            response = requests.get(url, stream=True)
            response.raise_for_status()
    
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
    
            # Barra de progresso
            with open("cats_vs_dogs.zip", "wb") as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc="Downloading"
            ) as bar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    bar.update(len(data))
    
            print("Download concluído e guardado como cats_vs_dogs.zip")
        
        return True


    def _extract_zip(_self):
        """
        Extraí a pasta que foi downloaded.
        """
        
        if not os.path.exists("./cats_vs_dogs/"):
            print("A extraír Zip")
            with zipfile.ZipFile("./cats_vs_dogs.zip", "r") as zip_ref:
                zip_ref.extractall("./cats_vs_dogs/")
    
            print("Zip Extraído")


    def _convert_to_png(_self):
        """
        Esta função converte as imagens de JPG para PNG, pois algumas estão corrompidas e PNG é melhor.
        """

        # Converter de jpg para png
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
    
    
    def _setup_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Esta função da cria aos dataset de Treino, Validação e Teste.
        """
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Train Dataset (80%)
        train_ds = image_dataset_from_directory(
            "./cats_vs_dogs_png/cats_vs_dogs/",
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=self.tamanho_img,
            batch_size=self.tamanho_batch,
            label_mode="binary",
            color_mode='rgb'
        )
    
        # Temp Dataset (20%)
        temp_ds = image_dataset_from_directory(
            "./cats_vs_dogs_png/cats_vs_dogs/",
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=self.tamanho_img,
            batch_size=self.tamanho_batch,
            label_mode="binary",
            color_mode='rgb'
        )
        
        # Separar o Temp Dataset em 2
        val_ds = temp_ds.take(len(temp_ds) // 2)
        test_ds = temp_ds.skip(len(temp_ds) // 2)
        
        # Apply optimization
        train_ds = train_ds.cache().shuffle(len(train_ds)).prefetch(AUTOTUNE)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)
        test_ds = test_ds.cache().prefetch(AUTOTUNE)
    
        # Debug prints
        for images, labels in train_ds.take(1):
            print("Shape das Imagens:", images.shape)
            print("Exemplo de Labels:", labels.numpy().T)
    
        print("Train:", len(train_ds)*self.tamanho_batch)
        print("Val:", len(val_ds)*self.tamanho_batch)
        print("Test:", len(test_ds)*self.tamanho_batch)
        
        return train_ds, val_ds, test_ds
    
    
    
    def _setup_model(self) -> Sequential:
        """
        Esta função cria um modelo de uma cnn com a efficientNetB0.
        """
        input_shape=(self.tamanho_img[0], self.tamanho_img[1], 3)
        
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling='avg'
        )
        base_model.trainable = False
    
        preprocess = Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.15, 0.15),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
            layers.Lambda(preprocess_input)
        ])
    
        model = Sequential([
            layers.Input(shape=input_shape),
            preprocess,
            base_model,

            layers.Dropout(0.3),
    
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.4),

            layers.Dense(128),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.3),
    
            layers.Dense(1, activation="sigmoid")
        ])
    
        model.base_model = base_model
        return model

    
    def _train_model(self) -> Sequential:
        # Enhanced callbacks
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=0.0001
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,     # Reduce learning rate by half
            patience=3,     # Wait 3 epochs
            min_lr=1e-7,    # Minimum learning rate
            verbose=1
        )
        
        # Initial training with frozen base
        self.modelo.compile(
            optimizer=Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        print("Stage 1: Training top layers...")
        history1 = self.modelo.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=30,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Gradual unfreezing for better fine-tuning
        print("\nStage 2: Gradual fine-tuning...")
        
        # Unfreeze last 50 layers gradually
        for layer in self.modelo.base_model.layers[-50:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False  # Keep BN frozen
            else:
                layer.trainable = True
        
        # Recompile with lower learning rate
        self.modelo.compile(
            optimizer=Adam(1e-4),  # Lower LR for fine-tuning
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        print("Fine-tuning last 50 layers...")
        history2 = self.modelo.fit(
            self.train_ds,
            validation_data=self.val_ds,
            initial_epoch=len(history1.epoch),
            epochs=len(history1.epoch) + 30,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Final fine-tuning with even more layers
        print("\nStage 3: Full fine-tuning...")
        self.modelo.base_model.trainable = True
        
        # Keep BatchNormalization layers frozen
        for layer in self.modelo.base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        
        # Even lower learning rate for final tuning
        self.modelo.compile(
            optimizer=Adam(1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        history3 = self.modelo.fit(
            self.train_ds,
            validation_data=self.val_ds,
            initial_epoch=len(history1.epoch) + len(history2.epoch),
            epochs=len(history1.epoch) + len(history2.epoch) + 20,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Save final model
        self.modelo.save("modelo.keras")
        self._plot_train_graph(history1, history2, history3)
        
        return self.modelo
    
    
    def _plot_train_graph(self, *histories):
        # Combine all histories
        all_loss = []
        all_val_loss = []
        all_acc = []
        all_val_acc = []
        epoch_offsets = [0]
        
        current_epoch = 0
        for i, history in enumerate(histories):
            all_loss.extend(history.history['loss'])
            all_val_loss.extend(history.history['val_loss'])
            all_acc.extend(history.history['accuracy'])
            all_val_acc.extend(history.history['val_accuracy'])
            
            if i < len(histories) - 1:
                current_epoch += len(history.history['loss'])
                epoch_offsets.append(current_epoch)
        
        # Plot do gráfico
        _, axs = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        epochs_range = range(1, len(all_loss) + 1)
        axs[0].plot(epochs_range, all_loss, label='Treino')
        axs[0].plot(epochs_range, all_val_loss, label='Validação')
        
        # Mark stage transitions
        for offset in epoch_offsets[1:]:
            axs[0].axvline(offset, color='green', linestyle='--')
        
        axs[0].set_xlabel('Épocas')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss do Modelo')
        axs[0].legend()
        
        # Precisão
        axs[1].plot(epochs_range, all_acc, label='Treino')
        axs[1].plot(epochs_range, all_val_acc, label='Validação')
        
        # Mark stage transitions
        for offset in epoch_offsets[1:]:
            axs[1].axvline(offset, color='green', linestyle='--')
        
        # Pontos Máximos na Precisão
        # Treino
        max_train_acc = max(all_acc)
        max_train_epoch = all_acc.index(max_train_acc) + 1
        
        axs[1].scatter(max_train_epoch, max_train_acc, color='blue', s=150, 
                      marker='o', edgecolors='black', linewidth=2, zorder=5, 
                      label=f'Melhor Treino: {max_train_acc:.4f}')
        
        # Validação
        max_val_acc = max(all_val_acc)
        max_val_epoch = all_val_acc.index(max_val_acc) + 1

        axs[1].scatter(max_val_epoch, max_val_acc, color='red', s=150, 
                      marker='s', edgecolors='black', linewidth=2, zorder=5, 
                      label=f'Melhor Validação: {max_val_acc:.4f}')
        
        # Offset
        offset_train = 0.02 if abs(max_train_epoch - max_val_epoch) < 5 else 0.01
        offset_val = -0.02 if abs(max_train_epoch - max_val_epoch) < 5 else 0.01

        # Valores escritos
        axs[1].annotate(f'{max_train_acc:.4f}', 
                       xy=(max_train_epoch, max_train_acc),
                       xytext=(max_train_epoch + 0.5, max_train_acc + offset_train),
                       color='blue', fontsize=10, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
        
        axs[1].annotate(f'{max_val_acc:.4f}', 
                       xy=(max_val_epoch, max_val_acc),
                       xytext=(max_val_epoch - 0.5, max_val_acc + offset_val),
                       color='red', fontsize=10, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        axs[1].set_xlabel('Épocas')
        axs[1].set_ylabel('Precisão')
        axs[1].set_title('Precisão do Modelo')
        
        # Melhorar a legenda
        handles, labels = axs[1].get_legend_handles_labels()
        # Remover labels duplicadas mantendo a ordem
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        axs[1].legend(unique_handles, unique_labels, loc='best')
        
        plt.tight_layout()
        plt.savefig("grafico_modelo.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Imprimir máximos de cada estágio
        for i, history in enumerate(histories):
            acc_train_stage = max(history.history['accuracy'])
            acc_val_stage = max(history.history['val_accuracy'])
            print(f"Estágio {i+1}: Treino = {acc_train_stage:.4f}, Validação = {acc_val_stage:.4f}")

        
    
    def _predict_image(self, img_array):
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
    
        pred = self.modelo.predict(img_array, verbose=0)[0][0]
    
        class_names = ['Cat', 'Dog']
    
        if pred < 0.5:
            return class_names[0], float(1 - pred)
        else:
            return class_names[1], float(pred)

