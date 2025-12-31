
#Extração e Conversão para PNG
import os
import numpy as np
from time import sleep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Pre-processamento
from sklearn.model_selection import train_test_split

#Relacionado a CNN
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

#Plot do Grafico e Imagens
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import sample
import quantus
from tqdm import tqdm


## Funcoes Importantes

class CNN_MNIST:
    def __init__(self):
        #Config
        self.tamanho_img = (28, 28)
        self.tamanho_batch = 16
        
        #ETC
        self.modelo: Sequential = None
        
        self.features: np.ndarray = None
        self.target: np.ndarray = None
        
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

        # Carrega o dataset
        print("A carregar dataset")
        self._load_dataset()
        print("Dataser carregado")
        
        #Criar Datasets
        print("A separar dataset")
        self._setup_datasets()
        print("Dataset separado")
        
        # Caso Exista Modelo Ja Treinado
        if os.path.exists("./assets/modelo_sensn.keras"):
            print("Modelo encontrado!")

            #Carregar Modelo
            self.modelo = keras.models.load_model("./assets/modelo_sensn.keras")


        # Caso Seja Preciso Treinar Modelo
        else:
            print("Modelo não encontrado, a treinar um em 3s")
            sleep(3)
    
            print("A criar modelo")
            self._setup_model()
            
            #Treinar Modelo
            print("A treinar modelo")
            self._train_model()
            print("Modelo treinado")


    def display_model_info(self):
        """
        Função que mostra a estrutura do modelo e da plot do gráfico de treino.
        """
        self.modelo.summary()

        plt.figure(figsize=(12, 8))
        img = mpimg.imread("./assets/grafico_modelo_sensn.png")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    
    def show_existing_classes(self):
        # Collect one random example per class
        examples_per_class = {}
        
        # Convert to list and shuffle
        all_samples = list(self.train_ds.unbatch().shuffle(10000))
        
        for image, label in all_samples:
            label_int = int(label.numpy())
            if label_int not in examples_per_class:
                examples_per_class[label_int] = image
            if len(examples_per_class) == 10:  # MNIST has 10 classes
                break
        
        # Plot
        plt.figure(figsize=(15, 3))
        for i in range(10):
            if i in examples_per_class:
                plt.subplot(2, 5, i+1)
                plt.imshow(examples_per_class[i].numpy().squeeze(), cmap='gray')
                plt.title(f"Classe: {i}")
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
        
        imagens_escolhidas = sample(test_images, qty_imgs)
        
        # Calculate rows needed
        cols = min(5, qty_imgs)
        rows = (qty_imgs + cols - 1) // cols
        
        plt.figure(figsize=(4*cols, 4*rows))
        
        for i, img_array in enumerate(imagens_escolhidas):
            classe, confianca = self._predict_image(img_array)
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img_array.squeeze(), cmap='gray')
            plt.title(f"Predição: {classe}\nConfiança: {confianca:.2%}")
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()


    def model_accuracy(self, test_dataset = None):
        """
        Testa a eficácia do modelo e devolve a sua precisão perantes os dados de teste
        """
        if test_dataset == None:
            test_dataset = self.test_ds

        test_dataset = (
            self.test_ds
            .batch(self.tamanho_batch)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
            
        _, test_accuracy = self.modelo.evaluate(test_dataset, verbose=0)
        return test_accuracy




    ## Funcoes Assistentes
    def _load_dataset(self):
        """
        Carrega o Dataset do MNIST.
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        features = tf.concat([x_train, x_test], axis=0)

        features = tf.cast(features, tf.float32) / 255.0

        self.features = tf.expand_dims(features, -1).numpy()
        self.target = tf.concat([y_train, y_test], axis=0).numpy()


    def _setup_datasets(self):
        """
        Cria os datasets de Treino, Validação e Teste.
        """        
        # Convert TensorFlow tensors to NumPy arrays for sklearn
        features_np = self.features
        target_np = self.target
        
        # 80% treino, 20% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            features_np,
            target_np,
            test_size=0.2,
            random_state=42,
            stratify=target_np
        )
        
        # 10% validação, 10% teste
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )
        
        # Criar tf.data.Dataset
        self.train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        self.test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # Debug
        for images, labels in self.train_ds.take(1):
            print("Shape das Imagens:", images.shape)
            print("Exemplo de Labels:", labels.numpy())
        
        print("Train:", len(X_train))
        print("Val:", len(X_val))
        print("Test:", len(X_test))

    
    
    
    def _setup_model(self) -> Sequential:
        """
        Esta função cria um modelo de uma cnn.
        """
        input_shape=(self.tamanho_img[0], self.tamanho_img[1], 1)
        
        self.modelo = Sequential([
            layers.Input(shape=input_shape),

            #Bloco 1
            layers.Conv2D(16, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            
            layers.Conv2D(16, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            #Bloco 2
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            #Bloco 3
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            #FCL
            layers.Flatten(),
            
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),

            layers.Dense(128),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Dropout(0.6),
            layers.Dense(10, activation='softmax')
        ])

    
    def _train_model(self) -> Sequential:
        """
        Esta função treina o modelo com vários hiper-parâmetros.
        """
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Batch + otimizações (equivalente ao original)
        train_ds = (
            self.train_ds
            .shuffle(len(self.features))
            .batch(self.tamanho_batch)
            .cache()
            .prefetch(AUTOTUNE)
        )
        
        val_ds = (
            self.val_ds
            .batch(self.tamanho_batch)
            .cache()
            .prefetch(AUTOTUNE)
        )
        
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        self.modelo.compile(
            optimizer=Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        history = self.modelo.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[early_stop],
            verbose=1
        )

        self.modelo.save("./assets/modelo_sensn.keras")
        self._save_train_graph(history)

    
    
    def _save_train_graph(self, history):
        """
        Plots training and validation metrics.
        """
        if not history or not history.history:
            print("No training history available.")
            return
        
        history_dict = history.history
        
        plt.figure(figsize=(14, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        if 'loss' in history_dict:
            plt.plot(history_dict['loss'], label='Treino')
        if 'val_loss' in history_dict:
            plt.plot(history_dict['val_loss'], label='Validação')
        
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.title('Loss do Modelo')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        if 'accuracy' in history_dict:
            plt.plot(history_dict['accuracy'], label='Treino')
        if 'val_accuracy' in history_dict:
            plt.plot(history_dict['val_accuracy'], label='Validação')
            
            # Add max validation accuracy annotation
            if history_dict['val_accuracy']:
                max_val_acc = max(history_dict['val_accuracy'])
                max_val_epoch = history_dict['val_accuracy'].index(max_val_acc)
                plt.scatter(max_val_epoch, max_val_acc, color='red', s=100, 
                          marker='s', edgecolors='black', linewidth=2, zorder=5)
                plt.annotate(f'{max_val_acc:.4f}', 
                           xy=(max_val_epoch, max_val_acc),
                           xytext=(max_val_epoch - 1, max_val_acc + 0.01),
                           color='red', fontsize=10, fontweight='bold')
        
        plt.xlabel('Épocas')
        plt.ylabel('Precisão')
        plt.title('Precisão do Modelo')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("./assets/grafico_modelo_sensn.png", dpi=300, bbox_inches='tight')
        
        # Print final metrics
        if history_dict:
            print(f"\nMétricas Finais:")
            if 'accuracy' in history_dict:
                print(f"Precisão Treino: {history_dict['accuracy'][-1]:.4f}")
            if 'val_accuracy' in history_dict:
                print(f"Precisão Validação: {history_dict['val_accuracy'][-1]:.4f}")

    
    def _predict_image(self, img_array):
        img_array = img_array / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = self.modelo.predict(img_array, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = float(np.max(predictions))
        
        return int(predicted_class), confidence
    
    def display_single_random_sample(self, dataset_name='train'):
        """
        Display a single random sample from a specific dataset.
        
        Parameters:
        dataset_name: 'train', 'val', 'test', or 'features'
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name == 'train':
            dataset = self.train_ds
        elif dataset_name == 'val':
            dataset = self.val_ds
        elif dataset_name == 'test':
            dataset = self.test_ds
        elif dataset_name == 'features':
            if self.features is not None:
                # Get random sample from features array
                idx = random.randint(0, len(self.features) - 1)
                img = self.features[idx].squeeze()
                label = self.target[idx]
                
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Random Sample from Original Features\nIndex: {idx}, Label: {label}')
                ax.axis('off')
                plt.tight_layout()
                plt.show()
                return
        else:
            print(f"Unknown dataset name: {dataset_name}")
            return
        
        # Get a batch and random sample for train_ds, val_ds, test_ds
        if dataset is not None:
            for batch in dataset.shuffle(1000).take(1):
                images, labels = batch
                
                if len(images) > 0:
                    idx = random.randint(0, len(images) - 1)
                    img = images[idx].numpy().squeeze()
                    label = labels[idx].numpy()
                    
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f'Random Sample from {dataset_name.capitalize()} Dataset\nLabel: {label}')
                    ax.axis('off')
                    plt.tight_layout()
                    plt.show()


#Extração e Conversão para PNG
import os
import numpy as np
from time import sleep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Pre-processamento
from sklearn.model_selection import train_test_split

#Relacionado a CNN
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

#Plot do Grafico e Imagens
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import sample
import quantus
from tqdm import tqdm


## Funcoes Importantes

class CNN_MNIST:
    def __init__(self):
        #Config
        self.tamanho_img = (28, 28)
        self.tamanho_batch = 16
        
        #ETC
        self.modelo: Sequential = None
        
        self.features: np.ndarray = None
        self.target: np.ndarray = None
        
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

        # Carrega o dataset
        print("A carregar dataset")
        self._load_dataset()
        print("Dataser carregado")
        
        #Criar Datasets
        print("A separar dataset")
        self._setup_datasets()
        print("Dataset separado")
        
        # Caso Exista Modelo Ja Treinado
        if os.path.exists("./assets/modelo_sensn.keras"):
            print("Modelo encontrado!")

            #Carregar Modelo
            self.modelo = keras.models.load_model("./assets/modelo_sensn.keras")


        # Caso Seja Preciso Treinar Modelo
        else:
            print("Modelo não encontrado, a treinar um em 3s")
            sleep(3)
    
            print("A criar modelo")
            self._setup_model()
            
            #Treinar Modelo
            print("A treinar modelo")
            self._train_model()
            print("Modelo treinado")


    def display_model_info(self):
        """
        Função que mostra a estrutura do modelo e da plot do gráfico de treino.
        """
        self.modelo.summary()

        plt.figure(figsize=(12, 8))
        img = mpimg.imread("./assets/grafico_modelo_sensn.png")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    
    def show_existing_classes(self):
        # Collect one random example per class
        examples_per_class = {}
        
        # Convert to list and shuffle
        all_samples = list(self.train_ds.unbatch().shuffle(10000))
        
        for image, label in all_samples:
            label_int = int(label.numpy())
            if label_int not in examples_per_class:
                examples_per_class[label_int] = image
            if len(examples_per_class) == 10:  # MNIST has 10 classes
                break
        
        # Plot
        plt.figure(figsize=(15, 3))
        for i in range(10):
            if i in examples_per_class:
                plt.subplot(2, 5, i+1)
                plt.imshow(examples_per_class[i].numpy().squeeze(), cmap='gray')
                plt.title(f"Classe: {i}")
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
        
        imagens_escolhidas = sample(test_images, qty_imgs)
        
        # Calculate rows needed
        cols = min(5, qty_imgs)
        rows = (qty_imgs + cols - 1) // cols
        
        plt.figure(figsize=(4*cols, 4*rows))
        
        for i, img_array in enumerate(imagens_escolhidas):
            classe, confianca = self._predict_image(img_array)
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img_array.squeeze(), cmap='gray')
            plt.title(f"Predição: {classe}\nConfiança: {confianca:.2%}")
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()


    def model_accuracy(self, test_dataset = None):
        """
        Testa a eficácia do modelo e devolve a sua precisão perantes os dados de teste
        """
        if test_dataset == None:
            test_dataset = self.test_ds

        test_dataset = (
            self.test_ds
            .batch(self.tamanho_batch)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
            
        _, test_accuracy = self.modelo.evaluate(test_dataset, verbose=0)
        return test_accuracy




    ## Funcoes Assistentes
    def _load_dataset(self):
        """
        Carrega o Dataset do MNIST.
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        features = tf.concat([x_train, x_test], axis=0)

        features = tf.cast(features, tf.float32) / 255.0

        self.features = tf.expand_dims(features, -1).numpy()
        self.target = tf.concat([y_train, y_test], axis=0).numpy()


    def _setup_datasets(self):
        """
        Cria os datasets de Treino, Validação e Teste.
        """        
        # Convert TensorFlow tensors to NumPy arrays for sklearn
        features_np = self.features
        target_np = self.target
        
        # 80% treino, 20% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            features_np,
            target_np,
            test_size=0.2,
            random_state=42,
            stratify=target_np
        )
        
        # 10% validação, 10% teste
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )
        
        # Criar tf.data.Dataset
        self.train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        self.test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # Debug
        for images, labels in self.train_ds.take(1):
            print("Shape das Imagens:", images.shape)
            print("Exemplo de Labels:", labels.numpy())
        
        print("Train:", len(X_train))
        print("Val:", len(X_val))
        print("Test:", len(X_test))

    
    
    
    def _setup_model(self) -> Sequential:
        """
        Esta função cria um modelo de uma cnn.
        """
        input_shape=(self.tamanho_img[0], self.tamanho_img[1], 1)
        
        self.modelo = Sequential([
            layers.Input(shape=input_shape),

            #Bloco 1
            layers.Conv2D(16, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            
            layers.Conv2D(16, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            #Bloco 2
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            #Bloco 3
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            #FCL
            layers.Flatten(),
            
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),

            layers.Dense(128),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Dropout(0.6),
            layers.Dense(10, activation='softmax')
        ])

    
    def _train_model(self) -> Sequential:
        """
        Esta função treina o modelo com vários hiper-parâmetros.
        """
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Batch + otimizações (equivalente ao original)
        train_ds = (
            self.train_ds
            .shuffle(len(self.features))
            .batch(self.tamanho_batch)
            .cache()
            .prefetch(AUTOTUNE)
        )
        
        val_ds = (
            self.val_ds
            .batch(self.tamanho_batch)
            .cache()
            .prefetch(AUTOTUNE)
        )
        
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        self.modelo.compile(
            optimizer=Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        history = self.modelo.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[early_stop],
            verbose=1
        )

        self.modelo.save("./assets/modelo_sensn.keras")
        self._save_train_graph(history)

    
    
    def _save_train_graph(self, history):
        """
        Plots training and validation metrics.
        """
        if not history or not history.history:
            print("No training history available.")
            return
        
        history_dict = history.history
        
        plt.figure(figsize=(14, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        if 'loss' in history_dict:
            plt.plot(history_dict['loss'], label='Treino')
        if 'val_loss' in history_dict:
            plt.plot(history_dict['val_loss'], label='Validação')
        
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.title('Loss do Modelo')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        if 'accuracy' in history_dict:
            plt.plot(history_dict['accuracy'], label='Treino')
        if 'val_accuracy' in history_dict:
            plt.plot(history_dict['val_accuracy'], label='Validação')
            
            # Add max validation accuracy annotation
            if history_dict['val_accuracy']:
                max_val_acc = max(history_dict['val_accuracy'])
                max_val_epoch = history_dict['val_accuracy'].index(max_val_acc)
                plt.scatter(max_val_epoch, max_val_acc, color='red', s=100, 
                          marker='s', edgecolors='black', linewidth=2, zorder=5)
                plt.annotate(f'{max_val_acc:.4f}', 
                           xy=(max_val_epoch, max_val_acc),
                           xytext=(max_val_epoch - 1, max_val_acc + 0.01),
                           color='red', fontsize=10, fontweight='bold')
        
        plt.xlabel('Épocas')
        plt.ylabel('Precisão')
        plt.title('Precisão do Modelo')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("./assets/grafico_modelo_sensn.png", dpi=300, bbox_inches='tight')
        
        # Print final metrics
        if history_dict:
            print(f"\nMétricas Finais:")
            if 'accuracy' in history_dict:
                print(f"Precisão Treino: {history_dict['accuracy'][-1]:.4f}")
            if 'val_accuracy' in history_dict:
                print(f"Precisão Validação: {history_dict['val_accuracy'][-1]:.4f}")

    
    def _predict_image(self, img_array):
        img_array = img_array / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = self.modelo.predict(img_array, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = float(np.max(predictions))
        
        return int(predicted_class), confidence
    
    def display_single_random_sample(self, dataset_name='train'):
        """
        Display a single random sample from a specific dataset.
        
        Parameters:
        dataset_name: 'train', 'val', 'test', or 'features'
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name == 'train':
            dataset = self.train_ds
        elif dataset_name == 'val':
            dataset = self.val_ds
        elif dataset_name == 'test':
            dataset = self.test_ds
        elif dataset_name == 'features':
            if self.features is not None:
                # Get random sample from features array
                idx = random.randint(0, len(self.features) - 1)
                img = self.features[idx].squeeze()
                label = self.target[idx]
                
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Random Sample from Original Features\nIndex: {idx}, Label: {label}')
                ax.axis('off')
                plt.tight_layout()
                plt.show()
                return
        else:
            print(f"Unknown dataset name: {dataset_name}")
            return
        
        # Get a batch and random sample for train_ds, val_ds, test_ds
        if dataset is not None:
            for batch in dataset.shuffle(1000).take(1):
                images, labels = batch
                
                if len(images) > 0:
                    idx = random.randint(0, len(images) - 1)
                    img = images[idx].numpy().squeeze()
                    label = labels[idx].numpy()
                    
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f'Random Sample from {dataset_name.capitalize()} Dataset\nLabel: {label}')
                    ax.axis('off')
                    plt.tight_layout()
                    plt.show()

    def run_sensitivity_n_analysis(self):
        """
        Executa a métrica Sensitivity-n num batch (self.tamanho_batch) do dataset de Teste.
        """
        print(f"\n--- A preparar dados para Sensitivity-n (Batch Size: {self.tamanho_batch}) ---")
        
        # 1. Carregar Batch diretamente usando o tamanho definido na classe
        ds_batched = self.test_ds.batch(self.tamanho_batch).take(1)
        x_batch, y_batch = next(iter(ds_batched))
        
        x_batch = x_batch.numpy()
        y_batch = y_batch.numpy()

        print(f"Dados prontos. Shape X: {x_batch.shape}, Shape Y: {y_batch.shape}")
    
        # 2. Gerar Explicações (Integrated Gradients)
        print("A gerar Integrated Gradients...")
        
        def get_integrated_gradients(img_input, label_idx, steps=50):
            baseline = tf.zeros_like(img_input)
            img_input = tf.cast(img_input, tf.float32)
            alphas = tf.linspace(0.0, 1.0, steps+1)[:, tf.newaxis, tf.newaxis, tf.newaxis]
            inputs = baseline + alphas * (img_input - baseline)
            
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                predictions = self.modelo(inputs)
                target_score = predictions[:, label_idx]
                
            grads = tape.gradient(target_score, inputs)
            avg_grads = tf.reduce_mean(grads, axis=0)
            return (img_input - baseline) * avg_grads

        # Calcular atribuições
        a_batch = []
        for i in tqdm(range(len(x_batch)), desc="Gerando Explicações"):
            img_tensor = tf.expand_dims(x_batch[i], axis=0) 
            attr = get_integrated_gradients(img_tensor, int(y_batch[i])).numpy()[0] # [0] para remover dim batch
            a_batch.append(attr)
            
        a_batch = np.array(a_batch)

        print("Explicações geradas. A iniciar Quantus...")
    
        # 3. Configurar e Executar Sensitivity-n (Quantus)
        metric = quantus.SensitivityN(
            n_max_percentage=0.2,
            features_in_step=28, 
            abs=True,            
            normalise=True,      
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            similarity_func=quantus.similarity_func.correlation_pearson,
            disable_warnings=True
        )
    
        scores = metric(
            model=self.modelo,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            device="gpu" if tf.config.list_physical_devices('GPU') else "cpu"
        )
        
        scores = [s for s in scores if not np.isnan(s)]
    
        if scores:
            mean_score = np.mean(scores)
            print(f"\nRESULTADO FINAL SENSITIVITY-N (Batch {len(x_batch)}):")
            print(f"Média do Score: {mean_score:.4f}")


    def plot_perturbation_curve(self, steps=10):
        """
        Gera gráfico de degradação da confiança removendo pixels importantes.
        Usa um batch do tamanho self.tamanho_batch.
        """
        print(f"\n--- A gerar Curva de Perturbação (Amostras: {self.tamanho_batch}) ---")
        
        # 1. Carregar Batch
        ds_batched = self.test_ds.batch(self.tamanho_batch).take(1)
        x_batch, y_batch = next(iter(ds_batched))
        x_batch, y_batch = x_batch.numpy(), y_batch.numpy()

        # 2. Definição IG (Interna para simplicidade)
        def get_integrated_gradients(img_input, label_idx, steps=50):
            baseline = tf.zeros_like(img_input)
            img_input = tf.cast(img_input, tf.float32)
            alphas = tf.linspace(0.0, 1.0, steps+1)[:, tf.newaxis, tf.newaxis, tf.newaxis]
            inputs = baseline + alphas * (img_input - baseline)
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                predictions = self.modelo(inputs)
                target_score = predictions[:, label_idx]
            grads = tape.gradient(target_score, inputs)
            avg_grads = tf.reduce_mean(grads, axis=0)
            return (img_input - baseline) * avg_grads

        # 3. Pré-calcular as explicações (MUITO MAIS RÁPIDO FAZER ISTO FORA DO LOOP DE STEPS)
        print("A calcular importâncias (IG) para todas as imagens...")
        batch_attributions = []
        for i in range(len(x_batch)):
            img_tensor = tf.expand_dims(x_batch[i], axis=0)
            attr = get_integrated_gradients(img_tensor, int(y_batch[i])).numpy()[0]
            batch_attributions.append(attr.flatten()) # Guardamos já flat
        
        # 4. Loop de Perturbação
        percentages = np.linspace(0, 1, steps + 1)
        scores_per_step = [] 

        print("A aplicar máscaras e prever...")
        for p in percentages:
            current_step_confs = []
            
            for i in range(len(x_batch)):
                flat_img = x_batch[i].flatten()
                flat_attr = batch_attributions[i]
                
                # Ordenar indices por importância
                sorted_indices = np.argsort(-np.abs(flat_attr))
                
                # Quantos mascarar
                num_to_mask = int(len(flat_img) * p)
                
                # Mascarar
                masked_img_flat = flat_img.copy()
                masked_img_flat[sorted_indices[:num_to_mask]] = 0.0
                
                # Prever
                masked_input = masked_img_flat.reshape(28, 28, 1)
                masked_input = np.expand_dims(masked_input, axis=0)
                
                pred = self.modelo.predict(masked_input, verbose=0)
                current_step_confs.append(pred[0][int(y_batch[i])])
            
            avg_conf = np.mean(current_step_confs)
            scores_per_step.append(avg_conf)
            print(f"Passo {p:.1%} -> Confiança Média: {avg_conf:.4f}")

        # 5. Plot
        plt.figure(figsize=(10, 6))
        plt.plot(percentages * 100, scores_per_step, marker='o', linestyle='-', color='b', linewidth=2)
        plt.axhline(y=0.1, color='r', linestyle='--', label='Aleatório (0.1)')
        plt.title(f'Degradação da Confiança (MoRF) - Batch: {self.tamanho_batch}')
        plt.xlabel('Percentagem de Pixels Removidos (%)')
        plt.ylabel('Confiança na Classe Correta')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()