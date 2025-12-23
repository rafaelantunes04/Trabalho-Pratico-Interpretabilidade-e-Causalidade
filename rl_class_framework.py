import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.callbacks import EarlyStopping
import os
import keras


class Modelo_RL_Classifier:
    def __init__(self):
        self.modelo: Sequential = None
        self.train_ds: pd.DataFrame = None
        self.val_ds: pd.DataFrame = None
        self.test_ds: pd.DataFrame = None
        self.scaler: StandardScaler = StandardScaler()

    def setup(self):
        # Carregamento Dataset
        print("A Carregar Dataset")
        self._carregar_datasets()
        print("Datast Carregado")

        #Caso Exista Modelo já Treinado
        if os.path.exists("./assets/modelo_rlclass.keras"):
            
            print("Modelo encontrado!")
            self.modelo = keras.models.load_model("./assets/modelo_rlclass.keras")

        
        # Caso Seja Preciso Treinar Modelo
        else:
            print("A Criar Modelo")
            self.modelo = self._setup_model()
            
            print("A Treinar Modelo")
            self._train_model()
            print("Modelo Treinado")


    def display_model_info(self):
        """
        Função que mostra a estrutura do modelo e da plot do gráfico de treino.
        """
        self.modelo.summary()
    
        plt.figure(figsize=(12, 8))
        img = mpimg.imread("./assets/grafico_modelo_rlclass.png")
        plt.imshow(img)
        plt.axis("off")
        plt.show()


    def show_existing_classes(self):
        """
        Mostra as Classes e a sua respectiva etiqueta. 
        """
        print("B (Benign) = 1 | M (Malign) = 0")

        
    def model_accuracy(self, test_dataset = None):
        """
        Testa a eficácia do modelo e devolve a sua precisão perantes os dados de teste
        """
        if test_dataset == None:
            test_dataset = self.test_ds

        X_test = test_dataset.drop("diagnosis", axis=1).values
        y_test = test_dataset["diagnosis"].values
        
        _, test_accuracy = self.modelo.evaluate(X_test, y_test, verbose=0)
        return test_accuracy



    ## Funcoes Assistentes
    def _carregar_datasets(self):
        """
        Carrega e faz o pre-processamento do dataset do cancro da mama.
        """
        #Carregamento
        df = pd.read_csv("./assets/breast_cancer.csv")
        df = df.drop("id", axis=1)

        df["diagnosis"] = df["diagnosis"].map({"M": 0, "B": 1})
        
        # Train Dataset (75%)
        train_df, temp_df = train_test_split(
            df,
            stratify=df["diagnosis"],
            test_size=0.25,
            random_state=42
        )

        # Validation & Test Dataset (25%)
        val_df, test_df = train_test_split(
            temp_df,
            stratify=temp_df["diagnosis"],
            test_size=0.5,
            random_state=42
        )

        #Tirar a Col. de Target
        X_train = train_df.drop("diagnosis", axis=1)
        X_val = val_df.drop("diagnosis", axis=1)
        X_test = test_df.drop("diagnosis", axis=1)
        
        self.scaler.fit(X_train)
        
        # Normalizar Features e Voltar a Juntar o Target
        #Train DS
        self.train_ds = pd.concat([
            pd.DataFrame(self.scaler.transform(X_train), columns=X_train.columns),
            train_df["diagnosis"].reset_index(drop=True)
        ], axis=1)

        #Val DS
        self.val_ds = pd.concat([
            pd.DataFrame(self.scaler.transform(X_val), columns=X_val.columns),
            val_df["diagnosis"].reset_index(drop=True)
        ], axis=1)

        #Test DS
        self.test_ds = pd.concat([
            pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns),
            test_df["diagnosis"].reset_index(drop=True)
        ], axis=1)

        print(f"Train: {len(self.train_ds)} samples")
        print(f"Val: {len(self.val_ds)} samples")
        print(f"Test: {len(self.test_ds)} samples")


    def _setup_model(self) -> Sequential:
        """
        Esta função cria um modelo simples de um classificador com regressão logistica.
        """
        return Sequential([
            layers.Input(shape=(self.train_ds.drop(columns=["diagnosis"]).shape[1],)),
            layers.Dense(1),
            layers.Activation('sigmoid')
        ])


    def _train_model(self):
        """
        Esta função treina o modelo a partir de hiperparâmetros.
        """
        X_train = self.train_ds.drop("diagnosis", axis=1).values
        y_train = self.train_ds["diagnosis"].values
        X_val = self.val_ds.drop("diagnosis", axis=1).values
        y_val = self.val_ds["diagnosis"].values

        self.modelo.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        history = self.modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            callbacks=[early_stop],
            verbose=1
        )

        self.modelo.save("./assets/modelo_rlclass.keras")
        self._plot_train_graph(history)


    def _plot_train_graph(self, history):
        # Usar diretamente o histórico fornecido
        all_loss = history.history['loss']
        all_val_loss = history.history['val_loss']
        all_acc = history.history['accuracy']
        all_val_acc = history.history['val_accuracy']
        
        # Plot do gráfico
        _, axs = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        epochs_range = range(1, len(all_loss) + 1)
        axs[0].plot(epochs_range, all_loss, label='Treino')
        axs[0].plot(epochs_range, all_val_loss, label='Validação')
        
        axs[0].set_xlabel('Épocas')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss do Modelo')
        axs[0].legend()
        
        # Precisão
        axs[1].plot(epochs_range, all_acc, label='Treino')
        axs[1].plot(epochs_range, all_val_acc, label='Validação')
        
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
        
        # Offset para evitar sobreposição
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
        plt.savefig("./assets/grafico_modelo_rlclass.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Imprimir os máximos
        print(f"Melhor precisão de treino: {max_train_acc:.4f} na época {max_train_epoch}")
        print(f"Melhor precisão de validação: {max_val_acc:.4f} na época {max_val_epoch}")