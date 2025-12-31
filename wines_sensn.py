import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import shap

class Modelo_XGB_Classifier:
    def __init__(self):
        self.dataset: pd.DataFrame = None
        self.scaler: StandardScaler = StandardScaler()
        
        self.train_ds: pd.DataFrame = None
        self.val_ds: pd.DataFrame = None
        self.test_ds: pd.DataFrame = None

        self.modelo: xgb.XGBClassifier = None
        self.explainer: shap.Explainer = None
        self.dataset_scaled: pd.DataFrame = None
        

    ## SETUPS
    def setup(self):
        # Carregamento Dataset
        print("A Carregar Dataset")
        self._carregar_dataset()
        self._split_dataset()

        # Caso Exista Modelo já Treinado
        if os.path.exists("./assets/modelo_xgbclass.pkl"):
            print("Modelo encontrado!")
            with open("./assets/modelo_xgbclass.pkl", "rb") as f:
                self.modelo = pickle.load(f)
        
        # Caso Seja Preciso Treinar Modelo
        else:
            print("A Criar Modelo")
            self._setup_model()
            
            print("A Treinar Modelo")
            self._train_model()
            print("Modelo Treinado")

    def setup_explainer(self):
        # Create scaled version of entire dataset for SHAP (same scaling as training)
        dataset_features = self.dataset.drop(columns=["good-quality"])
        dataset_scaled_features = pd.DataFrame(
            self.scaler.transform(dataset_features),
            columns=dataset_features.columns
        )
        
        # Store the scaled dataset for later use
        self.dataset_scaled = pd.concat([
            dataset_scaled_features,
            self.dataset["good-quality"].reset_index(drop=True)
        ], axis=1)
        
        # Use scaled features for SHAP explainer
        self.explainer = shap.Explainer(
            self.modelo,
            masker=dataset_scaled_features
        )


    ## DEBUGGING
    def display_model_info(self):
        """
        Função que mostra a estrutura do modelo e da plot do gráfico de treino.
        """
        print(f"Modelo: XGBClassifier")
        print(f"Número de estimadores: {self.modelo.n_estimators}")
        print(f"Profundidade máxima: {self.modelo.max_depth}")
        print(f"Taxa de aprendizagem: {self.modelo.learning_rate}")
        print(f"\nCaracterísticas importantes (primeiras 10):")
        
        # Obter importância das features
        feature_importance = self.modelo.feature_importances_
        feature_names = self.train_ds.drop("good-quality", axis=1).columns
        
        # Criar DataFrame com importância
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(importance_df.head(10).to_string(index=False))

    def show_existing_classes(self):
        """
        Mostra as Classes e a sua respectiva etiqueta. 
        """
        print("Boa Qualidade = 1 | Qualidade Amena/Má = 0")

    def model_accuracy(self, test_dataset=None):
        """
        Testa a eficácia do modelo e devolve a sua precisão perantes os dados de teste
        """
        if test_dataset is None:
            test_dataset = self.test_ds

        X_test = test_dataset.drop("good-quality", axis=1).values
        y_test = test_dataset["good-quality"].values
        
        y_pred = self.modelo.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        return test_accuracy

    ## Explicação
    
    def explain_sample(self, index):
        instance_scaled = self.dataset_scaled.drop(columns=["good-quality"]).iloc[[index]]
        
        # Get the true label from original dataset
        true_label = self.dataset.iloc[index]["good-quality"]
        
        # Make prediction on scaled data
        predicted_class = self.modelo.predict(instance_scaled.values)
        predicted_proba = self.modelo.predict_proba(instance_scaled.values)
        
        print(f"Sample index: {index}")
        print(f"True class: {true_label}")
        print(f"Predicted class: {predicted_class[0]}")
        print(f"Prediction probabilities: {predicted_proba[0]}")
        
        # SHAP explanation
        shap_value = self.explainer(instance_scaled)
        
        # Waterfall plot
        shap.plots.waterfall(shap_value[0])




    
    ## Funcoes Assistentes
    def _carregar_dataset(self):
        """
        Carrega e faz o tratamento do dataset.
        """
        dataset = pd.read_csv("./assets/winequality-red.csv", sep = ";")

        #Binarizar o target
        dataset.rename(columns={'quality': 'good-quality'}, inplace=True)
        
        dataset['good-quality'] = dataset['good-quality'].apply(lambda x: 1 if x >= 6 else 0)

        self.dataset = dataset

    def _split_dataset(self):
        """
        Da split e normaliza as features do dataset.
        """
        #Split
        train_df, temp_df = train_test_split(
            self.dataset,
            stratify=self.dataset["good-quality"],
            test_size=0.20,
            random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df,
            stratify=temp_df["good-quality"],
            test_size=0.5,
            random_state=42
        )

        #Normalização
        X_train = train_df.drop("good-quality", axis=1)
        X_val = val_df.drop("good-quality", axis=1)
        X_test = test_df.drop("good-quality", axis=1)
        
        self.scaler.fit(X_train)
        
        # Train DS
        self.train_ds = pd.concat([
            pd.DataFrame(self.scaler.transform(X_train), columns=X_train.columns),
            train_df["good-quality"].reset_index(drop=True)
        ], axis=1)

        # Val DS
        self.val_ds = pd.concat([
            pd.DataFrame(self.scaler.transform(X_val), columns=X_val.columns),
            val_df["good-quality"].reset_index(drop=True)
        ], axis=1)

        # Test DS
        self.test_ds = pd.concat([
            pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns),
            test_df["good-quality"].reset_index(drop=True)
        ], axis=1)

        #Debug
        print(f"Train: {len(self.train_ds)} samples")
        print(f"Val: {len(self.val_ds)} samples")
        print(f"Test: {len(self.test_ds)} samples")


    def _setup_model(self):
        """
        Esta função cria um modelo XGBoost Classifier.
        """
        self.modelo = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss'
        )
        

    def _train_model(self):
        """
        Esta função treina o modelo XGBoost.
        """
        X_train = self.train_ds.drop("good-quality", axis=1).values
        y_train = self.train_ds["good-quality"].values
        X_val = self.val_ds.drop("good-quality", axis=1).values
        y_val = self.val_ds["good-quality"].values

        # Treinar com early stopping
        self.modelo.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Salvar o modelo
        with open("./assets/modelo_xgbclass.pkl", "wb") as f:
            pickle.dump(self.modelo, f)
