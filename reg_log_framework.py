import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

class Modelo_MLPClassifier:
    def __init__(self):
        self.modelo: Sequential = None
        self.train_ds: pd.Dataset = None
        self.val_ds: pd.Dataset = None
        self.test_ds: pd.Dataset = None
        self.scaler = StandardScaler()

    def setup(self):
        self._carregar_datasets()
        print("Dataset Carregado")
        
        self.modelo = self._setup_model()
        self._train_model()


    def _carregar_datasets(self):
        df = pd.read_csv("./Iris.csv")
        df = df.drop("Id", axis=1)

        # Get mapping before converting to codes
        cat = df["Species"].astype("category")
        mapping = dict(enumerate(cat.cat.categories))
        print("Class mapping:", mapping)
        
        df["Species"] = cat.cat.codes
        
        # Separate features and target BEFORE normalization
        X = df.drop("Species", axis=1)
        y = df["Species"]
        
        # Normalize features before split
        X_normalized = self.scaler.fit_transform(X)
        
        # Convert back to DataFrame for easier handling
        X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)
        df_normalized = pd.concat([X_normalized_df, y.reset_index(drop=True)], axis=1)
        
        # Train Dataset (80%)
        self.train_ds, temp_ds = train_test_split(
            df_normalized,
            stratify=df_normalized["Species"],
            test_size=0.2,
            random_state=42
        )
    
        # Validation & Test Dataset (20%)
        self.val_ds, self.test_ds = train_test_split(
            temp_ds,
            stratify=temp_ds["Species"],
            test_size=0.5,
            random_state=42
        )

        print("Train:", len(self.train_ds))
        print("Val:", len(self.val_ds))
        print("Test:", len(self.test_ds))


    def _setup_model(self):
        return Sequential([
            layers.Input(shape=(self.train_ds.drop(columns=["Species"]).shape[1],)),
                
            layers.Dense(7, activation='sigmoid'),
                
            layers.Dense(6, activation='sigmoid'),
    
            layers.Dense(self.train_ds["Species"].nunique(), activation='softmax')
        ])

    def _train_model(self):
        # Prepare data
        X_train = self.train_ds.drop("Species", axis=1).values
        y_train = self.train_ds["Species"].values
        
        X_val = self.val_ds.drop("Species", axis=1).values
        y_val = self.val_ds["Species"].values
        
        X_test = self.test_ds.drop("Species", axis=1).values
        y_test = self.test_ds["Species"].values
        
        # Compile model
        self.modelo.compile(
            optimizer=Adam(learning_rate=0.01),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()]
        )
        
        # Train model
        print("\nTraining model...")
        history = self.modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc = self.modelo.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return history

    def predict(self, X_new=None):
        """Make predictions on new data"""
        if X_new is None:
            # Use test set if no new data provided
            X_new = self.test_ds.drop("Species", axis=1).values
        
        # Ensure new data is normalized using the same scaler
        if isinstance(X_new, pd.DataFrame):
            X_new = self.scaler.transform(X_new)
        elif isinstance(X_new, np.ndarray):
            X_new = self.scaler.transform(X_new)
            
        predictions = self.modelo.predict(X_new, verbose=0)
        return np.argmax(predictions, axis=1)


# Example usage
if __name__ == "__main__":
    modelo = Modelo_MLPClassifier()
    modelo.setup()
    
    # Make predictions on test set
    predictions = modelo.predict()
    print(f"Predictions on test set: {predictions}")