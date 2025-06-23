import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder,StandardScaler

# Termômetro encoding para numéricos
def thermometer_encode_numeric(X_scaled, n_bits=8):
    X_encoded = []
    for row in X_scaled:
        encoded_row = []
        for val in row:
            level = int(round(val * (n_bits - 1)))
            bits = [1] * level + [0] * (n_bits - level)
            encoded_row.extend(bits)
        X_encoded.append(encoded_row)
    return np.array(X_encoded, dtype=np.uint8)

def preprocess_data(X, y, categorical_cols, numeric_cols, n_bits=8):
    # OneHotEncoder para variáveis categóricas
    ohe = OneHotEncoder(sparse_output=False)
    if categorical_cols:
        X_categorical_encoded = ohe.fit_transform(X[categorical_cols])

    # Normalizar atributos numéricos para [0,1]
    scaler = MinMaxScaler()
    X_numeric_scaled = scaler.fit_transform(X[numeric_cols])

    # Aplica thermometer encoding (função externa)
    X_numeric_encoded = thermometer_encode_numeric(X_numeric_scaled, n_bits=n_bits)

    # Concatenar numéricos (termômetro) e categóricos (one-hot)
    if categorical_cols:
        X_processed = np.hstack((X_numeric_encoded, X_categorical_encoded.astype(np.uint8)))
    else:
        X_processed = X_numeric_encoded  # Não precisa hstack aqui

    # Label encode para y
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.iloc[:, 0])  # Garante 1D

    return X_processed, y_encoded, le

def preprocess_data_standart_scale(X, y, categorical_cols, numeric_cols):
    """
    Pré-processa os dados:
    - Aplica OneHotEncoder nas variáveis categóricas
    - Aplica StandardScaler nas variáveis numéricas
    - Concatena os dados
    - Codifica os rótulos do y
    """
    # OneHotEncoder para as variáveis categóricas
    ohe = OneHotEncoder(sparse_output=False)
    X_categorical_encoded = ohe.fit_transform(X[categorical_cols])
    categorical_column_names = ohe.get_feature_names_out(categorical_cols)

    # Normalizar atributos numéricos
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X[numeric_cols])

    # Concatenar dados
    X_processed = np.hstack((X_numeric_scaled, X_categorical_encoded))
    X_processed_df = pd.DataFrame(
        X_processed,
        columns=np.concatenate([numeric_cols, categorical_column_names])
    )

    # Ajustar y
    if len(y.shape) > 1 and y.shape[1] == 1:
        y = y.squeeze()


    # Codificar rótulos
    label_mapping = LabelEncoder()
    y_encoded = label_mapping.fit_transform(y)

    return X_processed_df, y_encoded, label_mapping

def preprocess_numeric_data_wizard(X, y, numeric_cols, n_bits=8):

    scaler = MinMaxScaler()
    X_numeric_scaled = scaler.fit_transform(X[numeric_cols])

    X_numeric_encoded = thermometer_encode_numeric(X_numeric_scaled, n_bits=n_bits)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y.iloc[:, 0])  # Garante 1D

    return X_numeric_encoded, y_encoded, le
