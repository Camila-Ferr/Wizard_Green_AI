import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

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
    y_encoded = le.fit_transform(y.ravel())  # Garante 1D

    return X_processed, y_encoded, le
