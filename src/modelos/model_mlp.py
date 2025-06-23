import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, dropout_rate, activation_fn):
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MLPClassifierTorch(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_sizes=(100,), dropout_rate=0.3, learning_rate=0.001,
                 activation_fn=nn.ReLU, max_epochs=100, weight_decay=0.0,
                 early_stopping=True, patience=10, verbose=False):
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.input_dim_ = X.shape[1]
        self.output_dim_ = len(np.unique(y))
        self.model_ = MLP(self.input_dim_, self.output_dim_, self.hidden_sizes,
                          self.dropout_rate, self.activation_fn)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            optimizer.zero_grad()
            outputs = self.model_(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss:.4f}")

            if self.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model_.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print("Early stopping triggered.")
                        break

        if best_model_state:
            self.model_.load_state_dict(best_model_state)

        return self

    def predict(self, X):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, preds = torch.max(outputs, 1)
        return preds.numpy()



def busca_melhores_mlp(X, y):
    """
    Cria um pipeline com MLP e realiza RandomizedSearchCV.
    """
    param_dist = {
        'mlp__hidden_sizes': [(64,), (128,), (64, 32), (128, 64), (256, 128)],
        'mlp__dropout_rate': uniform(0.1, 0.5),
        'mlp__learning_rate': uniform(0.0001, 0.01),
        'mlp__activation_fn': [nn.ReLU, nn.LeakyReLU, nn.Tanh],
        'mlp__max_epochs': randint(30, 300),
        'mlp__weight_decay': uniform(1e-5, 1e-2),
        'mlp__early_stopping': [True],
        'mlp__patience': [10],  # ou outro valor
        'mlp__verbose': [False],  # ou True se quiser logs
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifierTorch())
    ])

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=200,
        cv=StratifiedKFold(n_splits=5),
        n_jobs=-1,
        random_state=42,
        verbose=0,
        scoring='accuracy'
    )

    return random_search.fit(X, y)
