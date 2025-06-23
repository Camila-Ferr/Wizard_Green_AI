from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import torch
from torchwnn.classifiers import Wisard
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

class WiSARDClassifierTorch(BaseEstimator, ClassifierMixin):
    def __init__(self, tuple_size=4, bleaching=False, verbose=False):
        self.tuple_size = tuple_size
        self.bleaching = bleaching
        self.verbose = verbose

    def fit(self, X, y):
        # Assegura que X é binário int e y é int
        X = np.array(X, dtype=np.uint8)
        y = np.array(y, dtype=np.int64)

        X_tensor = torch.tensor(X, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)

        self.n_classes_ = len(np.unique(y))
        self.entry_size_ = X.shape[1]

        self.model_ = Wisard(
            entry_size=self.entry_size_,
            n_classes=self.n_classes_,
            tuple_size=self.tuple_size,
            bleaching=self.bleaching
        )

        self.model_.fit(X_tensor, y_tensor)

        return self

    def predict(self, X):
        X = np.array(X, dtype=np.uint8)
        X_tensor = torch.tensor(X, dtype=torch.long)

        y_pred_tensor = self.model_.predict(X_tensor)
        y_pred = y_pred_tensor.numpy()

        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

def buscar_melhor_wisard(X_train, y_train, param_dist=None, n_iter=100, cv=3, scoring='accuracy'):
    """
    Executa RandomizedSearchCV para otimizar o WiSARDClassifierTorch dentro de um pipeline.

    Args:
        X_train: Dados de entrada para treino.
        y_train: Labels correspondentes.
        param_dist: Dicionário com distribuições dos parâmetros para busca.
                    Se None, usa padrão {'wisard__tuple_size': [8,9,10,11], 'wisard__bleaching': [True, False]}.
        n_iter: Número de iterações RandomizedSearch.
        cv: Número de folds para validação cruzada.
        scoring: Métrica de avaliação.

    Returns:
        best_model: Pipeline com o melhor modelo encontrado.
        best_params: Dicionário dos melhores parâmetros.
        best_score: Melhor score obtido na validação cruzada.
    """

    if param_dist is None:
        param_dist = {
            'wisard__tuple_size': [8, 9, 10, 11],
            'wisard__bleaching': [True, False],
        }

    pipeline = Pipeline([
        ('wisard', WiSARDClassifierTorch())
    ])

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        verbose=0,
        n_jobs=-1,
        scoring=scoring
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    return best_model, best_params, best_score, random_search
