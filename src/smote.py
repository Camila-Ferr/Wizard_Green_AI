from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

class SmoteTransformer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.smote = SMOTE(random_state=self.random_state)

    def fit_transform(self, X, y):
        # Se X for ndarray, transforma em DataFrame
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        # Se y for ndarray, transforma em Series
        if isinstance(y, np.ndarray):
            y_series = pd.Series(y)
        else:
            y_series = y.copy()

        # Alinha os índices
        X_df = X_df.reset_index(drop=True)
        y_series = y_series.reset_index(drop=True)

        # Limpar NaN
        mask = X_df.notna().all(axis=1)
        X_cleaned_df = X_df.loc[mask].reset_index(drop=True)
        y_cleaned = y_series.loc[mask].reset_index(drop=True)

        # Aplicar SMOTE
        X_resampled, y_resampled = self.smote.fit_resample(X_cleaned_df, y_cleaned)

        # Retornar como DataFrame
        X_resampled_df = pd.DataFrame(
            X_resampled,
            columns=X_df.columns if hasattr(X_df, "columns") else None
        )

        return X_resampled_df, y_resampled
