
from sklearn.model_selection import ParameterGrid
from tqdm import trange, tqdm
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import trange
from torchhd.models import Centroid
from random import sample
from itertools import product
from sklearn.metrics import accuracy_score



class BinHD(nn.Module):
    def __init__(
            self,
            n_dimensions: int,
            n_classes: int,
            *,
            epochs: int = 30,
            device: torch.device = None,
    ) -> None:
        super().__init__()

        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.epochs = epochs
        self.device = device if device is not None else torch.device("cpu")
        self.classes_counter = torch.empty((n_classes, n_dimensions), device=self.device, dtype=torch.int8)
        self.classes_hv = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.classes_counter)

    def fit(self, input: Tensor, target: Tensor):
        input = 2 * input - 1
        self.classes_counter.index_add_(0, target, input)
        self.classes_hv = self.classes_counter.clamp(min=0, max=1)

    def fit_adapt(self, input: Tensor, target: Tensor):
        for _ in trange(0, self.epochs, desc="fit"):
            self.adapt(input, target)

    def adapt(self, input: Tensor, target: Tensor):
        pred = self.predict(input)
        is_wrong = target != pred

        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        input = 2 * input - 1
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.classes_counter.index_add_(0, target, input, alpha=1)
        self.classes_counter.index_add_(0, pred, input, alpha=-1)
        self.classes_hv = torch.where(self.classes_counter >= 0, 1, 0)

    def forward(self, samples: Tensor) -> Tensor:
        response = torch.empty((self.n_classes, samples.shape[0]), dtype=torch.int8, device=self.device)

        for i in range(self.n_classes):
            response[i] = torch.sum(torch.bitwise_xor(samples, self.classes_hv[i]), dim=1)  # Hamming distance

        return response.transpose(0, 1)

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmin(self(samples), dim=-1)

    @staticmethod
    def search_best_binhd(X_train_hv, y_train, X_test_hv, y_test, param_grid, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train_hv = X_train_hv.to(device)
        X_test_hv = X_test_hv.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)

        grid = list(ParameterGrid(param_grid))

        best_acc = 0.0
        best_params = None

        for params in tqdm(grid, desc="Grid Search"):
            model = BinHD(
                n_dimensions=params["n_dimensions"],
                n_classes=len(torch.unique(y_train)),
                epochs=params.get("epochs", 30),
                device=device
            ).to(device)

            model.reset_parameters()

            if params.get("adapt", False):
                model.fit_adapt(X_train_hv, y_train)
            else:
                model.fit(X_train_hv, y_train)

            y_pred = model.predict(X_test_hv)
            acc = accuracy_score(y_test.cpu(), y_pred.cpu())

            print(f"Params: {params} -> Accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_params = params

        print("\n✅ Melhor configuração encontrada:")
        print(f"Acurácia: {best_acc:.4f}")
        print(f"Parâmetros: {best_params}")

        return best_acc, best_params

class NeuralHD(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        regen_freq: int = 20,
        regen_rate: float = 0.04,
        epochs: int = 120,
        lr: float = 0.37,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.regen_freq = regen_freq
        self.regen_rate = regen_rate
        self.epochs = epochs
        self.lr = lr

        # REMOVIDO: self.encoder = nn.Identity()
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, input: Tensor, target: Tensor):
        # Assume que input já está codificado com RecordEncoder e em formato MAPTensor
        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)
        self.model.add(input, target)

        for epoch_idx in trange(1, self.epochs, desc="fit"):
            self.model.add_adapt(input, target, lr=self.lr)

            # Regeneração de dimensões (desabilitada pois não temos encoder interno)
            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                with torch.no_grad():
                    weight = F.normalize(self.model.weight, dim=1)
                    scores = torch.var(weight, dim=0)
                    regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices
                    self.model.weight.data[:, regen_dims].zero_()
                    # Comentado: regeneração de encoder interno que não existe
                    # self.encoder.weight.data[regen_dims, :].normal_()
                    # self.encoder.bias.data[:, regen_dims].uniform_(0, 2 * math.pi)

        return self

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(samples)  # Já é MAPTensor

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)

    @staticmethod
    def grid_search_encoded(
            X_train_hv: torch.Tensor,
            y_train: torch.Tensor,
            n_features: int,
            n_classes: int,
            device: torch.device,
            param_grid: dict,
            max_trials: int = None
    ):

        param_combinations = list(product(*param_grid.values()))
        if max_trials is not None and max_trials < len(param_combinations):
            param_combinations = sample(param_combinations, max_trials)

        best_acc = 0
        best_params = None
        best_model = None

        for params in param_combinations:
            dimension, num_levels, regen_rate, regen_freq, epochs, lr = params

            # Recorte da dimensão atual dos dados já codificados
            X_train_cut = X_train_hv[:, :dimension]

            model = NeuralHD(
                n_features,
                dimension,
                n_classes,
                regen_freq=regen_freq,
                regen_rate=regen_rate,
                epochs=epochs,
                lr=lr,
                device=device
            ).to(device)

            model.fit(X_train_cut, y_train)

            # Avaliação simples no treino (pode substituir por validação real)
            preds = model.predict(X_train_cut)
            acc = accuracy_score(preds.cpu(), y_train.cpu())

            print(
                f"[ACC={acc:.4f}] dim={dimension}, levels={num_levels}, rate={regen_rate}, freq={regen_freq}, epochs={epochs}, lr={lr}")

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = {
                    "dimension": dimension,
                    "num_levels": num_levels,
                    "regen_rate": regen_rate,
                    "regen_freq": regen_freq,
                    "epochs": epochs,
                    "lr": lr
                }

        return best_model, best_params, best_acc


