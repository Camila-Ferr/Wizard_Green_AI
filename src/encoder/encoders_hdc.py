from torchhd import embeddings, BSCTensor
from torch import Tensor
from torch.nn.parameter import Parameter
import torchhd.functional as functional
import torch
import torch.nn as nn
import torchhd
from torchhd.tensors.bsc import BSCTensor
from torchhd import functional as F
from torchhd.embeddings import Random




def bin_level(
        num_vectors: int,
        dimensions: int,
        *,
        requires_grad=False,
        **kwargs,
) -> BSCTensor:

    vsa_tensor = BSCTensor

    num_flipped_bits = dimensions // num_vectors
    idx_mapping = torch.randperm(dimensions)

    hv = torch.empty(
        num_vectors,
        dimensions,
        dtype=kwargs["dtype"],
        device=kwargs["device"],
    ).as_subclass(vsa_tensor)

    base = vsa_tensor.random(
        1,
        dimensions,
        **kwargs,
    )

    hv[0] = base[0]  # min hyper-vector
    slice_start = 0
    slice_end = num_flipped_bits

    for i in range(1, num_vectors):
        # Mark adding by 2 the num_flipped_bits position from base hyper-vector
        base.scatter_(1, idx_mapping[slice_start:slice_end].unsqueeze(0), 2, reduce='add')
        hv[i] = base[0]
        slice_start = slice_end
        slice_end += num_flipped_bits

    hv = hv.where(hv < 2, torch.logical_not(hv - 2))
    hv.requires_grad = requires_grad
    return hv


class ScatterCode(nn.Embedding):
    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "low",
        "high",
    ]

    low: float
    high: float
    vsa = "BSC"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        low: float = 0.0,
        high: float = 1.0,
        requires_grad: bool = False,
        device=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": torch.int8}
        # Have to call Module init explicitly in order not to use the Embedding init
        nn.Module.__init__(self)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.low = low
        self.high = high
        self.padding_idx = None  # required by nn.Embedding
        self.max_norm = None  # required by nn.Embedding
        self.norm_type = 2  # required by nn.Embedding
        self.scale_grad_by_freq = False  # required by nn.Embedding
        self.sparse = False  # required by nn.Embedding

        embeddings = bin_level(
            num_embeddings,
            embedding_dim,
            **factory_kwargs
        )
        # Have to provide requires grad at the creation of the parameters to
        # prevent errors when instantiating a non-float embedding
        self.weight = Parameter(embeddings, requires_grad=requires_grad)

    def reset_parameters(self) -> None:
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        with torch.no_grad():
            embeddings = bin_level(
                self.num_embeddings,
                self.embedding_dim,
                **factory_kwargs,
                **self.vsa_kwargs,
            )
            self.weight.copy_(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        index = functional.value_to_index(
            input, self.low, self.high, self.num_embeddings
        )
        index = index.clamp(min=0, max=self.num_embeddings - 1)
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        return super().forward(index).as_subclass(vsa_tensor)


class RecordEncoder(nn.Module):
    def __init__(self, out_features, size, levels, low, high):
        super(RecordEncoder, self).__init__()
        self.position = embeddings.Random(size, out_features, vsa="BSC", dtype=torch.uint8)
        self.value = ScatterCode(levels, out_features, low=low, high=high)

    def forward(self, x):
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return sample_hv

class NGramEncoder(nn.Module):
    def __init__(self, out_features, levels, low, high):
        super(NGramEncoder, self).__init__()
        self.value = ScatterCode(levels, out_features, low = low, high = high)

    def forward(self, x, oper = "bind"):
        if oper == "bind":
            sample_hv = torchhd.bind_sequence(self.value(x))
        elif oper == "bundle":
            sample_hv = torchhd.bundle_sequence(self.value(x))
        return sample_hv


