
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

def spectral_radius(mat: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(mat)
    return float(np.max(np.abs(eigvals)))



def _tanh_saturation_distance(h: torch.Tensor) -> torch.Tensor:
    """Distance to saturation for tanh outputs in [-1, 1].
    """
    return (1.0 - h.abs()).clamp(min=0.0)


def _sigmoid_saturation_distance(h: torch.Tensor) -> torch.Tensor:
    """Distance to saturation for sigmoid outputs in [0, 1].
    """
    return torch.min(h, 1.0 - h).clamp(min=0.0)


class VanillaRNN(nn.Module):
    """Vanilla RNN with the same heads used in the original code."""
    def __init__(
        self,
        nin: int,
        nout: int,
        nhid: int,
        init: str = "smart_tanh",
        classif_type: str = "lastSoftmax",
        rng=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.nhid = nhid
        self.init = init
        self.classif_type = classif_type
        if init == "sigmoid":
            W_uh = rng.normal(loc=0.0, scale=0.01, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.01, size=(nhid, nhid)).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.01, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "sigmoid"
        elif init == "test":
            W_uh = rng.normal(loc=0.0, scale=0.8, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.8, size=(nhid, nhid)).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.8, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "identity"
        elif init == "basic_tanh":
            W_uh = rng.normal(loc=0.0, scale=0.1, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.1, size=(nhid, nhid)).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.1, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "tanh"
        elif init == "smart_tanh":
            W_uh = rng.normal(loc=0.0, scale=0.01, size=(nin, nhid)).astype(np.float32)
            W_hh = rng.normal(loc=0.0, scale=0.01, size=(nhid, nhid)).astype(np.float32)
            for dx in range(nhid):
                spng = rng.permutation(nhid)
                W_hh[dx, spng[15:]] = 0.0
            sr = spectral_radius(W_hh)
            if sr > 0:
                W_hh = (0.95 * W_hh / sr).astype(np.float32)
            W_hy = rng.normal(loc=0.0, scale=0.01, size=(nhid, nout)).astype(np.float32)
            b_hh = np.zeros((nhid,), dtype=np.float32)
            b_hy = np.zeros((nout,), dtype=np.float32)
            self.act_name = "tanh"
        else:
            raise ValueError(f"Unknown init={init}. Choose from sigmoid, test, basic_tanh, smart_tanh")

        # Parameters with same naming as Theano code
        self.W_uh = nn.Parameter(torch.tensor(W_uh, dtype=dtype, device=device))
        self.W_hh = nn.Parameter(torch.tensor(W_hh, dtype=dtype, device=device))
        self.W_hy = nn.Parameter(torch.tensor(W_hy, dtype=dtype, device=device))
        self.b_hh = nn.Parameter(torch.tensor(b_hh, dtype=dtype, device=device))
        self.b_hy = nn.Parameter(torch.tensor(b_hy, dtype=dtype, device=device))

    def act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_name == "sigmoid":
            return torch.sigmoid(x)
        if self.act_name == "tanh":
            return torch.tanh(x)
        if self.act_name == "identity":
            return x
        raise RuntimeError("bad act_name")

    def act_deriv_from_h(self, h: torch.Tensor) -> torch.Tensor:
        """Derivative of the nonlinearity expressed using the hidden state."""
        if self.act_name == "sigmoid":
            return h * (1.0 - h)
        if self.act_name == "tanh":
            return 1.0 - h * h
        if self.act_name == "identity":
            return torch.ones_like(h)
        raise RuntimeError("bad act_name")

    def forward(self, u: torch.Tensor):
        """
        u: (T, B, nin)
        returns:
          logits:
            - lastSoftmax: (B, nout)
            - softmax: (T*B, nout)
            - lastLinear: (B, nout)
          h: (T, B, nhid)
        """
        T, B, _ = u.shape
        # h_{t-1}
        h_prev = u.new_zeros((B, self.nhid))

        h_seq = []
        logits_seq = []

        for t in range(T):
            preact = h_prev @ self.W_hh + u[t] @ self.W_uh + self.b_hh
            h_t = self.act(preact)
            y_t = h_t @ self.W_hy + self.b_hy

            h_seq.append(h_t)
            logits_seq.append(y_t)
            h_prev = h_t

        h = torch.stack(h_seq, dim=0)              # (T,B,nhid)
        logits_all = torch.stack(logits_seq, dim=0)  # (T,B,nout)

        if self.classif_type == "softmax":
            logits = logits_all.reshape(T * B, self.nout)
        elif self.classif_type in {"lastSoftmax", "lastLinear"}:
            logits = logits_all[-1]  # (B,nout)
        else:
            raise ValueError(f"Unknown classif_type={self.classif_type}")

        return logits, h

    supports_omega: bool = True

    def saturation_distance_from_h(self, h: torch.Tensor) -> torch.Tensor:
        """Elementwise distance-to-saturation for the nonlinearity.

        For tanh: min(1-h, 1+h). For sigmoid: min(h, 1-h).
        Smaller => more saturated.
        """
        if self.act_name == "sigmoid":
            return _sigmoid_saturation_distance(h)
        return _tanh_saturation_distance(h)

    def recurrent_weight_for_rho(self) -> torch.Tensor:
        return self.W_hh

    def numpy_state(self) -> dict:
        return {
            "W_hh": self.W_hh.detach().cpu().numpy(),
            "W_uh": self.W_uh.detach().cpu().numpy(),
            "W_hy": self.W_hy.detach().cpu().numpy(),
            "b_hh": self.b_hh.detach().cpu().numpy(),
            "b_hy": self.b_hy.detach().cpu().numpy(),
            "act_name": np.array(self.act_name),
            "classif_type": np.array(self.classif_type),
            "model_type": np.array("rnn"),
        }


class GRUModel(nn.Module):
    """Minimal GRU in the same (T, B, *) layout as VanillaRNN."""

    supports_omega: bool = False

    def __init__(
        self,
        nin: int,
        nout: int,
        nhid: int,
        init: str = "smart_tanh",
        classif_type: str = "lastSoftmax",
        rng: np.random.RandomState | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.nin = int(nin)
        self.nout = int(nout)
        self.nhid = int(nhid)
        self.init = init
        self.classif_type = classif_type

        if rng is None:
            rng = np.random.RandomState(1234)

        W_uz = rng.normal(0, 0.01, size=(nin, nhid))
        W_hz = rng.normal(0, 0.01 / 2, size=(nhid, nhid))
        b_z = np.zeros((nhid,), dtype=np.float64) - 2

        W_ur = rng.normal(0, 0.01, size=(nin, nhid))
        W_hr = rng.normal(0, 0.01 / 2, size=(nhid, nhid))
        b_r = np.zeros((nhid,), dtype=np.float64)

        W_hh = rng.normal(0, 0.01, size=(nhid, nhid))
        W_uh = rng.normal(0, 0.01, size=(nin, nhid))
        b_h = np.zeros((nhid,), dtype=np.float64)

        W_hy = rng.normal(0, 0.01, size=(nhid, nout))
        b_y = np.zeros((nout,), dtype=np.float64)

        self.W_uz = nn.Parameter(torch.tensor(W_uz, dtype=dtype, device=device))
        self.W_hz = nn.Parameter(torch.tensor(W_hz, dtype=dtype, device=device))
        self.b_z = nn.Parameter(torch.tensor(b_z, dtype=dtype, device=device))

        self.W_ur = nn.Parameter(torch.tensor(W_ur, dtype=dtype, device=device))
        self.W_hr = nn.Parameter(torch.tensor(W_hr, dtype=dtype, device=device))
        self.b_r = nn.Parameter(torch.tensor(b_r, dtype=dtype, device=device))

        self.W_uh = nn.Parameter(torch.tensor(W_uh, dtype=dtype, device=device))
        self.W_hh = nn.Parameter(torch.tensor(W_hh, dtype=dtype, device=device))
        self.b_h = nn.Parameter(torch.tensor(b_h, dtype=dtype, device=device))

        self.W_hy = nn.Parameter(torch.tensor(W_hy, dtype=dtype, device=device))
        self.b_y = nn.Parameter(torch.tensor(b_y, dtype=dtype, device=device))

    def saturation_distance_from_h(self, h: torch.Tensor) -> torch.Tensor:
        return _tanh_saturation_distance(h)

    def act_deriv_from_h(self, h: torch.Tensor) -> torch.Tensor:
        return 1.0 - h * h

    def recurrent_weight_for_rho(self) -> torch.Tensor:
        return self.W_hh

    def numpy_state(self) -> dict:
        return {
            "W_uz": self.W_uz.detach().cpu().numpy(),
            "W_hz": self.W_hz.detach().cpu().numpy(),
            "b_z": self.b_z.detach().cpu().numpy(),
            "W_ur": self.W_ur.detach().cpu().numpy(),
            "W_hr": self.W_hr.detach().cpu().numpy(),
            "b_r": self.b_r.detach().cpu().numpy(),
            "W_uh": self.W_uh.detach().cpu().numpy(),
            "W_hh": self.W_hh.detach().cpu().numpy(),
            "b_h": self.b_h.detach().cpu().numpy(),
            "W_hy": self.W_hy.detach().cpu().numpy(),
            "b_y": self.b_y.detach().cpu().numpy(),
            "init": np.array(self.init),
            "classif_type": np.array(self.classif_type),
            "model_type": np.array("gru"),
        }

    def forward(self, u: torch.Tensor, return_extras: bool = False):
        """u: (T,B,nin). Returns (logits, h) or (logits, h, extras)."""
        T, B, _ = u.shape
        h_prev = u.new_zeros((B, self.nhid))

        h_seq = []
        y_seq = []

        if return_extras:
            z_preact_seq = []
            r_preact_seq = []
            htilde_preact_seq = []

        for t in range(T):
            u_t = u[t]

            z_preact = u_t @ self.W_uz + h_prev @ self.W_hz + self.b_z
            r_preact = u_t @ self.W_ur + h_prev @ self.W_hr + self.b_r

            z = torch.sigmoid(z_preact)
            r = torch.sigmoid(r_preact)

            htilde_preact = u_t @ self.W_uh + (r * h_prev) @ self.W_hh + self.b_h
            h_tilde = torch.tanh(htilde_preact)

            h_t = (1.0 - z) * h_prev + z * h_tilde
            y_t = h_t @ self.W_hy + self.b_y

            h_seq.append(h_t)
            y_seq.append(y_t)

            if return_extras:
                z_preact_seq.append(z_preact)
                r_preact_seq.append(r_preact)
                htilde_preact_seq.append(htilde_preact)

            h_prev = h_t

        h = torch.stack(h_seq, dim=0)   # (T,B,nhid)
        y_all = torch.stack(y_seq, dim=0)  # (T,B,nout)

        if self.classif_type == "softmax":
            logits = y_all.reshape(T * B, self.nout)
        elif self.classif_type in {"lastSoftmax", "lastLinear"}:
            logits = y_all[-1]  # (B,nout)
        else:
            raise ValueError(f"Unknown classif_type={self.classif_type}")

        if not return_extras:
            return logits, h

        extras = {
            "z": torch.stack(z_preact_seq, dim=0),        # (T,B,nhid)
            "r": torch.stack(r_preact_seq, dim=0),        # (T,B,nhid)
            "h_tilde": torch.stack(htilde_preact_seq, dim=0),  # (T,B,nhid)
        }
        return logits, h, extras


def make_model(
    model_type: str,
    nin: int,
    nout: int,
    nhid: int,
    init: str,
    classif_type: str,
    rng: np.random.RandomState,
    dtype: torch.dtype,
    device: torch.device,
):
    model_type = model_type.lower()
    if model_type in {"rnn", "vanilla", "vanillarnn"}:
        return VanillaRNN(
            nin=nin,
            nout=nout,
            nhid=nhid,
            init=init,
            classif_type=classif_type,
            rng=rng,
            dtype=dtype,
            device=device,
        )
    if model_type in {"gru"}:
        return GRUModel(
            nin=nin,
            nout=nout,
            nhid=nhid,
            init=init,
            classif_type=classif_type,
            rng=rng,
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"Unknown model_type={model_type}")
