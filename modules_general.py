import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Optional
import torch.nn.functional as F
import pandas as pd


def get_data_from_trajectory_id(ids_df, data_df, trajectory_ids):
    """
    Return the exact portion(s) of `data_df` corresponding to one or more trajectory_ids.
    - Preserves data_df's original indexing (no reset).
    - If trajectory_ids contains all ids in ids_df, returns data_df unchanged.
    - If any requested id is missing, prints missing ids and returns None.
    """
    # normalize to list of ints (preserve order)
    if isinstance(trajectory_ids, (np.integer, int)):
        trajectory_ids = [int(trajectory_ids)]
    else:
        trajectory_ids = [int(x) for x in trajectory_ids]

    # ensure id columns are ints in ids_df
    ids_df = ids_df.copy()
    for col in ("start_index", "end_index", "generated_points", "trajectory_id"):
        if col in ids_df.columns:
            ids_df[col] = ids_df[col].astype("int64")

    existing_ids = list(ids_df["trajectory_id"].values)
    missing = [tid for tid in trajectory_ids if tid not in existing_ids]
    if missing:
        print(f"trajectory_id(s) {missing} not in ids_df. Pick from {existing_ids}")
        return None

    # special-case: request for all trajectories -> return full data_df unchanged
    if set(trajectory_ids) == set(existing_ids):
        return data_df

    parts = []
    for tid in trajectory_ids:
        row = ids_df.loc[ids_df["trajectory_id"] == tid].iloc[0]
        start = int(row["start_index"])
        end = int(row["end_index"])   # exclusive by your convention

        if start >= end:
            # empty trajectory: skip (or you can append an empty frame if you prefer)
            continue

        # label-based selection: .loc is inclusive on the right, so use end-1
        sub = data_df.loc[start : end - 1]
        parts.append(sub)

    if not parts:
        # nothing found (all requested trajectories empty)
        return data_df.iloc[0:0]   # empty DataFrame with same columns

    # if single part, return it directly (preserves original index)
    if len(parts) == 1:
        return parts[0]
    # multiple parts: concatenate preserving indices and order
    return pd.concat(parts)

class Step_1_general(nn.Module):
    """
    General-n Hamilton-Jacobi Layer, Step 1 (coordinate scaling, momentum shift).

    For each degree of freedom i in {1, ..., n}:
        Q_i^(1) = a * q_i + c1_i
        P_i^(1) = (1/a) * p_i - gamma * dG(q, t)/dq_i + c2_i

    where:
      - G(q, t) is a single scalar MLP taking all n coordinates + time  -> scalar,
      - a, gamma are scalar per layer (shared across all DOF; required scalar by
        the symplecticity proof: a per-coordinate scaling/gamma would need to
        commute with the full Hessian H_G, forcing a multiple of the identity),
      - c1, c2 are per-coordinate shift vectors of shape (n,) (they never enter
        the Jacobian, so any shape is symplectic; the paper specifies per-coord).

    Shapes:
        q: (batch, n)   p: (batch, n)   t: (batch, 1)
        Q: (batch, n)   P: (batch, n)   t: (batch, 1)  (t passed through)
    """

    def __init__(
        self,
        n: int,
        # MLP architecture
        hidden_dims: Union[int, List[int]] = 32,
        n_hidden_layers: int = 2,
        # activation
        activation: str = 'tanh',
        activation_params: Optional[dict] = None,
        final_activation: Optional[str] = None,
        # initialization
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        # architectural choices
        use_bias: bool = True,
        use_layer_norm: bool = False,
        # trainable scalar-parameter constraints for a
        a_eps_min: float = 0.1,
        a_eps_max: float = 10.0,
        a_k: float = 0.1,
        a_raw_innit: float = 0.0,
        gamma_innit: float = 5.0,
        # per-coordinate shift inits: scalar (broadcast-filled) or array-like of len n
        c1_innit: Union[float, List[float]] = 0.0,
        c2_innit: Union[float, List[float]] = 5.0,
    ):
        super(Step_1_general, self).__init__()

        self.n = n

        # scalar trainable params
        self.a_raw = nn.Parameter(torch.tensor(float(a_raw_innit)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_innit)))

        # per-coordinate shift params, shape (n,)
        self.c1 = nn.Parameter(self._make_vec(c1_innit, n))
        self.c2 = nn.Parameter(self._make_vec(c2_innit, n))

        self.a_eps_min = a_eps_min
        self.a_eps_max = a_eps_max
        self.a_k = a_k

        # process hidden dims
        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * n_hidden_layers
        else:
            self.hidden_dims = list(hidden_dims)

        # input_dim = n coordinates + time; output_dim = 1 (scalar G)
        self.G_network = self._build_mlp(
            input_dim=n + 1,
            output_dim=1,
            hidden_dims=self.hidden_dims,
            activation=activation,
            activation_params=activation_params,
            final_activation=final_activation,
            use_bias=use_bias,
            use_layer_norm=use_layer_norm,
        )

        self._initialize_weights(
            weight_init=weight_init,
            weight_init_params=weight_init_params,
            bias_init=bias_init,
            bias_init_value=bias_init_value,
        )

    @staticmethod
    def _make_vec(val, n):
        """Turn a scalar (broadcast-fill) or length-n array-like into a float32 (n,) tensor."""
        if isinstance(val, (int, float)):
            return torch.full((n,), float(val), dtype=torch.float32)
        t = torch.as_tensor(val, dtype=torch.float32).reshape(-1)
        if t.numel() != n:
            raise ValueError(f"Expected scalar or length-{n} init, got length {t.numel()}")
        return t.clone()

    @property
    def a(self):
        """Compute a from a_raw via bounded sigmoid to keep it non-zero (invertibility)."""
        return self.a_eps_min + (self.a_eps_max - self.a_eps_min) * torch.sigmoid(self.a_k * self.a_raw)

    def _get_activation(self, activation_name: str, params: Optional[dict] = None):
        params = params or {}
        activations = {
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'elu': lambda: nn.ELU(**params),
            'selu': nn.SELU,
            'gelu': nn.GELU,
            'softplus': lambda: nn.Softplus(**params),
            'swish': nn.SiLU,
            'mish': nn.Mish,
            'identity': nn.Identity,
            'none': nn.Identity,
        }
        if activation_name.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation_name}")
        activation = activations[activation_name.lower()]
        return activation() if callable(activation) else activation(**params)

    def _build_mlp(self, input_dim, output_dim, hidden_dims, activation,
                   activation_params, final_activation, use_bias, use_layer_norm):
        layers = []
        all_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(all_dims) - 1):
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]
            is_last_layer = (i == len(all_dims) - 2)
            layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            if use_layer_norm and not is_last_layer:
                layers.append(nn.LayerNorm(out_dim))
            if is_last_layer:
                if final_activation is not None:
                    layers.append(self._get_activation(final_activation, activation_params))
            else:
                layers.append(self._get_activation(activation, activation_params))
        return nn.Sequential(*layers)

    def _initialize_weights(self, weight_init, weight_init_params, bias_init, bias_init_value):
        weight_init_params = weight_init_params or {}

        def init_weights(m):
            if isinstance(m, nn.Linear):
                if weight_init == 'xavier_uniform':
                    init.xavier_uniform_(m.weight, **weight_init_params)
                elif weight_init == 'xavier_normal':
                    init.xavier_normal_(m.weight, **weight_init_params)
                elif weight_init == 'kaiming_uniform':
                    init.kaiming_uniform_(m.weight, **weight_init_params)
                elif weight_init == 'kaiming_normal':
                    init.kaiming_normal_(m.weight, **weight_init_params)
                elif weight_init == 'normal':
                    init.normal_(m.weight, **weight_init_params)
                elif weight_init == 'uniform':
                    init.uniform_(m.weight, **weight_init_params)
                elif weight_init == 'orthogonal':
                    init.orthogonal_(m.weight, **weight_init_params)
                if m.bias is not None:
                    if bias_init == 'zeros':
                        init.zeros_(m.bias)
                    elif bias_init == 'ones':
                        init.ones_(m.bias)
                    elif bias_init == 'uniform':
                        init.uniform_(m.bias, -bias_init_value, bias_init_value)
                    elif bias_init == 'normal':
                        init.normal_(m.bias, std=bias_init_value)
                    elif bias_init == 'constant':
                        init.constant_(m.bias, bias_init_value)

        self.G_network.apply(init_weights)

    def compute_G_and_gradient(self, q, t):
        """
        Compute dG(q, t)/dq for all n coordinates via a single vectorized autograd.grad.

        Args:
            q: (batch, n)
            t: (batch, 1)
        Returns:
            dG_dq: (batch, n)
        """
        q = q.requires_grad_(True)
        mlp_input = torch.cat([q, t], dim=1)          # (batch, n+1)
        G = self.G_network(mlp_input)                 # (batch, 1)
        dG_dq = torch.autograd.grad(
            outputs=G,
            inputs=q,
            grad_outputs=torch.ones_like(G),
            create_graph=self.training,
            retain_graph=self.training,
        )[0]                                          # (batch, n)
        return dG_dq

    def forward(self, q, p, t):
        """
        Args:
            q: (batch, n)   positions
            p: (batch, n)   momenta
            t: (batch, 1)   time
        Returns:
            Q: (batch, n)   Q = a * q + c1
            P: (batch, n)   P = (1/a) * p - gamma * dG/dq + c2
            t: (batch, 1)   unchanged
        """
        assert q.shape == p.shape, "q and p must have the same shape (batch, n)"
        assert q.dim() == 2 and q.shape[1] == self.n, f"q must be (batch, {self.n})"
        assert t.dim() == 2 and t.shape[1] == 1, "t must be (batch, 1)"
        assert t.shape[0] == q.shape[0], "batch dimension of t must match q"

        dG_dq = self.compute_G_and_gradient(q, t)     # (batch, n)
        a_val = self.a                                # scalar

        Q = a_val * q + self.c1                        # (batch, n), c1 broadcasts (n,)
        P = (1.0 / a_val) * p - self.gamma * dG_dq + self.c2

        return Q, P, t

    def get_config(self):
        return {
            'n': self.n,
            'hidden_dims': self.hidden_dims,
            'a': self.a.item(),
            'gamma': self.gamma.item(),
            'c1': self.c1.detach().cpu().numpy().tolist(),
            'c2': self.c2.detach().cpu().numpy().tolist(),
        }


class Step_2_general(nn.Module):
    """
    General-n Hamilton-Jacobi Layer, Step 2 (momentum scaling, coordinate shift).

    For each degree of freedom i in {1, ..., n}:
        P_i^(2) = a' * P_i^(1) + c1_i
        Q_i^(2) = (1/a') * Q_i^(1) + gamma' * dF(P^(1), t)/dP_i^(1) + c2_i

    where F(P^(1), t) is a single scalar MLP taking all n (step-1-transformed)
    momenta + time -> scalar. a', gamma' scalar per layer; c1, c2 per-coordinate (n,).

    Note on argument naming: to keep a uniform (coord, mom, t) call signature across
    both steps, forward takes (q, p, t) where here `p` is the momentum-like input that
    gets scaled and `q` is the coordinate-like input that gets shifted. This matches
    the original Step_2 forward(x, u, t) convention.

    Shapes:
        q: (batch, n)   p: (batch, n)   t: (batch, 1)
        Q: (batch, n)   P: (batch, n)   t: (batch, 1)
    """

    def __init__(
        self,
        n: int,
        hidden_dims: Union[int, List[int]] = 32,
        n_hidden_layers: int = 2,
        activation: str = 'tanh',
        activation_params: Optional[dict] = None,
        final_activation: Optional[str] = None,
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        use_bias: bool = True,
        use_layer_norm: bool = False,
        a_eps_min: float = 0.1,
        a_eps_max: float = 10.0,
        a_k: float = 0.1,
        a_raw_innit: float = 0.0,
        gamma_innit: float = 5.0,
        c1_innit: Union[float, List[float]] = 0.0,
        c2_innit: Union[float, List[float]] = 5.0,
    ):
        super(Step_2_general, self).__init__()

        self.n = n

        self.a_eps_min = a_eps_min
        self.a_eps_max = a_eps_max
        self.a_k = a_k

        self.a_raw = nn.Parameter(torch.tensor(float(a_raw_innit)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_innit)))
        self.c1 = nn.Parameter(self._make_vec(c1_innit, n))
        self.c2 = nn.Parameter(self._make_vec(c2_innit, n))

        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * n_hidden_layers
        else:
            self.hidden_dims = list(hidden_dims)

        self.F_network = self._build_mlp(
            input_dim=n + 1,
            output_dim=1,
            hidden_dims=self.hidden_dims,
            activation=activation,
            activation_params=activation_params,
            final_activation=final_activation,
            use_bias=use_bias,
            use_layer_norm=use_layer_norm,
        )

        self._initialize_weights(
            weight_init=weight_init,
            weight_init_params=weight_init_params,
            bias_init=bias_init,
            bias_init_value=bias_init_value,
        )

    @staticmethod
    def _make_vec(val, n):
        """Turn a scalar (broadcast-fill) or length-n array-like into a float32 (n,) tensor."""
        if isinstance(val, (int, float)):
            return torch.full((n,), float(val), dtype=torch.float32)
        t = torch.as_tensor(val, dtype=torch.float32).reshape(-1)
        if t.numel() != n:
            raise ValueError(f"Expected scalar or length-{n} init, got length {t.numel()}")
        return t.clone()

    def _get_activation(self, activation_name: str, params: Optional[dict] = None):
        params = params or {}
        activations = {
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'elu': lambda: nn.ELU(**params),
            'selu': nn.SELU,
            'gelu': nn.GELU,
            'softplus': lambda: nn.Softplus(**params),
            'swish': nn.SiLU,
            'mish': nn.Mish,
            'identity': nn.Identity,
            'none': nn.Identity,
        }
        if activation_name.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation_name}")
        activation = activations[activation_name.lower()]
        return activation() if callable(activation) else activation(**params)

    def _build_mlp(self, input_dim, output_dim, hidden_dims, activation,
                   activation_params, final_activation, use_bias, use_layer_norm):
        layers = []
        all_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(all_dims) - 1):
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]
            is_last_layer = (i == len(all_dims) - 2)
            layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            if use_layer_norm and not is_last_layer:
                layers.append(nn.LayerNorm(out_dim))
            if is_last_layer:
                if final_activation is not None:
                    layers.append(self._get_activation(final_activation, activation_params))
            else:
                layers.append(self._get_activation(activation, activation_params))
        return nn.Sequential(*layers)

    def _initialize_weights(self, weight_init, weight_init_params, bias_init, bias_init_value):
        weight_init_params = weight_init_params or {}

        def init_weights(m):
            if isinstance(m, nn.Linear):
                if weight_init == 'xavier_uniform':
                    init.xavier_uniform_(m.weight, **weight_init_params)
                elif weight_init == 'xavier_normal':
                    init.xavier_normal_(m.weight, **weight_init_params)
                elif weight_init == 'kaiming_uniform':
                    init.kaiming_uniform_(m.weight, **weight_init_params)
                elif weight_init == 'kaiming_normal':
                    init.kaiming_normal_(m.weight, **weight_init_params)
                elif weight_init == 'normal':
                    init.normal_(m.weight, **weight_init_params)
                elif weight_init == 'uniform':
                    init.uniform_(m.weight, **weight_init_params)
                elif weight_init == 'orthogonal':
                    init.orthogonal_(m.weight, **weight_init_params)
                if m.bias is not None:
                    if bias_init == 'zeros':
                        init.zeros_(m.bias)
                    elif bias_init == 'ones':
                        init.ones_(m.bias)
                    elif bias_init == 'uniform':
                        init.uniform_(m.bias, -bias_init_value, bias_init_value)
                    elif bias_init == 'normal':
                        init.normal_(m.bias, std=bias_init_value)
                    elif bias_init == 'constant':
                        init.constant_(m.bias, bias_init_value)

        self.F_network.apply(init_weights)

    @property
    def a(self):
        return self.a_eps_min + (self.a_eps_max - self.a_eps_min) * torch.sigmoid(self.a_k * self.a_raw)

    def compute_F_and_gradient(self, p, t):
        """
        Compute dF(p, t)/dp for all n momenta via a single vectorized autograd.grad.

        Args:
            p: (batch, n)   the step-1-transformed momenta P^(1)
            t: (batch, 1)
        Returns:
            dF_dp: (batch, n)
        """
        p = p.requires_grad_(True)
        mlp_input = torch.cat([p, t], dim=1)          # (batch, n+1)
        F = self.F_network(mlp_input)                 # (batch, 1)
        dF_dp = torch.autograd.grad(
            outputs=F,
            inputs=p,
            grad_outputs=torch.ones_like(F),
            create_graph=self.training,
            retain_graph=self.training,
        )[0]                                          # (batch, n)
        return dF_dp

    def forward(self, q, p, t):
        """
        Args:
            q: (batch, n)   coordinate-like input (gets shifted)
            p: (batch, n)   momentum-like input  (gets scaled)
            t: (batch, 1)
        Returns:
            Q: (batch, n)   Q = (1/a') * q + gamma' * dF/dp + c2
            P: (batch, n)   P = a' * p + c1
            t: (batch, 1)   unchanged
        """
        assert q.shape == p.shape, "q and p must have the same shape (batch, n)"
        assert q.dim() == 2 and q.shape[1] == self.n, f"q must be (batch, {self.n})"
        assert t.dim() == 2 and t.shape[1] == 1, "t must be (batch, 1)"
        assert t.shape[0] == q.shape[0], "batch dimension of t must match q"

        dF_dp = self.compute_F_and_gradient(p, t)     # (batch, n)
        a_val = self.a

        P = a_val * p + self.c1
        Q = (1.0 / a_val) * q + self.gamma * dF_dp + self.c2

        return Q, P, t

    def get_config(self):
        return {
            'n': self.n,
            'hidden_dims': self.hidden_dims,
            'a': self.a.item(),
            'gamma': self.gamma.item(),
            'c1': self.c1.detach().cpu().numpy().tolist(),
            'c2': self.c2.detach().cpu().numpy().tolist(),
        }


class CombinedHamiltonianLayer_general(nn.Module):
    """
    General-n Hamilton-Jacobi Layer (HJL): one full HJL = Step 1 followed by Step 2.

    Forward (applied in this order, per the paper):
        Step 1 (coordinate scaling, momentum shift):
            Q^(1) = a  * q      + c1
            P^(1) = (1/a) * p - gamma  * dG(q, t)/dq   + c2
        Step 2 (momentum scaling, coordinate shift):
            P^(2) = a' * P^(1)  + c1'
            Q^(2) = (1/a') * Q^(1) + gamma' * dF(P^(1), t)/dP^(1) + c2'

    a, a', gamma, gamma' are scalar per layer (shared across DOF; required scalar
    by the symplecticity proof). c1, c2, c1', c2' are per-coordinate (n,) shifts.
    G and F are each a single scalar MLP over all n coordinates/momenta + time.

    Shapes:  q, p : (batch, n) ;  t : (batch, 1)  ->  Q, P : (batch, n) ;  t : (batch, 1)
    """

    def __init__(
        self,
        n: int,
        # Step 1 (G network) architecture
        step1_hidden_dims: Union[int, List[int]] = 32,
        step1_n_hidden_layers: int = 2,
        step1_activation: str = 'tanh',
        step1_activation_params: Optional[dict] = None,
        step1_final_activation: Optional[str] = None,
        # Step 2 (F network) architecture
        step2_hidden_dims: Union[int, List[int]] = 32,
        step2_n_hidden_layers: int = 2,
        step2_activation: str = 'tanh',
        step2_activation_params: Optional[dict] = None,
        step2_final_activation: Optional[str] = None,
        # shared init/architecture
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        use_bias: bool = True,
        use_layer_norm: bool = False,
        # scalar-a reparam constraints (shared by both steps)
        a_eps_min: float = 0.1,
        a_eps_max: float = 10.0,
        a_k: float = 0.1,
        # per-step scalar inits
        step_1_a_raw_innit: float = 0.0,
        step_1_gamma_innit: float = 5.0,
        step_2_a_raw_innit: float = 0.0,
        step_2_gamma_innit: float = 5.0,
        # per-step, per-coordinate shift inits (scalar broadcast or length-n array)
        step_1_c1_innit: Union[float, List[float]] = 0.0,
        step_1_c2_innit: Union[float, List[float]] = 5.0,
        step_2_c1_innit: Union[float, List[float]] = 0.0,
        step_2_c2_innit: Union[float, List[float]] = 5.0,
    ):
        super(CombinedHamiltonianLayer_general, self).__init__()

        self.n = n

        self.step_1 = Step_1_general(
            n=n,
            hidden_dims=step1_hidden_dims,
            n_hidden_layers=step1_n_hidden_layers,
            activation=step1_activation,
            activation_params=step1_activation_params,
            final_activation=step1_final_activation,
            weight_init=weight_init,
            weight_init_params=weight_init_params,
            bias_init=bias_init,
            bias_init_value=bias_init_value,
            use_bias=use_bias,
            use_layer_norm=use_layer_norm,
            a_eps_min=a_eps_min,
            a_eps_max=a_eps_max,
            a_k=a_k,
            a_raw_innit=step_1_a_raw_innit,
            gamma_innit=step_1_gamma_innit,
            c1_innit=step_1_c1_innit,
            c2_innit=step_1_c2_innit,
        )

        self.step_2 = Step_2_general(
            n=n,
            hidden_dims=step2_hidden_dims,
            n_hidden_layers=step2_n_hidden_layers,
            activation=step2_activation,
            activation_params=step2_activation_params,
            final_activation=step2_final_activation,
            weight_init=weight_init,
            weight_init_params=weight_init_params,
            bias_init=bias_init,
            bias_init_value=bias_init_value,
            use_bias=use_bias,
            use_layer_norm=use_layer_norm,
            a_eps_min=a_eps_min,
            a_eps_max=a_eps_max,
            a_k=a_k,
            a_raw_innit=step_2_a_raw_innit,
            gamma_innit=step_2_gamma_innit,
            c1_innit=step_2_c1_innit,
            c2_innit=step_2_c2_innit,
        )

    def forward(self, q, p, t):
        """
        Args:  q, p : (batch, n) ;  t : (batch, 1)
        Returns: Q, P : (batch, n) ;  t : (batch, 1)
        """
        q1, p1, t1 = self.step_1(q, p, t)
        Q, P, t2 = self.step_2(q1, p1, t1)
        return Q, P, t2


import numpy as np


# ---------------------------------------------------------------------------
# NOTE: solve_a_raw is identical to the one already in your module (scalar,
# n-independent). Included here so this file is self-contained for testing.
# On integration into your single module, keep only ONE copy.
# ---------------------------------------------------------------------------
def solve_a_raw(a_eps_min, a_eps_max, a_k, a_target):
    if (a_target <= a_eps_min) or (a_target >= a_eps_max):
        raise ValueError(f"a_target must be between a_eps_min and a_eps_max. Got {a_target}")
    s = (a_target - a_eps_min) / (a_eps_max - a_eps_min)
    a_raw = (1 / a_k) * np.log(s / (1 - s))
    return a_raw


def a_raw_std_from_induced(sigma_a, a_mean, a_eps_min, a_eps_max, a_k):
    """
    Option A (delta method): map a desired std of the *induced* a onto the std of
    the raw parameter a_raw, using the local slope of the sigmoid reparameterisation
    a = a_min + (a_max - a_min) * sigmoid(a_k * a_raw) at the mean.

        da/da_raw |_mean = (a_max - a_min) * a_k * s * (1 - s),   s = (a_mean - a_min)/(a_max - a_min)
        sigma_raw = sigma_a / |da/da_raw|_mean

    First-order/approximate (a nonlinear map of a Gaussian is not Gaussian), matching
    the paper's own approximate "induced mean/std" specification.
    """
    s = (a_mean - a_eps_min) / (a_eps_max - a_eps_min)
    slope = (a_eps_max - a_eps_min) * a_k * s * (1.0 - s)
    return sigma_a / slope


def init_step_params_general(num_layers, step_1_mean, step_2_mean,
                             step_1_variance, step_2_variance,
                             n_coords=None, seed=None):
    """
    Draw per-layer init values from N(mean, variance) with ABSOLUTE variance
    (paper scheme), i.i.d. per layer and (when n_coords is set) per coordinate.

    Draw order within a call: step_1 first, then step_2 (preserves the global
    RNG stream ordering used across the a -> gamma -> c1 -> c2 calls).

    Args:
        num_layers:     number of HJLs (L)
        step_1_mean, step_2_mean:         scalar means
        step_1_variance, step_2_variance: scalar ABSOLUTE variances (std = sqrt(var))
        n_coords: None -> shape (num_layers,)        [scalar params: a_raw, gamma]
                  n    -> shape (num_layers, n)       [per-coord params: c1, c2]
        seed: optional; if set, seeds np.random before drawing
    Returns:
        (step_1_arr, step_2_arr) as float32 numpy arrays of the shape above.
    """
    if seed is not None:
        np.random.seed(seed)
    step_1_std = np.sqrt(step_1_variance)
    step_2_std = np.sqrt(step_2_variance)
    shape = (num_layers,) if n_coords is None else (num_layers, n_coords)
    step_1 = np.random.randn(*shape) * step_1_std + step_1_mean
    step_2 = np.random.randn(*shape) * step_2_std + step_2_mean
    return step_1.astype(np.float32), step_2.astype(np.float32)


class SimpleStackedHamiltonianNetwork_general(nn.Module):
    """
    General-n Mapping Network: a stack of L identical-in-config
    CombinedHamiltonianLayer_general (HJL) blocks.

    Init scheme (paper-faithful, per design decisions):
      - a:     mean per step via solve_a_raw; spread specified as a RATIO of the
               induced-a mean (a_std_ratio), mapped to raw space by the delta method.
      - gamma: N(mean, gamma_variance), absolute variance.
      - c1,c2: N(mean, c*_variance) per layer AND per coordinate -> (L, n).

    Draw order (reproducible with `seed`): a -> gamma -> c1 -> c2, step_1 before step_2.

    Final-HJL linear output (paper): with final_activation_only_on_final_layer=True and
    final_activation=None, intermediate HJLs' MLPs are fully nonlinear and only the
    L-th HJL's MLPs use a linear output layer.

    Shapes: q, p : (batch, n) ; t : (batch, 1) -> Q, P : (batch, n) ; t : (batch, 1)
    """

    def __init__(
        self,
        n: int,
        n_layers: int = 3,
        hidden_dims: Union[int, List[int]] = 32,
        n_hidden_layers: Optional[int] = 2,
        activation: str = 'tanh',
        activation_params: Optional[dict] = None,
        final_activation: Optional[str] = None,
        final_activation_only_on_final_layer: bool = True,
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        use_bias: bool = True,
        use_layer_norm: bool = False,
        # scalar-a reparam constraints
        a_eps_min: float = 0.1,
        a_eps_max: float = 10.0,
        a_k: float = 0.1,
        # a: induced-space means (per step) + spread as ratio of the mean
        step_1_a_mean_innit: float = 1.0,
        step_2_a_mean_innit: float = 1.0,
        a_std_ratio: float = 0.0,
        # gamma: means (per step) + absolute variance (shared across steps)
        step_1_gamma_mean_innit: float = 5.0,
        step_2_gamma_mean_innit: float = 5.0,
        gamma_variance: float = 0.0,
        # c1: means (per step) + absolute variance
        step_1_c1_mean_innit: float = 0.0,
        step_2_c1_mean_innit: float = 0.0,
        c1_variance: float = 0.0,
        # c2: means (per step) + absolute variance
        step_1_c2_mean_innit: float = 5.0,
        step_2_c2_mean_innit: float = 5.0,
        c2_variance: float = 0.0,
        seed: Optional[int] = None,
    ):
        super(SimpleStackedHamiltonianNetwork_general, self).__init__()

        self.n = n
        self.n_layers = n_layers
        self.layers = nn.ModuleList()

        # ----- reproducible draw stream: seed once, then draw a -> gamma -> c1 -> c2 -----
        if seed is not None:
            np.random.seed(seed)

        # a: mean handled in raw space via solve_a_raw; spread via delta method (Option A)
        step_1_a_raw_mean = solve_a_raw(a_eps_min, a_eps_max, a_k, step_1_a_mean_innit)
        step_2_a_raw_mean = solve_a_raw(a_eps_min, a_eps_max, a_k, step_2_a_mean_innit)
        sig_a_raw_1 = a_raw_std_from_induced(a_std_ratio * step_1_a_mean_innit,
                                             step_1_a_mean_innit, a_eps_min, a_eps_max, a_k)
        sig_a_raw_2 = a_raw_std_from_induced(a_std_ratio * step_2_a_mean_innit,
                                             step_2_a_mean_innit, a_eps_min, a_eps_max, a_k)
        s1_a, s2_a = init_step_params_general(
            n_layers, step_1_a_raw_mean, step_2_a_raw_mean,
            sig_a_raw_1 ** 2, sig_a_raw_2 ** 2, n_coords=None, seed=None)

        s1_g, s2_g = init_step_params_general(
            n_layers, step_1_gamma_mean_innit, step_2_gamma_mean_innit,
            gamma_variance, gamma_variance, n_coords=None, seed=None)

        s1_c1, s2_c1 = init_step_params_general(
            n_layers, step_1_c1_mean_innit, step_2_c1_mean_innit,
            c1_variance, c1_variance, n_coords=n, seed=None)

        s1_c2, s2_c2 = init_step_params_general(
            n_layers, step_1_c2_mean_innit, step_2_c2_mean_innit,
            c2_variance, c2_variance, n_coords=n, seed=None)

        # ----- build layers -----
        for i in range(n_layers):
            if final_activation_only_on_final_layer:
                # intermediate HJLs fully nonlinear (end in `activation`);
                # final HJL uses `final_activation` (None -> linear output)
                layer_final_activation = activation if i < n_layers - 1 else final_activation
            else:
                layer_final_activation = final_activation

            layer = CombinedHamiltonianLayer_general(
                n=n,
                step1_hidden_dims=hidden_dims,
                step1_n_hidden_layers=n_hidden_layers,
                step1_activation=activation,
                step1_activation_params=activation_params,
                step1_final_activation=layer_final_activation,
                step2_hidden_dims=hidden_dims,
                step2_n_hidden_layers=n_hidden_layers,
                step2_activation=activation,
                step2_activation_params=activation_params,
                step2_final_activation=layer_final_activation,
                weight_init=weight_init,
                weight_init_params=weight_init_params,
                bias_init=bias_init,
                bias_init_value=bias_init_value,
                use_bias=use_bias,
                use_layer_norm=use_layer_norm,
                a_eps_min=a_eps_min,
                a_eps_max=a_eps_max,
                a_k=a_k,
                step_1_a_raw_innit=float(s1_a[i]),
                step_1_gamma_innit=float(s1_g[i]),
                step_2_a_raw_innit=float(s2_a[i]),
                step_2_gamma_innit=float(s2_g[i]),
                step_1_c1_innit=s1_c1[i],   # length-n array
                step_1_c2_innit=s1_c2[i],
                step_2_c1_innit=s2_c1[i],
                step_2_c2_innit=s2_c2[i],
            )
            self.layers.append(layer)

    def forward(self, q, p, t):
        Q, P, T = q, p, t
        for layer in self.layers:
            Q, P, T = layer(Q, P, T)
        return Q, P, T


class ReverseStep1_general(nn.Module):
    """
    Analytic inverse of Step_1_general. SHARES the G network and all parameters
    with the forward module (no new parameters).

    Forward Step 1:   Q = a*q + c1 ;   P = (1/a)*p - gamma*dG(q,t)/dq + c2
    Inverse:          q = (Q - c1)/a
                      p = a * (P + gamma*dG(q,t)/dq - c2)   [gradient at recovered q]

    Shapes: Q, P : (batch, n) ; t : (batch, 1) -> q, p : (batch, n) ; t : (batch, 1)
    """
    def __init__(self, forward_step1_module: nn.Module):
        super(ReverseStep1_general, self).__init__()
        self.forward_module = forward_step1_module

    def forward(self, Q, P, t):
        a = self.forward_module.a
        c1 = self.forward_module.c1
        c2 = self.forward_module.c2
        gamma = self.forward_module.gamma

        q = (Q - c1) / a                                      # recover q first
        dG_dq = self.forward_module.compute_G_and_gradient(q, t)
        p = a * (P + gamma * dG_dq - c2)
        return q, p, t


class ReverseStep2_general(nn.Module):
    """
    Analytic inverse of Step_2_general. SHARES the F network and all parameters
    with the forward module.

    Forward Step 2:   P = a'*p + c1' ;   Q = (1/a')*q + gamma'*dF(p,t)/dp + c2'
    Inverse:          p = (P - c1')/a'
                      q = a' * (Q - gamma'*dF(p,t)/dp - c2')  [gradient at recovered p]

    Shapes: Q, P : (batch, n) ; t : (batch, 1) -> q, p : (batch, n) ; t : (batch, 1)
    """
    def __init__(self, forward_step2_module: nn.Module):
        super(ReverseStep2_general, self).__init__()
        self.forward_module = forward_step2_module

    def forward(self, Q, P, t):
        a = self.forward_module.a
        c1 = self.forward_module.c1
        c2 = self.forward_module.c2
        gamma = self.forward_module.gamma

        p = (P - c1) / a                                      # recover p first
        dF_dp = self.forward_module.compute_F_and_gradient(p, t)
        q = a * (Q - gamma * dF_dp - c2)
        return q, p, t


class ReverseCombinedHamiltonianLayer_general(nn.Module):
    """
    Inverse of one full HJL. Shares weights with the forward
    CombinedHamiltonianLayer_general. Order is reversed relative to forward:
    forward is Step_1 -> Step_2, so inverse is Reverse_Step_2 -> Reverse_Step_1.
    """
    def __init__(self, forward_combined_layer: nn.Module):
        super(ReverseCombinedHamiltonianLayer_general, self).__init__()
        self.reverse_step2 = ReverseStep2_general(forward_combined_layer.step_2)
        self.reverse_step1 = ReverseStep1_general(forward_combined_layer.step_1)

    def forward(self, Q, P, t):
        q1, p1, t1 = self.reverse_step2(Q, P, t)
        q0, p0, t0 = self.reverse_step1(q1, p1, t1)
        return q0, p0, t0


class InverseStackedHamiltonianNetwork_general(nn.Module):
    """
    Inverse of SimpleStackedHamiltonianNetwork_general (the decoder). Each reverse
    layer shares all weights with its forward counterpart; layers are applied in
    reverse order so that M^{-1}(M(q,p,t)) = (q,p,t) up to numerical precision.
    """
    def __init__(self, forward_network: nn.Module):
        super(InverseStackedHamiltonianNetwork_general, self).__init__()
        self.n = forward_network.n
        self.n_layers = forward_network.n_layers
        self.reverse_layers = nn.ModuleList()
        for i in reversed(range(self.n_layers)):
            self.reverse_layers.append(
                ReverseCombinedHamiltonianLayer_general(forward_network.layers[i])
            )

    def forward(self, Q, P, t):
        q, p, T = Q, P, t
        for reverse_layer in self.reverse_layers:
            q, p, T = reverse_layer(q, p, T)
        return q, p, T
    





class SegmentDataLoader_general:
    """
    Paper-faithful training dataloader for the Hamilton-Jacobi Network.

    Data model (per the paper's training section, generalised to variable-length
    segments): a "segment" is a contiguous, time-normalised piece of a trajectory,
    identified by a globally-unique segment_id. N indexes over segments.

    Epoch/batch structure:
      - An epoch has Lmax batches, where Lmax = length of the longest segment.
      - Each segment draws a fresh random permutation of its own samples per epoch.
      - At batch b, segment s is ACTIVE iff b < len(s). Active segments contribute
        their perm[b]-th sample as the single ENCODE observation. So every sample of
        every segment is used as an encode observation exactly once per epoch, and
        exhausted (shorter) segments drop out of later batches -> the active count
        N_active(b) shrinks over the epoch.
      - DECODE targets for an active segment are ALL of that segment's samples
        (constant across batches), zero-padded to Lmax with a boolean mask.

    Per-epoch reshuffle guarantee (D3): each epoch's permutation for a segment is
    guaranteed different from the immediately previous epoch's (rejection resample;
    skipped for length-1 segments, which have only one ordering).

    Per-batch output dict (all tensors float32 unless noted), with Na = N_active(b):
        q_enc, p_enc   : (Na, n)          encode observation (positions, momenta)
        t_enc          : (Na, 1)          encode time
        q_tgt, p_tgt   : (Na, Lmax, n)    decode targets (full segment, zero-padded)
        t_pred         : (Na, Lmax, 1)    query times (zero-padded)
        mask           : (Na, Lmax) bool  True on real samples, False on padding
        segment_ids    : (Na,) long       segment id per row (for logging/joins)
        trajectory_ids : (Na,) long       trajectory id per row
        encode_indices : (Na,) long       which in-segment sample index was encoded

    Notes:
      - Time is assumed already segment-normalised in the dataframe (starts at ~0);
        the loader passes t through unchanged.
      - Rows within a segment are sorted by time_col (stable sort) so decode targets
        are chronological regardless of dataframe row order.
      - tau_L / energy are intentionally NOT emitted; segment_ids/trajectory_ids let
        later steps join back to a segment-info dataframe for those.
    """

    def __init__(self, seg_df, pos_cols=('x', 'y'), mom_cols=('px', 'py'),
                 time_col='t', seg_col='segment_id', traj_col='trajectory_id',
                 seed=42, device='cpu'):
        if len(pos_cols) != len(mom_cols):
            raise ValueError(f"pos_cols ({len(pos_cols)}) and mom_cols ({len(mom_cols)}) "
                             f"must have the same length n")
        self.n = len(pos_cols)
        self.pos_cols = list(pos_cols)
        self.mom_cols = list(mom_cols)
        self.time_col = time_col
        self.seg_col = seg_col
        self.traj_col = traj_col
        self.device = device
        self.rng = np.random.default_rng(seed)

        # ----- group by segment, sort each by time -----
        self.seg_ids = sorted(int(s) for s in seg_df[seg_col].unique())
        self.S = len(self.seg_ids)

        self.q_full, self.p_full, self.t_full = [], [], []
        self.len, self.seg_id_arr, self.traj_id_arr = [], [], []
        for sid in self.seg_ids:
            sub = seg_df[seg_df[seg_col] == sid].sort_values(time_col, kind='mergesort')
            self.q_full.append(torch.tensor(sub[self.pos_cols].to_numpy(), dtype=torch.float32))
            self.p_full.append(torch.tensor(sub[self.mom_cols].to_numpy(), dtype=torch.float32))
            self.t_full.append(torch.tensor(sub[time_col].to_numpy(), dtype=torch.float32).unsqueeze(1))
            self.len.append(len(sub))
            self.seg_id_arr.append(int(sid))
            self.traj_id_arr.append(int(sub[traj_col].iloc[0]))
        self.len = np.asarray(self.len, dtype=np.int64)
        self.Lmax = int(self.len.max())

        # ----- precompute padded decode blocks + masks (constant per segment) -----
        S, Lmax, n = self.S, self.Lmax, self.n
        self.Q_pad = torch.zeros(S, Lmax, n, dtype=torch.float32)
        self.P_pad = torch.zeros(S, Lmax, n, dtype=torch.float32)
        self.T_pad = torch.zeros(S, Lmax, 1, dtype=torch.float32)
        self.mask = torch.zeros(S, Lmax, dtype=torch.bool)
        for j in range(S):
            L = int(self.len[j])
            self.Q_pad[j, :L] = self.q_full[j]
            self.P_pad[j, :L] = self.p_full[j]
            self.T_pad[j, :L] = self.t_full[j]
            self.mask[j, :L] = True
        self.seg_id_tensor = torch.tensor(self.seg_id_arr, dtype=torch.long)
        self.traj_id_tensor = torch.tensor(self.traj_id_arr, dtype=torch.long)

        # active segment indices per batch (len > b)
        self.active = [np.nonzero(self.len > b)[0] for b in range(self.Lmax)]

        # epoch-0 permutations (no previous epoch to differ from)
        self._prev_perms = None
        self.perms = self._draw_permutations(guarantee_diff=False)

    def _draw_permutations(self, guarantee_diff=True):
        perms = []
        for j in range(self.S):
            L = int(self.len[j])
            perm = self.rng.permutation(L)
            if guarantee_diff and self._prev_perms is not None and L >= 2:
                while np.array_equal(perm, self._prev_perms[j]):
                    perm = self.rng.permutation(L)
            perms.append(perm)
        self._prev_perms = perms
        return perms

    def on_epoch_start(self):
        """Redraw per-segment permutations, guaranteed different from previous epoch."""
        self.perms = self._draw_permutations(guarantee_diff=True)
        return len(self)

    def __len__(self):
        return self.Lmax

    def get_batch(self, b):
        active = self.active[b]                      # np array of segment indices, len > b
        idx = torch.as_tensor(active, dtype=torch.long)

        enc_pos = [int(self.perms[j][b]) for j in active]   # in-segment index encoded
        q_enc = torch.stack([self.q_full[j][k] for j, k in zip(active, enc_pos)])  # (Na, n)
        p_enc = torch.stack([self.p_full[j][k] for j, k in zip(active, enc_pos)])
        t_enc = torch.stack([self.t_full[j][k] for j, k in zip(active, enc_pos)])  # (Na, 1)

        batch = {
            'q_enc': q_enc, 'p_enc': p_enc, 't_enc': t_enc,
            'q_tgt': self.Q_pad[idx], 'p_tgt': self.P_pad[idx], 't_pred': self.T_pad[idx],
            'mask': self.mask[idx],
            'segment_ids': self.seg_id_tensor[idx],
            'trajectory_ids': self.traj_id_tensor[idx],
            'encode_indices': torch.tensor(enc_pos, dtype=torch.long),
        }
        if self.device != 'cpu':
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        return batch

    def __getitem__(self, b):
        return self.get_batch(b)

    def __iter__(self):
        for b in range(len(self)):
            yield self.get_batch(b)






def prediction_loss_masked(q_hat, p_hat, q_tgt, p_tgt, mask, eps=1e-12):
    """
    Masked MAE prediction loss (paper L_pred, generalised to variable-length segments).

    Paper:  L_pred = (1 / (N * N_pred * n)) * sum_{j,m,i} (|q_hat - q| + |p_hat - p|)

    With variable-length segments, N_pred is replaced by a per-segment valid count
    encoded in `mask`; we use decision (a): one global denominator = total number of
    VALID (segment, query-time, DOF) triples. Reduces exactly to the paper formula
    when all segments have equal length (mask all-True).

    Args:
        q_hat, p_hat : (N, Lmax, n)  decoder predictions at the query times
        q_tgt, p_tgt : (N, Lmax, n)  ground-truth targets (padded)
        mask         : (N, Lmax) bool/float  True on real samples, False on padding
    Returns:
        scalar loss
    """
    n = q_hat.shape[-1]
    m3 = mask.unsqueeze(-1).to(q_hat.dtype)                 # (N, Lmax, 1)
    abs_err = (q_hat - q_tgt).abs() + (p_hat - p_tgt).abs()  # (N, Lmax, n)
    numer = (abs_err * m3).sum()
    denom = mask.sum().to(q_hat.dtype) * n                  # count of valid (j,m,i) triples
    return numer / (denom + eps)


def hsic_loss_general(Q, P, use_unbiased=True, epsilon=1e-10, verbose=False):
    """
    Per-DOF HSIC independence loss, vectorised over degrees of freedom.

    For each DOF i in {1..n}, computes HSIC(Q^(i), P^(i)) over the N segments, then
    averages over the n DOF (paper L_HSIC = (1/n) sum_i HSIC(Q^(i), P^(i))).

    RBF kernel + median-heuristic bandwidth follow the ORIGINAL n=1 code (decision 1):
    bandwidth_sq = median{||.||^2 : r<s} (detached), K = exp(-0.5 * ||.||^2 / bandwidth_sq),
    diagonal zeroed for the unbiased estimator. The unbiased estimator matches the paper's
    Algorithm 1(c). Each per-DOF HSIC is clamped to >= 0 before averaging (matching the
    old code's per-call clamp); at n=1 this reduces exactly to the old single-DOF HSIC.

    Args:
        Q, P : (N, n)   encoder-output latents (one value per segment per DOF)
        use_unbiased : True -> unbiased estimator (needs N >= 4); False -> biased (N >= 2)
    Returns:
        scalar loss (0.0, gradient-connected, if N below the minimum)
    """
    if Q.shape != P.shape:
        raise ValueError(f"Q and P must have the same shape. Got {Q.shape} vs {P.shape}")
    if Q.dim() != 2:
        raise ValueError(f"Q, P must be 2D (N, n). Got {Q.dim()}D")

    N, n = Q.shape
    min_batch = 4 if use_unbiased else 2
    if N < min_batch:
        if verbose:
            print(f"Number of segments {N} < {min_batch}, returning 0.0 at hsic loss")
        return Q.sum() * 0.0 + P.sum() * 0.0

    A = Q.t().contiguous()   # (n, N)
    B = P.t().contiguous()   # (n, N)

    def batched_dist_sq(M):  # M: (n, N) -> (n, N, N), Gram method (mirrors old code)
        zz = torch.bmm(M.unsqueeze(2), M.unsqueeze(1))        # (n, N, N): M[i,r]*M[i,s]
        zsq = M * M                                           # (n, N)
        d = zsq.unsqueeze(2) + zsq.unsqueeze(1) - 2.0 * zz    # (n, N, N)
        return F.relu(d)

    DQ = batched_dist_sq(A)
    DP = batched_dist_sq(B)

    triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=Q.device), diagonal=1)
    eps_t = torch.tensor(epsilon, dtype=Q.dtype, device=Q.device)

    def bandwidth(D):  # per-DOF median of off-diagonal (r<s), detached, floored at epsilon
        off = D[:, triu]                                     # (n, M)
        med = off.detach().median(dim=1).values             # (n,)
        return torch.maximum(med, eps_t).view(n, 1, 1)

    K = torch.exp(-0.5 * DQ / bandwidth(DQ))                 # (n, N, N)
    L = torch.exp(-0.5 * DP / bandwidth(DP))

    b = float(N)
    if use_unbiased:
        offdiag = (~torch.eye(N, dtype=torch.bool, device=Q.device)).to(Q.dtype)  # (N,N)
        K = K * offdiag                                      # zero diagonal (autograd-safe)
        L = L * offdiag
        KL = torch.bmm(K, L)                                 # (n, N, N)
        K_sum = K.sum(dim=(1, 2))                            # (n,)
        L_sum = L.sum(dim=(1, 2))
        KL_sum = KL.sum(dim=(1, 2))
        KL_trace = torch.diagonal(KL, dim1=1, dim2=2).sum(dim=1)   # (n,)
        hsic = (KL_trace + (K_sum * L_sum) / ((b - 1) * (b - 2))
                - 2.0 * KL_sum / (b - 2)) / (b * (b - 3))    # (n,)
    else:
        K_mean = K.mean(dim=(1, 2), keepdim=True)            # (n,1,1)
        L_mean = L.mean(dim=(1, 2), keepdim=True)
        K_row = K.mean(dim=2, keepdim=True)                  # (n,N,1)
        L_row = L.mean(dim=2, keepdim=True)
        Kc = K - K_row - K_row.transpose(1, 2) + K_mean
        Lc = L - L_row - L_row.transpose(1, 2) + L_mean
        hsic = (Kc * Lc).sum(dim=(1, 2)) / (b * b)           # (n,)

    hsic = torch.maximum(hsic, torch.zeros_like(hsic))       # per-DOF clamp
    return hsic.mean()


 
def compute_total_loss_general(prediction_loss, hsic_loss, lambda_pred,
                               pred_scale, hsic_slope):
    """
    Total training loss (paper Eq. L_total), same algebra as the original
    compute_total_loss with paper-aligned names.
 
        L_total = lambda_pred * (L_pred / pred_scale) + hsic_slope * L_HSIC
 
    where, matching the original code's naming:
        lambda_pred  = prediction_coefficient          (paper lambda_pred)
        pred_scale   = prediction_loss_scale = L_bar_pred (Phase-I calibration const)
        hsic_slope   = hsic_loss_max_want / H_max = lambda_HSIC / H_max
                       (so hsic_slope * L_HSIC = lambda_HSIC * L_HSIC / H_max)
 
    Gradients flow through prediction_loss and hsic_loss; the scales/coeffs are
    fixed floats.
    """
    prediction_loss_scaled = prediction_loss / pred_scale
    hsic_loss_scaled = hsic_loss * hsic_slope
    return hsic_loss_scaled + lambda_pred * prediction_loss_scaled
 
 
def hsic_max_for_N(N, cache=None, floor=0.05, use_unbiased=True, epsilon=1e-10):
    """
    Per-batch HSIC normalisation constant H_max (Decision A2: depends on N_active).
 
    Reference signal s = [1, 2, ..., N]; H_max = max(HSIC(s, s), floor). Because s is
    perfectly (identity) dependent with itself, this is the maximal-dependence
    reference the paper uses to normalise L_HSIC. Cached by N.
 
    Uses hsic_loss_general on a single-DOF (N,1) reference, which equals the old
    hsic_loss_statistics_only(s, s) exactly (verified).
 
    Args:
        N     : number of segments in the batch (N_active)
        cache : optional dict {N: H_max} to memoise across batches
    Returns:
        float H_max
    """
    if cache is not None and N in cache:
        return cache[N]
    s = torch.arange(1, N + 1, dtype=torch.float32).view(N, 1)
    h = hsic_loss_general(s, s, use_unbiased=use_unbiased, epsilon=epsilon).item()
    h = max(h, floor)
    if cache is not None:
        cache[N] = h
    return h

import os
import json
from datetime import datetime




# ---------------------------------------------------------------------------
# Checkpointing (kept nearly verbatim from the original; model-agnostic via
# state_dict(), so it works unchanged for the general model).
# ---------------------------------------------------------------------------
def save_checkpoint(path, mapping_net, optimizer, scheduler, epoch, best_val_loss, extra_info=None):
    """Save a checkpoint with all model states and training information."""
    state_dict = {
        'epoch': epoch,
        'model': mapping_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }
    if extra_info is not None:
        state_dict['extra_info'] = extra_info
    if scheduler is not None:
        state_dict['scheduler'] = scheduler.state_dict()
    if best_val_loss is not None:
        state_dict['best_val_loss'] = best_val_loss
    torch.save(state_dict, path)


def load_checkpoint(path, mapping_net, device, optimizer=None, scheduler=None):
    """Load a checkpoint into models and training state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if 'model' in checkpoint:
        try:
            mapping_net.load_state_dict(checkpoint['model'])
            print("Successfully loaded model")
        except Exception as e:
            print(f"Failed to load model, because of: {e}")

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Successfully loaded optimizer, beginning with learning rate {optimizer.param_groups[0]['lr']}")

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Successfully loaded scheduler")

    return (
        checkpoint.get('epoch', -1),
        checkpoint.get('extra_info', None),
        checkpoint.get('best_val_loss', None),
    )


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------
def add_gaussian_noise(q, p, variance=0.1):
    """
    Add zero-mean Gaussian noise (paper: sigma^2 to each observation) to the encode
    inputs. Operates on (N, n) tensors; targets are NOT noised by the caller.
    """
    std = float(variance) ** 0.5
    return q + torch.randn_like(q) * std, p + torch.randn_like(p) * std


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    else:
        return obj


# ---------------------------------------------------------------------------
# Hyperparameter collection for reproducible runs.
#
# The general model stores only {n, n_layers, layers}; the rest of its
# configuration is not persisted as attributes. So we:
#   (1) recover everything reliably readable from the built model/layers, and
#   (2) log an optional `model_config` dict (the exact kwargs used to build the
#       net) verbatim -- this is the source of truth for reproducibility.
# We also snapshot the ACTUAL per-layer scaling parameters (a, gamma, c1, c2),
# which capture what was really initialised regardless of the init scheme.
# ---------------------------------------------------------------------------
def _recover_hidden_activation(mlp):
    """Activation types used on hidden layers of an MLP (excludes Linear/LayerNorm)."""
    acts = {type(m).__name__ for m in mlp if not isinstance(m, (nn.Linear, nn.LayerNorm))}
    return sorted(acts)


def _recover_output_is_linear(mlp):
    """True if the MLP's output layer is linear (last module is nn.Linear, no trailing activation)."""
    modules = list(mlp)
    return isinstance(modules[-1], nn.Linear)


def collect_run_hyperparameters(mapping_net, loader=None, model_config=None,
                                train_config=None, include_param_snapshot=True):
    """
    Assemble a JSON-serialisable dict fully describing a training run.

    Args:
        mapping_net : SimpleStackedHamiltonianNetwork_general
        loader      : SegmentDataLoader_general (optional) -> logs data identity
        model_config: dict of exact model-construction kwargs (optional but
                      recommended; logged verbatim as the reproducibility source)
        train_config: dict of training/loss/optimiser hyperparameters (logged verbatim)
        include_param_snapshot: if True, record actual per-layer a/gamma/c1/c2
    """
    hp = {}

    # ----- recoverable-from-model -----
    hp['model_n'] = int(mapping_net.n)
    hp['model_n_layers'] = int(mapping_net.n_layers)
    hp['model_trainable_parameter_count'] = count_parameters(mapping_net)
    L0 = mapping_net.layers[0]
    s1 = L0.step_1
    hp['model_hidden_dims'] = list(s1.hidden_dims)
    hp['model_a_eps_min'] = float(s1.a_eps_min)
    hp['model_a_eps_max'] = float(s1.a_eps_max)
    hp['model_a_k'] = float(s1.a_k)
    hp['model_hidden_activation'] = _recover_hidden_activation(s1.G_network)
    # final-layer output linearity (paper: the final HJL's MLPs use a linear output)
    last_s1 = mapping_net.layers[-1].step_1
    hp['model_final_layer_output_linear'] = _recover_output_is_linear(last_s1.G_network)
    hp['model_intermediate_layer_output_linear'] = _recover_output_is_linear(s1.G_network)

    # ----- verbatim config dicts (the real reproducibility record) -----
    if model_config is not None:
        hp['model_config'] = dict(model_config)
    if train_config is not None:
        hp['train_config'] = dict(train_config)

    # ----- loader identity -----
    if loader is not None:
        hp['data_n'] = int(getattr(loader, 'n', -1))
        hp['data_num_segments'] = int(getattr(loader, 'S', -1))
        hp['data_Lmax_num_batches'] = int(getattr(loader, 'Lmax', -1))
        hp['data_pos_cols'] = list(getattr(loader, 'pos_cols', []))
        hp['data_mom_cols'] = list(getattr(loader, 'mom_cols', []))
        hp['data_time_col'] = getattr(loader, 'time_col', None)
        if hasattr(loader, 'len'):
            lengths = np.asarray(loader.len)
            uniq, cnt = np.unique(lengths, return_counts=True)
            hp['data_segment_length_histogram'] = {int(u): int(c) for u, c in zip(uniq, cnt)}

    # ----- actual initialised scaling params (per layer) -----
    if include_param_snapshot:
        snap = []
        for i, layer in enumerate(mapping_net.layers):
            snap.append({
                'layer': i,
                'step_1': {'a': float(layer.step_1.a.detach()),
                           'gamma': float(layer.step_1.gamma.detach()),
                           'c1': layer.step_1.c1.detach().cpu().tolist(),
                           'c2': layer.step_1.c2.detach().cpu().tolist()},
                'step_2': {'a': float(layer.step_2.a.detach()),
                           'gamma': float(layer.step_2.gamma.detach()),
                           'c1': layer.step_2.c1.detach().cpu().tolist(),
                           'c2': layer.step_2.c2.detach().cpu().tolist()},
            })
        hp['initial_scaling_param_snapshot'] = snap

    return make_json_serializable(hp)





def broadcast_decode(inverse_net, Q, P, t_pred):
    """
    Broadcast per-segment latents across query times and decode in parallel.

    Args:
        inverse_net : InverseStackedHamiltonianNetwork_general (decoder)
        Q, P        : (N, n)        per-segment conserved latents (encode output)
        t_pred      : (N, Lmax, 1)  query times per segment (padded)
    Returns:
        q_hat, p_hat : (N, Lmax, n)  predictions at every query time
    """
    N, Lmax, _ = t_pred.shape
    n = Q.shape[1]
    # broadcast latents along the query-time axis: (N, n) -> (N, Lmax, n)
    Q_full = Q.unsqueeze(1).expand(N, Lmax, n)
    P_full = P.unsqueeze(1).expand(N, Lmax, n)
    # flatten (N, Lmax, .) -> (N*Lmax, .) for the decoder's (batch, n) contract
    Q_flat = Q_full.reshape(N * Lmax, n)
    P_flat = P_full.reshape(N * Lmax, n)
    t_flat = t_pred.reshape(N * Lmax, 1)
    q_hat_flat, p_hat_flat, _ = inverse_net(Q_flat, P_flat, t_flat)
    # reshape back to (N, Lmax, n)
    q_hat = q_hat_flat.reshape(N, Lmax, n)
    p_hat = p_hat_flat.reshape(N, Lmax, n)
    return q_hat, p_hat


def training_step(batch, mapping_net, inverse_net, *,
                  lambda_pred, lambda_hsic, pred_scale,
                  add_noise=False, noise_variance=0.1,
                  hsic_cache=None, use_unbiased_hsic=True):
    """
    One paper Algorithm-1 Phase-II forward pass (loss only; caller does backward/step).

    Steps:
      1. (optional) add Gaussian noise to the encode inputs q_enc, p_enc.
      2. Encode: Q, P = M(q_enc, p_enc, t_enc)   -> (N, n)
      3. HSIC on the encode latents (Decision 4): L_HSIC = hsic_loss_general(Q, P)
      4. Broadcast + decode at all query times -> q_hat, p_hat : (N, Lmax, n)
      5. Masked MAE prediction loss over valid (segment, time, dof) entries.
      6. Total loss with per-batch H_max (Decision A2): hsic_slope = lambda_hsic / H_max.

    Returns: (total_loss, hsic_loss, prediction_loss)  -- all tensors with grad.
    """
    q_enc, p_enc, t_enc = batch['q_enc'], batch['p_enc'], batch['t_enc']
    q_tgt, p_tgt, t_pred = batch['q_tgt'], batch['p_tgt'], batch['t_pred']
    mask = batch['mask']

    if add_noise:
        q_enc, p_enc = add_gaussian_noise(q_enc, p_enc, variance=noise_variance)

    # encode -> latents (N, n)
    Q, P, _ = mapping_net(q_enc, p_enc, t_enc)

    # independence loss on the encode latents
    hsic = hsic_loss_general(Q, P, use_unbiased=use_unbiased_hsic)

    # broadcast + decode + masked MAE
    q_hat, p_hat = broadcast_decode(inverse_net, Q, P, t_pred)
    pred = prediction_loss_masked(q_hat, p_hat, q_tgt, p_tgt, mask)

    # per-batch H_max (A2): reference signal of length N_active
    N = Q.shape[0]
    H_max = hsic_max_for_N(N, cache=hsic_cache, use_unbiased=use_unbiased_hsic)
    hsic_slope = lambda_hsic / H_max

    total = compute_total_loss_general(pred, hsic, lambda_pred=lambda_pred,
                                       pred_scale=pred_scale, hsic_slope=hsic_slope)
    return total, hsic, pred


def make_validation_batch(val_seg_df, pos_cols=('x', 'y'), mom_cols=('px', 'py'),
                          time_col='t', seg_col='segment_id', traj_col='trajectory_id',
                          device='cpu'):
    """
    Build the single deterministic validation batch (paper val criterion):
    encode from the FIRST (earliest-time) sample of each segment; decode ALL samples.

    Returns a batch dict with the same keys/shapes as SegmentDataLoader_general.get_batch,
    with N = number of validation segments and one fixed (non-random) encode point.
    """
    n = len(pos_cols)
    seg_ids = sorted(int(s) for s in val_seg_df[seg_col].unique())
    lengths, q_full, p_full, t_full, traj_ids = [], [], [], [], []
    for sid in seg_ids:
        sub = val_seg_df[val_seg_df[seg_col] == sid].sort_values(time_col, kind='mergesort')
        q_full.append(torch.tensor(sub[list(pos_cols)].to_numpy(), dtype=torch.float32))
        p_full.append(torch.tensor(sub[list(mom_cols)].to_numpy(), dtype=torch.float32))
        t_full.append(torch.tensor(sub[time_col].to_numpy(), dtype=torch.float32).unsqueeze(1))
        lengths.append(len(sub))
        traj_ids.append(int(sub[traj_col].iloc[0]))

    S = len(seg_ids)
    Lmax = int(max(lengths))
    Q_pad = torch.zeros(S, Lmax, n); P_pad = torch.zeros(S, Lmax, n)
    T_pad = torch.zeros(S, Lmax, 1); mask = torch.zeros(S, Lmax, dtype=torch.bool)
    q_enc = torch.zeros(S, n); p_enc = torch.zeros(S, n); t_enc = torch.zeros(S, 1)
    for j in range(S):
        Lj = lengths[j]
        Q_pad[j, :Lj] = q_full[j]; P_pad[j, :Lj] = p_full[j]
        T_pad[j, :Lj] = t_full[j]; mask[j, :Lj] = True
        q_enc[j] = q_full[j][0]           # first (earliest-time) sample
        p_enc[j] = p_full[j][0]
        t_enc[j] = t_full[j][0]

    batch = {
        'q_enc': q_enc, 'p_enc': p_enc, 't_enc': t_enc,
        'q_tgt': Q_pad, 'p_tgt': P_pad, 't_pred': T_pad, 'mask': mask,
        'segment_ids': torch.tensor(seg_ids, dtype=torch.long),
        'trajectory_ids': torch.tensor(traj_ids, dtype=torch.long),
        'encode_indices': torch.zeros(S, dtype=torch.long),   # always the first sample
    }
    if device != 'cpu':
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return batch


def validation_step(val_batch, mapping_net, inverse_net):
    """
    Paper validation criterion: encode first point of each segment, decode all points,
    masked MAE. Returns a scalar prediction-loss tensor (no HSIC, no grad needed).
    """
    Q, P, _ = mapping_net(val_batch['q_enc'], val_batch['p_enc'], val_batch['t_enc'])
    q_hat, p_hat = broadcast_decode(inverse_net, Q, P, val_batch['t_pred'])
    return prediction_loss_masked(q_hat, p_hat, val_batch['q_tgt'],
                                  val_batch['p_tgt'], val_batch['mask'])

def validation_step_per_segment(val_batch, mapping_net, inverse_net):
    """
    Per-segment version of validation_step: same deterministic first-point encode +
    decode-all forward pass, but the masked MAE is NOT reduced across segments.
 
    Returns:
        seg_ids : (N,) long tensor of segment ids (from val_batch['segment_ids'])
        losses  : (N,) tensor, per-segment masked MAE:
                  loss_j = sum_{m,i} M_jm (|dq| + |dp|) / (n * sum_m M_jm)
 
    The mean of `losses` over segments equals the scalar validation_step loss ONLY when
    all segments share the same valid-count; with variable lengths they differ (the
    aggregate weights every observation equally, this weights every segment equally).
    Both are reported so the per-segment view is available without changing the criterion.
    """
    Q, P, _ = mapping_net(val_batch['q_enc'], val_batch['p_enc'], val_batch['t_enc'])
    q_hat, p_hat = broadcast_decode(inverse_net, Q, P, val_batch['t_pred'])
 
    q_tgt, p_tgt, mask = val_batch['q_tgt'], val_batch['p_tgt'], val_batch['mask']
    n = q_hat.shape[-1]
    m3 = mask.unsqueeze(-1).to(q_hat.dtype)                       # (N, Lmax, 1)
    abs_err = ((q_hat - q_tgt).abs() + (p_hat - p_tgt).abs()) * m3  # (N, Lmax, n)
    per_seg_sum = abs_err.sum(dim=(1, 2))                          # (N,)
    valid_counts = mask.sum(dim=1).to(q_hat.dtype) * n            # (N,)  = n * sum_m M_jm
    per_seg_loss = per_seg_sum / valid_counts.clamp_min(1e-12)     # (N,)
    return val_batch['segment_ids'], per_seg_loss.detach()
 



import os
import sys
import json
import time
import signal
import traceback
 
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
 

 
 
# ---------------------------------------------------------------------------
# Parameter grouping (Decision C): weight decay on MLP (G/F network) params only;
# zero weight decay on HJL scaling params (a_raw, gamma, c1, c2).
# ---------------------------------------------------------------------------
def split_param_groups(mapping_net):
    mlp_params, scaling_params = [], []
    for name, p in mapping_net.named_parameters():
        if ('G_network' in name) or ('F_network' in name):
            mlp_params.append(p)
        else:  # a_raw, gamma, c1, c2
            scaling_params.append(p)
    return mlp_params, scaling_params
 
 
# ---------------------------------------------------------------------------
# Phase I calibration: L_bar_pred (mean over calibration batches, or median if the
# per-batch losses are "too noisy": |mean/std| < noise_threshold). Forward-only.
# (H_max is per-batch/Decision A2, computed live in training_step; not part of Phase I.)
# ---------------------------------------------------------------------------
def calibrate_scales(train_loader, mapping_net, inverse_net, *, add_noise=False,
                     noise_variance=0.1, noise_threshold=2.0, device='cpu', save_path=None):
    import numpy as np
    mapping_net.eval()
    pred_losses = []
    # NOTE: cannot use torch.no_grad() here -- the model computes dG/dq via autograd
    # internally, which requires grad to be enabled. We just read scalar .item() values
    # and never call backward, so no training update happens (forward-only in effect).
    for b in range(len(train_loader)):
        batch = train_loader.get_batch(b)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        q_enc, p_enc, t_enc = batch['q_enc'], batch['p_enc'], batch['t_enc']
        if add_noise:
            q_enc, p_enc = add_gaussian_noise(q_enc, p_enc, variance=noise_variance)
        Q, P, _ = mapping_net(q_enc, p_enc, t_enc)
        q_hat, p_hat = broadcast_decode(inverse_net, Q, P, batch['t_pred'])
        pl = prediction_loss_masked(q_hat, p_hat, batch['q_tgt'], batch['p_tgt'], batch['mask'])
        pred_losses.append(pl.item())
 
    mean = float(np.mean(pred_losses))
    std = float(np.std(pred_losses))
    if std > 0 and abs(mean / std) < noise_threshold:
        pred_scale = float(np.median(pred_losses))
        print(f"Calibration L_pred: {mean:.4f}+/-{std:.4f} (noisy: |mean/std|<{noise_threshold}); using median {pred_scale:.4f}")
    else:
        pred_scale = mean
        print(f"Calibration L_pred: {mean:.4f}+/-{std:.4f}; using mean")

    if save_path is not None:
        import pickle
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"pred_scale": pred_scale}, f)
        print(f"Saved pred_scale to {save_path}")

    return pred_scale
 
 
# ---------------------------------------------------------------------------
# Main training loop.
# ---------------------------------------------------------------------------
def train_model_general(
    train_loader,
    mapping_net,
    *,
    # validation batches (deterministic first-point batches from make_validation_batch)
    val_batch=None,
    val_train_batch=None,
    # loss hyperparameters (paper names)
    lambda_pred,               # paper lambda_pred
    lambda_hsic,               # paper lambda_HSIC
    pred_scale=None,           # L_bar_pred; if None, computed via calibrate_scales
    add_noise=False,
    noise_variance=0.1,
    noise_threshold=2.0,
    use_unbiased_hsic=True,
    # model construction record (for reproducible logging)
    model_config=None,
    # optimiser
    optimizer_type='AdamW',
    learning_rate=1e-4,
    weight_decay=1e-6,
    scheduler_type='plateau',
    scheduler_params=None,
    grad_clip_value=1.0,
    # training control
    num_epochs=30,
    reshuffle_each_epoch=True,
    early_stopping=True,
    patience=5,
    min_delta=1e-3,
    # checkpointing / logging
    save_dir='./checkpoints',
    save_best_only=True,
    save_freq_epochs=1,
    auto_rescue=True,
    log_freq_batches=10,
    verbose=1,
    device='cpu',
    # resume
    optimizer=None,
    scheduler=None,
    continue_from_epoch=None,
    best_validation_criterio_loss_till_now=None,
):
    """
    Train the general HJN. Assembles the verified parts: Phase-I calibration,
    per-batch training_step, two-group AdamW (MLP weight-decay only), grad clip,
    NaN/Inf grad scrubbing, plateau scheduler, both-loader validation, checkpoint/
    rescue/hyperparameter logging.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    mapping_net.to(device)
    os.makedirs(save_dir, exist_ok=True)
 
    inverse_net = InverseStackedHamiltonianNetwork_general(mapping_net).to(device)
 
    mlp_params, scaling_params = split_param_groups(mapping_net)
 
    if optimizer is None:
        if optimizer_type == 'AdamW':
            optimizer = AdamW([
                {'params': mlp_params, 'weight_decay': weight_decay},
                {'params': scaling_params, 'weight_decay': 0.0},
            ], lr=learning_rate)
        elif optimizer_type == 'Adam':
            optimizer = Adam([
                {'params': mlp_params}, {'params': scaling_params},
            ], lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
        print(f"Initialised {optimizer_type} (lr={learning_rate}, MLP wd={weight_decay}, scaling wd=0)")
 
    if scheduler is None:
        if scheduler_type == 'plateau':
            scheduler_params = scheduler_params or {'mode': 'min', 'factor': 0.1, 'patience': 5}
            scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")
 
    # Phase I calibration if not supplied
    if pred_scale is None:
        pred_scale = calibrate_scales(train_loader, mapping_net, inverse_net,
                                      add_noise=add_noise, noise_variance=noise_variance,
                                      noise_threshold=noise_threshold, device=device)
    print(f"Using L_bar_pred (pred_scale) = {pred_scale:.6f}")
 
    hsic_cache = {}
    best_val_loss = float('inf') if best_validation_criterio_loss_till_now is None \
        else best_validation_criterio_loss_till_now
    patience_counter = 0
    start_epoch = 0 if continue_from_epoch is None else continue_from_epoch
 
    # module-level state for the rescue handler
    epoch = start_epoch
    batch_idx = 0
 
    def _log_run(tag):
        hp = collect_run_hyperparameters(
            mapping_net, loader=train_loader, model_config=model_config,
            train_config=dict(lambda_pred=lambda_pred, lambda_hsic=lambda_hsic,
                              pred_scale=pred_scale, add_noise=add_noise,
                              noise_variance=noise_variance, use_unbiased_hsic=use_unbiased_hsic,
                              optimizer_type=optimizer_type, learning_rate=learning_rate,
                              weight_decay=weight_decay, scheduler_type=scheduler_type,
                              scheduler_params=scheduler_params, grad_clip_value=grad_clip_value,
                              num_epochs=num_epochs, reshuffle_each_epoch=reshuffle_each_epoch,
                              early_stopping=early_stopping, patience=patience, min_delta=min_delta))
        with open(os.path.join(save_dir, f"run_hyperparameters_{tag}.json"), "w") as f:
            json.dump(make_json_serializable(hp), f, indent=2)
 
    def save_rescue_checkpoint(signal_received=None, frame=None):
        try:
            rescue_path = os.path.join(save_dir, f"rescue_epoch_{epoch}_batch_{batch_idx}.pt")
            save_checkpoint(rescue_path, mapping_net, optimizer, scheduler, epoch, best_val_loss,
                            extra_info={'interrupted_at_batch': batch_idx,
                                        'exception': traceback.format_exc()})
            _log_run(f"rescue_epoch_{epoch}")
            print(f"Rescue checkpoint saved to {rescue_path}")
            if signal_received:
                sys.exit(0)
        except Exception as e:
            print(f"Failed to save rescue checkpoint: {e}")
 
    if auto_rescue:
        try:
            signal.signal(signal.SIGINT, save_rescue_checkpoint)
            signal.signal(signal.SIGTERM, save_rescue_checkpoint)
        except (ValueError, OSError):
            pass  # not in main thread (e.g. notebook worker); skip signal handlers


# seed the accumulator from any leftover file, keeping only rows BEFORE the resume point
    prog_path     = os.path.join(save_dir, "segment_progress.parquet")
    prog_csv_path = os.path.join(save_dir, "segment_progress.csv")
    segment_progress_rows = []
    _old = None
    try:
        if os.path.exists(prog_path):
            _old = pd.read_parquet(prog_path)
        elif os.path.exists(prog_csv_path):
            _old = pd.read_csv(prog_csv_path)
    except Exception as e:
        print(f"Could not load leftover segment_progress ({e}); starting fresh")
        _old = None
    if _old is not None:
        _old = _old[_old['epoch'] < start_epoch]
        segment_progress_rows = _old.to_dict('records')
        print(f"Seeded segment_progress with {len(segment_progress_rows)} leftover rows (epoch < {start_epoch})")

    history = []
    try:
        for epoch in range(start_epoch, start_epoch + num_epochs):
            if epoch > start_epoch and reshuffle_each_epoch:
                train_loader.on_epoch_start()
 
            mapping_net.train()
            t0 = time.time()
            tot_sum = hsic_sum = pred_sum = 0.0
            gradnorm_sum = 0.0
            clipped = 0
            n_batches = 0
 
            for batch_idx in range(len(train_loader)):
                batch = train_loader.get_batch(batch_idx)
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
 
                optimizer.zero_grad()
                total, hsic, pred = training_step(
                    batch, mapping_net, inverse_net,
                    lambda_pred=lambda_pred, lambda_hsic=lambda_hsic, pred_scale=pred_scale,
                    add_noise=add_noise, noise_variance=noise_variance,
                    hsic_cache=hsic_cache, use_unbiased_hsic=use_unbiased_hsic)
 
                if not torch.isfinite(total):
                    print(f"Non-finite total loss at epoch {epoch} batch {batch_idx}; skipping")
                    continue
 
                total.backward()
 
                # NaN/Inf gradient scrubbing

                for name, p in mapping_net.named_parameters():
                    if p.grad is not None:

                        if torch.isnan(p.grad).any():
                            print(f"NaN gradients detected in {name}")
                            p.grad[torch.isnan(p.grad)] = 0.0

                        if torch.isinf(p.grad).any():
                            print(f"Inf gradients detected in {name}")
                            p.grad[torch.isinf(p.grad)] = 0.0

                        if verbose > 1:
                            grad_norm_print = p.grad.norm().item()

                            if grad_norm_print > 60.0:
                                print(f"High gradient norm in {name}: {grad_norm_print}")

                            if grad_norm_print < 0.00001:
                                print(f"Low gradient norm in {name}: {grad_norm_print}")
 
                if grad_clip_value is not None:
                    gn = torch.nn.utils.clip_grad_norm_(mlp_params + scaling_params, grad_clip_value)
                    gradnorm_sum += float(gn)
                    if gn > grad_clip_value:
                        clipped += 1
                        if verbose > 0:
                            print(f"Grad clipping activated, grad_norm before clipping: {gn}")
 
                optimizer.step()
 
                tot_sum += total.item(); hsic_sum += hsic.item(); pred_sum += pred.item()
                n_batches += 1
                if verbose > 0 and batch_idx % log_freq_batches == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)} - Prediction Loss: {pred.item():.4f}")
 
            nb = max(1, n_batches)
            train_total, train_hsic, train_pred = tot_sum/nb, hsic_sum/nb, pred_sum/nb
            mean_gradnorm = gradnorm_sum/nb
            pct_clipped = 100.0*clipped/nb
 
            # ---- validation (paper criterion: first point -> predict rest -> masked MAE) ----
            mapping_net.eval()
            val_loss = float('nan'); val_train_loss = float('nan')
            # no torch.no_grad(): the model needs autograd internally for dG/dq. We read
            # .item() and never backward on these, so no validation gradient step occurs.
            if val_batch is not None:
                val_loss = validation_step(val_batch, mapping_net, inverse_net).item()
            if val_train_batch is not None:
                seg_ids_t, seg_losses_t = validation_step_per_segment(
                    val_train_batch, mapping_net, inverse_net)
                for sid, sl in zip(seg_ids_t.tolist(), seg_losses_t.tolist()):
                    segment_progress_rows.append(
                        {'epoch': epoch, 'segment_id': int(sid), 'val_train_pred_loss': float(sl)})
                # flush every epoch (append-safe): rewrite the full accumulated frame
                prog_df = pd.DataFrame(segment_progress_rows)
                prog_path = os.path.join(save_dir, "segment_progress.parquet")
                try:
                    prog_df.to_parquet(prog_path, index=False)
                except Exception:
                    prog_df.to_csv(os.path.join(save_dir, "segment_progress.csv"), index=False)
                val_train_loss = validation_step(val_train_batch, mapping_net, inverse_net).item()
 
            criterion = val_loss if val_batch is not None else train_pred
 
            epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            metrics = dict(epoch=epoch,
               train_total_loss_=train_total,
               train_hsic_loss_=train_hsic,
               train_prediction_loss_=train_pred,
               val_prediction_loss_=val_loss,
               val_prediction_loss_training_set=val_train_loss,
               validation_criterio=criterion,
               mean_grad_norm_=mean_gradnorm,
               pct_batches_clipped=pct_clipped,
               lr=optimizer.param_groups[0]['lr'])
            with open(os.path.join(epoch_dir, "epoch_metrics.json"), "w") as f:
                json.dump(make_json_serializable(metrics), f, indent=2)
            history.append(metrics)
 
            if criterion < best_val_loss - min_delta:
                best_val_loss = criterion
                print(f"Found best validation criterio loss till now:{criterion:.4f}, on epoch {epoch}")
                save_checkpoint(os.path.join(save_dir, "best_model.pt"),
                                mapping_net, optimizer, scheduler, epoch, best_val_loss)
                patience_counter = 0
                if verbose > 0:
                    print(f"  new best criterion {best_val_loss:.4f} at epoch {epoch}")
            else:
                if early_stopping:
                    patience_counter += 1
                    if verbose > 0:
                        print(f"Early stopping patience: {patience_counter}/{patience}")
 
            if scheduler is not None:
                scheduler.step(criterion)
 


            epoch_time = time.time() - t0

            if verbose > 0:
                print(f"\nEpoch {epoch}/{start_epoch+num_epochs} completed in {epoch_time:.2f}s")

                print(f"Percentage of batches clipped in epoch: {pct_clipped:.1f}%")

                if grad_clip_value is not None:
                    print(f"Mean grad norm: {mean_gradnorm:.4f}")

                print(f"Mean total train loss: {train_total:.4f}")

                print(f"Mean HSIC train loss: {train_hsic:.4f}")

                print(f"Mean prediction train loss: {train_pred:.4f}")

                print(f"Mean total val loss: {val_loss:.4f}")

                print(f"Mean total val loss training set: {val_train_loss:.4f}")

                print(f"Validation criterio: {criterion:.4f}")

                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
 
            if (epoch % save_freq_epochs == 0) and not save_best_only:
                save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"),
                                mapping_net, optimizer, scheduler, epoch, best_val_loss)
 
            if early_stopping and patience_counter >= patience:
                if verbose > 0:
                    print(f"Early stopping at epoch {epoch}")
                break
 
        _log_run(f"epochs_{start_epoch}_to_{epoch}")
 
    except Exception as e:
        if auto_rescue:
            print(f"Training interrupted by exception: {e}")
            save_rescue_checkpoint()
        raise
    finally:
        try:
            save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"),
                            mapping_net, optimizer, scheduler, epoch, best_val_loss)
        except Exception as e:
            print(f"Failed to save final checkpoint: {e}")
 
    if verbose > 0:
        print(f"Training complete. Best validation criterion: {best_val_loss:.4f}")
    return dict(history=history, best_val_loss=best_val_loss, pred_scale=pred_scale,
                optimizer=optimizer, scheduler=scheduler, inverse_net=inverse_net)
 


def resume_training_from_checkpoint_general(
    checkpoint_path,
    train_loader,
    mapping_net,
    *,
    val_batch=None,
    val_train_batch=None,
    # loss hyperparameters (paper names) -- pred_scale MUST be the ORIGINAL run's
    # calibrated L_bar_pred (do NOT recalibrate on resume; the model state has changed).
    lambda_pred,
    lambda_hsic,
    pred_scale,
    add_noise=False,
    noise_variance=0.1,
    use_unbiased_hsic=True,
    model_config=None,
    # === resume mode ===
    # Mode A (load_scheduler_and_optimizer=True): restore exact optimizer/scheduler state.
    #   learning_rate/weight_decay: None -> use checkpoint values, or override.
    #   optimizer_type/scheduler_type MUST match the original run.
    # Mode B (load_scheduler_and_optimizer=False): fresh optimizer/scheduler.
    #   learning_rate/weight_decay REQUIRED.
    load_scheduler_and_optimizer=False,
    learning_rate=None,
    weight_decay=None,
    optimizer_type='AdamW',
    scheduler_type='plateau',
    scheduler_params=None,
    reset_scheduler_patience=False,   # Mode A only: reset num_bad_epochs/cooldown
    # training control
    num_epochs=30,                    # ADDITIONAL epochs
    grad_clip_value=1.0,
    reshuffle_each_epoch=True,
    early_stopping=False,
    patience=20,
    min_delta=1e-3,
    # checkpointing / logging
    save_dir='./checkpoints',
    save_best_only=False,
    save_freq_epochs=2,
    auto_rescue=True,
    log_freq_batches=10,
    verbose=1,
    device='cpu',
):
    """
    Resume training from a checkpoint. Rebuilds the inverse net from mapping_net,
    restores state per the selected mode, and continues via train_model_general.
 
    Returns the same dict as train_model_general.
    """
    mlp_params, scaling_params = split_param_groups(mapping_net)
 
    if load_scheduler_and_optimizer:
        # ----- Mode A: reconstruct optimizer/scheduler, then load their state -----
        if optimizer_type == 'AdamW':
            temp_optimizer = AdamW([
                {'params': mlp_params, 'weight_decay': (weight_decay if weight_decay is not None else 0.0)},
                {'params': scaling_params, 'weight_decay': 0.0},
            ])
        elif optimizer_type == 'Adam':
            temp_optimizer = Adam([{'params': mlp_params}, {'params': scaling_params}])
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
 
        temp_scheduler = ReduceLROnPlateau(temp_optimizer, mode='min')  # overwritten by load
        epoch, extra_info, best_val_loss = load_checkpoint(
            checkpoint_path, mapping_net, device, temp_optimizer, temp_scheduler)
 
        # optional scheduler-param overrides
        if scheduler_params is not None:
            for key, value in scheduler_params.items():
                if hasattr(temp_scheduler, key):
                    setattr(temp_scheduler, key, value)
                    print(f"  scheduler.{key} -> {value}")
                else:
                    print(f"  warning: scheduler has no attribute '{key}'")
 
        if reset_scheduler_patience:
            temp_scheduler.num_bad_epochs = 0
            temp_scheduler.cooldown_counter = 0
            print(f"Reset scheduler patience; will wait full {temp_scheduler.patience} epochs before LR reduction.")
 
        optimizer, scheduler = temp_optimizer, temp_scheduler
    else:
        # ----- Mode B: fresh optimizer/scheduler; LR/WD required -----
        if learning_rate is None:
            raise ValueError("Mode B (load_scheduler_and_optimizer=False) requires learning_rate.")
        if weight_decay is None:
            raise ValueError("Mode B (load_scheduler_and_optimizer=False) requires weight_decay.")
        epoch, extra_info, best_val_loss = load_checkpoint(
            checkpoint_path, mapping_net, device, None, None)
        optimizer, scheduler = None, None   # train_model_general will build fresh
 
    # ----- determine the epoch to continue from -----
    if extra_info and 'interrupted_at_batch' in extra_info:
        continue_from_epoch = epoch          # incomplete epoch -> redo it
        print(f"Resuming incomplete epoch {epoch}")
    else:
        continue_from_epoch = epoch + 1      # epoch completed -> next one
        print(f"Epoch {epoch} completed; starting from epoch {epoch + 1}")
 
    # ----- resolve LR/WD for Mode A (None -> use checkpoint values) -----
    if load_scheduler_and_optimizer:
        if learning_rate is None:
            learning_rate = optimizer.param_groups[0]['lr']
            print(f"Using checkpoint learning rate {learning_rate}")
        if weight_decay is None:
            weight_decay = optimizer.param_groups[0]['weight_decay']
            print(f"Using checkpoint weight decay {weight_decay}")
        for g in optimizer.param_groups:
            g['lr'] = learning_rate
        optimizer.param_groups[0]['weight_decay'] = weight_decay
 
    print(f"Training {num_epochs} additional epochs "
          f"(epochs {continue_from_epoch} to {continue_from_epoch + num_epochs - 1})")
 
    return train_model_general(
        train_loader, mapping_net,
        val_batch=val_batch, val_train_batch=val_train_batch,
        lambda_pred=lambda_pred, lambda_hsic=lambda_hsic,
        pred_scale=pred_scale,                       # reuse original scale; no recalibration
        add_noise=add_noise, noise_variance=noise_variance,
        use_unbiased_hsic=use_unbiased_hsic, model_config=model_config,
        optimizer_type=optimizer_type, learning_rate=learning_rate, weight_decay=weight_decay,
        scheduler_type=scheduler_type, scheduler_params=scheduler_params,
        grad_clip_value=grad_clip_value,
        num_epochs=num_epochs, reshuffle_each_epoch=reshuffle_each_epoch,
        early_stopping=early_stopping, patience=patience, min_delta=min_delta,
        save_dir=save_dir, save_best_only=save_best_only, save_freq_epochs=save_freq_epochs,
        auto_rescue=auto_rescue, log_freq_batches=log_freq_batches, verbose=verbose, device=device,
        optimizer=optimizer, scheduler=scheduler,
        continue_from_epoch=continue_from_epoch,
        best_validation_criterio_loss_till_now=best_val_loss,
    )
 

"""
Notebook-side analysis of per-segment training progress.

Reads the long-format `segment_progress.parquet` (columns: epoch, segment_id,
val_train_pred_loss) written each epoch by train_model_general, joins it to
train_seg_id_df for per-segment structure (frac_regular, macro_tau_L), and
visualises how training progresses for chaotic vs regular segments.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def load_segment_progress(save_dir, seg_id_df, seg_col='segment_id'):
    """Load the progress file and join per-segment structure columns."""
    path = os.path.join(save_dir, "segment_progress.parquet")
    if not os.path.exists(path):
        path_csv = os.path.join(save_dir, "segment_progress.csv")
        if os.path.exists(path_csv):
            prog = pd.read_csv(path_csv)
        else:
            raise FileNotFoundError(f"No segment_progress.parquet/.csv in {save_dir}")
    else:
        prog = pd.read_parquet(path)
    keep = [c for c in [seg_col, 'frac_regular', 'macro_tau_L'] if c in seg_id_df.columns]
    merged = prog.merge(seg_id_df[keep], on=seg_col, how='left')
    return merged

from matplotlib.ticker import LogLocator, LogFormatter

def plot_segment_progress(merged, loss_col='val_train_pred_loss',
                          n_sample_lines=50, color_by='frac_regular',
                          n_bins=3, log_y=True, seed=0, figsize=(13, 5)):
    """
    Two panels:
      (left)  a subsample of individual segment curves, colour-coded by `color_by`
              (continuous [0,1] colormap; e.g. frac_regular).
      (right) binned aggregate curves (mean +/- std band) over `color_by` terciles,
              so the chaotic-vs-regular trend is legible despite many segments.
    All segments are used for the aggregate; only the left panel subsamples for clarity.
    """
    seg_ids = merged['segment_id'].unique()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)

    # ---------- LEFT: subsample of individual segment curves ----------
    has_color = color_by in merged.columns and merged[color_by].notna().any()
    if has_color:
        cvals = merged.groupby('segment_id')[color_by].first()
        norm = Normalize(vmin=float(np.nanmin(cvals)), vmax=float(np.nanmax(cvals)))
        cmap = cm.get_cmap('coolwarm_r')  # regular (high frac) -> cool, chaotic (low) -> warm
    # pick a subsample spanning the color range
    rng = np.random.default_rng(seed)
    if has_color:
        order = cvals.sort_values().index.to_numpy()
        idx = np.linspace(0, len(order) - 1, min(n_sample_lines, len(order))).astype(int)
        sample_ids = order[idx]
    else:
        sample_ids = rng.choice(seg_ids, size=min(n_sample_lines, len(seg_ids)), replace=False)

    for sid in sample_ids:
        d = merged[merged['segment_id'] == sid].sort_values('epoch')
        color = cmap(norm(cvals.loc[sid])) if has_color else 'gray'
        axL.plot(d['epoch'], d[loss_col], color=color, alpha=0.5, linewidth=0.8)
    axL.set_xlabel('epoch'); axL.set_ylabel(loss_col)
    axL.set_title(f'{len(sample_ids)} sampled segments'
                  + (f' (colour = {color_by})' if has_color else ''))
    if log_y:
        axL.set_yscale('log')
        axL.yaxis.set_major_locator(LogLocator(base=10))
        axL.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        axL.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False))
    axL.grid(True, ls='--', alpha=0.5)
    if has_color:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=axL, label=color_by)

    # ---------- RIGHT: binned aggregate curves over terciles of color_by ----------
    if has_color:
        cvals_all = merged.groupby('segment_id')[color_by].first()
        # tercile edges on the color variable
        qs = np.nanquantile(cvals_all.values, np.linspace(0, 1, n_bins + 1))
        qs[0] -= 1e-9; qs[-1] += 1e-9
        labels = [f'{color_by} [{qs[b]:.2f},{qs[b+1]:.2f}]' for b in range(n_bins)]
        seg_bin = np.digitize(cvals_all.values, qs[1:-1])  # 0..n_bins-1
        bin_of = dict(zip(cvals_all.index, seg_bin))
        merged = merged.copy()
        merged['_bin'] = merged['segment_id'].map(bin_of)
        _palette = ['#d62728', '#9467bd', '#1f77b4', '#2ca02c', '#ff7f0e', '#8c564b']
        bin_colors = [_palette[b % len(_palette)] for b in range(n_bins)]
        for b in range(n_bins):
            sub = merged[merged['_bin'] == b]
            g = sub.groupby('epoch')[loss_col]
            mean = g.mean(); std = g.std()
            axR.plot(mean.index, mean.values, color=bin_colors[b], label=labels[b], lw=1.8)
            axR.fill_between(mean.index, (mean-std).values, (mean+std).values,
                             color=bin_colors[b], alpha=0.15)
        axR.legend(fontsize=8)
        axR.set_title(f'mean +/- std by {color_by} tercile (all segments)')
    else:
        g = merged.groupby('epoch')[loss_col]
        axR.plot(g.mean().index, g.mean().values, color='tab:blue')
        axR.set_title('mean over all segments')
    axR.set_xlabel('epoch'); axR.set_ylabel(loss_col)
    if log_y:
        axR.set_yscale('log')
        axR.yaxis.set_major_locator(LogLocator(base=10))
        axR.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        axR.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False))
    axR.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    return fig


def summarise_segment_progress(merged, loss_col='val_train_pred_loss', color_by='frac_regular', n_bins=3):
    """Text summary: final-epoch loss statistics per color_by tercile."""
    last_epoch = merged['epoch'].max()
    final = merged[merged['epoch'] == last_epoch]
    if color_by not in final.columns:
        print(f"(no {color_by} column; overall final mean = {final[loss_col].mean():.4f})")
        return
    cv = final[color_by]
    qs = np.nanquantile(cv.values, np.linspace(0, 1, n_bins + 1))
    print(f"Final-epoch ({last_epoch}) {loss_col} by {color_by} tercile:")
    for b in range(n_bins):
        lo, hi = qs[b], qs[b+1]
        m = final[(cv >= lo) & (cv <= hi if b == n_bins-1 else cv < hi)]
        print(f"  {color_by} in [{lo:.3f},{hi:.3f}]: n={len(m):3d}  "
              f"mean={m[loss_col].mean():.4f}  median={m[loss_col].median():.4f}  max={m[loss_col].max():.4f}")





def Autoregressive_Handoff_Inference_Protocol_general(
    mapping_net,
    inverse_net,
    x_obs,
    u_obs,
    t_obs,
    t_min,
    t_max,
    Tmax,
    query_times,
    query_indices,
    device,
    n,
):
    """
    General-n Autoregressive Handoff Inference Protocol.
 
    Args:
        mapping_net: trained general-n MappingNet. Expects (q, p, t) with
                     q, p of shape (batch, n) and t of shape (batch, 1).
        inverse_net: trained general-n InverseNet, same shape convention.
        x_obs: (n,) tensor, observed q vector.
        u_obs: (n,) tensor, observed p vector.
        t_obs: float, observation time.
        t_min: float, start of prediction interval.
        t_max: float, end of prediction interval.
        Tmax: float, maximum normalized temporal horizon.
        query_times: 1D numpy array of query times (excluding t_obs).
        query_indices: list of ints, original indices for query_times.
        device: torch device.
        n: int, number of degrees of freedom.
 
    Returns:
        pred_dict: dict mapping original index -> (q_pred, p_pred), each an
                   (n,) float32 numpy array.
        forward_pass_count: dict with 'mapping_net' and 'inverse_net' counts.
    """
    # -------- Input validation --------
    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")
    if not isinstance(query_indices, list):
        raise ValueError(f"query_indices must be a list, got {type(query_indices)}")
    if len(query_indices) != len(query_times):
        raise ValueError(
            f"query_indices and query_times must have the same length, "
            f"got {len(query_indices)} and {len(query_times)}"
        )
    if x_obs.dim() != 1 or u_obs.dim() != 1:
        raise ValueError(
            f"x_obs and u_obs must be 1D (n,) tensors, got dims "
            f"{x_obs.dim()} and {u_obs.dim()}"
        )
    if x_obs.shape[-1] != n or u_obs.shape[-1] != n:
        raise ValueError(
            f"x_obs and u_obs must have length n={n}, got "
            f"{x_obs.shape[-1]} and {u_obs.shape[-1]}"
        )
    if hasattr(mapping_net, "n") and mapping_net.n != n:
        raise ValueError(f"mapping_net.n={mapping_net.n} does not match n={n}")
    if hasattr(inverse_net, "n") and inverse_net.n != n:
        raise ValueError(f"inverse_net.n={inverse_net.n} does not match n={n}")
 
    forward_pass_count = {'mapping_net': 0, 'inverse_net': 0}
    pred_dict = {}
 
    if len(query_indices) == 0:
        return pred_dict, forward_pass_count
 
    # -------- Case 1: Direct Prediction --------
    if (t_max - t_min) <= Tmax:
        t_rel_obs = torch.tensor([[t_obs - t_min]], device=device, dtype=torch.float32)  # (1,1)
        Q, P, _ = mapping_net(x_obs.unsqueeze(0), u_obs.unsqueeze(0), t_rel_obs)          # (1,n)
        forward_pass_count['mapping_net'] += 1
 
        t_rel_query = torch.tensor(query_times - t_min, device=device,
                                   dtype=torch.float32).unsqueeze(1)                       # (Nq,1)
        Q_full = Q.expand(len(query_indices), n)                                           # (Nq,n)
        P_full = P.expand(len(query_indices), n)
        x_pred_query, u_pred_query, _ = inverse_net(Q_full, P_full, t_rel_query)           # (Nq,n)
        forward_pass_count['inverse_net'] += 1
 
        for j, idx in enumerate(query_indices):
            pred_dict[idx] = (
                x_pred_query[j].detach().cpu().numpy(),
                u_pred_query[j].detach().cpu().numpy(),
            )
        return pred_dict, forward_pass_count
 
    # -------- Case 2: Autoregressive Propagation --------
 
    # === Phase 1: Forward Propagation ===
    fwd = [(idx, query_times[j]) for j, idx in enumerate(query_indices) if query_times[j] > t_obs]
    if fwd:
        fwd_indices, fwd_times = zip(*fwd)
        fwd_indices = list(fwd_indices)
        fwd_times = np.array(fwd_times)
 
        K_fwd = int(np.ceil((t_max - t_obs) / Tmax - 1e-9))
        K_fwd = max(K_fwd, 1)
 
        q_curr = x_obs.clone()  # (n,)
        p_curr = u_obs.clone()  # (n,)
        t_curr = t_obs
 
        for k in range(K_fwd):
            T_start_k = t_curr
            T_end_k = min(t_curr + Tmax, t_max)
            if k == K_fwd - 1:
                T_end_k = t_max
 
            if k < K_fwd - 1:
                I_k_query = [(idx, ft) for idx, ft in zip(fwd_indices, fwd_times)
                             if T_start_k <= ft < T_end_k]
            else:
                I_k_query = [(idx, ft) for idx, ft in zip(fwd_indices, fwd_times)
                             if T_start_k <= ft <= T_end_k]
 
            need_boundary = (k < K_fwd - 1)
 
            t_rel_list = [ft - T_start_k for _, ft in I_k_query]
            if need_boundary:
                t_rel_list.append(T_end_k - T_start_k)
            if len(t_rel_list) == 0:
                continue
 
            t_rel_k = torch.tensor(t_rel_list, device=device, dtype=torch.float32).unsqueeze(1)  # (Nt,1)
 
            Q_k, P_k, _ = mapping_net(
                q_curr.unsqueeze(0), p_curr.unsqueeze(0),
                torch.tensor([[0.0]], device=device, dtype=torch.float32),  # (1,1)
            )
            forward_pass_count['mapping_net'] += 1
 
            Q_full_k = Q_k.expand(t_rel_k.shape[0], n)
            P_full_k = P_k.expand(t_rel_k.shape[0], n)
 
            q_hat_k, p_hat_k, _ = inverse_net(Q_full_k, P_full_k, t_rel_k)  # (Nt,n)
            forward_pass_count['inverse_net'] += 1
 
            for j, (idx, _) in enumerate(I_k_query):
                pred_dict[idx] = (
                    q_hat_k[j].detach().cpu().numpy(),
                    p_hat_k[j].detach().cpu().numpy(),
                )
 
            if need_boundary:
                q_curr = q_hat_k[-1].detach()  # (n,)
                p_curr = p_hat_k[-1].detach()  # (n,)
                t_curr = T_end_k
 
    # === Phase 2: Backward Propagation ===
    bwd = [(idx, query_times[j]) for j, idx in enumerate(query_indices) if query_times[j] < t_obs]
    if bwd:
        bwd_indices, bwd_times = zip(*bwd)
        bwd_indices = list(bwd_indices)
        bwd_times = np.array(bwd_times)
 
        K_bwd = int(np.ceil((t_obs - t_min) / Tmax - 1e-9))
        K_bwd = max(K_bwd, 1)
 
        q_curr = x_obs.clone()
        p_curr = u_obs.clone()
        t_curr = t_obs
 
        for k in range(K_bwd):
            T_start_k = max(t_curr - Tmax, t_min)
            T_end_k = t_curr
            if k == K_bwd - 1:
                T_start_k = t_min
 
            if k < K_bwd - 1:
                I_k_query = [(idx, bt) for idx, bt in zip(bwd_indices, bwd_times)
                             if T_start_k < bt <= T_end_k]
            else:
                I_k_query = [(idx, bt) for idx, bt in zip(bwd_indices, bwd_times)
                             if T_start_k <= bt <= T_end_k]
 
            need_boundary = (k < K_bwd - 1)
 
            t_rel_list = []
            if need_boundary:
                t_rel_list.append(0.0)
            t_rel_list.extend([bt - T_start_k for _, bt in I_k_query])
            if len(t_rel_list) == 0:
                continue
 
            t_rel_k = torch.tensor(t_rel_list, device=device, dtype=torch.float32).unsqueeze(1)  # (Nt,1)
 
            t_rel_obs_in_window = T_end_k - T_start_k
            Q_k, P_k, _ = mapping_net(
                q_curr.unsqueeze(0), p_curr.unsqueeze(0),
                torch.tensor([[t_rel_obs_in_window]], device=device, dtype=torch.float32),  # (1,1)
            )
            forward_pass_count['mapping_net'] += 1
 
            Q_full_k = Q_k.expand(t_rel_k.shape[0], n)
            P_full_k = P_k.expand(t_rel_k.shape[0], n)
 
            q_hat_k, p_hat_k, _ = inverse_net(Q_full_k, P_full_k, t_rel_k)  # (Nt,n)
            forward_pass_count['inverse_net'] += 1
 
            offset = 1 if need_boundary else 0
            for j, (idx, _) in enumerate(I_k_query):
                pred_dict[idx] = (
                    q_hat_k[offset + j].detach().cpu().numpy(),
                    p_hat_k[offset + j].detach().cpu().numpy(),
                )
 
            if need_boundary:
                q_curr = q_hat_k[0].detach()  # (n,)
                p_curr = p_hat_k[0].detach()  # (n,)
                t_curr = T_start_k
 
    missing = [idx for idx in query_indices if idx not in pred_dict]
    if missing:
        raise RuntimeError(
            f"Autoregressive_Handoff_Inference_Protocol_general failed to produce "
            f"predictions for indices {missing}. This is likely a floating point "
            f"boundary issue. t_obs={t_obs}, t_min={t_min}, t_max={t_max}, Tmax={Tmax}"
        )
    return pred_dict, forward_pass_count
 
 
def Batched_Autoregressive_Handoff_Inference_Protocol_general(
    mapping_net,
    inverse_net,
    x_obs,
    u_obs,
    t_obs,
    t_min,
    t_max,
    Tmax,
    query_times,
    query_indices,
    device,
    n,
):
    """
    General-n Batched Autoregressive Handoff Inference Protocol.
 
    All trajectories share the same time structure (same t values, same obs index).
 
    Args:
        x_obs: (N_traj, n) tensor, observed q vectors.
        u_obs: (N_traj, n) tensor, observed p vectors.
        (remaining args as in the single-trajectory version)
        n: int, number of degrees of freedom.
 
    Returns:
        x_pred_all: (N_traj, N_query, n) tensor of q predictions at query times.
        u_pred_all: (N_traj, N_query, n) tensor of p predictions at query times.
        forward_pass_count: dict with 'mapping_net' and 'inverse_net' counts.
    """
    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")
    if not isinstance(query_indices, list):
        raise ValueError(f"query_indices must be a list, got {type(query_indices)}")
    if len(query_indices) != len(query_times):
        raise ValueError(
            f"query_indices and query_times must have the same length, "
            f"got {len(query_indices)} and {len(query_times)}"
        )
    if x_obs.dim() != 2 or u_obs.dim() != 2:
        raise ValueError(
            f"x_obs and u_obs must be 2D (N_traj, n) tensors, got dims "
            f"{x_obs.dim()} and {u_obs.dim()}"
        )
    if x_obs.shape[0] != u_obs.shape[0]:
        raise ValueError(
            f"x_obs and u_obs must have the same N_traj, got "
            f"{x_obs.shape[0]} and {u_obs.shape[0]}"
        )
    if x_obs.shape[1] != n or u_obs.shape[1] != n:
        raise ValueError(
            f"x_obs and u_obs must have DOF dim n={n}, got "
            f"{x_obs.shape[1]} and {u_obs.shape[1]}"
        )
    if hasattr(mapping_net, "n") and mapping_net.n != n:
        raise ValueError(f"mapping_net.n={mapping_net.n} does not match n={n}")
    if hasattr(inverse_net, "n") and inverse_net.n != n:
        raise ValueError(f"inverse_net.n={inverse_net.n} does not match n={n}")
 
    N_traj = x_obs.shape[0]
    N_query = len(query_indices)
    forward_pass_count = {'mapping_net': 0, 'inverse_net': 0}
 
    x_pred_all = torch.empty(N_traj, N_query, n, device=device, dtype=torch.float32)
    u_pred_all = torch.empty(N_traj, N_query, n, device=device, dtype=torch.float32)
 
    if N_query == 0:
        return x_pred_all, u_pred_all, forward_pass_count
 
    # -------- Case 1: Direct Prediction --------
    if (t_max - t_min) <= Tmax:
        t_rel_obs = torch.full((N_traj, 1), float(t_obs - t_min),
                               device=device, dtype=torch.float32)  # (N_traj,1)
        Q, P, _ = mapping_net(x_obs, u_obs, t_rel_obs)              # (N_traj,n)
        forward_pass_count['mapping_net'] += 1
 
        # traj-major, query-minor flatten
        Q_full = Q.repeat_interleave(N_query, dim=0)               # (N_traj*Nq, n)
        P_full = P.repeat_interleave(N_query, dim=0)
 
        t_rel_query = torch.tensor(query_times - t_min, device=device, dtype=torch.float32)  # (Nq,)
        t_rel_full = t_rel_query.repeat(N_traj).unsqueeze(1)       # (N_traj*Nq, 1)
 
        x_pred_flat, u_pred_flat, _ = inverse_net(Q_full, P_full, t_rel_full)  # (N_traj*Nq, n)
        forward_pass_count['inverse_net'] += 1
 
        x_pred_all = x_pred_flat.reshape(N_traj, N_query, n)
        u_pred_all = u_pred_flat.reshape(N_traj, N_query, n)
        return x_pred_all, u_pred_all, forward_pass_count
 
    # -------- Case 2: Autoregressive Propagation --------
 
    # === Phase 1: Forward Propagation ===
    fwd_positions = [j for j, idx in enumerate(query_indices) if query_times[j] > t_obs]
    if len(fwd_positions) > 0:
        K_fwd = int(np.ceil((t_max - t_obs) / Tmax - 1e-9))
        K_fwd = max(K_fwd, 1)
        q_curr = x_obs.clone()  # (N_traj, n)
        p_curr = u_obs.clone()
        t_curr = t_obs
 
        for k in range(K_fwd):
            T_start_k = t_curr
            T_end_k = min(t_curr + Tmax, t_max)
            if k == K_fwd - 1:
                T_end_k = t_max
            if k < K_fwd - 1:
                seg_query_positions = [j for j in fwd_positions
                                       if T_start_k <= query_times[j] < T_end_k]
            else:
                seg_query_positions = [j for j in fwd_positions
                                       if T_start_k <= query_times[j] <= T_end_k]
 
            need_boundary = (k < K_fwd - 1)
 
            t_rel_list = [query_times[j] - T_start_k for j in seg_query_positions]
            if need_boundary:
                t_rel_list.append(T_end_k - T_start_k)
 
            N_times = len(t_rel_list)
            if N_times == 0:
                continue
 
            t_rel_k = torch.tensor(t_rel_list, device=device, dtype=torch.float32)  # (Nt,)
 
            t_zero = torch.zeros(N_traj, 1, device=device, dtype=torch.float32)     # (N_traj,1)
            Q_k, P_k, _ = mapping_net(q_curr, p_curr, t_zero)                       # (N_traj,n)
            forward_pass_count['mapping_net'] += 1
 
            Q_full_k = Q_k.repeat_interleave(N_times, dim=0)        # (N_traj*Nt, n)
            P_full_k = P_k.repeat_interleave(N_times, dim=0)
            t_rel_full_k = t_rel_k.repeat(N_traj).unsqueeze(1)      # (N_traj*Nt, 1)
 
            q_hat_flat, p_hat_flat, _ = inverse_net(Q_full_k, P_full_k, t_rel_full_k)
            forward_pass_count['inverse_net'] += 1
 
            q_hat_k = q_hat_flat.reshape(N_traj, N_times, n)
            p_hat_k = p_hat_flat.reshape(N_traj, N_times, n)
 
            for local_j, global_j in enumerate(seg_query_positions):
                x_pred_all[:, global_j, :] = q_hat_k[:, local_j, :]
                u_pred_all[:, global_j, :] = p_hat_k[:, local_j, :]
 
            if need_boundary:
                q_curr = q_hat_k[:, -1, :].detach()  # (N_traj, n)
                p_curr = p_hat_k[:, -1, :].detach()
                t_curr = T_end_k
 
    # === Phase 2: Backward Propagation ===
    bwd_positions = [j for j, idx in enumerate(query_indices) if query_times[j] < t_obs]
    if len(bwd_positions) > 0:
        K_bwd = int(np.ceil((t_obs - t_min) / Tmax - 1e-9))
        K_bwd = max(K_bwd, 1)
        q_curr = x_obs.clone()  # (N_traj, n)
        p_curr = u_obs.clone()
        t_curr = t_obs
 
        for k in range(K_bwd):
            T_start_k = max(t_curr - Tmax, t_min)
            T_end_k = t_curr
            if k == K_bwd - 1:
                T_start_k = t_min
 
            if k < K_bwd - 1:
                seg_query_positions = [j for j in bwd_positions
                                       if T_start_k < query_times[j] <= T_end_k]
            else:
                seg_query_positions = [j for j in bwd_positions
                                       if T_start_k <= query_times[j] <= T_end_k]
 
            need_boundary = (k < K_bwd - 1)
 
            t_rel_list = []
            if need_boundary:
                t_rel_list.append(0.0)
            t_rel_list.extend([query_times[j] - T_start_k for j in seg_query_positions])
 
            N_times = len(t_rel_list)
            if N_times == 0:
                continue
 
            t_rel_k = torch.tensor(t_rel_list, device=device, dtype=torch.float32)  # (Nt,)
 
            t_rel_obs_in_window = T_end_k - T_start_k
            t_obs_window = torch.full((N_traj, 1), float(t_rel_obs_in_window),
                                      device=device, dtype=torch.float32)           # (N_traj,1)
            Q_k, P_k, _ = mapping_net(q_curr, p_curr, t_obs_window)                 # (N_traj,n)
            forward_pass_count['mapping_net'] += 1
 
            Q_full_k = Q_k.repeat_interleave(N_times, dim=0)
            P_full_k = P_k.repeat_interleave(N_times, dim=0)
            t_rel_full_k = t_rel_k.repeat(N_traj).unsqueeze(1)
 
            q_hat_flat, p_hat_flat, _ = inverse_net(Q_full_k, P_full_k, t_rel_full_k)
            forward_pass_count['inverse_net'] += 1
 
            q_hat_k = q_hat_flat.reshape(N_traj, N_times, n)
            p_hat_k = p_hat_flat.reshape(N_traj, N_times, n)
 
            offset = 1 if need_boundary else 0
            for local_j, global_j in enumerate(seg_query_positions):
                x_pred_all[:, global_j, :] = q_hat_k[:, offset + local_j, :]
                u_pred_all[:, global_j, :] = p_hat_k[:, offset + local_j, :]
 
            if need_boundary:
                q_curr = q_hat_k[:, 0, :].detach()  # (N_traj, n)
                p_curr = p_hat_k[:, 0, :].detach()
                t_curr = T_start_k
 
    return x_pred_all, u_pred_all, forward_pass_count





POS_COLS = ("x", "y")
MOM_COLS = ("px", "py")


def build_omega_general(n, device=None, dtype=None):
    """
    Symplectic form for n degrees of freedom, in the grouped ordering
    z = (q_1, ..., q_n, p_1, ..., p_n) fixed by the paper:

        Omega = [[ 0,   I_n ],
                 [-I_n, 0   ]]      shape (2n, 2n)

    Reduces to [[0, 1], [-1, 0]] at n = 1.
    """
    I = torch.eye(n, device=device, dtype=dtype)
    Z = torch.zeros(n, n, device=device, dtype=dtype)
    return torch.cat([torch.cat([Z, I], dim=1),
                      torch.cat([-I, Z], dim=1)], dim=0)


def compute_jacobian_functional_general(model, x, u, t, n):
    """
    Compute the 2n x 2n Jacobian of a general-n network using
    torch.autograd.functional.jacobian, one sample at a time.

    Row/column ordering is the grouped convention z = (q_1..q_n, p_1..p_n):
        M[i, j] = d z'_i / d z_j
        rows    0..n-1   -> Q,   rows    n..2n-1 -> P
        columns 0..n-1   -> q,   columns n..2n-1 -> p

    IMPORTANT: the HJL layers compute their shear terms with
    create_graph=self.training. If `model` is in eval() mode the generating
    function gradients are detached, the off-diagonal blocks of M come out
    exactly zero, and M collapses to a (still symplectic, hence vacuously
    passing) diagonal scaling. Callers must run this in train() mode; the
    orchestrator below enforces that.

    Args:
        model: network taking (q, p, t) with q, p of shape (batch, n) and
               t of shape (batch, 1), returning (Q, P, t).
        x: (batch, n) tensor, coordinate-like input.
        u: (batch, n) tensor, momentum-like input.
        t: (batch, 1) tensor, time.
        n: int, number of degrees of freedom.

    Returns:
        jacobian: (2n, 2n, batch) tensor.
    """
    from torch.autograd.functional import jacobian as torch_jacobian

    if x.dim() != 2 or u.dim() != 2:
        raise ValueError(f"x and u must be 2D (batch, n) tensors, got dims {x.dim()} and {u.dim()}")
    if x.shape[1] != n or u.shape[1] != n:
        raise ValueError(f"x and u must have DOF dim n={n}, got {x.shape[1]} and {u.shape[1]}")
    if t.dim() != 2 or t.shape[1] != 1:
        raise ValueError(f"t must be a 2D (batch, 1) tensor, got shape {tuple(t.shape)}")
    if not (x.shape[0] == u.shape[0] == t.shape[0]):
        raise ValueError(
            f"batch dims must match, got {x.shape[0]}, {u.shape[0]}, {t.shape[0]}")
    if hasattr(model, "n") and model.n != n:
        raise ValueError(f"model.n={model.n} does not match n={n}")

    batch_size = x.shape[0]
    jacobians = []

    for i in range(batch_size):
        x_i = x[i:i + 1]          # (1, n)
        u_i = u[i:i + 1]          # (1, n)
        t_i = t[i:i + 1]          # (1, 1)

        # Function of (x, u) only; t is held fixed (the paper takes the
        # Jacobian at fixed segment-relative time).
        def func(x_val, u_val):
            X_final, U_final, _ = model(x_val, u_val, t_i)
            return torch.cat([X_final.reshape(-1), U_final.reshape(-1)], dim=0)   # (2n,)

        jac = torch_jacobian(func, (x_i, u_i))

        # jac[0]: d(out)/dx with shape (2n, 1, n);  jac[1]: d(out)/du likewise.
        # Explicit reshape rather than .squeeze(), which would over-collapse at n=1.
        jac_x = jac[0].reshape(2 * n, n)                       # (2n, n)
        jac_u = jac[1].reshape(2 * n, n)                       # (2n, n)

        jac_matrix = torch.cat([jac_x, jac_u], dim=1)          # (2n, 2n)
        jacobians.append(jac_matrix)

    jacobian = torch.stack(jacobians, dim=2)                   # (2n, 2n, batch)
    return jacobian


def compute_symplectic_product_general(jacobian, n):
    """
    Compute M^T Omega M for each 2n x 2n Jacobian in the batch, with
    Omega = [[0, I_n], [-I_n, 0]].

    Args:
        jacobian: (2n, 2n, batch) tensor.
        n: int, number of degrees of freedom.

    Returns:
        result: (2n, 2n, batch) tensor, each slice being M^T Omega M.
    """
    if jacobian.dim() != 3:
        raise ValueError(f"jacobian must be 3D (2n, 2n, batch), got dim {jacobian.dim()}")
    if jacobian.shape[0] != 2 * n or jacobian.shape[1] != 2 * n:
        raise ValueError(
            f"jacobian must be (2n, 2n, batch) with n={n} -> "
            f"({2*n}, {2*n}, batch), got {tuple(jacobian.shape)}")

    batch_size = jacobian.shape[2]
    device = jacobian.device
    dtype = jacobian.dtype

    Omega = build_omega_general(n, device=device, dtype=dtype)      # (2n, 2n)

    M = jacobian.permute(2, 0, 1)                                   # (batch, 2n, 2n)
    M_T = M.transpose(1, 2)                                         # (batch, 2n, 2n)
    Omega_batch = Omega.unsqueeze(0).expand(batch_size, -1, -1)

    Omega_M = torch.bmm(Omega_batch, M)                             # (batch, 2n, 2n)
    result_batch = torch.bmm(M_T, Omega_M)                          # (batch, 2n, 2n)

    result = result_batch.permute(1, 2, 0)                          # (2n, 2n, batch)
    return result


def test_canonical_transformation_on_trajectory_general(
    get_data_from_trajectory_id_function,
    compute_jacobian_functional_function,
    compute_symplectic_product_function,
    test_id_df,
    test_df,
    trajectory_id,
    mapping_net,
    inverse_net,
    device,
    Tmax,
    n,
    pos_cols=POS_COLS,
    mom_cols=MOM_COLS,
    tolerance=1e-5,
    verbose=True,
):
    """
    General-n symplecticity diagnostic on a single trajectory.

    Partitions the trajectory into segments of length Tmax and, at every
    observation, computes the 2n x 2n Jacobian of MappingNet w.r.t. (q, p) and
    of InverseNet w.r.t. (Q, P), both at the segment-relative time, then
    evaluates the residual ||M^T Omega M - Omega||_max per point.

    Both networks are forced into train() mode for the duration and restored
    on exit: the HJL shear gradients are built with create_graph=self.training,
    so in eval() mode the off-diagonal Jacobian blocks are identically zero and
    the residual is vacuously zero.

    Returns:
        results: dict with per-point residuals and summary statistics for both
                 networks (same keys as the n=1 version, plus 'n').
    """
    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")
    if len(pos_cols) != n:
        raise ValueError(f"pos_cols must have n={n} entries, got {len(pos_cols)}: {pos_cols}")
    if len(mom_cols) != n:
        raise ValueError(f"mom_cols must have n={n} entries, got {len(mom_cols)}: {mom_cols}")
    if hasattr(mapping_net, "n") and mapping_net.n != n:
        raise ValueError(f"mapping_net.n={mapping_net.n} does not match n={n}")
    if hasattr(inverse_net, "n") and inverse_net.n != n:
        raise ValueError(f"inverse_net.n={inverse_net.n} does not match n={n}")

    test_trajectory_data = get_data_from_trajectory_id_function(
        test_id_df, test_df, trajectory_ids=trajectory_id)

    missing = [c for c in list(pos_cols) + list(mom_cols) + ['t']
               if c not in test_trajectory_data.columns]
    if missing:
        raise ValueError(f"columns missing from trajectory data: {missing}")

    x = torch.as_tensor(test_trajectory_data[list(pos_cols)].to_numpy(dtype=np.float32),
                        device=device)                                   # (T, n)
    u = torch.as_tensor(test_trajectory_data[list(mom_cols)].to_numpy(dtype=np.float32),
                        device=device)                                   # (T, n)
    t = torch.as_tensor(test_trajectory_data['t'].to_numpy(dtype=np.float32),
                        device=device)                                   # (T,)

    if x.dim() != 2 or u.dim() != 2 or t.dim() != 1:
        raise ValueError(
            f"x, u must be 2-dimensional and t 1-dimensional, got dims "
            f"{x.dim()}, {u.dim()}, {t.dim()}")
    if x.shape[1] != n or u.shape[1] != n:
        raise ValueError(f"x, u must have DOF dim n={n}, got {x.shape[1]}, {u.shape[1]}")

    t_np = t.detach().cpu().numpy()
    t_max = t_np[-1]

    # Build segments (anchored at absolute zero, as in the n=1 version)
    num_segments = int(np.ceil(t_max / Tmax - 1e-9))
    num_segments = max(num_segments, 1)

    mapping_jacobians = []
    inverse_jacobians = []

    # ---- force train() so that create_graph=True inside the HJL shear grads ----
    mapping_was_training = mapping_net.training
    inverse_was_training = inverse_net.training
    try:
        mapping_net.train()
        inverse_net.train()

        for seg_idx in range(num_segments):
            t_start = seg_idx * Tmax
            t_end = min((seg_idx + 1) * Tmax, t_max)
            if seg_idx == num_segments - 1:
                t_end = t_max

            if seg_idx == 0:
                seg_mask = (t_np >= t_start) & (t_np <= t_end)
            else:
                seg_mask = (t_np > t_start) & (t_np <= t_end)

            seg_indices = np.where(seg_mask)[0]

            if len(seg_indices) == 0:
                continue

            x_seg = x[seg_indices]                                       # (S, n)
            u_seg = u[seg_indices]                                       # (S, n)
            t_seg_relative = (t[seg_indices] - t_start).unsqueeze(1)     # (S, 1)

            jac_mapping = compute_jacobian_functional_function(
                mapping_net, x=x_seg, u=u_seg, t=t_seg_relative, n=n)
            mapping_jacobians.append(jac_mapping)

            Q_seg, P_seg, _ = mapping_net(x_seg, u_seg, t_seg_relative)

            jac_inverse = compute_jacobian_functional_function(
                inverse_net, x=Q_seg.detach(), u=P_seg.detach(),
                t=t_seg_relative, n=n)
            inverse_jacobians.append(jac_inverse)
    finally:
        # inverse_net shares its modules with mapping_net, so restore the
        # inverse first and let mapping_net (the owner) have the last word.
        inverse_net.train(inverse_was_training)
        mapping_net.train(mapping_was_training)

    all_mapping_jac = torch.cat(mapping_jacobians, dim=2)                # (2n, 2n, N)
    all_inverse_jac = torch.cat(inverse_jacobians, dim=2)

    mapping_symplectic = compute_symplectic_product_function(all_mapping_jac, n)
    inverse_symplectic = compute_symplectic_product_function(all_inverse_jac, n)

    batch_size = mapping_symplectic.shape[2]
    device_sym = mapping_symplectic.device
    dtype_sym = mapping_symplectic.dtype

    Omega = build_omega_general(n, device=device_sym, dtype=dtype_sym)   # (2n, 2n)

    diff_mapping = torch.abs(mapping_symplectic - Omega.unsqueeze(-1))
    mapping_residuals = diff_mapping.amax(dim=(0, 1))

    diff_inverse = torch.abs(inverse_symplectic - Omega.unsqueeze(-1))
    inverse_residuals = diff_inverse.amax(dim=(0, 1))

    mapping_residuals_np = mapping_residuals.detach().cpu().numpy()
    inverse_residuals_np = inverse_residuals.detach().cpu().numpy()

    def compute_residual_stats(residuals, name):
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0,
            'median': np.median(residuals),
            'max': np.max(residuals),
            'min': np.min(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75),
            'n_failing': int(np.sum(residuals > tolerance)),
            'n_total': len(residuals),
            'fraction_passing': float(np.mean(residuals <= tolerance)),
        }

    mapping_stats = compute_residual_stats(mapping_residuals_np, 'mapping')
    inverse_stats = compute_residual_stats(inverse_residuals_np, 'inverse')

    if verbose:
        print(f"=== Canonical Transformation Test — Trajectory {trajectory_id} ===")
        print(f"(n={n}, Jacobians are {2*n}x{2*n}, {num_segments} segment(s), "
              f"{batch_size} total points, tolerance={tolerance:.1e})\n")

        print("MappingNet: d(Q,P)/d(q,p)")
        if mapping_stats['n_failing'] == 0:
            print(f"  ✓ All {batch_size} points pass (max residual: {mapping_stats['max']:.2e})")
        else:
            print(f"  ✗ {mapping_stats['n_failing']}/{batch_size} points fail "
                  f"({mapping_stats['fraction_passing']*100:.1f}% pass)")
        print(f"  Residual max|M^T Ω M - Ω| per point:")
        print(f"    Mean:   {mapping_stats['mean']:.2e} (std={mapping_stats['std']:.2e})")
        print(f"    Median: {mapping_stats['median']:.2e} [IQR: {mapping_stats['q25']:.2e}, {mapping_stats['q75']:.2e}]")
        print(f"    Min:    {mapping_stats['min']:.2e}")
        print(f"    Max:    {mapping_stats['max']:.2e}")

        print(f"\nInverseNet: d(q,p)/d(Q,P)")
        if inverse_stats['n_failing'] == 0:
            print(f"  ✓ All {batch_size} points pass (max residual: {inverse_stats['max']:.2e})")
        else:
            print(f"  ✗ {inverse_stats['n_failing']}/{batch_size} points fail "
                  f"({inverse_stats['fraction_passing']*100:.1f}% pass)")
        print(f"  Residual max|M^T Ω M - Ω| per point:")
        print(f"    Mean:   {inverse_stats['mean']:.2e} (std={inverse_stats['std']:.2e})")
        print(f"    Median: {inverse_stats['median']:.2e} [IQR: {inverse_stats['q25']:.2e}, {inverse_stats['q75']:.2e}]")
        print(f"    Min:    {inverse_stats['min']:.2e}")
        print(f"    Max:    {inverse_stats['max']:.2e}")

        print("\n=== Table row (Task = trajectory {}) ===".format(trajectory_id))
        print(f"  M  max    = {mapping_stats['max']:.2e}")
        print(f"  M  median = {mapping_stats['median']:.2e}")
        print(f"  M^-1 max    = {inverse_stats['max']:.2e}")
        print(f"  M^-1 median = {inverse_stats['median']:.2e}")
        print("  LaTeX: "
              f"{mapping_stats['max']:.2e} & {mapping_stats['median']:.2e} & "
              f"{inverse_stats['max']:.2e} & {inverse_stats['median']:.2e} \\\\")

    results = {
        'mapping_residuals': mapping_residuals_np,
        'inverse_residuals': inverse_residuals_np,
        'mapping_stats': mapping_stats,
        'inverse_stats': inverse_stats,
        'mapping_symplectic': mapping_symplectic,
        'inverse_symplectic': inverse_symplectic,
        't': t_np,
        'trajectory_id': trajectory_id,
        'tolerance': tolerance,
        'num_segments': num_segments,
        'n_points': batch_size,
        'n': n,
    }

    return results
 


def prediction_loss_euclidean_general(
    x_pred: torch.Tensor,
    u_pred: torch.Tensor,
    X_labels: torch.Tensor,
    U_labels: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Prediction loss: mean over prediction times of the Euclidean distance in the
    2n-dimensional phase space,

        d_t = sqrt( sum_i (q_hat_{t,i} - q_{t,i})^2 + sum_i (p_hat_{t,i} - p_{t,i})^2 )

    At n = 1 with (..., 1)-shaped inputs the sum over the DOF axis is a
    single-element sum, so this reduces exactly (bitwise) to the n = 1 function
        sqrt((x_pred - X_labels)^2 + (u_pred - U_labels)^2).mean()
    evaluated on the corresponding (...,)-shaped tensors.

    Args:
        x_pred:   (..., n) predicted positions
        u_pred:   (..., n) predicted momenta
        X_labels: (..., n) ground-truth positions
        U_labels: (..., n) ground-truth momenta
        n: int, number of degrees of freedom (last axis).

    Returns:
        loss: scalar tensor, mean Euclidean phase-space distance.
    """
    if not (x_pred.shape == u_pred.shape == X_labels.shape == U_labels.shape):
        raise ValueError(
            f"x_pred, u_pred, X_labels, U_labels must share a shape, got "
            f"{tuple(x_pred.shape)}, {tuple(u_pred.shape)}, "
            f"{tuple(X_labels.shape)}, {tuple(U_labels.shape)}")
    if x_pred.dim() < 1 or x_pred.shape[-1] != n:
        raise ValueError(
            f"last axis must be the DOF axis of size n={n}, got shape {tuple(x_pred.shape)}")

    euclidean_distances = torch.sqrt(
        ((x_pred - X_labels) ** 2).sum(dim=-1)
        + ((u_pred - U_labels) ** 2).sum(dim=-1)
    )
    return torch.mean(euclidean_distances)


def test_model_all_observations_all_trajectories_general(
    get_data_from_trajectory_id_function,
    prediction_loss_function,
    test_id_df,
    test_df,
    mapping_net,
    inverse_net,
    device,
    Tmax,
    n,
    pos_cols=POS_COLS,
    mom_cols=MOM_COLS,
    bootstrap_B=10000,
    confidence_level=0.95,
    normalize_by_energy=True,
    verbose=True,
):
    """
    General-n evaluation of all test trajectories from every possible single
    observation point.

    Identical in statistics and aggregation to the n = 1 version; the only change
    is that phase-space quantities carry a trailing DOF axis of size n, and the
    batched inference call is the general-n protocol.

    `prediction_loss_function` must accept
        (x_pred, u_pred, X_labels, U_labels, n)
    by keyword, with the four tensors of shape (N_obs, n), and return a scalar.

    Returns:
        results: dict, same keys as the n = 1 version plus 'n'.
    """
    from scipy.stats import bootstrap

    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")

    if normalize_by_energy and 'energy' not in test_id_df.columns:
        raise ValueError("test_id_df must contain an 'energy' column when normalize_by_energy=True")

    if len(pos_cols) != n:
        raise ValueError(f"pos_cols must have n={n} entries, got {len(pos_cols)}: {pos_cols}")
    if len(mom_cols) != n:
        raise ValueError(f"mom_cols must have n={n} entries, got {len(mom_cols)}: {mom_cols}")
    if hasattr(mapping_net, "n") and mapping_net.n != n:
        raise ValueError(f"mapping_net.n={mapping_net.n} does not match n={n}")
    if hasattr(inverse_net, "n") and inverse_net.n != n:
        raise ValueError(f"inverse_net.n={inverse_net.n} does not match n={n}")

    trajectory_ids = test_id_df['trajectory_id'].values.astype(int)
    N_traj = len(trajectory_ids)

    if normalize_by_energy:
        energies = test_id_df.set_index('trajectory_id').loc[trajectory_ids, 'energy'].values.astype(np.float64)
        if np.any(energies <= 1e-12):
            raise ValueError("Energy normalization by sqrt(E) requires positive nonzero energies.")
        sqrt_energies = np.sqrt(energies)
    else:
        energies = test_id_df.set_index('trajectory_id').loc[trajectory_ids, 'energy'].values.astype(np.float64) if 'energy' in test_id_df.columns else np.full(N_traj, np.nan)
        sqrt_energies = None

    # Load all trajectories and validate shared time structure
    all_x = []
    all_u = []
    ref_t = None

    for idx, row_data in test_id_df.iterrows():
        trajectory_id = int(row_data['trajectory_id'])
        test_trajectory_data = get_data_from_trajectory_id_function(
            test_id_df, test_df, trajectory_ids=trajectory_id)

        missing = [c for c in list(pos_cols) + list(mom_cols) + ['t']
                   if c not in test_trajectory_data.columns]
        if missing:
            raise ValueError(f"Trajectory {trajectory_id}: columns missing: {missing}")

        x_i = torch.as_tensor(test_trajectory_data[list(pos_cols)].to_numpy(dtype=np.float32),
                              device=device)                                   # (T, n)
        u_i = torch.as_tensor(test_trajectory_data[list(mom_cols)].to_numpy(dtype=np.float32),
                              device=device)                                   # (T, n)
        t_i = torch.as_tensor(test_trajectory_data['t'].to_numpy(dtype=np.float32),
                              device=device)                                   # (T,)

        if x_i.dim() != 2 or u_i.dim() != 2 or t_i.dim() != 1:
            raise ValueError(
                f"Trajectory {trajectory_id}: x, u must be 2D (T, n) and t 1D, got dims "
                f"{x_i.dim()}, {u_i.dim()}, {t_i.dim()}")
        if x_i.shape[1] != n or u_i.shape[1] != n:
            raise ValueError(
                f"Trajectory {trajectory_id}: DOF dim must be n={n}, got "
                f"{x_i.shape[1]} and {u_i.shape[1]}")

        if ref_t is None:
            ref_t = t_i
        else:
            if len(t_i) != len(ref_t):
                raise ValueError(
                    f"Trajectory {trajectory_id} has {len(t_i)} points, expected {len(ref_t)}. "
                    f"All trajectories must have the same number of points for batched inference."
                )
            if not torch.allclose(t_i, ref_t, atol=1e-6):
                raise ValueError(
                    f"Trajectory {trajectory_id} has different time values. "
                    f"All trajectories must share the same time values for batched inference."
                )

        all_x.append(x_i)
        all_u.append(u_i)

    # Stack: (N_traj, N_points, n)
    x_batch = torch.stack(all_x, dim=0)
    u_batch = torch.stack(all_u, dim=0)
    t_np = ref_t.detach().cpu().numpy()
    N_obs = len(ref_t)

    all_mapping_net_calls = np.empty(N_obs, dtype=int)
    all_inverse_net_calls = np.empty(N_obs, dtype=int)

    loss_matrix = np.empty((N_obs, N_traj))

    for obs_idx in range(N_obs):
        query_indices = [i for i in range(N_obs) if i != obs_idx]
        query_times = t_np[query_indices]

        x_pred_queries, u_pred_queries, fwd_count = \
            Batched_Autoregressive_Handoff_Inference_Protocol_general(
                mapping_net=mapping_net,
                inverse_net=inverse_net,
                x_obs=x_batch[:, obs_idx, :],          # (N_traj, n)
                u_obs=u_batch[:, obs_idx, :],          # (N_traj, n)
                t_obs=t_np[obs_idx],
                t_min=t_np[0],
                t_max=t_np[-1],
                Tmax=Tmax,
                query_times=query_times,
                query_indices=query_indices,
                device=device,
                n=n,
            )

        # Assemble full predictions: (N_traj, N_obs, n)
        x_pred_full = torch.empty(N_traj, N_obs, n, device=device, dtype=torch.float32)
        u_pred_full = torch.empty(N_traj, N_obs, n, device=device, dtype=torch.float32)
        x_pred_full[:, obs_idx, :] = x_batch[:, obs_idx, :]
        u_pred_full[:, obs_idx, :] = u_batch[:, obs_idx, :]
        for j, qi in enumerate(query_indices):
            x_pred_full[:, qi, :] = x_pred_queries[:, j, :]
            u_pred_full[:, qi, :] = u_pred_queries[:, j, :]

        # Per-trajectory loss for this observation index
        for traj_i in range(N_traj):
            loss_i = prediction_loss_function(
                x_pred=x_pred_full[traj_i],            # (N_obs, n)
                u_pred=u_pred_full[traj_i],
                X_labels=x_batch[traj_i],
                U_labels=u_batch[traj_i],
                n=n,
            )
            loss_matrix[obs_idx, traj_i] = loss_i.item()

        if verbose:
            mean_this_obs = np.mean(loss_matrix[obs_idx])
            print(f"Observation index {obs_idx}/{N_obs-1}, mean loss across trajectories: {mean_this_obs:.6f}")
        all_mapping_net_calls[obs_idx] = fwd_count['mapping_net']
        all_inverse_net_calls[obs_idx] = fwd_count['inverse_net']

    # Apply energy normalization if requested
    if normalize_by_energy:
        loss_matrix_normalized = loss_matrix / sqrt_energies[np.newaxis, :]
        loss_matrix_for_stats = loss_matrix_normalized
    else:
        loss_matrix_normalized = None
        loss_matrix_for_stats = loss_matrix

    # Aggregate: median over observation choices per trajectory
    per_trajectory_median_loss = np.median(loss_matrix_for_stats, axis=0)  # (N_traj,)
    per_obs_mean_loss = np.mean(loss_matrix_for_stats, axis=1)             # (N_obs,)

    # Statistics over trajectories (the independent units)
    mean_loss = np.mean(per_trajectory_median_loss)
    std_loss = np.std(per_trajectory_median_loss, ddof=1)
    median_loss = np.median(per_trajectory_median_loss)
    q1_loss = np.percentile(per_trajectory_median_loss, 25)
    q3_loss = np.percentile(per_trajectory_median_loss, 75)
    iqr_loss = q3_loss - q1_loss

    # BCa bootstrap CI over trajectories
    bootstrap_result = bootstrap(
        data=(per_trajectory_median_loss,),
        statistic=np.mean,
        n_resamples=bootstrap_B,
        confidence_level=confidence_level,
        method='BCa',
    )
    ci_lower = bootstrap_result.confidence_interval.low
    ci_upper = bootstrap_result.confidence_interval.high

    summary_stats = {
        'mean': mean_loss,
        'std': std_loss,
        'median': median_loss,
        'q1': q1_loss,
        'q3': q3_loss,
        'iqr': iqr_loss,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }

    loss_label = 'loss / √energy' if normalize_by_energy else 'loss'

    # Forward pass count statistics
    all_total_calls = all_mapping_net_calls + all_inverse_net_calls
    forward_pass_stats = {
        'mapping_net_mean': np.mean(all_mapping_net_calls),
        'mapping_net_std': np.std(all_mapping_net_calls, ddof=1),
        'mapping_net_min': int(np.min(all_mapping_net_calls)),
        'mapping_net_max': int(np.max(all_mapping_net_calls)),
        'inverse_net_mean': np.mean(all_inverse_net_calls),
        'inverse_net_std': np.std(all_inverse_net_calls, ddof=1),
        'inverse_net_min': int(np.min(all_inverse_net_calls)),
        'inverse_net_max': int(np.max(all_inverse_net_calls)),
        'total_mean': np.mean(all_total_calls),
        'total_std': np.std(all_total_calls, ddof=1),
        'total_min': int(np.min(all_total_calls)),
        'total_max': int(np.max(all_total_calls)),
        'total_across_all_obs': int(np.sum(all_total_calls)),
    }

    # Find trajectory closest to median
    median_traj_idx = int(np.argmin(np.abs(per_trajectory_median_loss - median_loss)))
    median_representative_id = int(trajectory_ids[median_traj_idx])
    median_representative_loss = per_trajectory_median_loss[median_traj_idx]

    if verbose:
        print(f"\n=== Cross-Trajectory Statistics ===")
        print(f"Metric: {loss_label}")
        print(f"Per-trajectory summary: median across obs. choices")
        print(f"Mean of medians: {mean_loss:.6f}")
        print(f"Std of medians: {std_loss:.6f}")
        print(f"Median of medians: {median_loss:.6f}")
        print(f"IQR of medians: [{q1_loss:.6f}, {q3_loss:.6f}]")
        print(f"{confidence_level*100:.0f}% BCa CI for mean of medians (over trajectories): [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"N_traj={N_traj}, N_obs={N_obs}, n={n}")
        print(f"\nMedian-representative trajectory: ID={median_representative_id} "
              f"(obs-median {loss_label}={median_representative_loss:.6f})")
        print(f"\n=== Forward Pass Count Statistics (over {N_obs} obs. choices, batched over {N_traj} trajectories) ===")
        print(f"MappingNet calls per obs:  mean={forward_pass_stats['mapping_net_mean']:.1f} "
              f"(std={forward_pass_stats['mapping_net_std']:.1f}), "
              f"range=[{forward_pass_stats['mapping_net_min']}, {forward_pass_stats['mapping_net_max']}]")
        print(f"InverseNet calls per obs:  mean={forward_pass_stats['inverse_net_mean']:.1f} "
              f"(std={forward_pass_stats['inverse_net_std']:.1f}), "
              f"range=[{forward_pass_stats['inverse_net_min']}, {forward_pass_stats['inverse_net_max']}]")
        print(f"Total calls per obs:       mean={forward_pass_stats['total_mean']:.1f} "
              f"(std={forward_pass_stats['total_std']:.1f}), "
              f"range=[{forward_pass_stats['total_min']}, {forward_pass_stats['total_max']}]")
        print(f"Total calls across all obs: {forward_pass_stats['total_across_all_obs']}")

    results = {
        'loss_matrix': loss_matrix,
        'loss_matrix_normalized': loss_matrix_normalized,
        'loss_matrix_for_stats': loss_matrix_for_stats,
        'per_trajectory_median_loss': per_trajectory_median_loss,
        'per_obs_mean_loss': per_obs_mean_loss,
        'summary_stats': summary_stats,
        'trajectory_ids': trajectory_ids,
        'energies': energies,
        't': t_np,
        'N_obs': N_obs,
        'N_traj': N_traj,
        'confidence_level': confidence_level,
        'normalized': normalize_by_energy,
        'forward_pass_stats': forward_pass_stats,
        'all_mapping_net_calls': all_mapping_net_calls,
        'all_inverse_net_calls': all_inverse_net_calls,
        'median_representative_id': median_representative_id,
        'median_representative_loss': median_representative_loss,
        'n': n,
    }

    return results


POS_COLS = ("x", "y")
MOM_COLS = ("px", "py")


# =====================================================================
#  Energy functions:  energy(q, p) with q, p numpy of shape (..., n) -> (...)
# =====================================================================

def make_energy_henon_heiles(lam=1.0):
    """
    H = 0.5(px^2 + py^2) + 0.5(x^2 + y^2) + lam*(x^2 y - y^3/3)   [n = 2]

    Identical to HenonHeiles.hamiltonian used to generate the data, and the
    exact Hamiltonian of build_hh_integrator's equations of motion.
    """
    def energy(q, p):
        x, y = q[..., 0], q[..., 1]
        px, py = p[..., 0], p[..., 1]
        return 0.5 * (px**2 + py**2) + 0.5 * (x**2 + y**2) \
            + lam * (x**2 * y - (1 / 3.) * y**3)
    energy.__name__ = f"henon_heiles(lam={lam})"
    return energy


def make_energy_harmonic_oscillator(k=1.0, mass=1.0):
    """H = 0.5 p^2 + 0.5 (k/m) q^2, summed over DOF. Bitwise-identical at n=1."""
    def energy(q, p):
        return 0.5 * (p**2).sum(-1) + 0.5 * (k / mass) * (q**2).sum(-1)
    energy.__name__ = f"harmonic_oscillator(k={k},mass={mass})"
    return energy


def make_energy_ideal_pendulum(g=9.81, length=3.0):
    """H = 0.5 p^2 + (g/l)(1 - cos q), summed over DOF. Bitwise-identical at n=1."""
    def energy(q, p):
        return 0.5 * (p**2).sum(-1) + (g / length) * (1 - np.cos(q)).sum(-1)
    energy.__name__ = f"ideal_pendulum(g={g},length={length})"
    return energy


def make_energy_real_pendulum(constant=2.4):
    """H = p^2 + c(1 - cos q), summed over DOF. Bitwise-identical at n=1."""
    def energy(q, p):
        return (p**2).sum(-1) + constant * (1 - np.cos(q)).sum(-1)
    energy.__name__ = f"real_pendulum(constant={constant})"
    return energy


# =====================================================================
#  Evaluation loss (NOT the training loss: no 1/n normalisation)
# =====================================================================

def prediction_loss_general(
    x_pred: torch.Tensor,
    u_pred: torch.Tensor,
    X_labels: torch.Tensor,
    U_labels: torch.Tensor,
    n: int,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    General-n MAE / MSE prediction loss.

    The DOF axis is SUMMED, not averaged, so that for loss_type='mse' this is
    exactly the paper's scalar
        L_m = mean over prediction times of
              SE(t) = sum_i (q_hat_i - q_i)^2 + sum_i (p_hat_i - p_i)^2 .

    NOTE this is the *evaluation* loss. The training loss of Sec. Training
    additionally divides by n; the two coincide at n = 1.

    At n = 1 with (..., 1)-shaped inputs the DOF sum is a single-element sum,
    so this reduces bitwise to the n = 1 `prediction_loss` on (...,)-shaped
    tensors, for both 'mae' and 'mse'.

    Args:
        x_pred, u_pred, X_labels, U_labels: (..., n) tensors
        n: int, number of degrees of freedom (last axis)
        loss_type: 'mae' or 'mse'

    Returns:
        loss: scalar tensor
    """
    if not (x_pred.shape == u_pred.shape == X_labels.shape == U_labels.shape):
        raise ValueError(
            f"x_pred, u_pred, X_labels, U_labels must share a shape, got "
            f"{tuple(x_pred.shape)}, {tuple(u_pred.shape)}, "
            f"{tuple(X_labels.shape)}, {tuple(U_labels.shape)}")
    if x_pred.dim() < 1 or x_pred.shape[-1] != n:
        raise ValueError(
            f"last axis must be the DOF axis of size n={n}, got shape {tuple(x_pred.shape)}")

    if loss_type.lower() == "mae":
        x_loss = torch.mean(torch.abs(x_pred - X_labels).sum(dim=-1))
        u_loss = torch.mean(torch.abs(u_pred - U_labels).sum(dim=-1))
    elif loss_type.lower() == "mse":
        x_loss = torch.mean(((x_pred - X_labels) ** 2).sum(dim=-1))
        u_loss = torch.mean(((u_pred - U_labels) ** 2).sum(dim=-1))
    else:
        raise ValueError(f"Invalid loss_type '{loss_type}'. Choose either 'mae' or 'mse'.")

    total_loss = x_loss + u_loss
    return total_loss


# =====================================================================
#  Single-trajectory inference from one observation index
# =====================================================================

def test_model_in_single_trajectory_general(
    get_data_from_trajectory_id_function,
    loss_type,
    test_id_df,
    test_df,
    trajectory_id,
    mapping_net,
    inverse_net,
    device,
    point_indexes_observed,
    Tmax,
    n,
    pos_cols=POS_COLS,
    mom_cols=MOM_COLS,
    verbose=False,
):
    """
    General-n version. Returns the same 7-tuple as the n = 1 function, with
    x_pred, u_pred, x, u now of shape (T, n) and t of shape (T,).
    """
    if not isinstance(point_indexes_observed, list) or len(point_indexes_observed) != 1:
        raise ValueError(f"point_indexes_observed must be a list with exactly one integer, got {point_indexes_observed}")

    if not isinstance(point_indexes_observed[0], (int, np.integer)):
        raise ValueError(f"point_indexes_observed[0] must be an integer, got {type(point_indexes_observed[0])}")

    if loss_type not in ("euclidean", "mae", "mse"):
        raise ValueError(f"Unknown loss_type '{loss_type}'. Must be 'euclidean', 'mae', or 'mse'.")

    if len(pos_cols) != n:
        raise ValueError(f"pos_cols must have n={n} entries, got {len(pos_cols)}: {pos_cols}")
    if len(mom_cols) != n:
        raise ValueError(f"mom_cols must have n={n} entries, got {len(mom_cols)}: {mom_cols}")

    test_trajectory_data = get_data_from_trajectory_id_function(
        test_id_df, test_df, trajectory_ids=trajectory_id)

    missing = [c for c in list(pos_cols) + list(mom_cols) + ['t']
               if c not in test_trajectory_data.columns]
    if missing:
        raise ValueError(f"columns missing from trajectory data: {missing}")

    x = torch.as_tensor(test_trajectory_data[list(pos_cols)].to_numpy(dtype=np.float32),
                        device=device)                                       # (T, n)
    u = torch.as_tensor(test_trajectory_data[list(mom_cols)].to_numpy(dtype=np.float32),
                        device=device)                                       # (T, n)
    t = torch.as_tensor(test_trajectory_data['t'].to_numpy(dtype=np.float32),
                        device=device)                                       # (T,)

    if x.dim() != 2 or u.dim() != 2 or t.dim() != 1:
        raise ValueError(
            f"x, u must be 2-dimensional (T, n) and t 1-dimensional, got dims "
            f"{x.dim()}, {u.dim()}, {t.dim()}")
    if x.shape[1] != n or u.shape[1] != n:
        raise ValueError(f"x, u must have DOF dim n={n}, got {x.shape[1]}, {u.shape[1]}")

    obs_idx = point_indexes_observed[0]
    T = len(t)

    if obs_idx < 0 or obs_idx >= T:
        raise ValueError(f"point_indexes_observed[0]={obs_idx} is out of bounds for trajectory of length {T}")

    t_np = t.detach().cpu().numpy()
    query_indices = [i for i in range(T) if i != obs_idx]
    query_times = t_np[query_indices]

    pred_dict, forward_pass_count = Autoregressive_Handoff_Inference_Protocol_general(
        mapping_net=mapping_net,
        inverse_net=inverse_net,
        x_obs=x[obs_idx],                    # (n,)
        u_obs=u[obs_idx],                    # (n,)
        t_obs=t_np[obs_idx],
        t_min=t_np[0],
        t_max=t_np[-1],
        Tmax=Tmax,
        query_times=query_times,
        query_indices=query_indices,
        device=device,
        n=n,
    )

    # Assemble predictions. pred_dict entries are (n,) float32 numpy arrays, so
    # we stage in numpy and transfer once (pure copies; bit-exact, but avoids
    # T-1 host<->device round trips that scalar assignment would incur).
    x_pred_np = np.empty((T, n), dtype=np.float32)
    u_pred_np = np.empty((T, n), dtype=np.float32)
    x_pred_np[obs_idx] = x[obs_idx].detach().cpu().numpy()
    u_pred_np[obs_idx] = u[obs_idx].detach().cpu().numpy()
    for idx in query_indices:
        x_pred_np[idx] = pred_dict[idx][0]
        u_pred_np[idx] = pred_dict[idx][1]
    x_pred = torch.as_tensor(x_pred_np, device=device)                       # (T, n)
    u_pred = torch.as_tensor(u_pred_np, device=device)                       # (T, n)

    if verbose:
        total_passes = forward_pass_count['mapping_net'] + forward_pass_count['inverse_net']
        print(f"\n=== Forward Pass Count ===")
        print(f"mapping_net calls: {forward_pass_count['mapping_net']}")
        print(f"inverse_net calls: {forward_pass_count['inverse_net']}")
        print(f"Total forward passes: {total_passes}")

    if loss_type == "euclidean":
        pred_loss_full_trajectory = prediction_loss_euclidean_general(
            x_pred=x_pred, u_pred=u_pred, X_labels=x, U_labels=u, n=n)
    elif loss_type in ("mae", "mse"):
        pred_loss_full_trajectory = prediction_loss_general(
            x_pred=x_pred, u_pred=u_pred, X_labels=x, U_labels=u, n=n, loss_type=loss_type)

    return pred_loss_full_trajectory, x_pred, u_pred, x, u, t, forward_pass_count


# =====================================================================
#  Varying-observation sensitivity analysis
# =====================================================================

def test_model_varying_observation_single_trajectory_general(
    get_data_from_trajectory_id_function,
    loss_type,
    test_id_df,
    test_df,
    trajectory_id,
    mapping_net,
    inverse_net,
    device,
    Tmax,
    n,
    pos_cols=POS_COLS,
    mom_cols=MOM_COLS,
    energy_function=None,
    verbose=False,
):
    """
    General-n single-trajectory diagnostic over every observation choice.

    `energy_function` replaces the n = 1 `case` + (k, mass, g, length, constant)
    parameters: a callable energy(q, p) taking numpy arrays of shape (..., n)
    and returning shape (...). None disables energy tracking, exactly as
    case=None did.

    Returns the same dict as the n = 1 version, with:
        'x', 'u'                     : (T, n)
        'all_x_pred', 'all_u_pred'   : (N, T, n)
        'pointwise_errors'           : (N, T)      [DOF axis collapsed]
        'energy_true'                : (T,)
        'all_energy_pred'            : (N, T)
        'case'                       : energy_function.__name__ or None
    plus 'n', 'pos_cols', 'mom_cols'.
    """
    if loss_type not in ("euclidean", "mae", "mse"):
        raise ValueError(f"Unknown loss_type '{loss_type}'. Must be 'euclidean', 'mae', or 'mse'.")
    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")
    if len(pos_cols) != n:
        raise ValueError(f"pos_cols must have n={n} entries, got {len(pos_cols)}: {pos_cols}")
    if len(mom_cols) != n:
        raise ValueError(f"mom_cols must have n={n} entries, got {len(mom_cols)}: {mom_cols}")
    if energy_function is not None and not callable(energy_function):
        raise ValueError(f"energy_function must be callable or None, got {type(energy_function)}")

    # Load trajectory once
    test_trajectory_data = get_data_from_trajectory_id_function(
        test_id_df, test_df, trajectory_ids=trajectory_id)

    missing = [c for c in list(pos_cols) + list(mom_cols) + ['t']
               if c not in test_trajectory_data.columns]
    if missing:
        raise ValueError(f"columns missing from trajectory data: {missing}")

    x = torch.as_tensor(test_trajectory_data[list(pos_cols)].to_numpy(dtype=np.float32),
                        device=device)                                       # (T, n)
    u = torch.as_tensor(test_trajectory_data[list(mom_cols)].to_numpy(dtype=np.float32),
                        device=device)                                       # (T, n)
    t = torch.as_tensor(test_trajectory_data['t'].to_numpy(dtype=np.float32),
                        device=device)                                       # (T,)

    T = len(t)
    if T < 2:
        raise ValueError(f"Need at least two time points to run observation-choice analysis, got {T}")
    if x.shape[1] != n or u.shape[1] != n:
        raise ValueError(f"x, u must have DOF dim n={n}, got {x.shape[1]}, {u.shape[1]}")
    N = T  # one run per observation point

    # Pre-convert ground truth to numpy once
    t_np = t.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()                                          # (T, n)
    u_np = u.detach().cpu().numpy()                                          # (T, n)

    per_run_losses = np.empty(N)
    pointwise_errors = np.empty((N, T))
    all_x_pred = np.empty((N, T, n))
    all_u_pred = np.empty((N, T, n))
    all_mapping_net_calls = np.empty(N, dtype=int)
    all_inverse_net_calls = np.empty(N, dtype=int)

    # Energy computation setup
    if energy_function is not None:
        all_energy_pred = np.empty((N, T))
        energy_true = energy_function(x_np, u_np)                            # (T,)
        energy_true = np.asarray(energy_true)
        if energy_true.shape != (T,):
            raise ValueError(
                f"energy_function must return shape (T,) = ({T},) for (T, n) inputs, "
                f"got {energy_true.shape}")

    for obs_idx in range(N):
        pred_loss, x_pred, u_pred, _, _, _, fwd_count = test_model_in_single_trajectory_general(
            get_data_from_trajectory_id_function=get_data_from_trajectory_id_function,
            loss_type=loss_type,
            test_id_df=test_id_df,
            test_df=test_df,
            trajectory_id=trajectory_id,
            mapping_net=mapping_net,
            inverse_net=inverse_net,
            device=device,
            point_indexes_observed=[obs_idx],
            Tmax=Tmax,
            n=n,
            pos_cols=pos_cols,
            mom_cols=mom_cols,
            verbose=False,
        )

        per_run_losses[obs_idx] = pred_loss.item()

        x_pred_np = x_pred.detach().cpu().numpy()                            # (T, n)
        u_pred_np = u_pred.detach().cpu().numpy()                            # (T, n)
        all_x_pred[obs_idx] = x_pred_np
        all_u_pred[obs_idx] = u_pred_np

        if energy_function is not None:
            all_energy_pred[obs_idx] = energy_function(x_pred_np, u_pred_np)

        # DOF axis summed (phase-space quantities), bitwise-identical at n = 1
        if loss_type == "euclidean":
            pointwise_errors[obs_idx] = np.sqrt(
                ((x_pred_np - x_np)**2).sum(-1) + ((u_pred_np - u_np)**2).sum(-1))
        elif loss_type == "mae":
            pointwise_errors[obs_idx] = (
                np.abs(x_pred_np - x_np).sum(-1) + np.abs(u_pred_np - u_np).sum(-1))
        elif loss_type == "mse":
            pointwise_errors[obs_idx] = (
                ((x_pred_np - x_np)**2).sum(-1) + ((u_pred_np - u_np)**2).sum(-1))

        if verbose:
            print(f"Observation index {obs_idx}/{N-1}, loss: {per_run_losses[obs_idx]:.6f}")

        all_mapping_net_calls[obs_idx] = fwd_count['mapping_net']
        all_inverse_net_calls[obs_idx] = fwd_count['inverse_net']

    # === Summary statistics over runs ===
    mean_loss = np.mean(per_run_losses)
    std_loss = np.std(per_run_losses, ddof=1)
    median_loss = np.median(per_run_losses)
    q1_loss = np.percentile(per_run_losses, 25)
    q3_loss = np.percentile(per_run_losses, 75)
    iqr_loss = q3_loss - q1_loss
    min_loss = np.min(per_run_losses)
    max_loss = np.max(per_run_losses)

    summary_stats = {
        'mean': mean_loss,
        'std': std_loss,
        'median': median_loss,
        'q1': q1_loss,
        'q3': q3_loss,
        'iqr': iqr_loss,
        'min': min_loss,
        'max': max_loss,
    }

    representative_idx = int(np.argmin(np.abs(per_run_losses - median_loss)))

    if verbose:
        print(f"\n=== Summary Statistics ===")
        print(f"Mean loss: {mean_loss:.6f}")
        print(f"Std loss: {std_loss:.6f}")
        print(f"Median loss: {median_loss:.6f}")
        print(f"IQR: [{q1_loss:.6f}, {q3_loss:.6f}]")
        print(f"Range: [{min_loss:.6f}, {max_loss:.6f}]")
        print(f"Representative run: obs_idx={representative_idx}, loss={per_run_losses[representative_idx]:.6f}")

    # === Per-time-point statistics, excluding self-observation ===
    per_timepoint_median = np.empty(T)
    per_timepoint_p25 = np.empty(T)
    per_timepoint_p75 = np.empty(T)

    for j in range(T):
        mask = np.ones(N, dtype=bool)
        mask[j] = False
        errors_at_j = pointwise_errors[mask, j]

        per_timepoint_median[j] = np.median(errors_at_j)
        per_timepoint_p25[j] = np.percentile(errors_at_j, 25)
        per_timepoint_p75[j] = np.percentile(errors_at_j, 75)

    per_timepoint_stats = {
        'median': per_timepoint_median,
        'p25': per_timepoint_p25,
        'p75': per_timepoint_p75,
    }

    # === Per-time-point energy statistics, excluding self-observation ===
    if energy_function is not None:
        energy_timepoint_median = np.empty(T)
        energy_timepoint_p25 = np.empty(T)
        energy_timepoint_p75 = np.empty(T)
        energy_timepoint_p5 = np.empty(T)
        energy_timepoint_p95 = np.empty(T)

        for j in range(T):
            mask = np.ones(N, dtype=bool)
            mask[j] = False
            energy_at_j = all_energy_pred[mask, j]

            energy_timepoint_median[j] = np.median(energy_at_j)
            energy_timepoint_p25[j] = np.percentile(energy_at_j, 25)
            energy_timepoint_p75[j] = np.percentile(energy_at_j, 75)
            energy_timepoint_p5[j] = np.percentile(energy_at_j, 5)
            energy_timepoint_p95[j] = np.percentile(energy_at_j, 95)

        energy_stats = {
            'median': energy_timepoint_median,
            'p25': energy_timepoint_p25,
            'p75': energy_timepoint_p75,
            'p5': energy_timepoint_p5,
            'p95': energy_timepoint_p95,
        }
    else:
        energy_true = None
        all_energy_pred = None
        energy_stats = None

    # === Forward pass count statistics ===
    all_total_calls = all_mapping_net_calls + all_inverse_net_calls
    forward_pass_stats = {
        'mapping_net_mean': np.mean(all_mapping_net_calls),
        'mapping_net_std': np.std(all_mapping_net_calls, ddof=1),
        'mapping_net_min': int(np.min(all_mapping_net_calls)),
        'mapping_net_max': int(np.max(all_mapping_net_calls)),
        'inverse_net_mean': np.mean(all_inverse_net_calls),
        'inverse_net_std': np.std(all_inverse_net_calls, ddof=1),
        'inverse_net_min': int(np.min(all_inverse_net_calls)),
        'inverse_net_max': int(np.max(all_inverse_net_calls)),
        'total_mean': np.mean(all_total_calls),
        'total_std': np.std(all_total_calls, ddof=1),
        'total_min': int(np.min(all_total_calls)),
        'total_max': int(np.max(all_total_calls)),
    }
    if verbose:
        print(f"\n=== Forward Pass Count Statistics (over {N} obs. choices) ===")
        print(f"MappingNet calls:  mean={forward_pass_stats['mapping_net_mean']:.1f} "
              f"(std={forward_pass_stats['mapping_net_std']:.1f}), "
              f"range=[{forward_pass_stats['mapping_net_min']}, {forward_pass_stats['mapping_net_max']}]")
        print(f"InverseNet calls:  mean={forward_pass_stats['inverse_net_mean']:.1f} "
              f"(std={forward_pass_stats['inverse_net_std']:.1f}), "
              f"range=[{forward_pass_stats['inverse_net_min']}, {forward_pass_stats['inverse_net_max']}]")
        print(f"Total calls:       mean={forward_pass_stats['total_mean']:.1f} "
              f"(std={forward_pass_stats['total_std']:.1f}), "
              f"range=[{forward_pass_stats['total_min']}, {forward_pass_stats['total_max']}]")

    results = {
        'per_run_losses': per_run_losses,
        'pointwise_errors': pointwise_errors,
        'summary_stats': summary_stats,
        'per_timepoint_stats': per_timepoint_stats,
        'representative_idx': representative_idx,
        't': t_np,
        'x': x_np,
        'u': u_np,
        'trajectory_id': trajectory_id,
        'loss_type': loss_type,
        'N': N,
        'T': T,
        'all_x_pred': all_x_pred,
        'all_u_pred': all_u_pred,
        'energy_true': energy_true,
        'all_energy_pred': all_energy_pred,
        'energy_stats': energy_stats,
        'case': (energy_function.__name__ if energy_function is not None else None),
        'forward_pass_stats': forward_pass_stats,
        'all_mapping_net_calls': all_mapping_net_calls,
        'all_inverse_net_calls': all_inverse_net_calls,
        'n': n,
        'pos_cols': tuple(pos_cols),
        'mom_cols': tuple(mom_cols),
    }

    return results


# =====================================================================
#  Plotting
# =====================================================================

def plot_varying_observation_results_general(
    results, obs_idx, phase_coords=None, figsize=(28, 7),
    connect_points=True, iqr_alpha=0.4,
):
    """
    General-n version. Identical to the n = 1 plot except that panel 3 shows the
    two phase-space coordinates named in `phase_coords`.

    Args:
        results: dict from test_model_varying_observation_single_trajectory_general
        obs_idx: int, which observation-run to display in panels 2 and 3
        phase_coords: tuple of exactly two distinct names drawn from
            pos_cols + mom_cols, e.g. ('x','px'), ('y','px'), ('x','y').
            Default (pos_cols[0], mom_cols[0]), which is the n = 1 (q, p) plot.
        figsize, connect_points, iqr_alpha: as in the n = 1 version.
    """
    t = results['t']
    per_timepoint_stats = results['per_timepoint_stats']
    summary_stats = results['summary_stats']
    trajectory_id = results['trajectory_id']
    loss_type = results['loss_type']
    N = results['N']
    T = results['T']
    x_gt = results['x']                       # (T, n)
    u_gt = results['u']                       # (T, n)
    all_x_pred = results['all_x_pred']        # (N, T, n)
    all_u_pred = results['all_u_pred']
    pointwise_errors = results['pointwise_errors']
    pos_cols = tuple(results['pos_cols'])
    mom_cols = tuple(results['mom_cols'])

    if not (0 <= obs_idx < N):
        raise ValueError(f"obs_idx={obs_idx} is out of bounds for N={N} runs")

    # ---- resolve the two phase-space coordinates ----
    all_names = list(pos_cols) + list(mom_cols)
    if len(set(all_names)) != len(all_names):
        raise ValueError(f"pos_cols + mom_cols contain duplicate names: {all_names}")
    if phase_coords is None:
        phase_coords = (pos_cols[0], mom_cols[0])
    if len(phase_coords) != 2:
        raise ValueError(f"phase_coords must name exactly two coordinates, got {phase_coords}")
    if phase_coords[0] == phase_coords[1]:
        raise ValueError(f"phase_coords must name two distinct coordinates, got {phase_coords}")
    for c in phase_coords:
        if c not in all_names:
            raise ValueError(f"unknown coordinate '{c}'; available: {all_names}")

    def _gt(name):
        return x_gt[:, pos_cols.index(name)] if name in pos_cols \
            else u_gt[:, mom_cols.index(name)]

    def _pred(name, run):
        return all_x_pred[run][:, pos_cols.index(name)] if name in pos_cols \
            else all_u_pred[run][:, mom_cols.index(name)]

    cA, cB = phase_coords
    A_gt, B_gt = _gt(cA), _gt(cB)
    A_pred, B_pred = _pred(cA, obs_idx), _pred(cB, obs_idx)

    energy_true = results.get('energy_true', None)
    all_energy_pred = results.get('all_energy_pred', None)
    energy_stats = results.get('energy_stats', None)

    obs_time = t[obs_idx]

    has_energy = energy_true is not None and energy_stats is not None
    n_plots = 3 if has_energy else 2

    if n_plots == 2:
        figsize = (20, 7)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    fig.text(
        0.005, 0.995,
        f"Trajectory {trajectory_id} | {N} obs. choices | obs. t={obs_time:.2f} (idx {obs_idx})",
        ha='left',
        va='top',
        fontsize=10,
        alpha=1.0,
    )

    if n_plots == 2:
        ax1, ax3 = axes
        ax2 = None
    else:
        ax1, ax2, ax3 = axes

    # === Plot 1: Per-time-point error IQR band ===
    ax1.fill_between(t, per_timepoint_stats['p25'], per_timepoint_stats['p75'],
                     alpha=iqr_alpha, color='red', label='IQR across obs. choices')
    ax1.plot(t, per_timepoint_stats['median'], color='red', linewidth=1.5,
             label='Median across obs. choices')
    ax1.plot(t, pointwise_errors[obs_idx], color='red', linewidth=1.2,
             linestyle='dotted', label='Selected obs. run')

    ax1.set_xlim(t[0], t[-1])
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel(f'Pointwise Error ({loss_type})')
    ax1.set_title('Prediction Error Over Time')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    selected_run_loss = results['per_run_losses'][obs_idx]
    textstr = (
        f"Scalar loss across obs. choices:\n"
        f"  Mean (std={summary_stats['std']:.4f}): {summary_stats['mean']:.4f}\n"
        f"  Median [IQR]: {summary_stats['median']:.4f} "
        f"[{summary_stats['q1']:.4f}, {summary_stats['q3']:.4f}]\n"
        f"  Range: [{summary_stats['min']:.4f}, {summary_stats['max']:.4f}]\n"
        f"Selected obs. run: {selected_run_loss:.4f}"
    )
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=8,
             ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # === Plot 2: Energy (if available) ===
    if has_energy:
        ax2.fill_between(t, energy_stats['p25'], energy_stats['p75'],
                         alpha=iqr_alpha, color='red',
                         label='IQR across obs. choices')

        energy_run = all_energy_pred[obs_idx]
        ax2.plot(t, energy_run, color='red', linewidth=1.5,
                 label='E_pred(t) (selected obs. run)')

        ax2.plot(t, energy_true, color='blue', linewidth=2, label='E_true(t)')

        ax2.set_xlim(t[0], t[-1])
        ax2.set_xlabel('Time (t)')
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy: Predicted vs True')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        energy_errors_run = np.abs(energy_run - energy_true)
        mean_energy_error = np.mean(energy_errors_run)
        max_energy_error = np.max(energy_errors_run)
        true_energy_mean = np.mean(np.abs(energy_true))
        relative_error = (
            mean_energy_error / true_energy_mean * 100
            if true_energy_mean > 1e-12
            else np.nan
        )

        energy_textstr = (
            f"Selected obs. run:\n"
            f"  Mean |E_pred - E_true|: {mean_energy_error:.4f}\n"
            f"  Max |E_pred - E_true|: {max_energy_error:.4f}\n"
            f"  Relative error: {relative_error:.2f}%"
        )
        ax2.text(0.02, 0.98, energy_textstr, transform=ax2.transAxes, fontsize=8,
                 ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # === Plot 3: Phase space (selected coordinate pair) ===
    scatter_gt = ax3.scatter(A_gt, B_gt, c=t, cmap='Blues', s=60, alpha=0.8,
                             edgecolors='darkblue', linewidths=0.5, zorder=3)
    scatter_pred = ax3.scatter(A_pred, B_pred, c=t, cmap='Reds', s=60,
                               alpha=0.8, edgecolors='darkred',
                               linewidths=0.5, zorder=3)

    if connect_points:
        ax3.plot(A_gt, B_gt, color='blue', alpha=0.3, linewidth=1.5, zorder=2)
        ax3.plot(A_pred, B_pred, color='red', alpha=0.3,
                 linewidth=1.5, zorder=2)

    ax3.scatter(A_gt[0], B_gt[0], c='blue', marker='*', s=400,
                edgecolors='black', linewidths=2.5, zorder=5)
    ax3.scatter(A_pred[0], B_pred[0], c='red', marker='*', s=400,
                edgecolors='black', linewidths=2.5, zorder=5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=10, markeredgecolor='darkblue',
               label='Ground Truth'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, markeredgecolor='darkred',
               label='Prediction'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='blue',
               markersize=15, markeredgecolor='black',
               label='Start (Ground Truth)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=15, markeredgecolor='black',
               label='Start (Prediction)'),
    ]
    ax3.legend(handles=legend_elements, fontsize=7, loc='best')

    cbar = plt.colorbar(scatter_pred, ax=ax3, pad=0.02, fraction=0.046)
    cbar.set_label('Time', fontsize=10)

    ax3.set_xlabel(cA, fontsize=12)
    ax3.set_ylabel(cB, fontsize=12)
    ax3.set_title(
        f'Phase Space ({cA}, {cB}): Ground Truth vs Prediction (selected obs. run)',
        fontsize=11,
    )
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()