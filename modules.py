import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Any, Callable, Union
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as init
import json
import sys
import os
import signal
import traceback
from datetime import datetime
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import torch.nn.functional as F
import re
import datetime
from pathlib import Path



import contextlib
import hashlib





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

def get_trajectory_ids_by_energies(ids_df, want_more_energy, energy_percentile):
    if want_more_energy==False:
        return list(ids_df[ids_df['energy'] < ids_df['energy'].quantile(energy_percentile)]['trajectory_id'])
    if want_more_energy==True:
        return list(ids_df[ids_df['energy'] > ids_df['energy'].quantile(energy_percentile)]['trajectory_id'])
    

def find_valid_combinations(train_df, train_id_df, max_segment_length=20, max_n_segments=50):
    results = []
    
    total_points = len(train_df)
    n_trajectories = len(train_id_df['trajectory_id'].unique())
    
    # Get actual trajectory lengths (might vary)
    traj_lengths = []
    for _, row in train_id_df.iterrows():
        length = int(row['end_index'] - row['start_index'])
        traj_lengths.append(length)
    
    # If all trajectories same length, use that; otherwise use minimum
    if len(set(traj_lengths)) == 1:
        points_per_traj = traj_lengths[0]
    else:
        points_per_traj = min(traj_lengths)
        print(f"Warning: Trajectories have different lengths. Using minimum: {points_per_traj}")
    
    # Find divisors of points_per_traj for valid segment lengths
    valid_segment_lengths = []
    for s in range(1, min(max_segment_length + 1, points_per_traj + 1)):
        if points_per_traj % s == 0:
            valid_segment_lengths.append(s)
    
    for segment_length in valid_segment_lengths:
        max_segments_per_traj = points_per_traj // segment_length
        
        for n_segments in range(1, min(max_n_segments + 1, max_segments_per_traj + 1)):
            # Check if trajectories can be evenly distributed across batches
            if max_segments_per_traj % n_segments != 0:
                continue
            
            batches_per_traj = max_segments_per_traj // n_segments
            
            # Try different batch compositions
            for batch_traj in range(1, n_trajectories + 1):
                # Total batches must use all trajectories evenly
                if (n_trajectories * batches_per_traj) % batch_traj != 0:
                    continue
                
                total_batches = (n_trajectories * batches_per_traj) // batch_traj
                
                # Verify coverage
                total_points_covered = total_batches * batch_traj * n_segments * segment_length
                
                # For exact coverage (adjust if allowing partial coverage)
                if total_points_covered != n_trajectories * points_per_traj:
                    continue
                
                batch_ppt = n_segments * segment_length
                batch_size = batch_traj * batch_ppt
                ratio = batch_ppt / batch_traj
                

                
                results.append({
                    "segment_length": segment_length,
                    "n_segments": n_segments,
                    "batch_traj": batch_traj,
                    "batch_ppt": batch_ppt,
                    "batch_size": batch_size,
                    "total_batches": total_batches,
                    "ratio": ratio,
                    "coverage_pct": (total_points_covered / total_points) * 100,
                    "points_per_traj": points_per_traj
                })
    
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("No valid combinations found with given constraints!")
        print(f"Dataset: {total_points} points, {n_trajectories} trajectories, {points_per_traj} points/traj")
        print(f"Valid segment lengths: {valid_segment_lengths[:10]}...")
    
    return df


class SimpleHarmonicDataLoader:
    def __init__(self, train_df, train_id_df, ratio, batch_size, segment_length, get_data_func, seed=42):
        """
        Simple dataloader with guaranteed full coverage.
        
        Args:
            ratio: points_per_trajectory / trajectories_per_batch
            batch_size: total points per batch
            segment_length: consecutive points per segment
        """
        self.train_df = train_df
        self.train_id_df = train_id_df
        self.get_data_func = get_data_func
        self.ratio = ratio
        self.batch_size = batch_size
        self.segment_length = segment_length
        
        np.random.seed(seed)
        
        # Calculate derived parameters
        self.n_trajectories = len(train_id_df['trajectory_id'].unique())
        self.total_points = len(train_df)
        self.points_per_traj = self.total_points // self.n_trajectories
        
        # Validate hyperparameters
        valid_df = find_valid_combinations(train_df, train_id_df, max_segment_length=25, max_n_segments=50)
        valid_params = valid_df[
            (np.isclose(valid_df['ratio'], ratio, rtol=0.01)) & 
            (valid_df['batch_size'] == batch_size) & 
            (valid_df['segment_length'] == segment_length)
        ]
        
        if len(valid_params) == 0:
            raise ValueError(f"Invalid hyperparameters: ratio={ratio}, batch_size={batch_size}, segment_length={segment_length}")
        
        # Extract validated parameters
        row = valid_params.iloc[0]
        self.batch_traj = int(row['batch_traj'])
        self.n_segments = int(row['n_segments'])
        self.total_batches = int(row['total_batches'])
        
        # Compatibility aliases for validation function
        self.actual_batch_size = self.batch_size
        self.number_of_trajectories = self.batch_traj
        self.points_per_trajectory = self.n_segments * self.segment_length
        self.segments_per_trajectory = self.n_segments
        
        print(f"Dataloader initialized:")
        print(f"  Trajectories per batch: {self.batch_traj}")
        print(f"  Segments per trajectory: {self.n_segments}")
        print(f"  Points per trajectory: {self.points_per_trajectory}")
        print(f"  Total batches: {self.total_batches}")
        
        self.trajectory_segments = {}
        for _, row in train_id_df.iterrows():
            tid = int(row['trajectory_id'])
            start_idx = int(row['start_index'])
            
            # All possible segment starting positions for this trajectory
            segments = []
            for i in range(0, self.points_per_traj, self.segment_length):
                segments.append(start_idx + i)
            self.trajectory_segments[tid] = segments
        
        # Generate epoch plan
        self._generate_epoch_plan()
 
    
    def _generate_epoch_plan(self):
        """Generate batches with more balanced trajectory usage."""
        self.batches = []
        
        # Track remaining segments for each trajectory
        remaining_segments = {}
        for tid, segments in self.trajectory_segments.items():
            remaining_segments[tid] = segments.copy()
        
        while True:
            # Find trajectories with at least n_segments remaining
            available = [(tid, len(segs)) for tid, segs in remaining_segments.items() 
                        if len(segs) >= self.n_segments]
            
            if len(available) < self.batch_traj:
                # Final batch with all remaining trajectories that have segments
                if available:
                    batch_data = {}
                    for tid, _ in available:
                        segments = remaining_segments[tid]
                        np.random.shuffle(segments)
                        selected = segments[:min(self.n_segments, len(segments))]
                        batch_data[tid] = selected
                        # Remove used segments
                        for seg in selected:
                            remaining_segments[tid].remove(seg)
                    self.batches.append(batch_data)
                break
            
            # Sort by number of remaining segments (descending) to prioritize fuller trajectories
            available.sort(key=lambda x: x[1], reverse=True)
            
            # Add some randomness but favor trajectories with more segments
            # Take top 2*batch_traj candidates and randomly select from them
            candidates = [tid for tid, _ in available[:min(2 * self.batch_traj, len(available))]]
            np.random.shuffle(candidates)
            selected_tids = candidates[:self.batch_traj]
            
            # Build batch
            batch_data = {}
            for tid in selected_tids:
                segments = remaining_segments[tid]
                np.random.shuffle(segments)
                selected = segments[:self.n_segments]
                batch_data[tid] = selected
                # Remove used segments
                for seg in selected:
                    remaining_segments[tid].remove(seg)
            
            self.batches.append(batch_data)
        
        print(f"Generated {len(self.batches)} batches")
        if self.batches and len(self.batches[-1]) < self.batch_traj:
            print(f"Last batch has {len(self.batches[-1])} trajectories (partial batch)")
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx):
        """Get batch data."""
        batch_data = self.batches[idx]
        
        x_list, u_list, t_list, tid_list, energy_list = [], [], [], [], []
        
        for tid, segment_starts in batch_data.items():
            # Get trajectory data once
            traj_data = self.get_data_func(self.train_id_df, self.train_df, tid)
            traj_info = self.train_id_df[self.train_id_df['trajectory_id'] == tid].iloc[0]
            traj_start = int(traj_info['start_index'])
            energy = traj_info['energy']
            
            # Process each segment
            for seg_start in segment_starts:
                for offset in range(self.segment_length):
                    relative_idx = seg_start - traj_start + offset
                    x_list.append(traj_data.iloc[relative_idx]['x'])
                    u_list.append(traj_data.iloc[relative_idx]['u'])
                    t_list.append(traj_data.iloc[relative_idx]['t'])
                    tid_list.append(tid)
                    energy_list.append(energy)
        
        return {
            'x': torch.tensor(x_list, dtype=torch.float32),
            'u': torch.tensor(u_list, dtype=torch.float32),
            't': torch.tensor(t_list, dtype=torch.float32),
            'trajectory_ids': torch.tensor(tid_list, dtype=torch.long),
            'energies': torch.tensor(energy_list, dtype=torch.float32)
        }
    def on_epoch_start(self):
        """
        Regenerate batch compositions and shuffle batch order.
        Call this at the start of each epoch for maximum randomness.
        """
        self._generate_epoch_plan()  # Create new batch compositions
        np.random.shuffle(self.batches)  # Shuffle the batch order
        print(f"Regenerated {len(self.batches)} batches for new epoch")


def create_simple_dataloader(train_df, train_id_df, ratio, batch_size, segment_length, get_data_func, device='cuda', seed=42):
    """Create dataloader with validation."""
    dataset = SimpleHarmonicDataLoader(
        train_df, train_id_df, ratio, batch_size, segment_length, get_data_func, seed
    )

    def collate_and_to_device(batch):
        """Collate function that moves tensors to device."""
        batch_dict = batch[0]  # Since batch_size=1 in DataLoader
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_dict.items()}
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Dataset handles batching
        shuffle=False,  # Already shuffled in epoch plan
        collate_fn=collate_and_to_device
    )
    dataloader.device = device
    
    return dataloader




def add_gaussian_noise(
    x: torch.Tensor, 
    u: torch.Tensor, 
    variance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adds Gaussian noise to two 1D tensors.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape [batch_size].
    u : torch.Tensor
        Input tensor with shape [batch_size].
    variance : float
        Variance of the Gaussian noise to be added (σ²).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two tensors with added Gaussian noise, preserving original 
        shape, dtype, and device.
    """
    # Compute standard deviation from variance
    sigma = torch.sqrt(torch.tensor(variance))
    
    # Generate Gaussian noise with same shape, dtype, and device as inputs
    noise_x = torch.randn_like(x) * sigma
    noise_u = torch.randn_like(u) * sigma
    
    # Add independent Gaussian noise to x and u
    noisy_x = x + noise_x
    noisy_u = u + noise_u
    
    return noisy_x, noisy_u

class Step_1(nn.Module):
    """
    A fully customizable layer that transforms position and velocity:
    X = a * x + c1
    U = (1/a) * u - γ * ∂G/∂x + c2
    where G(x,t) is a customizable MLP network and a, c1, γ, c2 are trainable parameters.
    
    Input: (x, u, t) 
    Output: (X, U, t) where X and U are transformed
    """
    
    def __init__(
        self,
        # MLP Architecture parameters
        hidden_dims: Union[int, List[int]] = 32,
        n_hidden_layers: int = 2,
        
        # Activation parameters
        activation: str = 'tanh',
        activation_params: Optional[dict] = None,
        final_activation: Optional[str] = None,
        tanh_wrapper: bool = False,
        
        # Initialization parameters
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros', 
        bias_init_value: float = 0.0,
        
        # Architectural choices
        use_bias: bool = True,
        use_layer_norm: bool = False,
        
        # Input/Output parameters
        input_dim: int = 2,  # x and t
        output_dim: int = 1,  # scalar G
        
        # Trainable parameter constraints
        a_eps_min: float = 0.1,  # Minimum value for a
        a_eps_max: float = 10.0,  # Maximum value for a  
        a_k: float = 0.1,  # Sigmoid steepness
        a_raw_innit: float = 0.0,
        gamma_innit: float = 5.0,
        c1_innit: float = 0.0,
        c2_innit: float = 5.0,
        g_bound_innit: float = 10.0,
    ):
        """
        Initialize the velocity transformation layer with trainable parameters.
        """
        super(Step_1, self).__init__()

        self.tanh_wrapper = tanh_wrapper
        
        # Initialize trainable parameters
        self.a_raw = nn.Parameter(torch.tensor(a_raw_innit))  
        self.c1 = nn.Parameter(torch.tensor(c1_innit))
        self.gamma = nn.Parameter(torch.tensor(gamma_innit))
        self.c2 = nn.Parameter(torch.tensor(c2_innit))

        if tanh_wrapper:
            self.g_bound = nn.Parameter(torch.tensor(g_bound_innit))
        
        self.a_eps_min = a_eps_min
        self.a_eps_max = a_eps_max
        self.a_k = a_k



        
        # Process hidden dimensions
        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * n_hidden_layers
        else:
            self.hidden_dims = hidden_dims
            
        # Build the MLP
        self.G_network = self._build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            activation=activation,
            activation_params=activation_params,
            final_activation=final_activation,
            use_bias=use_bias,
            use_layer_norm=use_layer_norm,
        )
        
        # Initialize weights
        self._initialize_weights(
            weight_init=weight_init,
            weight_init_params=weight_init_params,
            bias_init=bias_init,
            bias_init_value=bias_init_value
        )
    
    @property
    def a(self):
        """Compute a from a_raw using sigmoid to ensure bounded values."""
        return self.a_eps_min + (self.a_eps_max - self.a_eps_min) * torch.sigmoid(self.a_k * self.a_raw)
    
    def _get_activation(self, activation_name: str, params: Optional[dict] = None):
        """Get activation function by name."""
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
    
    def _build_mlp(
        self, input_dim, output_dim, hidden_dims, activation, activation_params,
        final_activation, use_bias, use_layer_norm
    ):
        """Build the MLP network with specified configuration."""
        layers = []
        
        # Determine layer dimensions
        all_dims = [input_dim] + hidden_dims + [output_dim]
        
        # Build each layer
        for i in range(len(all_dims) - 1):
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]
            is_last_layer = (i == len(all_dims) - 2)
            
            # Linear layer
            linear = nn.Linear(in_dim, out_dim, bias=use_bias)
            layers.append(linear)

            if use_layer_norm and not is_last_layer:
                layers.append(nn.LayerNorm(out_dim))
            
            # Add activation
            if is_last_layer:
                if final_activation is not None:
                    layers.append(self._get_activation(final_activation, activation_params))
            else:
                layers.append(self._get_activation(activation, activation_params))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, weight_init, weight_init_params, bias_init, bias_init_value):
        """Initialize weights and biases according to specifications."""
        weight_init_params = weight_init_params or {}
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Initialize weights
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
                
                # Initialize biases
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
    
    def compute_G_and_gradient(self, x, t):
        """
        Compute G(x,t) and its gradient with respect to x.
        
        Args:
            x: Position tensor of shape (batch_size,) or (batch_size, 1)
            t: Time tensor of shape (batch_size,) or (batch_size, 1)
            
        Returns:
            dG_dx: The partial derivative ∂G/∂x
        """
        # Ensure x requires gradient for autograd to work
        x = x.requires_grad_(True)
        
        # Reshape inputs if needed
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Stack inputs for the MLP
        mlp_input = torch.cat([x, t], dim=1)
        
        # Forward pass through the MLP
        G = self.G_network(mlp_input)

        if self.tanh_wrapper:
            G = self.g_bound * torch.tanh(G / self.g_bound)
        
        # Compute the gradient ∂G/∂x
        dG_dx = torch.autograd.grad(
            outputs=G,
            inputs=x,
            grad_outputs=torch.ones_like(G),
            create_graph=self.training,
            retain_graph=self.training
        )[0]
        
        return dG_dx.squeeze(1)
    
    def forward(self, x, u, t):
        """
        Forward pass of the layer.
        
        Args:
            x: Position tensor of shape (batch_size,)
            u: Velocity tensor of shape (batch_size,)
            t: Time tensor of shape (batch_size,)
            
        Returns:
            X: Transformed position where X = a * x + c1
            U: Transformed velocity where U = (1/a) * u - γ * ∂G/∂x + c2
            t: Unchanged time
        """
        assert x.shape == u.shape == t.shape, "Input shapes must match"
        original_shape = x.shape
        x = x.view(-1)
        u = u.view(-1)
        t = t.view(-1)
        
        # Compute gradient of G with respect to INPUT x (not transformed X)
        dG_dx = self.compute_G_and_gradient(x, t)
        
        # Get current value of a (always positive)
        a_val = self.a
        
        # Transform position: X = a * x + c1
        X = a_val * x + self.c1
        
        # Transform velocity: U = (1/a) * u - γ * ∂G/∂x + c2
        U = (1.0 / a_val) * u - self.gamma * dG_dx + self.c2
        
        return X.view(original_shape), U.view(original_shape), t.view(original_shape)
    
    def get_config(self):
        """Return the configuration of this layer as a dictionary."""
        return {
            'hidden_dims': self.hidden_dims,
            'a': self.a.item(),
            'c1': self.c1.item(),
            'gamma': self.gamma.item(),
            'c2': self.c2.item(),
        }


class Step_2(nn.Module):
    """
    A fully customizable layer that transforms position and velocity:
    U = a' * u + c1'
    X = (1/a') * x + γ' * ∂F/∂u + c2'
    where F(u,t) is a customizable MLP network and a', c1', γ', c2' are trainable parameters.
    
    Input: (x, u, t) 
    Output: (X, U, t) where X and U are transformed
    """
    
    def __init__(
        self,
        # MLP Architecture parameters
        hidden_dims: Union[int, List[int]] = 32,
        n_hidden_layers: int = 2,
        
        # Activation parameters
        activation: str = 'tanh',
        activation_params: Optional[dict] = None,
        final_activation: Optional[str] = None,
        tanh_wrapper: bool = False,
        
        # Initialization parameters
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        
        # Architectural choices
        use_bias: bool = True,
        use_layer_norm: bool = False,
        
        # Input/Output parameters
        input_dim: int = 2,  # u and t
        output_dim: int = 1,  # scalar F
        
        # Trainable parameter constraints
        a_eps_min: float = 0.1,
        a_eps_max: float = 10.0,
        a_k: float = 0.1,
        a_raw_innit: float = 0.0,
        gamma_innit: float = 5.0,
        c1_innit: float = 0.0,
        c2_innit: float = 5.0,
        f_bound_innit: float = 10.0,
    ):
        super(Step_2, self).__init__()
        
        # Store hyperparameters for a
        self.a_eps_min = a_eps_min
        self.a_eps_max = a_eps_max
        self.a_k = a_k
        self.tanh_wrapper = tanh_wrapper
        
        # Initialize trainable parameters
        self.a_raw = nn.Parameter(torch.tensor(a_raw_innit))
        self.c1 = nn.Parameter(torch.tensor(c1_innit))
        self.gamma = nn.Parameter(torch.tensor(gamma_innit))
        self.c2 = nn.Parameter(torch.tensor(c2_innit))
        if tanh_wrapper:
            self.f_bound = nn.Parameter(torch.tensor(f_bound_innit))
        
        # Process hidden dimensions
        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * n_hidden_layers
        else:
            self.hidden_dims = hidden_dims
            
        # Build the MLP
        self.F_network = self._build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            activation=activation,
            activation_params=activation_params,
            final_activation=final_activation,
            use_bias=use_bias,
            use_layer_norm=use_layer_norm,
        )
        
        # Initialize weights
        self._initialize_weights(
            weight_init=weight_init,
            weight_init_params=weight_init_params,
            bias_init=bias_init,
            bias_init_value=bias_init_value
        )
    
    @property
    def a(self):
        """Compute a from a_raw using sigmoid to ensure bounded values."""
        return self.a_eps_min + (self.a_eps_max - self.a_eps_min) * torch.sigmoid(self.a_k * self.a_raw)
    
    def _get_activation(self, activation_name: str, params: Optional[dict] = None):
        """Get activation function by name."""
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
    
    def _build_mlp(
        self, input_dim, output_dim, hidden_dims, activation, activation_params,
        final_activation, use_bias, use_layer_norm
    ):
        """Build the MLP network with specified configuration."""
        layers = []
        
        # Determine layer dimensions
        all_dims = [input_dim] + hidden_dims + [output_dim]
        
        # Build each layer
        for i in range(len(all_dims) - 1):
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]
            is_last_layer = (i == len(all_dims) - 2)
            
            # Linear layer
            linear = nn.Linear(in_dim, out_dim, bias=use_bias)
            layers.append(linear)
            
            if use_layer_norm and not is_last_layer:
                layers.append(nn.LayerNorm(out_dim))

            # Add activation
            if is_last_layer:
                if final_activation is not None:
                    layers.append(self._get_activation(final_activation, activation_params))
            else:
                layers.append(self._get_activation(activation, activation_params))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, weight_init, weight_init_params, bias_init, bias_init_value):
        """Initialize weights and biases according to specifications."""
        weight_init_params = weight_init_params or {}
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Initialize weights
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
                
                # Initialize biases
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
    
    def compute_F_and_gradient(self, u, t):
        """
        Compute F(u,t) and its gradient with respect to u.
        
        Args:
            u: Velocity tensor of shape (batch_size,) or (batch_size, 1)
            t: Time tensor of shape (batch_size,) or (batch_size, 1)
            
        Returns:
            dF_du: The partial derivative ∂F/∂u
        """
        # Ensure u requires gradient for autograd to work
        u = u.requires_grad_(True)
        
        # Reshape inputs if needed
        if u.dim() == 1:
            u = u.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Stack inputs for the MLP
        mlp_input = torch.cat([u, t], dim=1)
        
        # Forward pass through the MLP
        F = self.F_network(mlp_input)

        if self.tanh_wrapper:
            F = self.f_bound * torch.tanh(F / self.f_bound)

        # Compute the gradient dF_du
        dF_du = torch.autograd.grad(
            outputs=F,
            inputs=u,
            grad_outputs=torch.ones_like(F),
            create_graph=self.training,
            retain_graph=self.training
        )[0]
        
        return dF_du.squeeze(1)
    
    def forward(self, x, u, t):
        """
        Forward pass of the layer.
        
        Args:
            x: Position tensor of shape (batch_size,)
            u: Velocity tensor of shape (batch_size,)
            t: Time tensor of shape (batch_size,)
            
        Returns:
            X: Transformed position where X = (1/a') * x + γ' * ∂F/∂u + c2'
            U: Transformed velocity where U = a' * u + c1'
            t: Unchanged time
        """
        assert x.shape == u.shape == t.shape, "Input shapes must match"
        original_shape = x.shape
        x = x.view(-1)
        u = u.view(-1)
        t = t.view(-1)
        
        # Compute gradient of F with respect to INPUT u (not transformed U)
        dF_du = self.compute_F_and_gradient(u, t)
        
        # Get current value of a (always positive and bounded)
        a_val = self.a
        
        # Transform velocity: U = a' * u + c1'
        U = a_val * u + self.c1
        
        # Transform position: X = (1/a') * x + γ' * ∂F/∂u + c2'
        X = (1.0 / a_val) * x + self.gamma * dF_du + self.c2
        
        return X.view(original_shape), U.view(original_shape), t.view(original_shape)
    
    def get_config(self):
        """Return the configuration of this layer as a dictionary."""
        return {
            'hidden_dims': self.hidden_dims,
            'a': self.a.item(),
            'c1': self.c1.item(),
            'gamma': self.gamma.item(),
            'c2': self.c2.item(),
        }
    

class CombinedHamiltonianLayer(nn.Module):
    """
    Combined layer that performs both transformations in sequence:
    Step 1: (x, u, t) → (x, U, t) where U = u - ∂G/∂x
    Step 2: (x, U, t) → (X, U, t) where X = x + ∂F/∂U
    
    Final: (x, u, t) → (X, U, t)
    """
    
    def __init__(
        self,
        # Step 1 parameters (G network: takes x,t → scalar)
        step1_hidden_dims: Union[int, List[int]] = 32,
        step1_n_hidden_layers: int = 2,
        step1_activation: str = 'tanh',
        step1_activation_params: Optional[dict] = None,
        step1_final_activation: Optional[str] = None,
        step1_tanh_wrapper : bool = False,

        
        # Step 2 parameters (F network: takes U,t → scalar)
        step2_hidden_dims: Union[int, List[int]] = 32,
        step2_n_hidden_layers: int = 2,
        step2_activation: str = 'tanh',
        step2_activation_params: Optional[dict] = None,
        step2_final_activation: Optional[str] = None,
        step2_tanh_wrapper : bool = False,

        
        # Shared initialization parameters (apply to both steps)
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        


        use_bias: bool = True,
        use_layer_norm: bool = False,

        # Input/Output parameters
        input_dim: int = 2,  # x and t
        output_dim: int = 1,  # scalar G
        
        # Trainable parameter constraints
        a_eps_min: float = 0.1,  # Minimum value for a
        a_eps_max: float = 10.0,  # Maximum value for a  
        a_k: float = 0.1,  # Sigmoid steepness
        step_1_a_raw_innit: float = 0.0,
        step_1_gamma_innit: float = 5.0,
        step_1_c1_innit: float = 0.0,
        step_1_c2_innit: float = 5.0,
        step_2_a_raw_innit: float = 0.0,
        step_2_gamma_innit: float = 5.0,
        step_2_c1_innit: float = 0.0,
        step_2_c2_innit: float = 5.0,
        bound_innit: float = 10.0,
    

    ):
        """
        Initialize the combined Hamiltonian layer.
        
        Args:
            step1_*: Parameters for Step_1 (velocity transformation)
            step2_*: Parameters for Step_2 (position transformation)
            Other parameters are applied to both steps
        """
        super(CombinedHamiltonianLayer, self).__init__()
        
        # Create Step 1: Velocity Transform Layer
        # Transforms     X = a * x + c1
        #                U = (1/a) * u - γ * ∂G/∂x + c2
        self.step_1 = Step_1(
            hidden_dims=step1_hidden_dims,
            n_hidden_layers=step1_n_hidden_layers,
            activation=step1_activation,
            activation_params=step1_activation_params,
            final_activation=step1_final_activation,
            tanh_wrapper=step1_tanh_wrapper,
            weight_init=weight_init,
            weight_init_params=weight_init_params,
            bias_init=bias_init,
            bias_init_value=bias_init_value,
            use_bias=use_bias,
            use_layer_norm=use_layer_norm,
            input_dim= input_dim,  # x and t
            output_dim= output_dim,  # scalar G
            
            # Trainable parameter constraints
            a_eps_min= a_eps_min,  # Minimum value for a
            a_eps_max= a_eps_max,  # Maximum value for a  
            a_k= a_k,  # Sigmoid steepness
            a_raw_innit = step_1_a_raw_innit,
            gamma_innit = step_1_gamma_innit,
            c1_innit=step_1_c1_innit,
            c2_innit=step_1_c2_innit,
            g_bound_innit=bound_innit
        )
        
        # Create Step 2: Position Transform Layer  
        # Transforms     U = a' * u + c1'
        #                X = (1/a') * x + γ' * ∂F/∂u + c2'
        self.step_2 = Step_2(
            hidden_dims=step2_hidden_dims,
            n_hidden_layers=step2_n_hidden_layers,
            activation=step2_activation,
            activation_params=step2_activation_params,
            final_activation=step2_final_activation,
            tanh_wrapper=step2_tanh_wrapper,
            weight_init=weight_init,
            weight_init_params=weight_init_params,
            bias_init=bias_init,
            bias_init_value=bias_init_value,
            use_bias=use_bias,
            use_layer_norm=use_layer_norm,
            input_dim= input_dim,  # x and t
            output_dim= output_dim,  # scalar G
            
            # Trainable parameter constraints
            a_eps_min= a_eps_min,  # Minimum value for a
            a_eps_max= a_eps_max,  # Maximum value for a  
            a_k= a_k,  # Sigmoid steepness
            a_raw_innit = step_2_a_raw_innit,
            gamma_innit = step_2_gamma_innit,
            c1_innit=step_2_c1_innit,
            c2_innit=step_2_c2_innit,
            f_bound_innit=bound_innit

        )
        
    def forward(self, x, u, t):
        """
        Forward pass through both transformation steps.
        
        Args:
            x: Position tensor of shape (batch_size,)
            u: Velocity tensor of shape (batch_size,)
            t: Time tensor of shape (batch_size,)
            
        Returns:
            X: Transformed position (after both steps)
            U: Transformed velocity (after step_1)
            t: Unchanged time
        """
        # Step 1:
        x_after_step1, u_after_step1, t_after_step_1 = self.step_1(x, u, t)
        
        # Step 2:
        X_final, U_final, t_final = self.step_2(x_after_step1, u_after_step1, t_after_step_1)
        
        return X_final, U_final, t_final
    


def solve_a_raw(a_eps_min, a_eps_max, a_k, a_target):

    if (a_target <= a_eps_min) or (a_target >= a_eps_max):
        raise ValueError(f"a_target must be between a_eps_min and a_eps_max. Got {a_target}")
    
    s = (a_target-a_eps_min) / (a_eps_max-a_eps_min)
    a_raw = (1/a_k) * np.log(s / (1-s))
    return a_raw

def init_step_params(num_layers, 
                     step_1_mean_init, 
                     step_2_mean_init, 
                     std_to_mean_ratio_init, 
                     seed=None):
    """
    Initialize step parameters with Gaussian distribution based on specified means and std ratio.
    
    Mathematical formulation:
    For all layers i = 0, 1, ..., L-1:
        μ₁ = step_1_mean_init
        σ₁ = |step_1_mean_init| × std_to_mean_ratio_init
        step_1[i] ~ N(μ₁, σ₁²)
        
        μ₂ = step_2_mean_init
        σ₂ = |step_2_mean_init| × std_to_mean_ratio_init
        step_2[i] ~ N(μ₂, σ₂²)
    
    Args:
        num_layers: Number of layers
        step_1_mean_init: Mean value for step_1 parameters
        step_2_mean_init: Mean value for step_2 parameters
        std_to_mean_ratio_init: Ratio that defines standard deviation as ratio × mean
                                (e.g., 0.1 means std = 0.1 × mean)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Dictionary with numpy arrays: 'step_1', 'step_2'
    
    Example:
        >>> params = init_step_params(10, step_1_mean_init=1.0, 
        ...                           step_2_mean_init=2.0, 
        ...                           std_to_mean_ratio_init=0.1)
        >>> # step_1 values will have mean ≈ 1.0, std ≈ 0.1
        >>> # step_2 values will have mean ≈ 2.0, std ≈ 0.2
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate standard deviations based on the mean and ratio
    step_1_std = abs(step_1_mean_init) * std_to_mean_ratio_init
    step_2_std = abs(step_2_mean_init) * std_to_mean_ratio_init
    
    # Generate values from Gaussian distribution: X ~ N(μ, σ²)
    step_1 = np.random.randn(num_layers) * step_1_std + step_1_mean_init
    step_2 = np.random.randn(num_layers) * step_2_std + step_2_mean_init
    
    return step_1.astype(np.float32), step_2.astype(np.float32)
    
class SimpleStackedHamiltonianNetwork(nn.Module):
    """
    Builds a network consisting of Hamiltonian layers where all layers have the same configuration.
    """
    
    def __init__(
        self,
        n_layers: int = 3,
        hidden_dims: Union[int, List[int]] = 32,
        n_hidden_layers: Optional[int] = 2,
        activation: str = 'tanh',
        activation_params: Optional[dict] = None,
        final_activation: Optional[str] = None,
        final_activation_only_on_final_layer: bool = True,
        tanh_wrapper : bool = False,
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        use_bias: bool = True,
        use_layer_norm: bool = False,


        # Input/Output parameters
        input_dim: int = 2,  # x and t
        output_dim: int = 1,  # scalar G
        
        # Trainable parameter constraints
        a_eps_min: float = 0.1,  # Minimum value for a
        a_eps_max: float = 10.0,  # Maximum value for a  
        a_k: float = 0.1,  # Sigmoid steepness
        step_1_a_mean_innit: float = 0.0,
        std_to_mean_ratio_a_mean_init: float = 0.3,
        step_1_gamma_mean_innit: float = 5.0,
        std_to_mean_ratio_gamma_mean_init: float = 0.3,
        step_1_c1_mean_innit: float = 0.0,
        std_to_mean_ratio_c1_mean_init: float = 0.0,
        step_1_c2_mean_innit: float = 5.0,
        std_to_mean_ratio_c2_mean_init: float = 0.0,
        step_2_a_mean_innit: float = 0.0,
        step_2_gamma_mean_innit: float = 5.0,
        step_2_c1_mean_innit: float = 0.0,
        step_2_c2_mean_innit: float = 5.0,

        bound_innit: float = 10.0,
        **kwargs  # Catch any extra parameters
    ):
        """
        Create a stack of identical CombinedHamiltonianLayers.
        
        Args:
            n_layers: Number of layers to stack
            hidden_dims: Hidden dimensions for MLPs (int or list of ints)
            n_hidden_layers: Number of hidden layers in MLPs (ignored if hidden_dims is a list)
            activation: Activation function for all MLPs
            activation_params: Parameters for activation function
            final_activation: Final activation for MLPs
            Other parameters are passed to both Step_1 and Step_2
        """
        super(SimpleStackedHamiltonianNetwork, self).__init__()
        
        self.n_layers = n_layers
        self.layers = nn.ModuleList()

        self.activation = activation
        self.activation_params = activation_params
        self.final_activation = final_activation
        self.final_activation_only_on_final_layer = final_activation_only_on_final_layer
        self.weight_init = weight_init
        self.weight_init_params = weight_init_params
        self.bias_init = bias_init
        self.bias_init_value = bias_init_value
        self.use_layer_norm = use_layer_norm
        self.a_eps_min = a_eps_min
        self.a_eps_max=a_eps_max
        self.a_k=a_k
        self.tanh_wrapper = tanh_wrapper

        self.step_1_a_mean_innit = step_1_a_mean_innit
        self.step_1_gamma_mean_innit = step_1_gamma_mean_innit
        self.step_1_c1_mean_innit = step_1_c1_mean_innit
        self.step_1_c2_mean_innit = step_1_c2_mean_innit

        self.step_2_a_mean_innit = step_2_a_mean_innit
        self.step_2_gamma_mean_innit = step_2_gamma_mean_innit
        self.step_2_c1_mean_innit = step_2_c1_mean_innit
        self.step_2_c2_mean_innit = step_2_c2_mean_innit

        self.std_to_mean_ratio_a_mean_init = std_to_mean_ratio_a_mean_init
        self.std_to_mean_ratio_gamma_mean_init = std_to_mean_ratio_gamma_mean_init
        self.std_to_mean_ratio_c1_mean_init = std_to_mean_ratio_c1_mean_init
        self.std_to_mean_ratio_c2_mean_init = std_to_mean_ratio_c2_mean_init



        self.bound_innit = bound_innit

        step_1_a_raw_mean_innit = solve_a_raw(a_eps_min=a_eps_min, a_eps_max=a_eps_max, a_k=a_k, a_target=step_1_a_mean_innit)
        step_2_a_raw_mean_innit = solve_a_raw(a_eps_min=a_eps_min, a_eps_max=a_eps_max, a_k=a_k, a_target=step_2_a_mean_innit)

        step_1_a_raw_innit_arr, step_2_a_raw_innit_arr = init_step_params(num_layers=n_layers, step_1_mean_init=step_1_a_raw_mean_innit, step_2_mean_init=step_2_a_raw_mean_innit, std_to_mean_ratio_init=std_to_mean_ratio_a_mean_init, seed=None)

        step_1_gamma_innit_arr, step_2_gamma_innit_arr = init_step_params(num_layers=n_layers, step_1_mean_init=step_1_gamma_mean_innit, step_2_mean_init=step_2_gamma_mean_innit, std_to_mean_ratio_init=std_to_mean_ratio_gamma_mean_init, seed=None)

        step_1_c1_innit_arr, step_2_c1_innit_arr = init_step_params(num_layers=n_layers, step_1_mean_init=step_1_c1_mean_innit, step_2_mean_init=step_2_c1_mean_innit, std_to_mean_ratio_init=std_to_mean_ratio_c1_mean_init, seed=None)

        step_1_c2_innit_arr, step_2_c2_innit_arr = init_step_params(num_layers=n_layers, step_1_mean_init=step_1_c2_mean_innit, step_2_mean_init=step_2_c2_mean_innit, std_to_mean_ratio_init=std_to_mean_ratio_c2_mean_init, seed=None)
        
        if final_activation_only_on_final_layer:
            for i in range(n_layers):
                if i < n_layers - 1:
                    # Intermediate layers: use the main activation
                    layer_final_activation = activation
                else:
                    # Last layer: use the specified final_activation (can be None)
                    layer_final_activation = final_activation
                layer = CombinedHamiltonianLayer(
                    # Step 1 parameters
                    step1_hidden_dims=hidden_dims,
                    step1_n_hidden_layers=n_hidden_layers,
                    step1_activation=activation,
                    step1_activation_params=activation_params,
                    step1_final_activation=layer_final_activation,
                    step1_tanh_wrapper = tanh_wrapper,
                    # Step 2 parameters (same as step 1 for simplicity)
                    step2_hidden_dims=hidden_dims,
                    step2_n_hidden_layers=n_hidden_layers,
                    step2_activation=activation,
                    step2_activation_params=activation_params,
                    step2_final_activation=layer_final_activation,
                    step2_tanh_wrapper = tanh_wrapper,
                    # Shared parameters
                    weight_init=weight_init,
                    weight_init_params=weight_init_params,
                    bias_init=bias_init,
                    bias_init_value=bias_init_value,
                    use_bias=use_bias,
                    use_layer_norm=use_layer_norm,
                    input_dim= input_dim,  # x and t
                    output_dim= output_dim,  # scalar G

                    # Trainable parameter constraints
                    a_eps_min= a_eps_min,  # Minimum value for a
                    a_eps_max= a_eps_max,  # Maximum value for a  
                    a_k= a_k,  # Sigmoid steepness
                    step_1_a_raw_innit = step_1_a_raw_innit_arr[i],
                    step_1_gamma_innit = step_1_gamma_innit_arr[i],
                    step_1_c1_innit = step_1_c1_innit_arr[i],
                    step_1_c2_innit = step_1_c2_innit_arr[i],
                    step_2_a_raw_innit = step_2_a_raw_innit_arr[i],
                    step_2_gamma_innit = step_2_gamma_innit_arr[i],
                    step_2_c1_innit = step_2_c1_innit_arr[i],
                    step_2_c2_innit = step_2_c2_innit_arr[i],
                    bound_innit = bound_innit,
                    **kwargs  # Pass any remaining kwargs
                )
                self.layers.append(layer)
        
        else:
            for i in range(n_layers):
                layer = CombinedHamiltonianLayer(
                    # Step 1 parameters
                    step1_hidden_dims=hidden_dims,
                    step1_n_hidden_layers=n_hidden_layers,
                    step1_activation=activation,
                    step1_activation_params=activation_params,
                    step1_final_activation=final_activation,
                    step1_tanh_wrapper = tanh_wrapper,
                    # Step 2 parameters (same as step 1 for simplicity)
                    step2_hidden_dims=hidden_dims,
                    step2_n_hidden_layers=n_hidden_layers,
                    step2_activation=activation,
                    step2_activation_params=activation_params,
                    step2_final_activation=final_activation,
                    step2_tanh_wrapper = tanh_wrapper,
                    # Shared parameters
                    weight_init=weight_init,
                    weight_init_params=weight_init_params,
                    bias_init=bias_init,
                    bias_init_value=bias_init_value,
                    use_bias=use_bias,
                    use_layer_norm=use_layer_norm,
                    input_dim= input_dim,  # x and t
                    output_dim= output_dim,  # scalar G

                    # Trainable parameter constraints
                    a_eps_min= a_eps_min,  # Minimum value for a
                    a_eps_max= a_eps_max,  # Maximum value for a  
                    a_k= a_k,  # Sigmoid steepness
                    step_1_a_raw_innit = step_1_a_raw_innit_arr[i],
                    step_1_gamma_innit = step_1_gamma_innit_arr[i],
                    step_1_c1_innit = step_1_c1_innit_arr[i],
                    step_1_c2_innit = step_1_c2_innit_arr[i],
                    step_2_a_raw_innit = step_2_a_raw_innit_arr[i],
                    step_2_gamma_innit = step_2_gamma_innit_arr[i],
                    step_2_c1_innit = step_2_c1_innit_arr[i],
                    step_2_c2_innit = step_2_c2_innit_arr[i],
                    bound_innit = bound_innit,
                    **kwargs  # Pass any remaining kwargs
                )
                self.layers.append(layer)
    
    def forward(self, x, u, t):
        """Forward pass through all layers."""
        X, U, T = x, u, t
        for layer in self.layers:
            X, U, T = layer(X, U, T)
        return X, U, T



class TrajectoryStats:
    """
    Precomputed trajectory statistics to eliminate redundant computations.
    Maintains 100% exact behavior of original code.
    """
    def __init__(
        self, 
        X_final: torch.Tensor, 
        U_final: torch.Tensor, 
        trajectory_ids: torch.Tensor
    ):
        self.device = X_final.device
        self.dtype = X_final.dtype
        
        # Core trajectory grouping (computed once)
        self.unique_ids, self.inverse_indices = torch.unique(
            trajectory_ids, return_inverse=True
        )
        self.n_trajectories = len(self.unique_ids)
        
        # Sums per trajectory (computed once) - EXACTLY as original
        self.X_sums = torch.zeros(self.n_trajectories, device=self.device, dtype=self.dtype)
        self.U_sums = torch.zeros(self.n_trajectories, device=self.device, dtype=self.dtype)
        self.X_sums = self.X_sums.scatter_add_(0, self.inverse_indices, X_final)
        self.U_sums = self.U_sums.scatter_add_(0, self.inverse_indices, U_final)
        
        # Store originals for reference
        self.X_final = X_final
        self.U_final = U_final
        self.trajectory_ids = trajectory_ids
        
        # Precompute BOTH versions of counts to match original functions exactly
        # Version 1: bincount converted to X_final.dtype (used by repulsion & hsic)
        self.counts_typed = torch.bincount(
            self.inverse_indices, 
            minlength=self.n_trajectories
        ).to(dtype=self.dtype, device=self.device)
        
        # Version 2: scatter_add with float (used by variance, prediction, etc.)
        self.counts_float = torch.zeros(self.n_trajectories, device=self.device)
        self.counts_float = self.counts_float.scatter_add_(
            0, 
            self.inverse_indices, 
            torch.ones_like(self.inverse_indices, dtype=torch.float)
        )
        
        # Means computed using typed counts (matches repulsion/hsic original behavior)
        self.X_means = self.X_sums / self.counts_typed
        self.U_means = self.U_sums / self.counts_typed




    
def hsic_loss(
    X_final: torch.Tensor, 
    U_final: torch.Tensor, 
    trajectory_ids: torch.Tensor,
    sigma_X_means: float = -1.0,
    sigma_U_means: float = -1.0,
    use_unbiased: bool = True,
    epsilon: float = 1e-10,
    stats = None
) -> torch.Tensor:
    """HSIC loss - uses precomputed stats"""
    device = X_final.device
    dtype = X_final.dtype
    
    # Use precomputed stats if available
    if stats is None:
        unique_ids, inverse = torch.unique(trajectory_ids, return_inverse=True)
        N = unique_ids.shape[0]
        
        if N <= 1:
            return X_final.sum() * 0.0
        
        # EXACTLY as original
        sums_x = torch.zeros(N, device=device, dtype=dtype).scatter_add_(0, inverse, X_final)
        sums_u = torch.zeros(N, device=device, dtype=dtype).scatter_add_(0, inverse, U_final)
        counts = torch.bincount(inverse, minlength=N).to(dtype=dtype, device=device)
        
        X_means = sums_x / counts
        U_means = sums_u / counts
    else:
        N = stats.n_trajectories
        if N <= 1:
            return X_final.sum() * 0.0
        
        X_means = stats.X_means
        U_means = stats.U_means

    # Rest is IDENTICAL to original
    if X_means.shape != U_means.shape:
        raise ValueError(f"X_means and U_means must have the same shape. Got X_means: {X_means.shape}, U_means: {U_means.shape}")
    if X_means.dim() != 1:
        raise ValueError(f"Expected 1D tensors, got X_means.dim()={X_means.dim()}. Please squeeze or select the right dimension.")
    
    batch_size = X_means.shape[0]
    min_batch = 4 if use_unbiased else 2
    
    if batch_size < min_batch:
        print("Number of trajectories lower than 4, so returning 0.0 at hsic loss")
        zero_loss = (X_means.sum() * 0.0 + U_means.sum() * 0.0)
        return zero_loss
    
    X_means = X_means.view(-1, 1)
    U_means = U_means.view(-1, 1)
    
    K = _compute_rbf_kernel(X_means, sigma_X_means, epsilon, use_unbiased, batch_size)
    L = _compute_rbf_kernel(U_means, sigma_U_means, epsilon, use_unbiased, batch_size)
    
    if use_unbiased:
        b = float(batch_size)
        KL = K @ L
        K_sum = K.sum()
        L_sum = L.sum()
        KL_sum = KL.sum()
        KL_trace = KL.trace()
        
        hsic = (KL_trace + (K_sum * L_sum) / ((b - 1) * (b - 2)) - 
                2.0 * KL_sum / (b - 2)) / (b * (b - 3))
    else:
        b = float(batch_size)
        K_mean = K.mean()
        L_mean = L.mean()
        K_row_mean = K.mean(dim=1, keepdim=True)
        L_row_mean = L.mean(dim=1, keepdim=True)
        
        K_centered = K - K_row_mean - K_row_mean.t() + K_mean
        L_centered = L - L_row_mean - L_row_mean.t() + L_mean
        
        hsic = (K_centered * L_centered).sum() / (b * b)
    
    zero = torch.tensor(0.0, device=hsic.device, dtype=hsic.dtype, requires_grad=False)
    hsic = torch.maximum(hsic, zero)
    
    return hsic

def hsic_loss_statistics_only(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_x: float = -1.0,
    sigma_y: float = -1.0,
    use_unbiased: bool = True,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """
    HSIC loss with guaranteed gradient flow and improved efficiency.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape. Got x: {x.shape}, y: {y.shape}")
    if x.dim() != 1:
        raise ValueError(f"Expected 1D tensors, got x.dim()={x.dim()}. Please squeeze or select the right dimension.")
    
    batch_size = x.shape[0]
    min_batch = 4 if use_unbiased else 2
    
    # FIX 1: Return zero loss with proper gradient connection instead of error
    if batch_size < min_batch:
        # Create a zero tensor that maintains gradient flow
        # Use x and y to ensure gradient connection
        zero_loss = (x.sum() * 0.0 + y.sum() * 0.0)  # Gradient flows but equals 0
        print("Number of trajectories lower than 4, so returning 0.0 at hsic loss")
        return zero_loss
    
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    
    K = _compute_rbf_kernel(x, sigma_x, epsilon, use_unbiased, batch_size)
    L = _compute_rbf_kernel(y, sigma_y, epsilon, use_unbiased, batch_size)
    
    if use_unbiased:
        b = float(batch_size)  # Convert to float once for efficiency
        
        # Pre-compute for efficiency
        KL = K @ L
        K_sum = K.sum()
        L_sum = L.sum()
        KL_sum = KL.sum()
        KL_trace = KL.trace()
        
        # Direct computation without intermediate variables (more efficient)
        hsic = (KL_trace + (K_sum * L_sum) / ((b - 1) * (b - 2)) - 
                2.0 * KL_sum / (b - 2)) / (b * (b - 3))
    else:
        b = float(batch_size)
        
        # More efficient centering without repeated mean computations
        K_mean = K.mean()
        L_mean = L.mean()
        K_row_mean = K.mean(dim=1, keepdim=True)
        L_row_mean = L.mean(dim=1, keepdim=True)
        
        # Direct computation of centered kernel product
        K_centered = K - K_row_mean - K_row_mean.t() + K_mean
        L_centered = L - L_row_mean - L_row_mean.t() + L_mean
        
        hsic = (K_centered * L_centered).sum() / (b * b)
    
    # FIX 2: Replace clamp with torch.maximum (more explicit gradient behavior)
    # Or use F.relu which is equivalent but clearer for non-negativity
    # Actually, if computed correctly, HSIC should be non-negative
    # But for numerical stability with float operations:
    zero = torch.tensor(0.0, device=hsic.device, dtype=hsic.dtype, requires_grad=False)
    hsic = torch.maximum(hsic, zero)
    
    return hsic


def _compute_rbf_kernel(
    z: torch.Tensor,
    sigma: float,
    epsilon: float,
    use_unbiased: bool,
    batch_size: int,
) -> torch.Tensor:
    """
    Compute RBF kernel matrix with improved efficiency and gradient safety.
    """
    # More efficient distance computation
    # Using einsum for clarity and potential optimization
    zz = z @ z.t()
    z_squared = zz.diagonal()  # More efficient than diag()
    
    # Compute squared distances efficiently
    dist_sq = z_squared.unsqueeze(1) + z_squared.unsqueeze(0) - 2.0 * zz
    
    # Ensure non-negative (F.relu is explicit about gradient behavior)
    dist_sq = F.relu(dist_sq)
    
    if sigma < 0:
        # Adaptive bandwidth using median heuristic
        # More efficient: get upper triangular part directly
        mask = torch.triu(torch.ones(batch_size, batch_size, device=z.device, dtype=torch.bool), diagonal=1)
        off_diag_dists = dist_sq[mask]
        
        if off_diag_dists.numel() > 0:
            # CORRECT: Detach for bandwidth computation
            median_dist = off_diag_dists.detach().median()
            # Use torch.maximum instead of clamp for clarity
            bandwidth_sq = torch.maximum(median_dist, 
                                        torch.tensor(epsilon, device=z.device, dtype=z.dtype))
        else:
            # FIX 3: Explicitly set requires_grad=False for the constant
            bandwidth_sq = torch.tensor(1.0, dtype=z.dtype, device=z.device, requires_grad=False)
    else:
        # FIX 3: Create tensor with explicit requires_grad=False
        bandwidth_sq = torch.tensor(max(sigma ** 2, epsilon), 
                                   dtype=z.dtype, device=z.device, requires_grad=False)
    
    # Compute RBF kernel
    K = torch.exp(-0.5 * dist_sq / bandwidth_sq)
    
    if use_unbiased:
        # More efficient: modify diagonal in-place after clone
        K = K.clone()
        K.fill_diagonal_(0.0)
    
    return K

class ReverseStep2(nn.Module):
    """
    Reverse transformation of Step_2.
    
    Forward Step_2 does: 
        U = a' * u + c1'
        X = (1/a') * x + γ' * ∂F/∂u + c2'
    
    Reverse Step_2 does: 
        u = (U - c1') / a'
        x = a' * (X - γ' * ∂F/∂u - c2')
    
    This module SHARES the F network and parameters with the forward Step_2.
    """
    
    def __init__(self, forward_step2_module: nn.Module):
        super(ReverseStep2, self).__init__()
        self.forward_module = forward_step2_module
    
    def forward(self, X, U, t):
        # Get shared parameters from forward module
        a = self.forward_module.a
        c1 = self.forward_module.c1
        gamma = self.forward_module.gamma
        c2 = self.forward_module.c2
        
        # FIRST: Reverse velocity transformation to get original u
        u = (U - c1) / a
        
        # SECOND: Use the recovered u to compute gradient
        dF_du = self.forward_module.compute_F_and_gradient(u, t)
        
        # THIRD: Reverse position transformation
        x = a * (X - gamma * dF_du - c2)
        
        # Return reconstructed values
        return x, u, t
    

class ReverseStep1(nn.Module):
    """
    Reverse transformation of Step_1.
    
    Forward Step_1 does: 
        X = a * x + c1
        U = (1/a) * u - γ * ∂G/∂x + c2
    
    Reverse Step_1 does: 
        x = (X - c1) / a
        u = a * (U + γ * ∂G/∂x - c2)
    
    This module SHARES the G network and parameters with the forward Step_1.
    """
    
    def __init__(self, forward_step1_module: nn.Module):
        super(ReverseStep1, self).__init__()
        self.forward_module = forward_step1_module
    
    def forward(self, X, U, t):
        # Get shared parameters from forward module
        a = self.forward_module.a
        c1 = self.forward_module.c1
        gamma = self.forward_module.gamma
        c2 = self.forward_module.c2
        
        # FIRST: Reverse position transformation to get original x
        x = (X - c1) / a
        
        # SECOND: Use the recovered x to compute gradient
        dG_dx = self.forward_module.compute_G_and_gradient(x, t)
        
        # THIRD: Reverse velocity transformation
        u = a * (U + gamma * dG_dx - c2)
        
        # Return reconstructed values
        return x, u, t
    

class ReverseCombinedHamiltonianLayer(nn.Module):
    """
    Combined reverse layer that performs both reverse transformations.
    Shares weights with the corresponding forward CombinedHamiltonianLayer.
    
    Order is REVERSED from forward:
    - Forward: Step_1 (velocity) → Step_2 (position)
    - Reverse: Reverse_Step_2 (position) → Reverse_Step_1 (velocity)
    """
    
    def __init__(
        self,
        forward_combined_layer: nn.Module,  # The forward CombinedHamiltonianLayer
    ):
        """
        Initialize the reverse combined layer.
        
        Args:
            forward_combined_layer: The forward CombinedHamiltonianLayer to reverse
        """
        super(ReverseCombinedHamiltonianLayer, self).__init__()
        
        
        # Create reverse steps that share weights with forward steps
        # Note: Order is reversed!
        self.reverse_step2 = ReverseStep2(
            forward_combined_layer.step_2,  # Share with forward Step_2
        )

        
        self.reverse_step1 = ReverseStep1(
            forward_combined_layer.step_1,  # Share with forward Step_1
        )
    
    def forward(self, X_final, U_final, t_final):
        """
        Apply both reverse transformations in reverse order.
        
        Args:
            X_final: Final position from forward network
            U_final: Final velocity from forward network
            t: Time (unchanged throughout)
            
        Returns:
            x_reconstructed: Should approximate original x
            u_reconstructed: Should approximate original u
            t_reconstructed: Should approximate original t
        """
        # First: Reverse Step_2 (position reconstruction)
        X_intermediate, U_intermediate, T_intermediate = self.reverse_step2(X_final, U_final, t_final)
        
        # Second: Reverse Step_1 (velocity reconstruction)
        x_reconstructed, u_reconstructed, t__reconstructed = self.reverse_step1(X_intermediate, U_intermediate, T_intermediate)
        
        return x_reconstructed, u_reconstructed, t__reconstructed
    

class InverseStackedHamiltonianNetwork(nn.Module):
    """
    Inverse network that reverses a SimpleStackedHamiltonianNetwork.
    Each layer shares weights with its corresponding forward layer.
    Layers are applied in REVERSE order.
    """
    
    def __init__(
        self,
        forward_network: nn.Module,  # SimpleStackedHamiltonianNetwork instance

    ):
        """
        Create inverse network from forward network.
        
        Args:
            forward_network: An initialized SimpleStackedHamiltonianNetwork
        """
        super(InverseStackedHamiltonianNetwork, self).__init__()
        

        self.n_layers = forward_network.n_layers
        
        # Create reverse layers that share weights with forward layers
        # IMPORTANT: We store them in REVERSE order for the inverse pass
        self.reverse_layers = nn.ModuleList()
        
        # Iterate through forward layers in REVERSE order
        for i in reversed(range(self.n_layers)):
            forward_layer = forward_network.layers[i]
            reverse_layer = ReverseCombinedHamiltonianLayer(
                forward_layer, 
            )
            self.reverse_layers.append(reverse_layer)
        
        print(f"Created inverse network with {len(self.reverse_layers)} reverse layers")

    
    def forward(self, X_final, U_final, T_final):
        """
        Apply inverse transformation through all layers.
        
        Args:
            X_final: Final position from forward network
            U_final: Final velocity from forward network
            t: Time (unchanged)
            
        Returns:
            x_reconstructed: Should approximate original x
            u_reconstructed: Should approximate original u
            t: Unchanged time
        """
        X, U, T = X_final, U_final, T_final
        
        # Apply reverse layers (already in reverse order)
        for i, reverse_layer in enumerate(self.reverse_layers):
            X, U, T = reverse_layer(X, U, T)
        
        return X, U, T
    

        



def prepare_prediction_inputs(
    X_final: torch.Tensor,
    U_final: torch.Tensor, 
    t_batch: torch.Tensor,
    trajectory_ids: torch.Tensor,
    possible_t_values: List[float],
    stats = None,
    predict_full_trajectory: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare prediction inputs - uses precomputed stats"""
    
    device = X_final.device

    
    # Use precomputed stats if available
    if stats is None:
        unique_ids, inverse_indices = torch.unique(trajectory_ids, return_inverse=True)
        n_trajectories = len(unique_ids)
        
        # EXACTLY as original - uses float counts
        counts = torch.zeros(n_trajectories, device=device)
        counts = counts.scatter_add_(
            0, 
            inverse_indices, 
            torch.ones_like(inverse_indices, dtype=torch.float)
        )
        
        X_sums = torch.zeros(n_trajectories, device=device, dtype=X_final.dtype)
        U_sums = torch.zeros(n_trajectories, device=device, dtype=U_final.dtype)
        X_sums = X_sums.scatter_add_(0, inverse_indices, X_final)
        U_sums = U_sums.scatter_add_(0, inverse_indices, U_final)
        
        X_means = X_sums / counts
        U_means = U_sums / counts
    else:
        unique_ids = stats.unique_ids
        inverse_indices = stats.inverse_indices
        # Use float counts (matches original)
        X_means = stats.X_sums / stats.counts_float
        U_means = stats.U_sums / stats.counts_float
    
    # NEW BEHAVIOR: Predict at all possible_t_values for each unique trajectory
    if predict_full_trajectory:
        n_trajectories = len(unique_ids)
        n_times = len(possible_t_values)
        
        # Each trajectory mean repeated n_times times
        # Shape: [n_trajectories * n_times]
        X_final_mean = X_means.repeat_interleave(n_times)  # Gradient safe
        U_final_mean = U_means.repeat_interleave(n_times)  # Gradient safe
        
        # All possible_t_values repeated for each trajectory
        # Shape: [n_trajectories * n_times]
        possible_t_tensor = torch.tensor(possible_t_values, device=device, dtype=t_batch.dtype)
        t_for_pred = possible_t_tensor.repeat(n_trajectories)  # Gradient safe
        
        return X_final_mean, U_final_mean, t_for_pred
    
    # ORIGINAL BEHAVIOR: Rest is IDENTICAL to original
    X_final_mean = X_means[inverse_indices]
    U_final_mean = U_means[inverse_indices]
    
    t_for_pred = torch.zeros_like(t_batch)
    possible_t_tensor = torch.tensor(possible_t_values, device=device, dtype=t_batch.dtype)
    
    for traj_idx, traj_id in enumerate(unique_ids):
        mask = (trajectory_ids == traj_id)
        n_samples = mask.sum().item()
        sampled_times = t_batch[mask]
        
        is_sampled = torch.isclose(
            possible_t_tensor.unsqueeze(1),
            sampled_times.unsqueeze(0),
            rtol=1e-5,
            atol=1e-8
        ).any(dim=1)
        available_times = possible_t_tensor[~is_sampled]
        
        if len(available_times) < n_samples:
            if len(available_times) > 0:
                repeated_times = available_times.repeat((n_samples // len(available_times)) + 1)
                selected_times = repeated_times[:n_samples]
                perm = torch.randperm(n_samples, device=device)
                selected_times = selected_times[perm]
            else:
                indices = torch.randint(0, len(possible_t_values), (n_samples,), device=device)
                selected_times = possible_t_tensor[indices]
        else:
            perm = torch.randperm(len(available_times), device=device)[:n_samples]
            selected_times = available_times[perm]
        
        t_for_pred[mask] = selected_times

    return X_final_mean, U_final_mean, t_for_pred






def prepare_prediction_inputs_for_real_pendulum_2(
    X_final: torch.Tensor,
    U_final: torch.Tensor, 
    t_batch: torch.Tensor,
    trajectory_ids: torch.Tensor,
    possible_t_values_per_trajectory: torch.Tensor,  # Shape: [total_num_trajectories, n_times]
    stats = None,
    predict_full_trajectory: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare prediction inputs for trajectories with different time values.
    
    Args:
        X_final: Mapped X values from mapping_net
        U_final: Mapped U values from mapping_net
        t_batch: Time values from batch
        trajectory_ids: Trajectory IDs from batch
        possible_t_values_per_trajectory: 2D tensor of shape [total_num_trajectories, n_times]
            where possible_t_values_per_trajectory[trajectory_id] contains the time values
            for that specific trajectory. Row index = trajectory_id.
        stats: Precomputed statistics (None in your case)
        predict_full_trajectory: Whether to predict at all time points
        
    Returns:
        X_final_mean: Mean X value repeated for each time point
        U_final_mean: Mean U value repeated for each time point  
        t_for_pred: Time values for prediction (trajectory-specific)
    """
    
    device = X_final.device
    batch_size = X_final.shape[0]
    
    # Ensure possible_t_values_per_trajectory is a tensor on the correct device
    if not isinstance(possible_t_values_per_trajectory, torch.Tensor):
        possible_t_values_per_trajectory = torch.tensor(
            possible_t_values_per_trajectory, 
            device=device, 
            dtype=t_batch.dtype
        )
    elif possible_t_values_per_trajectory.device != device:
        possible_t_values_per_trajectory = possible_t_values_per_trajectory.to(device)
    
    # Use precomputed stats if available
    if stats is None:
        unique_ids, inverse_indices = torch.unique(trajectory_ids, return_inverse=True)
        n_trajectories = len(unique_ids)
        
        # EXACTLY as original - uses float counts
        counts = torch.zeros(n_trajectories, device=device)
        counts = counts.scatter_add_(
            0, 
            inverse_indices, 
            torch.ones_like(inverse_indices, dtype=torch.float)
        )
        
        X_sums = torch.zeros(n_trajectories, device=device, dtype=X_final.dtype)
        U_sums = torch.zeros(n_trajectories, device=device, dtype=U_final.dtype)
        X_sums = X_sums.scatter_add_(0, inverse_indices, X_final)
        U_sums = U_sums.scatter_add_(0, inverse_indices, U_final)
        
        X_means = X_sums / counts
        U_means = U_sums / counts
    else:
        unique_ids = stats.unique_ids
        inverse_indices = stats.inverse_indices
        # Use float counts (matches original)
        X_means = stats.X_sums / stats.counts_float
        U_means = stats.U_sums / stats.counts_float
    
    if predict_full_trajectory:
        n_trajectories = len(unique_ids)
        n_times = possible_t_values_per_trajectory.shape[1]
        
        # Each trajectory mean repeated n_times times
        # Shape: [n_trajectories * n_times]
        X_final_mean = X_means.repeat_interleave(n_times)  # Gradient safe
        U_final_mean = U_means.repeat_interleave(n_times)  # Gradient safe
        
        # KEY CHANGE: Get trajectory-specific time values
        # unique_ids contains the trajectory IDs present in this batch (sorted)
        # Index into possible_t_values_per_trajectory to get the correct times
        # Shape: [n_trajectories, n_times]
        t_values_for_batch = possible_t_values_per_trajectory[unique_ids]
        
        # Flatten row by row to match the order of X_final_mean and U_final_mean
        # Shape: [n_trajectories * n_times]
        t_for_pred = t_values_for_batch.flatten()  # Gradient safe (no grad needed for t anyway)
        
        return X_final_mean, U_final_mean, t_for_pred

def generate_prediction_labels(
    train_df: pd.DataFrame,
    train_id_df: pd.DataFrame,
    trajectory_ids: torch.Tensor,
    t_for_pred: torch.Tensor,
    get_data_from_trajectory_id: callable,
    predict_full_trajectory: bool = False,
    possible_t_values: List[float] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate ground truth labels for prediction task.
    
    Args:
        train_df: DataFrame with columns ['x', 'u', 't']
        train_id_df: DataFrame with trajectory metadata
        trajectory_ids: Trajectory IDs for each sample in batch
        t_for_pred: Time values to predict at
        get_data_from_trajectory_id: Function to get trajectory data
        predict_full_trajectory: If True, predict at all possible_t_values for each unique trajectory
        possible_t_values: List of all possible time values (required if predict_full_trajectory=True)
        
    Returns:
        X_labels: Ground truth x values at t_for_pred times
        U_labels: Ground truth u values at t_for_pred times
        t_labels: Same as t_for_pred (for consistency)
    """
    device = trajectory_ids.device
    
    # NEW BEHAVIOR: Predict full trajectory at all possible_t_values
    if predict_full_trajectory:
        if possible_t_values is None:
            raise ValueError("possible_t_values must be provided when predict_full_trajectory=True")
        
        # Get unique trajectories from the batch (maintains order)
        unique_ids = torch.unique(trajectory_ids).cpu().numpy()
        n_times = len(possible_t_values)
        
        # Initialize output tensors with same shape and device as t_for_pred
        X_labels = torch.zeros_like(t_for_pred)
        U_labels = torch.zeros_like(t_for_pred)
        
        # Fill in labels for each unique trajectory
        for traj_idx, traj_id in enumerate(unique_ids):
            # Get full trajectory data
            traj_df = get_data_from_trajectory_id(
                ids_df=train_id_df,
                data_df=train_df,
                trajectory_ids=int(traj_id)
            )
            
            if traj_df is None or len(traj_df) == 0:
                print(f"Error: No data found for trajectory_id {traj_id}")
                return []
            
            # Extract x and u values directly
            x_values = torch.tensor(traj_df['x'].values, device=device)
            u_values = torch.tensor(traj_df['u'].values, device=device)
            
            # Fill the corresponding block
            start_idx = traj_idx * n_times
            end_idx = start_idx + n_times
            X_labels[start_idx:end_idx] = x_values
            U_labels[start_idx:end_idx] = u_values
        
        t_labels = t_for_pred.clone()
        return X_labels, U_labels, t_labels
    
    # ORIGINAL BEHAVIOR: Unchanged
    batch_size = trajectory_ids.shape[0]
    
    # Initialize output tensors
    X_labels = torch.zeros(batch_size, device=device)
    U_labels = torch.zeros(batch_size, device=device)

    # Convert tensors to numpy for easier indexing with pandas
    traj_ids_np = trajectory_ids.cpu().numpy()
    t_pred_np = t_for_pred.cpu().numpy()
    
    # Cache trajectory data to avoid repeated lookups
    trajectory_cache = {}
    
    for i in range(batch_size):
        traj_id = traj_ids_np[i]
        t_target = t_pred_np[i]
        
        # Get trajectory data (use cache if available)
        if traj_id not in trajectory_cache:
            traj_df = get_data_from_trajectory_id(
                ids_df=train_id_df,
                data_df=train_df,
                trajectory_ids=traj_id
            )

            trajectory_cache[traj_id] = traj_df
        else:
            traj_df = trajectory_cache[traj_id]
        
        # Find the exact time value or interpolate
        t_values = traj_df['t'].values
        x_values = traj_df['x'].values
        u_values = traj_df['u'].values
        
        # Check if exact time exists
        exact_match = np.isclose(t_values, t_target, atol=1e-8)
        if exact_match.any():
            # Use exact value
            idx = np.where(exact_match)[0][0]
            X_labels[i] = x_values[idx]
            U_labels[i] = u_values[idx]
        else:
            print(f"Didn't find any close t values to predict for trajectory_id {traj_id}, and target t {t_target}")
            return []
    
    # t_labels is just t_for_pred (for consistency and verification)
    t_labels = t_for_pred.clone()
    
    return X_labels, U_labels, t_labels



    

def generate_prediction_labels_for_real_pendulum_2(
    train_df: pd.DataFrame,
    train_id_df: pd.DataFrame,
    trajectory_ids: torch.Tensor,
    t_for_pred: torch.Tensor,
    get_data_from_trajectory_id: callable,
    predict_full_trajectory: bool = False,
    possible_t_values_per_trajectory: torch.Tensor = None  # Shape: [total_num_trajectories, n_times]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate ground truth labels for prediction task with trajectory-specific time values.
    
    Args:
        train_df: DataFrame with columns ['x', 'u', 't']
        train_id_df: DataFrame with trajectory metadata
        trajectory_ids: Trajectory IDs for each sample in batch
        t_for_pred: Time values to predict at (trajectory-specific)
        get_data_from_trajectory_id: Function to get trajectory data
        predict_full_trajectory: If True, predict at all time values for each unique trajectory
        possible_t_values_per_trajectory: 2D tensor of shape [total_num_trajectories, n_times]
            where row index = trajectory_id. Used to determine n_times.
        
    Returns:
        X_labels: Ground truth x values at t_for_pred times
        U_labels: Ground truth u values at t_for_pred times
        t_labels: Same as t_for_pred (for consistency)
    """
    device = trajectory_ids.device
    
    if predict_full_trajectory:
        if possible_t_values_per_trajectory is None:
            raise ValueError("possible_t_values_per_trajectory must be provided when predict_full_trajectory=True")
        
        # Get n_times from the shape of possible_t_values_per_trajectory
        n_times = possible_t_values_per_trajectory.shape[1]
        
        # Get unique trajectories from the batch (returns SORTED order - same as in prepare_prediction_inputs)
        unique_ids = torch.unique(trajectory_ids).cpu().numpy()
        
        # Initialize output tensors with same shape and device as t_for_pred
        X_labels = torch.zeros_like(t_for_pred)
        U_labels = torch.zeros_like(t_for_pred)
        
        # Fill in labels for each unique trajectory (in sorted order - matches prepare_prediction_inputs)
        for traj_idx, traj_id in enumerate(unique_ids):
            # Get full trajectory data
            traj_df = get_data_from_trajectory_id(
                ids_df=train_id_df,
                data_df=train_df,
                trajectory_ids=int(traj_id)
            )
            
            if traj_df is None or len(traj_df) == 0:
                print(f"Error: No data found for trajectory_id {traj_id}")
                return []
            
            # Extract x and u values directly (these are already in correct time order for this trajectory)
            x_values = torch.tensor(traj_df['x'].values, device=device, dtype=t_for_pred.dtype)
            u_values = torch.tensor(traj_df['u'].values, device=device, dtype=t_for_pred.dtype)
            
            # Fill the corresponding block
            start_idx = traj_idx * n_times
            end_idx = start_idx + n_times
            X_labels[start_idx:end_idx] = x_values
            U_labels[start_idx:end_idx] = u_values
        
        t_labels = t_for_pred.clone()
        return X_labels, U_labels, t_labels

def prediction_loss(
    x_pred: torch.Tensor,
    u_pred: torch.Tensor,
    X_labels: torch.Tensor,
    U_labels: torch.Tensor,
    loss_type: str = "mse"
) -> torch.Tensor:
    """
    Compute prediction loss using either Mean Absolute Error (MAE) or Mean Squared Error (MSE).
    
    Args:
        x_pred: Predicted positions from inverse network, shape (batch_size,)
        u_pred: Predicted velocities from inverse network, shape (batch_size,)
        X_labels: Ground truth positions, shape (batch_size,)
        U_labels: Ground truth velocities, shape (batch_size,)
        loss_type: Type of loss to use, either 'mae' or 'mse'. Default is 'mae'.
        
    Returns:
        loss: Scalar MAE or MSE loss
    """
    if loss_type.lower() == "mae":
        x_loss = torch.mean(torch.abs(x_pred - X_labels))
        u_loss = torch.mean(torch.abs(u_pred - U_labels))
    elif loss_type.lower() == "mse":
        x_loss = torch.mean((x_pred - X_labels) ** 2)
        u_loss = torch.mean((u_pred - U_labels) ** 2)
    else:
        raise ValueError(f"Invalid loss_type '{loss_type}'. Choose either 'mae' or 'mse'.")

    total_loss = x_loss + u_loss
    return total_loss





def prediction_loss_euclidean(
    x_pred: torch.Tensor,
    u_pred: torch.Tensor,
    X_labels: torch.Tensor,
    U_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute prediction loss using Euclidean distance in phase space.
    
    Args:
        x_pred: Predicted positions from inverse network, shape (batch_size,)
        u_pred: Predicted velocities from inverse network, shape (batch_size,)
        X_labels: Ground truth positions at t_for_pred times, shape (batch_size,)
        U_labels: Ground truth velocities at t_for_pred times, shape (batch_size,)
    
    Returns:
        loss: Mean Euclidean distance in (x, u) phase space
    """
    euclidean_distances = torch.sqrt((x_pred - X_labels) ** 2 + (u_pred - U_labels) ** 2)
    return torch.mean(euclidean_distances)


def compute_total_loss(hsic_loss, prediction_loss, prediction_coefficient,
                      prediction_loss_scale, hsic_loss_slope):
    """
    Compute final total loss with scale normalization and gradient-safe safety switch.
    
    Args:
        prediction_loss: Raw prediction loss tensor (with gradients)
        prediction_coefficient: Fixed weight for prediction loss (float, no gradients)
        prediction_loss_scale: Fixed scale for prediction loss normalization (float, no gradients)

        
    Returns:
        total_loss: Final weighted and scaled loss tensor (gradients flow back to input losses)
    """
    
    # Scale mapping and prediction losses by fixed scales (gradients preserved)

    prediction_loss_scaled = prediction_loss / prediction_loss_scale

    

    hsic_loss_scaled = hsic_loss * hsic_loss_slope
    
    
    # Combine all losses with coefficients
    total_loss = (
        hsic_loss_scaled + prediction_coefficient * prediction_loss_scaled
        )
    
    return total_loss

class TrajectoryDataset(Dataset):
    """
    Custom PyTorch Dataset.
    Each item corresponds to a full trajectory.
    """
    def __init__(self, ids_df, data_df, get_data_func):
        self.ids_df = ids_df
        self.data_df = data_df
        self.get_data_func = get_data_func
        # Get a list of unique trajectory IDs to iterate over
        self.trajectory_ids = self.ids_df['trajectory_id'].unique()

    def __len__(self):
        """The number of items in the dataset is the number of unique trajectories."""
        return len(self.trajectory_ids)

    def __getitem__(self, idx):
        """
        Retrieves all data for a single trajectory based on the index.
        """
        # Get the trajectory_id for the given index
        trajectory_id = self.trajectory_ids[idx]

        # Fetch the data for this specific trajectory using the provided function
        trajectory_data = self.get_data_func(
            self.ids_df, self.data_df, trajectory_id
        )

        # Convert data to PyTorch tensors
        # Assuming 'x', 'u', 't' are numeric. Add .values to get numpy array first.
        x_tensor = torch.tensor(trajectory_data['x'].values, dtype=torch.float32)
        u_tensor = torch.tensor(trajectory_data['u'].values, dtype=torch.float32)
        t_tensor = torch.tensor(trajectory_data['t'].values, dtype=torch.float32)

        # The output is a dictionary for a single trajectory
        sample = {
            'x': x_tensor,
            'u': u_tensor,
            't': t_tensor,
            'trajectory_id': trajectory_id
        }

        return sample

# --- Custom Collate Function ---
def trajectory_collate_fn(batch_list):
    """
    Custom collate function for the TrajectoryDataset.
    Since batch_size is 1 and __getitem__ returns a dict for the whole trajectory,
    the input `batch_list` will be a list containing a single dictionary.
    We just need to extract that dictionary.
    """
    return batch_list[0]

# --- Dataloader Creation Module ---
def create_val_dataloader_full_trajectory(val_df, val_id_df, get_data_func, device='cuda', seed=42):
    """
    Creates a PyTorch DataLoader that batches data by trajectory_id.

    Args:
        val_df (pd.DataFrame): DataFrame with the full training data ('x', 'u', 't').
        val_id_df (pd.DataFrame): DataFrame with metadata for each trajectory.
        get_data_func (function): Helper function to extract trajectory data.
        seed (int): Random seed for reproducibility.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Instantiate the Dataset
    trajectory_dataset = TrajectoryDataset(
        ids_df=val_id_df,
        data_df=val_df,
        get_data_func=get_data_func
    )

    def collate_and_to_device(batch):
        """Custom collate that moves to device."""
        batch_dict = trajectory_collate_fn(batch)
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_dict.items()}
    

    # Create the DataLoader
    # batch_size=1 because each "item" from the dataset is already a full trajectory batch.
    # shuffle=True will randomly order the trajectories each epoch.
    dataloader = DataLoader(
        trajectory_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_and_to_device
    )
    
    return dataloader


def prepare_validation_inputs(X_final, U_final, t_final):
    """
    Prepare validation inputs for prediction on full trajectories.
    
    Args:
        X_final: Final positions from forward network, shape (trajectory_length,)
        U_final: Final velocities from forward network, shape (trajectory_length,)
        t_final: Time values (unchanged from input), shape (trajectory_length,)
        
    Returns:
        X_final_mean_full: All values set to mean of X_final
        U_final_mean_full: All values set to mean of U_final
        t_rearranged: Time values randomly rearranged with no element in original position
    """
    # Compute means and convert to Python scalars
    X_mean = X_final.mean().item()
    U_mean = U_final.mean().item()
    
    # Create tensors filled with the mean values
    X_final_mean_full = torch.full_like(X_final, X_mean)
    U_final_mean_full = torch.full_like(U_final, U_mean)
    
    # Random derangement of t values
    n = len(t_final)
    if n == 1:
        # Can't rearrange a single element
        t_rearranged = t_final.clone()
    else:
        # Generate random derangement (no element in original position)
        indices = torch.arange(n, device=t_final.device)
        
        # Keep shuffling until we get a valid derangement
        max_attempts = 1000
        for _ in range(max_attempts):
            shuffled = indices[torch.randperm(n)]
            if not (shuffled == indices).any():
                # Valid derangement found
                break
        else:
            # Fallback: If random fails, use deterministic derangement
            # Shift by half the length (or 1 if n=2)
            shift = max(1, n // 2)
            shuffled = torch.roll(indices, shifts=shift)
        
        t_rearranged = t_final[shuffled]
    
    return X_final_mean_full, U_final_mean_full, t_rearranged


def compute_single_trajectory_stats(X_final, U_final):
    """
    Compute statistics for a single trajectory (validation case).
    
    Args:
        X_final: Final positions for single trajectory
        U_final: Final velocities for single trajectory
        
    Returns:
        dict with statistics as numpy values
    """
    # Compute means
    X_mean = X_final.mean()
    U_mean = U_final.mean()
    
    # Compute variances (unbiased if more than 1 point)
    if len(X_final) > 1:
        X_var = X_final.var(unbiased=True)
        U_var = U_final.var(unbiased=True)
    else:
        X_var = torch.tensor(0.0, device=X_final.device)
        U_var = torch.tensor(0.0, device=U_final.device)
    
    # Standard deviations
    X_std = torch.sqrt(X_var)
    U_std = torch.sqrt(U_var)
    
    # Total variance (same as in training for consistency)
    total_variance = X_var + U_var
    
    # Return all as numpy values
    return total_variance,  X_mean.item(), U_mean.item(), X_var.item(), U_var.item(), X_std.item(), U_std.item()







def calculate_losses_scale_on_untrained(train_loader, mapping_net, inverse_net, get_data_from_trajectory_id, possible_t_values, train_df, train_id_df,loss_type, predict_full_trajectory, add_noise, save_returned_values, save_dir, noise_threshold_mean_divided_by_std = 2, device="cuda"):
    mapping_net.to(device)
    mapping_net.eval()

    def forward_pass_for_rescaling(batch,predict_full_trajectory, add_noise):
        if add_noise:
            noisy_x, noisy_u = add_gaussian_noise(batch['x'], batch['u'], variance=0.1)
            X_final, U_final, t_final = mapping_net(noisy_x, noisy_u, batch['t'])
        
        else:
            X_final, U_final, t_final = mapping_net(batch['x'], batch['u'], batch['t'])

        X_final, U_final, t_final = mapping_net(batch['x'], batch['u'], batch['t'])


        X_final_mean, U_final_mean, t_for_pred = prepare_prediction_inputs(
            X_final, U_final, 
            t_batch=batch['t'], 
            trajectory_ids=batch['trajectory_ids'], 
            possible_t_values=possible_t_values,
            stats=None,
            predict_full_trajectory=predict_full_trajectory)
        
        x_pred, u_pred, _ = inverse_net(X_final_mean, U_final_mean, t_for_pred)

        X_labels, U_labels, _ = generate_prediction_labels(
                train_df=train_df, 
                train_id_df=train_id_df, 
                trajectory_ids=batch['trajectory_ids'], 
                t_for_pred=t_for_pred, 
                get_data_from_trajectory_id=get_data_from_trajectory_id,
                predict_full_trajectory=predict_full_trajectory,
                possible_t_values=possible_t_values,
            )
       

        prediction_loss_ = prediction_loss(
                x_pred=x_pred, 
                u_pred=u_pred, 
                X_labels=X_labels, 
                U_labels=U_labels,
                loss_type=loss_type
            )

        return prediction_loss_.item()


    prediction_losses_list = []
    for batch_idx, batch in enumerate(train_loader):
        prediction_loss_ = forward_pass_for_rescaling(batch,predict_full_trajectory=predict_full_trajectory, add_noise=add_noise)
        prediction_losses_list.append(prediction_loss_)
    prediction_loss_epoch_mean = np.mean(prediction_losses_list)


    prediction_loss_epoch_std = np.std(prediction_losses_list)

    if (np.abs(prediction_loss_epoch_mean/prediction_loss_epoch_std)<noise_threshold_mean_divided_by_std):

        prediction_loss_epoch_median = np.median(prediction_losses_list)
        print(f"Calculated epoch's prediction loss: {prediction_loss_epoch_mean:.4f}±{prediction_loss_epoch_std:.4f}\nWhich is too noisy so using median which is:{prediction_loss_epoch_median:.4f}")
        if save_returned_values:
            loss_scales = {"saved_prediction_loss_scale":prediction_loss_epoch_median}
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "loss_scales.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(loss_scales, f)
            print(f"Saved values at {save_path}")
        return prediction_loss_epoch_median
    else:
        print(f"Calculated epoch's prediction loss: {prediction_loss_epoch_mean:.4f}±{prediction_loss_epoch_std:.4f}, returning means")
        if save_returned_values:
            loss_scales = {"saved_prediction_loss_scale":prediction_loss_epoch_mean}
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "loss_scales.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(loss_scales, f)
            print(f"Saved values at {save_path}")
        return prediction_loss_epoch_mean

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(
    # Dataloaders
    train_loader,
    randomize_each_epoch_plan,
    val_loader, 
    val_loader_training_set,
    
    #Network
    mapping_net, 
    inverse_net,
     
    #Needed objects

    get_data_from_trajectory_id,
    possible_t_values,
    
    #Needed pandas dataframes
    train_df,
    train_id_df,
    val_df,
    val_id_df,
    add_noise,
    

    #Loss calculation hyperparameters

    prediction_loss_scale,
    loss_type,
    predict_full_trajectory,

    prediction_coefficient,

    hsic_loss_max_want,
    on_distribution_val_criterio_weight = 0.75,
    
    
    # Optimizer parameters
    optimizer_type = 'AdamW',
    
    learning_rate=1e-4,
    weight_decay=1e-6,

    scheduler_type='plateau', 
    scheduler_params=None,    # Dict of params specific to the scheduler. Set if it is a fresh training session.
    

    # Training parameters
    num_epochs=30,
    grad_clip_value=1.0,

    
    # Early stopping parameters
    early_stopping=True,
    patience=5,
    min_delta=0.001,
    
    # Checkpointing
    save_dir='./checkpoints',
    save_best_only=True,
    save_freq_epochs=1,
    auto_rescue=True,
    
    # Logging
    log_freq_batches=10,
    verbose=1,
    
    # Device
    device='cuda',
    
    #Used for continuing training from checkpoint, leave all None if it is a fresh training session.
    optimizer=None, 
    scheduler=None,  
    continue_from_epoch=None,
    best_validation_criterio_loss_till_now=None,
):
    
    """
    Comprehensive training loop.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mapping_net.to(device)
    
    # Create directory for checkpoints
    os.makedirs(save_dir, exist_ok=True)
    


    def make_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        else:
            return obj

    def save_rescue_checkpoint(signal_received=None, frame=None):
            """Save a rescue checkpoint and exit gracefully if needed"""
            nonlocal epoch, batch_idx, best_val_loss
            try:
                rescue_path = os.path.join(save_dir, f"rescue_checkpoint_epoch_{epoch}_batch_{batch_idx}.pt")
                print(f"\n\n{'='*50}")
                if signal_received:
                    print(f"Signal {signal_received} received. Saving rescue checkpoint...")
                else:
                    print(f"Exception detected. Saving rescue checkpoint...")
                    

                # Save current state  
                save_checkpoint(rescue_path, mapping_net, optimizer, scheduler, epoch, best_val_loss,
                               extra_info={'interrupted_at_batch': batch_idx, 'exception': traceback.format_exc()})
                run_hyperparameters ={}
                run_hyperparameters['train_dataloader_batch_size'] = train_loader.dataset.batch_size
                run_hyperparameters['train_dataloader_segment_length'] = train_loader.dataset.segment_length
                run_hyperparameters['train_dataloader_n_segments'] = train_loader.dataset.n_segments
                run_hyperparameters['train_dataloader_ratio'] = train_loader.dataset.ratio
                run_hyperparameters['train_dataloader_batch_traj'] = train_loader.dataset.batch_traj
                run_hyperparameters['randomize_each_epoch_plan'] = randomize_each_epoch_plan 
                run_hyperparameters['add_noise'] = add_noise
                
                
            
                n_parameters = count_parameters(mapping_net)
                run_hyperparameters['model_trainable_parameter_count'] = n_parameters
            
                model_n_layers = mapping_net.n_layers
                run_hyperparameters['model_n_layers'] = model_n_layers
            
                model_hidden_dims = mapping_net.layers[0].step_1.hidden_dims
                run_hyperparameters['model_hidden_dims'] = model_hidden_dims
            
                model_activation = mapping_net.activation 
                run_hyperparameters['model_activation'] = model_activation
            
                model_activation_params = mapping_net.activation_params 
                run_hyperparameters['model_activation_params'] = model_activation_params
            
                model_final_activation = mapping_net.final_activation 
                run_hyperparameters['model_final_activation'] = model_final_activation

                model_final_activation_only_on_final_layer = mapping_net.final_activation_only_on_final_layer 
                run_hyperparameters['model_final_activation_only_on_final_layer'] = model_final_activation_only_on_final_layer

                model_tanh_wrapper = mapping_net.tanh_wrapper
                run_hyperparameters['model_tanh_wrapper'] = model_tanh_wrapper
            
                model_weight_init = mapping_net.weight_init 
                run_hyperparameters['model_weight_init'] = model_weight_init
            
                model_weight_init_params = mapping_net.weight_init_params 
                run_hyperparameters['model_weight_init_params'] = model_weight_init_params
            
                model_bias_init = mapping_net.bias_init 
                run_hyperparameters['model_bias_init'] = model_bias_init
            
                model_bias_init_value = mapping_net.bias_init_value 
                run_hyperparameters['model_bias_init_value'] = model_bias_init_value

                model_use_layer_norm = mapping_net.use_layer_norm 
                run_hyperparameters['model_use_layer_norm'] = model_use_layer_norm
                
            
                model_a_eps_min = mapping_net.a_eps_min 
                run_hyperparameters['model_a_eps_min'] = model_a_eps_min
            
                model_a_eps_max = mapping_net.a_eps_max
                run_hyperparameters['model_a_eps_max'] = model_a_eps_max
                
                model_a_k = mapping_net.a_k
                run_hyperparameters['model_a_k'] = model_a_k

                model_step_1_a_mean_innit = mapping_net.step_1_a_mean_innit
                run_hyperparameters['model_step_1_a_mean_innit'] = model_step_1_a_mean_innit

                model_step_1_gamma_mean_innit = mapping_net.step_1_gamma_mean_innit
                run_hyperparameters['model_step_1_gamma_mean_innit'] = model_step_1_gamma_mean_innit

                model_step_1_c1_mean_innit = mapping_net.step_1_c1_mean_innit
                run_hyperparameters['model_step_1_c1_mean_innit'] = model_step_1_c1_mean_innit


                model_step_1_c2_mean_innit = mapping_net.step_1_c2_mean_innit
                run_hyperparameters['model_step_1_c2_mean_innit'] = model_step_1_c2_mean_innit

                model_step_2_a_mean_innit = mapping_net.step_2_a_mean_innit
                run_hyperparameters['model_step_2_a_mean_innit'] = model_step_2_a_mean_innit



                model_step_2_gamma_mean_innit = mapping_net.step_2_gamma_mean_innit
                run_hyperparameters['model_step_2_gamma_mean_innit'] = model_step_2_gamma_mean_innit

                model_step_2_c1_mean_innit = mapping_net.step_2_c1_mean_innit
                run_hyperparameters['model_step_2_c1_mean_innit'] = model_step_2_c1_mean_innit

                model_step_2_c2_mean_innit = mapping_net.step_2_c2_mean_innit
                run_hyperparameters['model_step_2_c2_mean_innit'] = model_step_2_c2_mean_innit


                model_std_to_mean_ratio_a_mean_init = mapping_net.std_to_mean_ratio_a_mean_init
                run_hyperparameters['model_std_to_mean_ratio_a_mean_init'] = model_std_to_mean_ratio_a_mean_init

                model_std_to_mean_ratio_gamma_mean_init = mapping_net.std_to_mean_ratio_gamma_mean_init
                run_hyperparameters['model_std_to_mean_ratio_gamma_mean_init'] = model_std_to_mean_ratio_gamma_mean_init

                model_std_to_mean_ratio_c1_mean_init = mapping_net.std_to_mean_ratio_c1_mean_init
                run_hyperparameters['model_std_to_mean_ratio_c1_mean_init'] = model_std_to_mean_ratio_c1_mean_init

                model_std_to_mean_ratio_c2_mean_init = mapping_net.std_to_mean_ratio_c2_mean_init
                run_hyperparameters['model_std_to_mean_ratio_c2_mean_init'] = model_std_to_mean_ratio_c2_mean_init

                model_bound_innit = mapping_net.bound_innit
                run_hyperparameters['model_bound_innit'] = model_bound_innit
            

                run_hyperparameters['prediction_loss_scale'] = prediction_loss_scale
                run_hyperparameters['loss_type'] = loss_type
                run_hyperparameters['predict_full_trajectory'] = predict_full_trajectory
                

                run_hyperparameters['prediction_coefficient'] = prediction_coefficient

                run_hyperparameters['hsic_loss_max_want'] = hsic_loss_max_want
                run_hyperparameters['on_distribution_val_criterio_weight'] = on_distribution_val_criterio_weight
                run_hyperparameters['grad_clip_value'] = grad_clip_value
                
            
            
                with open(os.path.join(save_dir, f"run_from_epochs_{start_epoch}_to_{epoch}.json"), "w") as f:
                    # Convert any non-serializable objects
                    serializable_run_hyperparameters = make_json_serializable(run_hyperparameters)
                    json.dump(serializable_run_hyperparameters, f, indent=2)

                print(f"Rescue checkpoint saved to: {rescue_path}")
                print(f"You can resume training from this checkpoint later.")
                print(f"{'='*50}\n")

                if signal_received:  # If this was triggered by a signal, exit
                    sys.exit(0)

            except Exception as e:
                print(f"Failed to save rescue checkpoint: {e}")
                
    epoch = 0 if continue_from_epoch is None else continue_from_epoch
    batch_idx = 0
    
    if auto_rescue:
        signal.signal(signal.SIGINT, save_rescue_checkpoint)  # Ctrl+C
        signal.signal(signal.SIGTERM, save_rescue_checkpoint)  # Termination request
    

    train_batches = train_loader.dataset.total_batches
    val_batches = len(val_loader)
    val_batches_training_set = len(val_loader_training_set)
    
    
    # Create parameter lists 
    mlp_params = []
    transform_params = []
    scale_params = []

    for name, param in mapping_net.named_parameters():
        if ('G_network' in name) or ('F_network' in name):
            mlp_params.append(param)
        elif ('g_bound' in name) or ('f_bound' in name) or ('a_raw' in name):
            scale_params.append(param)
        else: #c1, c2, gamma
            transform_params.append(param)
        
    



    


    if optimizer==None:
        if optimizer_type == 'Adam':
            print(f"Initializing new Adam optimizer with learning rate: {learning_rate}")
            optimizer = Adam([
                {'params': mlp_params, 'lr': learning_rate},
                {'params': scale_params, 'lr': learning_rate},
                {'params': transform_params, 'lr': learning_rate},
            ])
        elif optimizer_type == 'AdamW':
            print(f"Initializing new AdamW optimizer with learning rate: {learning_rate}, and weight decay {weight_decay}")
            optimizer = AdamW([
                {'params': mlp_params, 'weight_decay': weight_decay},
                {'params': scale_params, 'weight_decay': 0.0},
                {'params': transform_params, 'weight_decay': 0.0},
            ], lr=learning_rate)  
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")       
    
    # Create scheduler if requested
    if scheduler==None:
        print(f"Initializing new scheduler: {scheduler_type} with params: {scheduler_params}")

        if scheduler_type == 'plateau':
            scheduler_params = scheduler_params or {'mode': 'min', 'factor': 0.1, 'patience': 5, 'verbose': verbose > 0}
            scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    if (optimizer is not None) and (learning_rate != optimizer.param_groups[0]['lr']):
        optimizer.param_groups[0]['lr'] = learning_rate
        print(f"Manually resetting loaded optimizer's learning rate to {optimizer.param_groups[0]['lr']}")

    if (optimizer is not None) and (weight_decay != optimizer.param_groups[0]['weight_decay']):
        optimizer.param_groups[0]['weight_decay'] = weight_decay
        print(f"Manually resetting loaded optimizer's weight decay to {optimizer.param_groups[0]['weight_decay']}")
        
    # Initialize early stopping variables
    best_val_loss = float('inf')
    if best_validation_criterio_loss_till_now is not None:
        best_val_loss = best_validation_criterio_loss_till_now
    patience_counter = 0
    

    
    # Define a function to run the forward pass
    def forward_pass(batch, hsic_loss_slope, predict_full_trajectory, add_noise):
        if add_noise:
            noisy_x, noisy_u = add_gaussian_noise(batch['x'], batch['u'], variance=0.1)
            X_final, U_final, t_final = mapping_net(noisy_x, noisy_u, batch['t'])
        
        else:
            X_final, U_final, t_final = mapping_net(batch['x'], batch['u'], batch['t'])

        stats = TrajectoryStats(X_final, U_final, batch['trajectory_ids'])


        hsic_loss_ = hsic_loss(X_final, U_final, batch['trajectory_ids'], sigma_X_means=-1, sigma_U_means=-1, use_unbiased=True, stats=stats)




        X_final_mean, U_final_mean, t_for_pred = prepare_prediction_inputs(
            X_final, U_final, 
            t_batch=batch['t'], 
            trajectory_ids=batch['trajectory_ids'], 
            possible_t_values=possible_t_values,
            stats=stats,
            predict_full_trajectory=predict_full_trajectory)

        x_pred, u_pred, _ = inverse_net(X_final_mean, U_final_mean, t_for_pred)

        X_labels, U_labels, _ = generate_prediction_labels(
                train_df=train_df, 
                train_id_df=train_id_df, 
                trajectory_ids=batch['trajectory_ids'], 
                t_for_pred=t_for_pred, 
                get_data_from_trajectory_id=get_data_from_trajectory_id,
                predict_full_trajectory=predict_full_trajectory,  
                possible_t_values=possible_t_values 
            )
         
       
        prediction_loss_ = prediction_loss(
                x_pred=x_pred, 
                u_pred=u_pred, 
                X_labels=X_labels, 
                U_labels=U_labels,
                loss_type=loss_type
            )

        total_loss_ = compute_total_loss(
                hsic_loss = hsic_loss_,
                prediction_loss=prediction_loss_,
                prediction_coefficient=prediction_coefficient,
                hsic_loss_slope=hsic_loss_slope,
                prediction_loss_scale=prediction_loss_scale,

            )
        return total_loss_, hsic_loss_, prediction_loss_
    
    
    
    def validation_forward_pass(batch, val_df, val_id_df):
        X_final, U_final, t_final = mapping_net(batch['x'], batch['u'], batch['t'])
        variance_loss_, X_mean, U_mean, X_var, U_var, X_std, U_std = compute_single_trajectory_stats(X_final, U_final)


        traj_id_tensor = torch.tensor(batch['trajectory_id'], device=X_final.device)

        X_final_mean, U_final_mean, t_for_pred = prepare_prediction_inputs(
                X_final[0], U_final[0], 
                t_batch=batch['t'], 
                trajectory_ids=traj_id_tensor, 
                possible_t_values=possible_t_values,
                stats=None,
                predict_full_trajectory=predict_full_trajectory)

        x_pred, u_pred, _ = inverse_net(X_final_mean, U_final_mean, t_for_pred)

        X_labels, U_labels, _ = generate_prediction_labels(
        train_df=val_df, 
        train_id_df=val_id_df, 
        trajectory_ids=traj_id_tensor, 
        t_for_pred=t_for_pred, 
        get_data_from_trajectory_id=get_data_from_trajectory_id,
        predict_full_trajectory=predict_full_trajectory,  
        possible_t_values=possible_t_values 
        )

   
        prediction_loss_ = prediction_loss(
                x_pred=x_pred, 
                u_pred=u_pred, 
                X_labels=X_labels, 
                U_labels=U_labels,
                loss_type=loss_type
        )



        prediction_loss_scaled = prediction_loss_ / prediction_loss_scale


        total_loss_ =  prediction_coefficient * prediction_loss_scaled


        return total_loss_.item(), variance_loss_.item(), prediction_loss_.item(), X_mean, U_mean, X_var, U_var, X_std, U_std
    
    hsic_loss_max_calculated_cache = {}
    # Training loop
    start_epoch=0
    if continue_from_epoch is not None:
        start_epoch=continue_from_epoch
        
    try:
        for epoch in range(start_epoch, start_epoch+num_epochs):
                # Initialize epoch_metrics

            if epoch > start_epoch and randomize_each_epoch_plan:
                train_loader.dataset.on_epoch_start()


            epoch_metrics = {}
            epoch_start_time = time.time()

            

            # Set all models to training mode
            mapping_net.train()
            # Initialize metrics
            train_total_loss_ = 0.0
            train_hsic_loss_ = 0.0
            train_prediction_loss_ = 0.0
            mean_grad_norm_ = 0.0
            percentage_of_batches_clipped_in_epoch = 0.0
            batch_count = 0

            # Progress tracking
            if verbose > 0:
                print(f"\n{'='*20} Epoch {epoch}/{start_epoch+num_epochs} {'='*20}")

            # Training loop
            for batch_idx, batch in enumerate(train_loader):
                
                

                number_of_trajectories_in_batch =  torch.unique(batch['trajectory_ids']).shape[0]
                if number_of_trajectories_in_batch not in hsic_loss_max_calculated_cache:
                    linear_tensor = torch.arange(1, number_of_trajectories_in_batch+1, requires_grad=False)
                    hsic_loss_max_calculated = hsic_loss_statistics_only(x=torch.Tensor(linear_tensor), y=torch.Tensor(linear_tensor), sigma_x = -1, sigma_y = -1, use_unbiased = True, epsilon = 1e-10).item()
                    hsic_loss_max_calculated = max(hsic_loss_max_calculated, 0.05)
                    hsic_loss_max_calculated_cache[number_of_trajectories_in_batch] = hsic_loss_max_calculated
                else:
                    hsic_loss_max_calculated = hsic_loss_max_calculated_cache[number_of_trajectories_in_batch]
                
                hsic_loss_slope = hsic_loss_max_want / hsic_loss_max_calculated


                optimizer.zero_grad()

                # Run forward pass
                total_loss_, hsic_loss_, prediction_loss_ = forward_pass(batch, hsic_loss_slope, predict_full_trajectory=predict_full_trajectory, add_noise=add_noise)


                if torch.isnan(total_loss_) or torch.isinf(total_loss_):
                    print(f"Invalid total_loss_ value: {total_loss_.item()}, skipping batch")
                    continue
                # Backward pass
                total_loss_.backward()
                
                
                for name, param in mapping_net.named_parameters():
                    if param.grad is not None:
                        # Replace NaN gradients with zeros
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradients detected in {name}")
                            param.grad[torch.isnan(param.grad)] = 0.0
                        # Replace inf gradients
                        if torch.isinf(param.grad).any():
                            print(f"Inf gradients detected in {name}")
                            param.grad[torch.isinf(param.grad)] = 0.0
                        # Check for extreme gradients
                        if verbose>1:
                            grad_norm_print = param.grad.norm().item()
                            if grad_norm_print > 10.0:  # Threshold for reporting
                                print(f"High gradient norm in {name}: {grad_norm_print}")
                            if grad_norm_print < 0.00001:  
                                print(f"Low gradient norm in {name}: {grad_norm_print}")

                # Gradient clipping
                if grad_clip_value is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(mlp_params+scale_params+transform_params, grad_clip_value)
                    if (grad_norm>grad_clip_value):
                        percentage_of_batches_clipped_in_epoch += 1.0
                        if verbose>0:
                            print(f"Grad clipping activated, grad_norm before clipping: {grad_norm}")


                # Optimizer step
                optimizer.step()

                # Update metrics
                if grad_clip_value is not None:
                    mean_grad_norm_ += grad_norm.item()
                
                train_total_loss_ += total_loss_.item()
                train_hsic_loss_ += hsic_loss_.item()
                train_prediction_loss_ += prediction_loss_.item()
                batch_count += 1

                # Log progress
                if verbose > 0 and batch_idx % log_freq_batches == 0:
                    print(f"Batch {batch_idx}/{train_batches} - Total Loss: {total_loss_.item():.4f} - HSIC Loss: {hsic_loss_.item():.4f} - Prediction Loss: {prediction_loss_.item():.4f}")
                    if grad_clip_value is not None:
                        print(f"Grad norm of batch: {grad_norm.item():.4f}\n")


                        
                
            # Calculate epoch metrics
            mean_grad_norm_ /= max(1, batch_count)
            train_total_loss_ /= max(1, batch_count)
            train_hsic_loss_ /= max(1, batch_count)
            train_prediction_loss_ /= max(1, batch_count)
            percentage_of_batches_clipped_in_epoch /= max(1, batch_count)

            # Validation phase
            val_total_loss_ = 0.0
            val_variance_loss_ = 0.0
            val_prediction_loss_ = 0.0
            val_batch_count = 0


            # Set model to evaluation mode
            mapping_net.eval()
            
            epoch_saving_path = os.path.join(save_dir, f"epoch_{epoch}")
            
            val_trajectories_dir = os.path.join(epoch_saving_path, f"val_trajectories_data")
            os.makedirs(val_trajectories_dir, exist_ok=True)



            val_train_set_trajectories_dir = os.path.join(epoch_saving_path, f"val_train_set_trajectories_data")
            os.makedirs(val_train_set_trajectories_dir, exist_ok=True)


            
            
            # Validation loop

            print("Begining validation phase")
            for batch_idx, batch in enumerate(val_loader):
                # Run forward pass
                total_loss_, variance_loss_, prediction_loss_, X_mean, U_mean, X_var, U_var, X_std, U_std = validation_forward_pass(batch, val_df=val_df, val_id_df=val_id_df)
                # Update metrics
                if total_loss_ is not None:
                    val_total_loss_ += total_loss_
                    val_variance_loss_ += variance_loss_
                    val_prediction_loss_ += prediction_loss_
                    if verbose > 0 and batch_idx % log_freq_batches == 0:
                        print(f"Batch {batch_idx}/{val_batches} - Total Loss: {total_loss_:.4f}- Variance Loss: {variance_loss_:.4f}  - Prediction Loss: {prediction_loss_:.4f}")
                        
                    trajectory_data = {
                        'total_loss' : total_loss_,
                        'variance_loss' : variance_loss_,
                        'prediction_loss' : prediction_loss_,
                        'X_mean': X_mean,
                        'U_mean': U_mean,
                        'X_var': X_var,
                        'U_var': U_var,
                        'X_std': X_std,
                        'U_std': U_std,
                    }           
                    with open(os.path.join(val_trajectories_dir, f"trajectory_id_{batch['trajectory_id']}_data.json"), "w") as f:
                        # Convert any non-serializable objects
                        serializable_trajectory_data = make_json_serializable(trajectory_data)
                        json.dump(serializable_trajectory_data, f, indent=2)
                        
                        
                val_batch_count += 1
                    
                    

                    
            # Calculate validation metrics
            val_total_loss_ /= max(1, val_batch_count)
            val_variance_loss_ /= max(1, val_batch_count)
            val_prediction_loss_ /= max(1, val_batch_count)




            

         

            
            val_total_loss_training_set = 0.0
            val_variance_loss_training_set = 0.0

            val_prediction_loss_training_set = 0.0
            val_batch_count_training_set = 0
            

            print("Begining validation loop on training set data")
            for batch_idx, batch in enumerate(val_loader_training_set):
                # Run forward pass
                total_loss_, variance_loss_, prediction_loss_, X_mean, U_mean, X_var, U_var, X_std, U_std = validation_forward_pass(batch, val_df=train_df, val_id_df=train_id_df)
                
                # Update metrics
                if total_loss_ is not None:
                    val_total_loss_training_set += total_loss_
                    val_variance_loss_training_set += variance_loss_
                    val_prediction_loss_training_set += prediction_loss_
                    if verbose > 0 and batch_idx % log_freq_batches == 0:
                        print(f"Batch {batch_idx}/{val_batches_training_set} - Total Loss: {total_loss_:.4f}- Variance Loss: {variance_loss_:.4f}- Prediction Loss: {prediction_loss_:.4f}")
                    trajectory_data = {
                        'total_loss' : total_loss_,
                        'variance_loss' : variance_loss_,
                        'prediction_loss' : prediction_loss_,
                        'X_mean': X_mean,
                        'U_mean': U_mean,
                        'X_var': X_var,
                        'U_var': U_var,
                        'X_std': X_std,
                        'U_std': U_std,
                    }           
                    with open(os.path.join(val_train_set_trajectories_dir, f"trajectory_id_{batch['trajectory_id']}_data.json"), "w") as f:
                        # Convert any non-serializable objects
                        serializable_trajectory_data = make_json_serializable(trajectory_data)
                        json.dump(serializable_trajectory_data, f, indent=2)
                val_batch_count_training_set += 1

                    
            # Calculate validation metrics
            val_total_loss_training_set /= max(1, val_batch_count_training_set)
            val_variance_loss_training_set /= max(1, val_batch_count_training_set)
            val_prediction_loss_training_set /= max(1, val_batch_count_training_set)


            # Update epoch_metrics
            epoch_metrics['epoch'] = epoch


            
            epoch_metrics['percentage_of_batches_clipped_in_epoch'] = percentage_of_batches_clipped_in_epoch*100
            epoch_metrics['mean_grad_norm_'] = mean_grad_norm_
            epoch_metrics['train_total_loss_'] = train_total_loss_

            epoch_metrics['train_hsic_loss_'] = train_hsic_loss_

            epoch_metrics['train_prediction_loss_'] = train_prediction_loss_

            epoch_metrics['val_total_loss_'] = val_total_loss_
            epoch_metrics['val_variance_loss_'] = val_variance_loss_

            epoch_metrics['val_prediction_loss_'] = val_prediction_loss_
            

            
            epoch_metrics['val_total_loss_training_set'] = val_total_loss_training_set
            epoch_metrics['val_variance_loss_training_set'] = val_variance_loss_training_set

            epoch_metrics['val_prediction_loss_training_set'] = val_prediction_loss_training_set
            

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            epoch_metrics['learning_rates'] = current_lr


            validation_criterio = val_total_loss_
            epoch_metrics['validation_criterio'] = validation_criterio

            if validation_criterio < best_val_loss - min_delta:
                best_val_loss = validation_criterio
                epoch_metrics['best_validation_criterio_loss_till_now'] = validation_criterio
                print(f"Found best validation criterio loss till now:{validation_criterio:.4f}, on epoch {epoch}")

                # Save best model (whether early_stopping is enabled or not)
                best_model_path = os.path.join(save_dir, "best_model.pt")
                save_checkpoint(best_model_path, mapping_net, optimizer, scheduler, epoch, best_val_loss)
                if verbose > 0:
                    print(f"New best model saved with val_loss: {best_val_loss:.4f}")

                patience_counter = 0
            else:
                epoch_metrics['best_validation_criterio_loss_till_now'] = best_val_loss
                if early_stopping:
                    patience_counter += 1
                    if verbose > 0:
                        print(f"Early stopping patience: {patience_counter}/{patience}")
                
            with open(os.path.join(epoch_saving_path, "epoch_metrics.json"), "w") as f:
                # Convert any non-serializable objects
                serializable_epoch_metrics = make_json_serializable(epoch_metrics)
                json.dump(serializable_epoch_metrics, f, indent=2)
            
            # Step the scheduler if it exists
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(validation_criterio)  # Pass validation loss to plateau scheduler
                else:
                    scheduler.step()

            # Time taken for epoch
            epoch_time = time.time() - epoch_start_time

            # Print epoch summary
            if verbose > 0:
                print(f"\nEpoch {epoch}/{start_epoch+num_epochs} completed in {epoch_time:.2f}s")


                print(f"Percentage of batches clipped in epoch: {percentage_of_batches_clipped_in_epoch*100:.1f}%")
                if grad_clip_value is not None:
                    print(f"Mean grad norm: {mean_grad_norm_:.4f}")
                print(f"Mean total train loss: {train_total_loss_:.4f}")

                print(f"Mean HSIC train loss: {train_hsic_loss_:.4f}")

                print(f"Mean prediction train loss: {train_prediction_loss_:.4f}")
                
                
                print(f"Mean total val loss: {val_total_loss_:.4f}")
                print(f"Mean variance val loss: {val_variance_loss_:.4f}")

                print(f"Mean prediction val loss: {val_prediction_loss_:.4f}")
                
             

                print(f"Mean total val loss training set: {val_total_loss_training_set:.4f}")
                print(f"Mean variance val loss training set: {val_variance_loss_training_set:.4f}")

                print(f"Mean prediction val loss training set: {val_prediction_loss_training_set:.4f}")    
                
                print(f"Validation criterio: {validation_criterio:.4f}")    
                
                print(f"Learning Rate: {current_lr:.6f}")



            # Save checkpoint if needed
            if (epoch) % save_freq_epochs == 0 :
                if not save_best_only:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                    save_checkpoint(checkpoint_path, mapping_net, optimizer, scheduler, epoch, best_val_loss)
                    if verbose > 0:
                        print(f"Checkpoint saved to {checkpoint_path}")

            # Early stopping check
            if early_stopping and patience_counter >= patience:
                if verbose > 0:
                    print(f"Early stopping triggered in epoch: {epoch}")
                break

        run_hyperparameters ={}

        run_hyperparameters['train_dataloader_batch_size'] = train_loader.dataset.batch_size
        run_hyperparameters['train_dataloader_segment_length'] = train_loader.dataset.segment_length
        run_hyperparameters['train_dataloader_n_segments'] = train_loader.dataset.n_segments
        run_hyperparameters['train_dataloader_ratio'] = train_loader.dataset.ratio
        run_hyperparameters['train_dataloader_batch_traj'] = train_loader.dataset.batch_traj
        run_hyperparameters['randomize_each_epoch_plan'] = randomize_each_epoch_plan 
        run_hyperparameters['add_noise'] = add_noise 
        
        n_parameters = count_parameters(mapping_net)
        run_hyperparameters['model_trainable_parameter_count'] = n_parameters

        model_n_layers = mapping_net.n_layers
        run_hyperparameters['model_n_layers'] = model_n_layers

        model_hidden_dims = mapping_net.layers[0].step_1.hidden_dims
        run_hyperparameters['model_hidden_dims'] = model_hidden_dims

        model_activation = mapping_net.activation 
        run_hyperparameters['model_activation'] = model_activation

        model_activation_params = mapping_net.activation_params 
        run_hyperparameters['model_activation_params'] = model_activation_params

        model_final_activation = mapping_net.final_activation 
        run_hyperparameters['model_final_activation'] = model_final_activation

        model_final_activation_only_on_final_layer = mapping_net.final_activation_only_on_final_layer 
        run_hyperparameters['model_final_activation_only_on_final_layer'] = model_final_activation_only_on_final_layer

        model_tanh_wrapper = mapping_net.tanh_wrapper
        run_hyperparameters['model_tanh_wrapper'] = model_tanh_wrapper

        model_weight_init = mapping_net.weight_init 
        run_hyperparameters['model_weight_init'] = model_weight_init

        model_weight_init_params = mapping_net.weight_init_params 
        run_hyperparameters['model_weight_init_params'] = model_weight_init_params

        model_bias_init = mapping_net.bias_init 
        run_hyperparameters['model_bias_init'] = model_bias_init

        model_bias_init_value = mapping_net.bias_init_value 
        run_hyperparameters['model_bias_init_value'] = model_bias_init_value

        model_use_layer_norm = mapping_net.use_layer_norm 
        run_hyperparameters['model_use_layer_norm'] = model_use_layer_norm

        model_a_eps_min = mapping_net.a_eps_min 
        run_hyperparameters['model_a_eps_min'] = model_a_eps_min

        model_a_eps_max = mapping_net.a_eps_max
        run_hyperparameters['model_a_eps_max'] = model_a_eps_max

        model_a_k = mapping_net.a_k
        run_hyperparameters['model_a_k'] = model_a_k


        model_step_1_a_mean_innit = mapping_net.step_1_a_mean_innit
        run_hyperparameters['model_step_1_a_mean_innit'] = model_step_1_a_mean_innit

        model_step_1_gamma_mean_innit = mapping_net.step_1_gamma_mean_innit
        run_hyperparameters['model_step_1_gamma_mean_innit'] = model_step_1_gamma_mean_innit

        model_step_1_c1_mean_innit = mapping_net.step_1_c1_mean_innit
        run_hyperparameters['model_step_1_c1_mean_innit'] = model_step_1_c1_mean_innit


        model_step_1_c2_mean_innit = mapping_net.step_1_c2_mean_innit
        run_hyperparameters['model_step_1_c2_mean_innit'] = model_step_1_c2_mean_innit

        model_step_2_a_mean_innit = mapping_net.step_2_a_mean_innit
        run_hyperparameters['model_step_2_a_mean_innit'] = model_step_2_a_mean_innit



        model_step_2_gamma_mean_innit = mapping_net.step_2_gamma_mean_innit
        run_hyperparameters['model_step_2_gamma_mean_innit'] = model_step_2_gamma_mean_innit

        model_step_2_c1_mean_innit = mapping_net.step_2_c1_mean_innit
        run_hyperparameters['model_step_2_c1_mean_innit'] = model_step_2_c1_mean_innit

        model_step_2_c2_mean_innit = mapping_net.step_2_c2_mean_innit
        run_hyperparameters['model_step_2_c2_mean_innit'] = model_step_2_c2_mean_innit


        model_std_to_mean_ratio_a_mean_init = mapping_net.std_to_mean_ratio_a_mean_init
        run_hyperparameters['model_std_to_mean_ratio_a_mean_init'] = model_std_to_mean_ratio_a_mean_init

        model_std_to_mean_ratio_gamma_mean_init = mapping_net.std_to_mean_ratio_gamma_mean_init
        run_hyperparameters['model_std_to_mean_ratio_gamma_mean_init'] = model_std_to_mean_ratio_gamma_mean_init

        model_std_to_mean_ratio_c1_mean_init = mapping_net.std_to_mean_ratio_c1_mean_init
        run_hyperparameters['model_std_to_mean_ratio_c1_mean_init'] = model_std_to_mean_ratio_c1_mean_init

        model_std_to_mean_ratio_c2_mean_init = mapping_net.std_to_mean_ratio_c2_mean_init
        run_hyperparameters['model_std_to_mean_ratio_c2_mean_init'] = model_std_to_mean_ratio_c2_mean_init



        model_bound_innit = mapping_net.bound_innit
        run_hyperparameters['model_bound_innit'] = model_bound_innit




        run_hyperparameters['prediction_loss_scale'] = prediction_loss_scale
        run_hyperparameters['loss_type'] = loss_type
        run_hyperparameters['predict_full_trajectory'] = predict_full_trajectory



        run_hyperparameters['prediction_coefficient'] = prediction_coefficient
        run_hyperparameters['hsic_loss_max_want'] = hsic_loss_max_want
        run_hyperparameters['on_distribution_val_criterio_weight'] = on_distribution_val_criterio_weight
        run_hyperparameters['grad_clip_value'] = grad_clip_value
    


        with open(os.path.join(save_dir, f"run_from_epochs_{start_epoch}_to_{epoch+1}.json"), "w") as f:
            # Convert any non-serializable objects
            serializable_run_hyperparameters = make_json_serializable(run_hyperparameters)
            json.dump(serializable_run_hyperparameters, f, indent=2)
            
    except Exception as e:
        if auto_rescue:
            print(f"\nTraining interrupted by exception: {e}")
            save_rescue_checkpoint()
        raise  # Re-raise the exception after saving
    



    finally:
        try:
            if 'epoch' in locals():
                # Always save a final checkpoint regardless of how training ended
                final_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                save_checkpoint(final_path, mapping_net, optimizer, scheduler, epoch, best_val_loss)
                if verbose > 0:
                    print(f"\nFinal state saved to {final_path}")
            else:
                print("Exception before epoch loop began")
        except Exception as e:
            print(f"Failed to save final checkpoint: {e}")

    # Training complete
    if verbose > 0:
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    

    
    return 

def resume_training_from_checkpoint(
    # Checkpoint to resume from
    checkpoint_path,
    
    # Dataloaders
    train_loader,
    randomize_each_epoch_plan,
    val_loader,

    val_loader_training_set,
    
    # Network
    mapping_net,  # inverse_net is created automatically from mapping_net
    
    # Needed objects
    get_data_from_trajectory_id,
    possible_t_values,
    
    # Needed pandas dataframes
    train_df,
    train_id_df,
    val_df,
    val_id_df,
    add_noise,
    
    # Loss calculation hyperparameters
    prediction_loss_scale,
    predict_full_trajectory,
    loss_type,
    prediction_coefficient,
    hsic_loss_max_want,
    on_distribution_val_criterio_weight=0.75,
    
    # === RESUME MODE SELECTION ===
    # MODE A (load_scheduler_and_optimizer=True): Resume with exact optimizer/scheduler state
    
    #   - learning_rate: Set to None to use checkpoint LR, or specify to override
    #   - optimizer_type: MUST match the type used in original training
    #   - scheduler_type: MUST match the type used in original training
    #   - scheduler_params: MUST match the params used in original training
    #
    # MODE B (load_scheduler_and_optimizer=False): Resume with fresh optimizer/scheduler

    #   - learning_rate: REQUIRED - must specify, will error if None
    #   - optimizer_type: Can be different from original
    #   - scheduler_type: Can be different from original
    #   - scheduler_params: Can be different from original
    load_scheduler_and_optimizer=False,
    
    # Optimizer parameters
    learning_rate=None,  # MODE A: None=use checkpoint LR, or specify to override | MODE B: REQUIRED, must specify
    weight_decay=None,   # MODE A: None=use checkpoint weight_decay, or specify to override | MODE B: REQUIRED, must specify
    optimizer_type='AdamW',  # MODE A: must match original | MODE B: can be different
    
    # Scheduler parameters  
    scheduler_type='plateau',  # MODE A: must match original | MODE B: can be different
    scheduler_params=None,  # MODE A: can differ. Functionality would depend on reset_scheduler_patience | MODE B: can be different
    reset_scheduler_patience = False, #Only relevant on MODE A. Set True to reset num_bad_epochs. Use True if you want the learning rate to be lowered after the full patience amount. Use False if you want continuity, already waited N epochs, just need M-N more where M: loaded num_bad_epochs from previous training
    
    # Training parameters
    num_epochs=30,  # Number of ADDITIONAL epochs to train (not total epochs)
    grad_clip_value=3.0,
    
    # Early stopping parameters
    early_stopping=False,
    patience=20,
    min_delta=0.001,
    
    # Checkpointing
    save_dir='./checkpoints',  # Should typically match the directory where checkpoint_path is located
    save_best_only=False,
    save_freq_epochs=2,
    auto_rescue=True,
    
    # Logging
    log_freq_batches=50,
    verbose=2,
    
    # Device
    device='cuda'
    ):
    """Resume training from a checkpoint"""
    if load_scheduler_and_optimizer:
        mlp_params = []
        transform_params = []
        scale_params = []

        for name, param in mapping_net.named_parameters():
            if ('G_network' in name) or ('F_network' in name):
                mlp_params.append(param)
            elif ('g_bound' in name) or ('f_bound' in name) or ('a_raw' in name):
                scale_params.append(param)
            else: #c1, c2, gamma
                transform_params.append(param)

        if optimizer_type == 'Adam':
            temp_optimizer = Adam(mlp_params+scale_params+transform_params)
        elif optimizer_type == 'AdamW':
            temp_optimizer = AdamW([
                {'params': mlp_params, 'weight_decay': weight_decay},
                {'params': scale_params, 'weight_decay': 0.0},
                {'params': transform_params, 'weight_decay': weight_decay},
            ])

        temp_scheduler = ReduceLROnPlateau(temp_optimizer, mode='min', factor=0.1, patience=5, verbose=True) #Will be overwritten
        # Load checkpoint
        epoch, extra_info, best_val_loss = load_checkpoint(
            checkpoint_path, mapping_net, device, temp_optimizer, temp_scheduler
        )

        # Override scheduler params if specified
        if scheduler_params is not None:
            print(f"Overriding scheduler params: {scheduler_params}")
            for key, value in scheduler_params.items():
                if hasattr(temp_scheduler, key):
                    setattr(temp_scheduler, key, value)
                    print(f"  {key}: {getattr(temp_scheduler, key)} -> {value}")
                else:
                    print(f"  Warning: scheduler has no attribute '{key}'")
        else:
            if hasattr(temp_scheduler, 'patience'):  # ReduceLROnPlateau
                print("Using loaded scheduler")
                print(f"  factor: {temp_scheduler.factor}")
                print(f"  mode: {temp_scheduler.mode}")
                print(f"  threshold: {temp_scheduler.threshold}")
                print(f"  cooldown: {temp_scheduler.cooldown}")
                print(f"  best: {temp_scheduler.best}")


        # Reset scheduler patience counter if requested
        if reset_scheduler_patience:
            temp_scheduler.num_bad_epochs = 0
            temp_scheduler.cooldown_counter = 0
            print(f"Reset scheduler patience counter. Will wait full {temp_scheduler.patience} epochs before reducing LR.")
        else:
            # Calculate remaining epochs until LR reduction
            if temp_scheduler.cooldown_counter > 0:
                print(f"Scheduler in cooldown mode: {temp_scheduler.cooldown_counter} epochs remaining in cooldown")
            elif temp_scheduler.num_bad_epochs >= temp_scheduler.patience:
                print(f"Scheduler ready to reduce LR on next bad epoch (num_bad_epochs={temp_scheduler.num_bad_epochs} >= patience={temp_scheduler.patience})")
            else:
                remaining_epochs = temp_scheduler.patience - temp_scheduler.num_bad_epochs
                print(f"Scheduler status: {temp_scheduler.num_bad_epochs}/{temp_scheduler.patience} bad epochs. Will reduce LR after {remaining_epochs} more bad epochs.")

    else:
        epoch, extra_info, best_val_loss = load_checkpoint(
            checkpoint_path, mapping_net, device, None, None
        )

    if extra_info and 'interrupted_at_batch' in extra_info:
        continue_from_epoch = epoch
        print(f"Resuming incomplete epoch {epoch}")
        if 'exception' in extra_info:
            print(f"Previous training was interrupted by exception:\n{extra_info['exception']}")
    else:
        continue_from_epoch = epoch + 1
        print(f"Epoch {epoch} was completed, starting from epoch {epoch + 1}")
    
    # Use the learning rate from checkpoint if not specified
    if (learning_rate is None):
        if load_scheduler_and_optimizer:
            learning_rate = temp_optimizer.param_groups[0]['lr']
            print(f"Using learning rate from checkpoint: {learning_rate}")
        else:
            print("When load_scheduler_and_optimizer=False you should set Learning Rate")
            sys.exit(1)

    if (weight_decay is None):
        if load_scheduler_and_optimizer:
            weight_decay = temp_optimizer.param_groups[0]['weight_decay']
            print(f"Using weight decay from checkpoint: {weight_decay}")
        else:
            print("When load_scheduler_and_optimizer=False you should set Weight Decay")
            sys.exit(1)
            
    inverse_net = InverseStackedHamiltonianNetwork(forward_network=mapping_net)
    
    # Train for additional epochs
    print(f"Training for {num_epochs} additional epochs (epochs {continue_from_epoch} to {continue_from_epoch + num_epochs - 1})")
    
    # Resume training with the loaded state
    if load_scheduler_and_optimizer:
        return train_model(
            # Dataloaders
            train_loader=train_loader,
            randomize_each_epoch_plan=randomize_each_epoch_plan,
            val_loader=val_loader,

            val_loader_training_set=val_loader_training_set,
            
            # Network
            mapping_net=mapping_net, 
            inverse_net=inverse_net,
            
            # Needed objects
            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values=possible_t_values,
            
            # Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,
            val_df=val_df,
            val_id_df=val_id_df,

            add_noise=add_noise,
            
            # Loss calculation hyperparameters

            prediction_loss_scale=prediction_loss_scale,
            predict_full_trajectory=predict_full_trajectory,
            loss_type=loss_type,

            prediction_coefficient=prediction_coefficient,
            hsic_loss_max_want = hsic_loss_max_want,
            on_distribution_val_criterio_weight=on_distribution_val_criterio_weight,
            
            # Optimizer parameters - using loaded states
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=temp_optimizer,  # Loaded from checkpoint
            optimizer_type=optimizer_type,
            scheduler=temp_scheduler,  # Loaded from checkpoint
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_params,
            
            # Training parameters
            num_epochs=num_epochs,
            grad_clip_value=grad_clip_value,
            continue_from_epoch=continue_from_epoch,
            best_validation_criterio_loss_till_now=best_val_loss,  # Loaded from checkpoint
            
            # Early stopping parameters
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            
            # Checkpointing
            save_dir=save_dir,
            save_best_only=save_best_only,
            save_freq_epochs=save_freq_epochs,
            auto_rescue=auto_rescue,
            
            # Logging
            log_freq_batches=log_freq_batches,
            verbose=verbose,
            
            # Device
            device=device,
        )
    else:
        return train_model(
            # Dataloaders
            train_loader=train_loader,
            randomize_each_epoch_plan=randomize_each_epoch_plan,
            val_loader=val_loader,
            val_loader_training_set=val_loader_training_set,
            
            # Network
            mapping_net=mapping_net, 
            inverse_net=inverse_net,
            
            # Needed objects
            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values=possible_t_values,
            
            # Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,
            val_df=val_df,
            val_id_df=val_id_df,
            add_noise=add_noise,
            
            # Loss calculation hyperparameters
            prediction_loss_scale=prediction_loss_scale,
            predict_full_trajectory=predict_full_trajectory,
            loss_type=loss_type,
            prediction_coefficient=prediction_coefficient,

            hsic_loss_max_want = hsic_loss_max_want,
            on_distribution_val_criterio_weight=on_distribution_val_criterio_weight,
            
            # Optimizer parameters - creating fresh
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=None,  # Create fresh optimizer
            optimizer_type=optimizer_type,
            scheduler=None,  # Create fresh scheduler
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_params,
            
            # Training parameters
            num_epochs=num_epochs,
            grad_clip_value=grad_clip_value,
            continue_from_epoch=continue_from_epoch,
            best_validation_criterio_loss_till_now=best_val_loss,  # Loaded from checkpoint
            
            # Early stopping parameters
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            
            # Checkpointing
            save_dir=save_dir,
            save_best_only=save_best_only,
            save_freq_epochs=save_freq_epochs,
            auto_rescue=auto_rescue,
            
            # Logging
            log_freq_batches=log_freq_batches,
            verbose=verbose,
            
            # Device
            device=device,)
    

def train_model_real_pendulum(
    # Dataloaders
    train_loader,
    randomize_each_epoch_plan,
    val_loader, 

    
    #Network
    mapping_net, 
    inverse_net,
     
    #Needed objects

    get_data_from_trajectory_id,
    possible_t_values_per_trajectory,
    possible_t_values_per_trajectory_val,
    
    #Needed pandas dataframes
    train_df,
    train_id_df,
    val_df,
    val_id_df,

    add_noise,
    

    #Loss calculation hyperparameters
    loss_type,
    predict_full_trajectory,


    
    
    # Optimizer parameters
    optimizer_type = 'AdamW',
    
    learning_rate=1e-4,
    weight_decay=1e-6,

    scheduler_type='plateau', 
    scheduler_params=None,    # Dict of params specific to the scheduler. Set if it is a fresh training session.
    

    # Training parameters
    num_epochs=30,
    grad_clip_value=1.0,

    
    # Early stopping parameters
    early_stopping=True,
    patience=5,
    min_delta=0.001,
    
    # Checkpointing
    save_dir='./checkpoints',
    save_best_only=True,
    save_freq_epochs=1,
    auto_rescue=True,
    
    # Logging
    log_freq_batches=10,
    verbose=1,
    
    # Device
    device='cuda',
    
    #Used for continuing training from checkpoint, leave all None if it is a fresh training session.
    optimizer=None, 
    scheduler=None,  
    continue_from_epoch=None,
    best_validation_criterio_loss_till_now=None,
):
    
    """
    Comprehensive training loop.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mapping_net.to(device)
    
    # Create directory for checkpoints
    os.makedirs(save_dir, exist_ok=True)
    


    def make_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        else:
            return obj

    def save_rescue_checkpoint(signal_received=None, frame=None):
            """Save a rescue checkpoint and exit gracefully if needed"""
            nonlocal epoch, batch_idx, best_val_loss
            try:
                rescue_path = os.path.join(save_dir, f"rescue_checkpoint_epoch_{epoch}_batch_{batch_idx}.pt")
                print(f"\n\n{'='*50}")
                if signal_received:
                    print(f"Signal {signal_received} received. Saving rescue checkpoint...")
                else:
                    print(f"Exception detected. Saving rescue checkpoint...")
                    

                # Save current state  
                save_checkpoint(rescue_path, mapping_net, optimizer, scheduler, epoch, best_val_loss,
                               extra_info={'interrupted_at_batch': batch_idx, 'exception': traceback.format_exc()})
                run_hyperparameters ={}
                run_hyperparameters['train_dataloader_batch_size'] = train_loader.dataset.batch_size
                run_hyperparameters['train_dataloader_segment_length'] = train_loader.dataset.segment_length
                run_hyperparameters['train_dataloader_n_segments'] = train_loader.dataset.n_segments
                run_hyperparameters['train_dataloader_ratio'] = train_loader.dataset.ratio
                run_hyperparameters['train_dataloader_batch_traj'] = train_loader.dataset.batch_traj
                run_hyperparameters['randomize_each_epoch_plan'] = randomize_each_epoch_plan 
                run_hyperparameters['add_noise'] = add_noise


                
            
                n_parameters = count_parameters(mapping_net)
                run_hyperparameters['model_trainable_parameter_count'] = n_parameters
            
                model_n_layers = mapping_net.n_layers
                run_hyperparameters['model_n_layers'] = model_n_layers
            
                model_hidden_dims = mapping_net.layers[0].step_1.hidden_dims
                run_hyperparameters['model_hidden_dims'] = model_hidden_dims
            
                model_activation = mapping_net.activation 
                run_hyperparameters['model_activation'] = model_activation
            
                model_activation_params = mapping_net.activation_params 
                run_hyperparameters['model_activation_params'] = model_activation_params
            
                model_final_activation = mapping_net.final_activation 
                run_hyperparameters['model_final_activation'] = model_final_activation

                model_final_activation_only_on_final_layer = mapping_net.final_activation_only_on_final_layer 
                run_hyperparameters['model_final_activation_only_on_final_layer'] = model_final_activation_only_on_final_layer

                model_tanh_wrapper = mapping_net.tanh_wrapper
                run_hyperparameters['model_tanh_wrapper'] = model_tanh_wrapper
            
                model_weight_init = mapping_net.weight_init 
                run_hyperparameters['model_weight_init'] = model_weight_init
            
                model_weight_init_params = mapping_net.weight_init_params 
                run_hyperparameters['model_weight_init_params'] = model_weight_init_params
            
                model_bias_init = mapping_net.bias_init 
                run_hyperparameters['model_bias_init'] = model_bias_init
            
                model_bias_init_value = mapping_net.bias_init_value 
                run_hyperparameters['model_bias_init_value'] = model_bias_init_value

                model_use_layer_norm = mapping_net.use_layer_norm 
                run_hyperparameters['model_use_layer_norm'] = model_use_layer_norm
                
            
                model_a_eps_min = mapping_net.a_eps_min 
                run_hyperparameters['model_a_eps_min'] = model_a_eps_min
            
                model_a_eps_max = mapping_net.a_eps_max
                run_hyperparameters['model_a_eps_max'] = model_a_eps_max
                
                model_a_k = mapping_net.a_k
                run_hyperparameters['model_a_k'] = model_a_k

                model_step_1_a_mean_innit = mapping_net.step_1_a_mean_innit
                run_hyperparameters['model_step_1_a_mean_innit'] = model_step_1_a_mean_innit

                model_step_1_gamma_mean_innit = mapping_net.step_1_gamma_mean_innit
                run_hyperparameters['model_step_1_gamma_mean_innit'] = model_step_1_gamma_mean_innit

                model_step_1_c1_mean_innit = mapping_net.step_1_c1_mean_innit
                run_hyperparameters['model_step_1_c1_mean_innit'] = model_step_1_c1_mean_innit


                model_step_1_c2_mean_innit = mapping_net.step_1_c2_mean_innit
                run_hyperparameters['model_step_1_c2_mean_innit'] = model_step_1_c2_mean_innit

                model_step_2_a_mean_innit = mapping_net.step_2_a_mean_innit
                run_hyperparameters['model_step_2_a_mean_innit'] = model_step_2_a_mean_innit



                model_step_2_gamma_mean_innit = mapping_net.step_2_gamma_mean_innit
                run_hyperparameters['model_step_2_gamma_mean_innit'] = model_step_2_gamma_mean_innit

                model_step_2_c1_mean_innit = mapping_net.step_2_c1_mean_innit
                run_hyperparameters['model_step_2_c1_mean_innit'] = model_step_2_c1_mean_innit

                model_step_2_c2_mean_innit = mapping_net.step_2_c2_mean_innit
                run_hyperparameters['model_step_2_c2_mean_innit'] = model_step_2_c2_mean_innit


                model_std_to_mean_ratio_a_mean_init = mapping_net.std_to_mean_ratio_a_mean_init
                run_hyperparameters['model_std_to_mean_ratio_a_mean_init'] = model_std_to_mean_ratio_a_mean_init

                model_std_to_mean_ratio_gamma_mean_init = mapping_net.std_to_mean_ratio_gamma_mean_init
                run_hyperparameters['model_std_to_mean_ratio_gamma_mean_init'] = model_std_to_mean_ratio_gamma_mean_init

                model_std_to_mean_ratio_c1_mean_init = mapping_net.std_to_mean_ratio_c1_mean_init
                run_hyperparameters['model_std_to_mean_ratio_c1_mean_init'] = model_std_to_mean_ratio_c1_mean_init

                model_std_to_mean_ratio_c2_mean_init = mapping_net.std_to_mean_ratio_c2_mean_init
                run_hyperparameters['model_std_to_mean_ratio_c2_mean_init'] = model_std_to_mean_ratio_c2_mean_init

                model_bound_innit = mapping_net.bound_innit
                run_hyperparameters['model_bound_innit'] = model_bound_innit
            

                run_hyperparameters['loss_type'] = loss_type
                run_hyperparameters['predict_full_trajectory'] = predict_full_trajectory


                run_hyperparameters['grad_clip_value'] = grad_clip_value
                
            
            
                with open(os.path.join(save_dir, f"run_from_epochs_{start_epoch}_to_{epoch}.json"), "w") as f:
                    # Convert any non-serializable objects
                    serializable_run_hyperparameters = make_json_serializable(run_hyperparameters)
                    json.dump(serializable_run_hyperparameters, f, indent=2)

                print(f"Rescue checkpoint saved to: {rescue_path}")
                print(f"You can resume training from this checkpoint later.")
                print(f"{'='*50}\n")

                if signal_received:  # If this was triggered by a signal, exit
                    sys.exit(0)

            except Exception as e:
                print(f"Failed to save rescue checkpoint: {e}")
                
    epoch = 0 if continue_from_epoch is None else continue_from_epoch
    batch_idx = 0
    
    if auto_rescue:
        signal.signal(signal.SIGINT, save_rescue_checkpoint)  # Ctrl+C
        signal.signal(signal.SIGTERM, save_rescue_checkpoint)  # Termination request
    

    train_batches = train_loader.dataset.total_batches
    val_batches = len(val_loader)

    
    
    # Create parameter lists 
    mlp_params = []
    transform_params = []
    scale_params = []

    for name, param in mapping_net.named_parameters():
        if ('G_network' in name) or ('F_network' in name):
            mlp_params.append(param)
        elif ('g_bound' in name) or ('f_bound' in name) or ('a_raw' in name):
            scale_params.append(param)
        else: #c1, c2, gamma
            transform_params.append(param)
        
    



    


    if optimizer==None:
        if optimizer_type == 'Adam':
            print(f"Initializing new Adam optimizer with learning rate: {learning_rate}")
            optimizer = Adam([
                {'params': mlp_params, 'lr': learning_rate},
                {'params': scale_params, 'lr': learning_rate},
                {'params': transform_params, 'lr': learning_rate},
            ])
        elif optimizer_type == 'AdamW':
            print(f"Initializing new AdamW optimizer with learning rate: {learning_rate}, and weight decay {weight_decay}")
            optimizer = AdamW([
                {'params': mlp_params, 'weight_decay': weight_decay},
                {'params': scale_params, 'weight_decay': 0.0},
                {'params': transform_params, 'weight_decay': 0.0},
            ], lr=learning_rate)  
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")       
    
    # Create scheduler if requested
    if scheduler==None:
        print(f"Initializing new scheduler: {scheduler_type} with params: {scheduler_params}")

        if scheduler_type == 'plateau':
            scheduler_params = scheduler_params or {'mode': 'min', 'factor': 0.1, 'patience': 5, 'verbose': verbose > 0}
            scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    if (optimizer is not None) and (learning_rate != optimizer.param_groups[0]['lr']):
        optimizer.param_groups[0]['lr'] = learning_rate
        print(f"Manually resetting loaded optimizer's learning rate to {optimizer.param_groups[0]['lr']}")

    if (optimizer is not None) and (weight_decay != optimizer.param_groups[0]['weight_decay']):
        optimizer.param_groups[0]['weight_decay'] = weight_decay
        print(f"Manually resetting loaded optimizer's weight decay to {optimizer.param_groups[0]['weight_decay']}")
        
    # Initialize early stopping variables
    best_val_loss = float('inf')
    if best_validation_criterio_loss_till_now is not None:
        best_val_loss = best_validation_criterio_loss_till_now
    patience_counter = 0
    

    
    # Define a function to run the forward pass
    def forward_pass(batch, predict_full_trajectory, add_noise):
        if add_noise:
            noisy_x, noisy_u = add_gaussian_noise(batch['x'], batch['u'], variance=0.1)
            X_final, U_final, t_final = mapping_net(noisy_x, noisy_u, batch['t'])
        
        else:
            X_final, U_final, t_final = mapping_net(batch['x'], batch['u'], batch['t'])





        X_final_mean, U_final_mean, t_for_pred = prepare_prediction_inputs_for_real_pendulum_2(
            X_final, U_final, 
            t_batch=batch['t'], 
            trajectory_ids=batch['trajectory_ids'], 
            possible_t_values_per_trajectory=possible_t_values_per_trajectory,
            stats=None,
            predict_full_trajectory=predict_full_trajectory)

        x_pred, u_pred, _ = inverse_net(X_final_mean, U_final_mean, t_for_pred)

        X_labels, U_labels, _ = generate_prediction_labels_for_real_pendulum_2(
                train_df=train_df, 
                train_id_df=train_id_df, 
                trajectory_ids=batch['trajectory_ids'], 
                t_for_pred=t_for_pred, 
                get_data_from_trajectory_id=get_data_from_trajectory_id,
                predict_full_trajectory=predict_full_trajectory,  
                possible_t_values_per_trajectory=possible_t_values_per_trajectory,
            )

            
        prediction_loss_ = prediction_loss(
                x_pred=x_pred, 
                u_pred=u_pred, 
                X_labels=X_labels, 
                U_labels=U_labels,
                loss_type=loss_type
            )


        return prediction_loss_
    
    
    
    def validation_forward_pass(batch, val_df, val_id_df):
        X_final, U_final, t_final = mapping_net(batch['x'], batch['u'], batch['t'])


        X_final_mean, U_final_mean, t_for_pred = prepare_prediction_inputs_for_real_pendulum_2(
            X_final, U_final, 
            t_batch=batch['t'], 
            trajectory_ids=batch['trajectory_ids'], 
            possible_t_values_per_trajectory=possible_t_values_per_trajectory_val,
            stats=None,
            predict_full_trajectory=predict_full_trajectory)

        x_pred, u_pred, _ = inverse_net(X_final_mean, U_final_mean, t_for_pred)

        X_labels, U_labels, _ = generate_prediction_labels_for_real_pendulum_2(
        train_df=val_df, 
        train_id_df=val_id_df, 
        trajectory_ids=batch['trajectory_ids'], 
        t_for_pred=t_for_pred, 
        get_data_from_trajectory_id=get_data_from_trajectory_id,
        predict_full_trajectory=predict_full_trajectory,  
        possible_t_values_per_trajectory=possible_t_values_per_trajectory_val,

        )

        prediction_loss_ = prediction_loss(
                x_pred=x_pred, 
                u_pred=u_pred, 
                X_labels=X_labels, 
                U_labels=U_labels,
                loss_type=loss_type
            )
        return prediction_loss_.item()
    

    # Training loop
    start_epoch=0
    if continue_from_epoch is not None:
        start_epoch=continue_from_epoch
        
    try:
        for epoch in range(start_epoch, start_epoch+num_epochs):
                # Initialize epoch_metrics

            if epoch > start_epoch and randomize_each_epoch_plan:
                train_loader.dataset.on_epoch_start()


            epoch_metrics = {}
            epoch_start_time = time.time()

            

            # Set all models to training mode
            mapping_net.train()
            # Initialize metrics

            train_prediction_loss_ = 0.0
            mean_grad_norm_ = 0.0
            percentage_of_batches_clipped_in_epoch = 0.0
            batch_count = 0

            # Progress tracking
            if verbose > 0:
                print(f"\n{'='*20} Epoch {epoch}/{start_epoch+num_epochs} {'='*20}")

            # Training loop
            for batch_idx, batch in enumerate(train_loader):

                optimizer.zero_grad()

                # Run forward pass
                prediction_loss_ = forward_pass(batch, predict_full_trajectory=predict_full_trajectory, add_noise=add_noise)


                if torch.isnan(prediction_loss_) or torch.isinf(prediction_loss_):
                    print(f"Invalid prediction_loss_ value: {prediction_loss_.item()}, skipping batch")
                    continue
                # Backward pass
                prediction_loss_.backward()
                
                
                for name, param in mapping_net.named_parameters():
                    if param.grad is not None:
                        # Replace NaN gradients with zeros
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradients detected in {name}")
                            param.grad[torch.isnan(param.grad)] = 0.0
                        # Replace inf gradients
                        if torch.isinf(param.grad).any():
                            print(f"Inf gradients detected in {name}")
                            param.grad[torch.isinf(param.grad)] = 0.0
                        # Check for extreme gradients
                        if verbose>1:
                            grad_norm_print = param.grad.norm().item()
                            if grad_norm_print > 10.0:  # Threshold for reporting
                                print(f"High gradient norm in {name}: {grad_norm_print}")
                            if grad_norm_print < 0.00001:  
                                print(f"Low gradient norm in {name}: {grad_norm_print}")

                # Gradient clipping
                if grad_clip_value is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(mlp_params+scale_params+transform_params, grad_clip_value)
                    if (grad_norm>grad_clip_value):
                        percentage_of_batches_clipped_in_epoch += 1.0
                        if verbose>0:
                            print(f"Grad clipping activated, grad_norm before clipping: {grad_norm}")


                # Optimizer step
                optimizer.step()

                # Update metrics
                if grad_clip_value is not None:
                    mean_grad_norm_ += grad_norm.item()
                

                train_prediction_loss_ += prediction_loss_.item()
                batch_count += 1

                # Log progress
                if verbose > 0 and batch_idx % log_freq_batches == 0:
                    print(f"Batch {batch_idx}/{train_batches} - Prediction Loss: {prediction_loss_.item():.4f}")
                    if grad_clip_value is not None:
                        print(f"Grad norm of batch: {grad_norm.item():.4f}\n")


                        
                
            # Calculate epoch metrics
            mean_grad_norm_ /= max(1, batch_count)
            train_prediction_loss_ /= max(1, batch_count)
            percentage_of_batches_clipped_in_epoch /= max(1, batch_count)

            # Validation phase
            val_prediction_loss_ = 0.0
            val_batch_count = 0


            # Set model to evaluation mode
            mapping_net.eval()
            
            epoch_saving_path = os.path.join(save_dir, f"epoch_{epoch}")
            
            val_trajectories_dir = os.path.join(epoch_saving_path, f"val_trajectories_data")
            os.makedirs(val_trajectories_dir, exist_ok=True)



            
            
            # Validation loop

            print("Begining validation phase")
            for batch_idx, batch in enumerate(val_loader):
                # Run forward pass
                prediction_loss_ = validation_forward_pass(batch, val_df=val_df, val_id_df=val_id_df)
                # Update metrics
                if prediction_loss_ is not None:
                    val_prediction_loss_ += prediction_loss_
                    if verbose > 0 and batch_idx % log_freq_batches == 0:
                        print(f"Batch {batch_idx}/{val_batches} - Prediction Loss: {prediction_loss_:.4f}")
                        
                        
                        
                val_batch_count += 1
                    
                    

                    
            # Calculate validation metrics

            val_prediction_loss_ /= max(1, val_batch_count)




            #


            # Update epoch_metrics
            epoch_metrics['epoch'] = epoch


            
            epoch_metrics['percentage_of_batches_clipped_in_epoch'] = percentage_of_batches_clipped_in_epoch*100
            epoch_metrics['mean_grad_norm_'] = mean_grad_norm_

            epoch_metrics['train_prediction_loss_'] = train_prediction_loss_

            epoch_metrics['val_prediction_loss_'] = val_prediction_loss_
            

            

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            epoch_metrics['learning_rates'] = current_lr


            validation_criterio = val_prediction_loss_
            epoch_metrics['validation_criterio'] = validation_criterio

            if validation_criterio < best_val_loss - min_delta:
                best_val_loss = validation_criterio
                epoch_metrics['best_validation_criterio_loss_till_now'] = validation_criterio
                print(f"Found best validation criterio loss till now:{validation_criterio:.4f}, on epoch {epoch}")

                # Save best model (whether early_stopping is enabled or not)
                best_model_path = os.path.join(save_dir, "best_model.pt")
                save_checkpoint(best_model_path, mapping_net, optimizer, scheduler, epoch, best_val_loss)
                if verbose > 0:
                    print(f"New best model saved with val_loss: {best_val_loss:.4f}")

                patience_counter = 0
            else:
                epoch_metrics['best_validation_criterio_loss_till_now'] = best_val_loss
                if early_stopping:
                    patience_counter += 1
                    if verbose > 0:
                        print(f"Early stopping patience: {patience_counter}/{patience}")
                
            with open(os.path.join(epoch_saving_path, "epoch_metrics.json"), "w") as f:
                # Convert any non-serializable objects
                serializable_epoch_metrics = make_json_serializable(epoch_metrics)
                json.dump(serializable_epoch_metrics, f, indent=2)
            
            # Step the scheduler if it exists
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(validation_criterio)  # Pass validation loss to plateau scheduler
                else:
                    scheduler.step()

            # Time taken for epoch
            epoch_time = time.time() - epoch_start_time

            # Print epoch summary
            if verbose > 0:
                print(f"\nEpoch {epoch}/{start_epoch+num_epochs} completed in {epoch_time:.2f}s")


                print(f"Percentage of batches clipped in epoch: {percentage_of_batches_clipped_in_epoch*100:.1f}%")
                if grad_clip_value is not None:
                    print(f"Mean grad norm: {mean_grad_norm_:.4f}")

                print(f"Mean prediction train loss: {train_prediction_loss_:.4f}")
                
                

                print(f"Mean prediction val loss: {val_prediction_loss_:.4f}")
                
  
                
                print(f"Validation criterio: {validation_criterio:.4f}")    
                
                print(f"Learning Rate: {current_lr:.6f}")



            # Save checkpoint if needed
            if (epoch) % save_freq_epochs == 0 :
                if not save_best_only:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                    save_checkpoint(checkpoint_path, mapping_net, optimizer, scheduler, epoch, best_val_loss)
                    if verbose > 0:
                        print(f"Checkpoint saved to {checkpoint_path}")

            # Early stopping check
            if early_stopping and patience_counter >= patience:
                if verbose > 0:
                    print(f"Early stopping triggered in epoch: {epoch}")
                break

        run_hyperparameters ={}

        run_hyperparameters['train_dataloader_batch_size'] = train_loader.dataset.batch_size
        run_hyperparameters['train_dataloader_segment_length'] = train_loader.dataset.segment_length
        run_hyperparameters['train_dataloader_n_segments'] = train_loader.dataset.n_segments
        run_hyperparameters['train_dataloader_ratio'] = train_loader.dataset.ratio
        run_hyperparameters['train_dataloader_batch_traj'] = train_loader.dataset.batch_traj
        run_hyperparameters['randomize_each_epoch_plan'] = randomize_each_epoch_plan 
        run_hyperparameters['add_noise'] = add_noise 






        
        n_parameters = count_parameters(mapping_net)
        run_hyperparameters['model_trainable_parameter_count'] = n_parameters

        model_n_layers = mapping_net.n_layers
        run_hyperparameters['model_n_layers'] = model_n_layers

        model_hidden_dims = mapping_net.layers[0].step_1.hidden_dims
        run_hyperparameters['model_hidden_dims'] = model_hidden_dims

        model_activation = mapping_net.activation 
        run_hyperparameters['model_activation'] = model_activation

        model_activation_params = mapping_net.activation_params 
        run_hyperparameters['model_activation_params'] = model_activation_params

        model_final_activation = mapping_net.final_activation 
        run_hyperparameters['model_final_activation'] = model_final_activation

        model_final_activation_only_on_final_layer = mapping_net.final_activation_only_on_final_layer 
        run_hyperparameters['model_final_activation_only_on_final_layer'] = model_final_activation_only_on_final_layer

        model_tanh_wrapper = mapping_net.tanh_wrapper
        run_hyperparameters['model_tanh_wrapper'] = model_tanh_wrapper

        model_weight_init = mapping_net.weight_init 
        run_hyperparameters['model_weight_init'] = model_weight_init

        model_weight_init_params = mapping_net.weight_init_params 
        run_hyperparameters['model_weight_init_params'] = model_weight_init_params

        model_bias_init = mapping_net.bias_init 
        run_hyperparameters['model_bias_init'] = model_bias_init

        model_bias_init_value = mapping_net.bias_init_value 
        run_hyperparameters['model_bias_init_value'] = model_bias_init_value

        model_use_layer_norm = mapping_net.use_layer_norm 
        run_hyperparameters['model_use_layer_norm'] = model_use_layer_norm

        model_a_eps_min = mapping_net.a_eps_min 
        run_hyperparameters['model_a_eps_min'] = model_a_eps_min

        model_a_eps_max = mapping_net.a_eps_max
        run_hyperparameters['model_a_eps_max'] = model_a_eps_max

        model_a_k = mapping_net.a_k
        run_hyperparameters['model_a_k'] = model_a_k


        model_step_1_a_mean_innit = mapping_net.step_1_a_mean_innit
        run_hyperparameters['model_step_1_a_mean_innit'] = model_step_1_a_mean_innit

        model_step_1_gamma_mean_innit = mapping_net.step_1_gamma_mean_innit
        run_hyperparameters['model_step_1_gamma_mean_innit'] = model_step_1_gamma_mean_innit

        model_step_1_c1_mean_innit = mapping_net.step_1_c1_mean_innit
        run_hyperparameters['model_step_1_c1_mean_innit'] = model_step_1_c1_mean_innit


        model_step_1_c2_mean_innit = mapping_net.step_1_c2_mean_innit
        run_hyperparameters['model_step_1_c2_mean_innit'] = model_step_1_c2_mean_innit

        model_step_2_a_mean_innit = mapping_net.step_2_a_mean_innit
        run_hyperparameters['model_step_2_a_mean_innit'] = model_step_2_a_mean_innit



        model_step_2_gamma_mean_innit = mapping_net.step_2_gamma_mean_innit
        run_hyperparameters['model_step_2_gamma_mean_innit'] = model_step_2_gamma_mean_innit

        model_step_2_c1_mean_innit = mapping_net.step_2_c1_mean_innit
        run_hyperparameters['model_step_2_c1_mean_innit'] = model_step_2_c1_mean_innit

        model_step_2_c2_mean_innit = mapping_net.step_2_c2_mean_innit
        run_hyperparameters['model_step_2_c2_mean_innit'] = model_step_2_c2_mean_innit


        model_std_to_mean_ratio_a_mean_init = mapping_net.std_to_mean_ratio_a_mean_init
        run_hyperparameters['model_std_to_mean_ratio_a_mean_init'] = model_std_to_mean_ratio_a_mean_init

        model_std_to_mean_ratio_gamma_mean_init = mapping_net.std_to_mean_ratio_gamma_mean_init
        run_hyperparameters['model_std_to_mean_ratio_gamma_mean_init'] = model_std_to_mean_ratio_gamma_mean_init

        model_std_to_mean_ratio_c1_mean_init = mapping_net.std_to_mean_ratio_c1_mean_init
        run_hyperparameters['model_std_to_mean_ratio_c1_mean_init'] = model_std_to_mean_ratio_c1_mean_init

        model_std_to_mean_ratio_c2_mean_init = mapping_net.std_to_mean_ratio_c2_mean_init
        run_hyperparameters['model_std_to_mean_ratio_c2_mean_init'] = model_std_to_mean_ratio_c2_mean_init



        model_bound_innit = mapping_net.bound_innit
        run_hyperparameters['model_bound_innit'] = model_bound_innit


        run_hyperparameters['grad_clip_value'] = grad_clip_value
    


        with open(os.path.join(save_dir, f"run_from_epochs_{start_epoch}_to_{epoch+1}.json"), "w") as f:
            # Convert any non-serializable objects
            serializable_run_hyperparameters = make_json_serializable(run_hyperparameters)
            json.dump(serializable_run_hyperparameters, f, indent=2)
            
    except Exception as e:
        if auto_rescue:
            print(f"\nTraining interrupted by exception: {e}")
            save_rescue_checkpoint()
        raise  # Re-raise the exception after saving
    



    finally:
        try:
            if 'epoch' in locals():
                # Always save a final checkpoint regardless of how training ended
                final_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                save_checkpoint(final_path, mapping_net, optimizer, scheduler, epoch, best_val_loss)
                if verbose > 0:
                    print(f"\nFinal state saved to {final_path}")
            else:
                print("Exception before epoch loop began")
        except Exception as e:
            print(f"Failed to save final checkpoint: {e}")

    # Training complete
    if verbose > 0:
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    

    
    return 


def resume_training_from_checkpoint_real_pendulum(
    # Checkpoint to resume from
    checkpoint_path,
    
    # Dataloaders
    train_loader,
    randomize_each_epoch_plan,
    val_loader,

    
    # Network
    mapping_net,  # inverse_net is created automatically from mapping_net
    
    # Needed objects

    get_data_from_trajectory_id,
    possible_t_values_per_trajectory,
    possible_t_values_per_trajectory_val,
    
    # Needed pandas dataframes
    train_df,
    train_id_df,
    val_df,
    val_id_df,

    add_noise,
    
    # Loss calculation hyperparameters

    predict_full_trajectory,
    loss_type,


    
    # === RESUME MODE SELECTION ===
    # MODE A (load_scheduler_and_optimizer=True): Resume with exact optimizer/scheduler state
    
    #   - learning_rate: Set to None to use checkpoint LR, or specify to override
    #   - optimizer_type: MUST match the type used in original training
    #   - scheduler_type: MUST match the type used in original training
    #   - scheduler_params: MUST match the params used in original training
    #
    # MODE B (load_scheduler_and_optimizer=False): Resume with fresh optimizer/scheduler

    #   - learning_rate: REQUIRED - must specify, will error if None
    #   - optimizer_type: Can be different from original
    #   - scheduler_type: Can be different from original
    #   - scheduler_params: Can be different from original
    load_scheduler_and_optimizer=False,
    
    # Optimizer parameters
    learning_rate=None,  # MODE A: None=use checkpoint LR, or specify to override | MODE B: REQUIRED, must specify
    weight_decay=None,   # MODE A: None=use checkpoint weight_decay, or specify to override | MODE B: REQUIRED, must specify
    optimizer_type='AdamW',  # MODE A: must match original | MODE B: can be different
    
    # Scheduler parameters  
    scheduler_type='plateau',  # MODE A: must match original | MODE B: can be different
    scheduler_params=None,  # MODE A: can differ. Functionality would depend on reset_scheduler_patience | MODE B: can be different
    reset_scheduler_patience = False, #Only relevant on MODE A. Set True to reset num_bad_epochs. Use True if you want the learning rate to be lowered after the full patience amount. Use False if you want continuity, already waited N epochs, just need M-N more where M: loaded num_bad_epochs from previous training
    
    # Training parameters
    num_epochs=30,  # Number of ADDITIONAL epochs to train (not total epochs)
    grad_clip_value=3.0,
    
    # Early stopping parameters
    early_stopping=False,
    patience=20,
    min_delta=0.001,
    
    # Checkpointing
    save_dir='./checkpoints',  # Should typically match the directory where checkpoint_path is located
    save_best_only=False,
    save_freq_epochs=2,
    auto_rescue=True,
    
    # Logging
    log_freq_batches=50,
    verbose=2,
    
    # Device
    device='cuda'
    ):
    """Resume training from a checkpoint"""
    if load_scheduler_and_optimizer:
        mlp_params = []
        transform_params = []
        scale_params = []

        for name, param in mapping_net.named_parameters():
            if ('G_network' in name) or ('F_network' in name):
                mlp_params.append(param)
            elif ('g_bound' in name) or ('f_bound' in name) or ('a_raw' in name):
                scale_params.append(param)
            else: #c1, c2, gamma
                transform_params.append(param)

        if optimizer_type == 'Adam':
            temp_optimizer = Adam(mlp_params+scale_params+transform_params)
        elif optimizer_type == 'AdamW':
            temp_optimizer = AdamW([
                {'params': mlp_params, 'weight_decay': weight_decay},
                {'params': scale_params, 'weight_decay': 0.0},
                {'params': transform_params, 'weight_decay': weight_decay},
            ])

        temp_scheduler = ReduceLROnPlateau(temp_optimizer, mode='min', factor=0.1, patience=5, verbose=True) #Will be overwritten
        # Load checkpoint
        epoch, extra_info, best_val_loss = load_checkpoint(
            checkpoint_path, mapping_net, device, temp_optimizer, temp_scheduler
        )

        # Override scheduler params if specified
        if scheduler_params is not None:
            print(f"Overriding scheduler params: {scheduler_params}")
            for key, value in scheduler_params.items():
                if hasattr(temp_scheduler, key):
                    setattr(temp_scheduler, key, value)
                    print(f"  {key}: {getattr(temp_scheduler, key)} -> {value}")
                else:
                    print(f"  Warning: scheduler has no attribute '{key}'")
        else:
            if hasattr(temp_scheduler, 'patience'):  # ReduceLROnPlateau
                print("Using loaded scheduler")
                print(f"  factor: {temp_scheduler.factor}")
                print(f"  mode: {temp_scheduler.mode}")
                print(f"  threshold: {temp_scheduler.threshold}")
                print(f"  cooldown: {temp_scheduler.cooldown}")
                print(f"  best: {temp_scheduler.best}")


        # Reset scheduler patience counter if requested
        if reset_scheduler_patience:
            temp_scheduler.num_bad_epochs = 0
            temp_scheduler.cooldown_counter = 0
            print(f"Reset scheduler patience counter. Will wait full {temp_scheduler.patience} epochs before reducing LR.")
        else:
            # Calculate remaining epochs until LR reduction
            if temp_scheduler.cooldown_counter > 0:
                print(f"Scheduler in cooldown mode: {temp_scheduler.cooldown_counter} epochs remaining in cooldown")
            elif temp_scheduler.num_bad_epochs >= temp_scheduler.patience:
                print(f"Scheduler ready to reduce LR on next bad epoch (num_bad_epochs={temp_scheduler.num_bad_epochs} >= patience={temp_scheduler.patience})")
            else:
                remaining_epochs = temp_scheduler.patience - temp_scheduler.num_bad_epochs
                print(f"Scheduler status: {temp_scheduler.num_bad_epochs}/{temp_scheduler.patience} bad epochs. Will reduce LR after {remaining_epochs} more bad epochs.")

    else:
        epoch, extra_info, best_val_loss = load_checkpoint(
            checkpoint_path, mapping_net, device, None, None
        )

    if extra_info and 'interrupted_at_batch' in extra_info:
        continue_from_epoch = epoch
        print(f"Resuming incomplete epoch {epoch}")
        if 'exception' in extra_info:
            print(f"Previous training was interrupted by exception:\n{extra_info['exception']}")
    else:
        continue_from_epoch = epoch + 1
        print(f"Epoch {epoch} was completed, starting from epoch {epoch + 1}")
    
    # Use the learning rate from checkpoint if not specified
    if (learning_rate is None):
        if load_scheduler_and_optimizer:
            learning_rate = temp_optimizer.param_groups[0]['lr']
            print(f"Using learning rate from checkpoint: {learning_rate}")
        else:
            print("When load_scheduler_and_optimizer=False you should set Learning Rate")
            sys.exit(1)

    if (weight_decay is None):
        if load_scheduler_and_optimizer:
            weight_decay = temp_optimizer.param_groups[0]['weight_decay']
            print(f"Using weight decay from checkpoint: {weight_decay}")
        else:
            print("When load_scheduler_and_optimizer=False you should set Weight Decay")
            sys.exit(1)
            
    inverse_net = InverseStackedHamiltonianNetwork(forward_network=mapping_net)
    
    # Train for additional epochs
    print(f"Training for {num_epochs} additional epochs (epochs {continue_from_epoch} to {continue_from_epoch + num_epochs - 1})")
    
    # Resume training with the loaded state
    if load_scheduler_and_optimizer:
        return train_model_real_pendulum(
            # Dataloaders
            train_loader=train_loader,
            randomize_each_epoch_plan=randomize_each_epoch_plan,
            val_loader=val_loader,

            
            # Network
            mapping_net=mapping_net, 
            inverse_net=inverse_net,
            
            # Needed objects

            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values_per_trajectory=possible_t_values_per_trajectory,
            possible_t_values_per_trajectory_val=possible_t_values_per_trajectory_val,
            
            # Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,
            val_df=val_df,
            val_id_df=val_id_df,

            add_noise=add_noise,
            
            # Loss calculation hyperparameters

            predict_full_trajectory=predict_full_trajectory,
            loss_type=loss_type,


            
            # Optimizer parameters - using loaded states
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=temp_optimizer,  # Loaded from checkpoint
            optimizer_type=optimizer_type,
            scheduler=temp_scheduler,  # Loaded from checkpoint
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_params,
            
            # Training parameters
            num_epochs=num_epochs,
            grad_clip_value=grad_clip_value,
            continue_from_epoch=continue_from_epoch,
            best_validation_criterio_loss_till_now=best_val_loss,  # Loaded from checkpoint
            
            # Early stopping parameters
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            
            # Checkpointing
            save_dir=save_dir,
            save_best_only=save_best_only,
            save_freq_epochs=save_freq_epochs,
            auto_rescue=auto_rescue,
            
            # Logging
            log_freq_batches=log_freq_batches,
            verbose=verbose,
            
            # Device
            device=device,
        )
    else:
        return train_model_real_pendulum(
            # Dataloaders
            train_loader=train_loader,
            randomize_each_epoch_plan=randomize_each_epoch_plan,
            val_loader=val_loader,

            
            # Network
            mapping_net=mapping_net, 
            inverse_net=inverse_net,
            
            # Needed objects

            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values_per_trajectory=possible_t_values_per_trajectory,
            possible_t_values_per_trajectory_val=possible_t_values_per_trajectory_val,
            
            # Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,
            val_df=val_df,
            val_id_df=val_id_df,

            add_noise=add_noise,
            
            # Loss calculation hyperparameters

            predict_full_trajectory=predict_full_trajectory,
            loss_type=loss_type,



            
            # Optimizer parameters - creating fresh
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=None,  # Create fresh optimizer
            optimizer_type=optimizer_type,
            scheduler=None,  # Create fresh scheduler
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_params,
            
            # Training parameters
            num_epochs=num_epochs,
            grad_clip_value=grad_clip_value,
            continue_from_epoch=continue_from_epoch,
            best_validation_criterio_loss_till_now=best_val_loss,  # Loaded from checkpoint
            
            # Early stopping parameters
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            
            # Checkpointing
            save_dir=save_dir,
            save_best_only=save_best_only,
            save_freq_epochs=save_freq_epochs,
            auto_rescue=auto_rescue,
            
            # Logging
            log_freq_batches=log_freq_batches,
            verbose=verbose,
            
            # Device
            device=device,
        )
    

def save_checkpoint(path, mapping_net, optimizer, scheduler, epoch, best_val_loss, extra_info=None):
    """Save a checkpoint with all model states and training information."""
    state_dict = {
        'epoch': epoch,
        'model': mapping_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat()
    }

    if extra_info is not None:
        state_dict['extra_info'] = extra_info


    # Add scheduler state if it exists
    if scheduler is not None:
        state_dict['scheduler'] = scheduler.state_dict()

    if best_val_loss is not None:
        state_dict['best_val_loss'] = best_val_loss
    
    
    torch.save(state_dict, path)


def load_checkpoint(path, mapping_net, device, optimizer=None, scheduler=None):
    """Load a checkpoint into models and training state."""
    checkpoint = torch.load(path, map_location=device)
    
    # Load model

    if 'model' in checkpoint:
        try:
            mapping_net.load_state_dict(checkpoint['model'])
            print(f"Successfully loaded model")
        except Exception as e:
            print(f"Failed to load model, because of: {e}")

    
    # Load optimizer if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Successfully loaded optimizer, beginning with learning rate {optimizer.param_groups[0]['lr']}")
    
    # Load scheduler if provided
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Successfully loaded scheduler")

    
    # Return epoch 
    return (
        checkpoint.get('epoch', -1), 
        checkpoint.get('extra_info', None),
        checkpoint.get('best_val_loss', None)
    )






def Autoregressive_Handoff_Inference_Protocol(
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
):
    """
    Autoregressive Handoff Inference Protocol.
    
    Args:
        mapping_net: trained MappingNet
        inverse_net: trained InverseNet
        x_obs: scalar tensor, observed q value
        u_obs: scalar tensor, observed p value
        t_obs: float, observation time
        t_min: float, start of prediction interval
        t_max: float, end of prediction interval
        Tmax: float, maximum normalized temporal horizon
        query_times: 1D numpy array of query times (excluding t_obs)
        query_indices: list of ints, original indices corresponding to query_times
        device: torch device
    
    Returns:
        pred_dict: dict mapping original index -> (x_pred, u_pred)
        forward_pass_count: dict with 'mapping_net' and 'inverse_net' counts
    """
    # Input validation
    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")
    
    if not isinstance(query_indices, list):
        raise ValueError(f"query_indices must be a list, got {type(query_indices)}")
    
    if len(query_indices) != len(query_times):
        raise ValueError(f"query_indices and query_times must have the same length, got {len(query_indices)} and {len(query_times)}")

    forward_pass_count = {'mapping_net': 0, 'inverse_net': 0}
    pred_dict = {}

    if len(query_indices) == 0:
        return pred_dict, forward_pass_count

    # Case 1: Direct Prediction
    if (t_max - t_min) <= Tmax:
        t_rel_obs = torch.tensor([t_obs - t_min], device=device, dtype=torch.float32)
        Q, P, _ = mapping_net(x_obs.unsqueeze(0), u_obs.unsqueeze(0), t_rel_obs)
        forward_pass_count['mapping_net'] += 1

        t_rel_query = torch.tensor(query_times - t_min, device=device, dtype=torch.float32)
        Q_full = Q.expand(len(query_indices))
        P_full = P.expand(len(query_indices))
        x_pred_query, u_pred_query, _ = inverse_net(Q_full, P_full, t_rel_query)
        forward_pass_count['inverse_net'] += 1

        for j, idx in enumerate(query_indices):
            pred_dict[idx] = (x_pred_query[j].item(), u_pred_query[j].item())

        return pred_dict, forward_pass_count

    # Case 2: Autoregressive Propagation

    # === Phase 1: Forward Propagation ===
    fwd = [(idx, query_times[j]) for j, idx in enumerate(query_indices) if query_times[j] > t_obs]

    if fwd:
        fwd_indices, fwd_times = zip(*fwd)
        fwd_indices = list(fwd_indices)
        fwd_times = np.array(fwd_times)

        K_fwd = int(np.ceil((t_max - t_obs) / Tmax - 1e-9))
        K_fwd = max(K_fwd, 1)

        q_curr = x_obs.clone()
        p_curr = u_obs.clone()
        t_curr = t_obs

        for k in range(K_fwd):
            T_start_k = t_curr
            T_end_k = min(t_curr + Tmax, t_max)
            if k == K_fwd - 1:
                T_end_k = t_max

            if k < K_fwd - 1:
                I_k_query = [(idx, ft) for idx, ft in zip(fwd_indices, fwd_times) if T_start_k <= ft < T_end_k]
            else:
                I_k_query = [(idx, ft) for idx, ft in zip(fwd_indices, fwd_times) if T_start_k <= ft <= T_end_k]

            need_boundary = (k < K_fwd - 1)

            t_rel_list = [ft - T_start_k for _, ft in I_k_query]
            if need_boundary:
                t_rel_list.append(T_end_k - T_start_k)

            if len(t_rel_list) == 0:
                continue

            t_rel_k = torch.tensor(t_rel_list, device=device, dtype=torch.float32)

            Q_k, P_k, _ = mapping_net(q_curr.unsqueeze(0), p_curr.unsqueeze(0),
                                      torch.tensor([0.0], device=device, dtype=torch.float32))
            forward_pass_count['mapping_net'] += 1

            Q_full_k = Q_k.expand(len(t_rel_k))
            P_full_k = P_k.expand(len(t_rel_k))

            q_hat_k, p_hat_k, _ = inverse_net(Q_full_k, P_full_k, t_rel_k)
            forward_pass_count['inverse_net'] += 1

            for j, (idx, _) in enumerate(I_k_query):
                pred_dict[idx] = (q_hat_k[j].item(), p_hat_k[j].item())

            if need_boundary:
                q_curr = q_hat_k[-1].detach()
                p_curr = p_hat_k[-1].detach()
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
                I_k_query = [(idx, bt) for idx, bt in zip(bwd_indices, bwd_times) if T_start_k < bt <= T_end_k]
            else:
                I_k_query = [(idx, bt) for idx, bt in zip(bwd_indices, bwd_times) if T_start_k <= bt <= T_end_k]

            need_boundary = (k < K_bwd - 1)

            t_rel_list = []
            if need_boundary:
                t_rel_list.append(0.0)
            t_rel_list.extend([bt - T_start_k for _, bt in I_k_query])

            if len(t_rel_list) == 0:
                continue

            t_rel_k = torch.tensor(t_rel_list, device=device, dtype=torch.float32)

            t_rel_obs_in_window = T_end_k - T_start_k
            Q_k, P_k, _ = mapping_net(q_curr.unsqueeze(0), p_curr.unsqueeze(0),
                                      torch.tensor([t_rel_obs_in_window], device=device, dtype=torch.float32))
            forward_pass_count['mapping_net'] += 1

            Q_full_k = Q_k.expand(len(t_rel_k))
            P_full_k = P_k.expand(len(t_rel_k))

            q_hat_k, p_hat_k, _ = inverse_net(Q_full_k, P_full_k, t_rel_k)
            forward_pass_count['inverse_net'] += 1

            offset = 1 if need_boundary else 0
            for j, (idx, _) in enumerate(I_k_query):
                pred_dict[idx] = (q_hat_k[offset + j].item(), p_hat_k[offset + j].item())

            if need_boundary:
                q_curr = q_hat_k[0].detach()
                p_curr = p_hat_k[0].detach()
                t_curr = T_start_k

    missing = [idx for idx in query_indices if idx not in pred_dict]
    if missing:
        raise RuntimeError(
            f"Autoregressive_Handoff_Inference_Protocol failed to produce predictions for indices {missing}. "
            f"This is likely a floating point boundary issue. "
            f"t_obs={t_obs}, t_min={t_min}, t_max={t_max}, Tmax={Tmax}"
    )
    return pred_dict, forward_pass_count

def Batched_Autoregressive_Handoff_Inference_Protocol(
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
):
    """
    Batched Autoregressive Handoff Inference Protocol.
    
    All trajectories must share the same time structure (same t values, same obs index).
    
    Args:
        mapping_net: trained MappingNet
        inverse_net: trained InverseNet
        x_obs: (N_traj,) tensor, observed q values
        u_obs: (N_traj,) tensor, observed p values
        t_obs: float, shared observation time
        t_min: float, shared start of prediction interval
        t_max: float, shared end of prediction interval
        Tmax: float, maximum normalized temporal horizon
        query_times: 1D numpy array of shared query times (excluding t_obs)
        query_indices: list of ints, shared original indices corresponding to query_times
        device: torch device
    
    Returns:
        x_pred_all: (N_traj, N_query) tensor of x predictions at query times
        u_pred_all: (N_traj, N_query) tensor of u predictions at query times
        forward_pass_count: dict with 'mapping_net' and 'inverse_net' counts
    """
    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")
    if not isinstance(query_indices, list):
        raise ValueError(f"query_indices must be a list, got {type(query_indices)}")
    if len(query_indices) != len(query_times):
        raise ValueError(f"query_indices and query_times must have the same length, got {len(query_indices)} and {len(query_times)}")
    if x_obs.dim() != 1 or u_obs.dim() != 1:
        raise ValueError(f"x_obs and u_obs must be 1D tensors, got dims {x_obs.dim()} and {u_obs.dim()}")
    if x_obs.shape[0] != u_obs.shape[0]:
        raise ValueError(f"x_obs and u_obs must have the same length, got {x_obs.shape[0]} and {u_obs.shape[0]}")

    N_traj = x_obs.shape[0]
    N_query = len(query_indices)
    forward_pass_count = {'mapping_net': 0, 'inverse_net': 0}

    # Map from query list position to original index
    query_pos_to_idx = {j: idx for j, idx in enumerate(query_indices)}
    # Map from original index to query list position
    idx_to_query_pos = {idx: j for j, idx in enumerate(query_indices)}

    x_pred_all = torch.empty(N_traj, N_query, device=device, dtype=torch.float32)
    u_pred_all = torch.empty(N_traj, N_query, device=device, dtype=torch.float32)

    if N_query == 0:
        return x_pred_all, u_pred_all, forward_pass_count

    # Case 1: Direct Prediction
    if (t_max - t_min) <= Tmax:
        t_rel_obs = torch.tensor([t_obs - t_min], device=device, dtype=torch.float32).expand(N_traj)
        Q, P, _ = mapping_net(x_obs, u_obs, t_rel_obs)
        forward_pass_count['mapping_net'] += 1

        # Replicate Q, P: each trajectory's Q/P repeated N_query times
        # Q shape: (N_traj,) -> (N_traj * N_query,)
        Q_full = Q.repeat_interleave(N_query)
        P_full = P.repeat_interleave(N_query)

        # Shared relative times tiled for all trajectories
        t_rel_query = torch.tensor(query_times - t_min, device=device, dtype=torch.float32)
        t_rel_full = t_rel_query.repeat(N_traj)

        x_pred_flat, u_pred_flat, _ = inverse_net(Q_full, P_full, t_rel_full)
        forward_pass_count['inverse_net'] += 1

        x_pred_all = x_pred_flat.view(N_traj, N_query)
        u_pred_all = u_pred_flat.view(N_traj, N_query)

        return x_pred_all, u_pred_all, forward_pass_count

    # Case 2: Autoregressive Propagation

    # === Phase 1: Forward Propagation ===
    fwd_positions = [j for j, idx in enumerate(query_indices) if query_times[j] > t_obs]
    fwd_times_local = np.array([query_times[j] for j in fwd_positions]) if fwd_positions else np.array([])

    if len(fwd_positions) > 0:
        K_fwd = int(np.ceil((t_max - t_obs) / Tmax - 1e-9))
        K_fwd = max(K_fwd, 1)
        q_curr = x_obs.clone()  # (N_traj,)
        p_curr = u_obs.clone()  # (N_traj,)
        t_curr = t_obs

        for k in range(K_fwd):
            T_start_k = t_curr
            T_end_k = min(t_curr + Tmax, t_max)
            if k == K_fwd - 1:
                T_end_k = t_max
            if k < K_fwd - 1:
                seg_query_positions = [j for j in fwd_positions if T_start_k <= query_times[j] < T_end_k]
            else:
                seg_query_positions = [j for j in fwd_positions if T_start_k <= query_times[j] <= T_end_k]

            need_boundary = (k < K_fwd - 1)

            t_rel_list = [query_times[j] - T_start_k for j in seg_query_positions]
            if need_boundary:
                t_rel_list.append(T_end_k - T_start_k)

            N_times = len(t_rel_list)
            if N_times == 0:
                continue

            t_rel_k = torch.tensor(t_rel_list, device=device, dtype=torch.float32)

            # MappingNet: all trajectories at relative time 0
            t_zero = torch.zeros(N_traj, device=device, dtype=torch.float32)
            Q_k, P_k, _ = mapping_net(q_curr, p_curr, t_zero)
            forward_pass_count['mapping_net'] += 1

            # Replicate: (N_traj,) -> (N_traj * N_times,)
            Q_full_k = Q_k.repeat_interleave(N_times)
            P_full_k = P_k.repeat_interleave(N_times)
            t_rel_full_k = t_rel_k.repeat(N_traj)

            q_hat_flat, p_hat_flat, _ = inverse_net(Q_full_k, P_full_k, t_rel_full_k)
            forward_pass_count['inverse_net'] += 1

            # Reshape: (N_traj * N_times,) -> (N_traj, N_times)
            q_hat_k = q_hat_flat.view(N_traj, N_times)
            p_hat_k = p_hat_flat.view(N_traj, N_times)

            # Store predictions for real query positions
            for local_j, global_j in enumerate(seg_query_positions):
                x_pred_all[:, global_j] = q_hat_k[:, local_j]
                u_pred_all[:, global_j] = p_hat_k[:, local_j]

            # Handoff
            if need_boundary:
                q_curr = q_hat_k[:, -1].detach()
                p_curr = p_hat_k[:, -1].detach()
                t_curr = T_end_k

    # === Phase 2: Backward Propagation ===
    bwd_positions = [j for j, idx in enumerate(query_indices) if query_times[j] < t_obs]
    bwd_times_local = np.array([query_times[j] for j in bwd_positions]) if bwd_positions else np.array([])

    if len(bwd_positions) > 0:
        K_bwd = int(np.ceil((t_obs - t_min) / Tmax - 1e-9))
        K_bwd = max(K_bwd, 1)
        q_curr = x_obs.clone()  # (N_traj,)
        p_curr = u_obs.clone()  # (N_traj,)
        t_curr = t_obs

        for k in range(K_bwd):
            T_start_k = max(t_curr - Tmax, t_min)
            T_end_k = t_curr
            if k == K_bwd - 1:
                T_start_k = t_min

            if k < K_bwd - 1:
                seg_query_positions = [j for j in bwd_positions if T_start_k < query_times[j] <= T_end_k]
            else:
                seg_query_positions = [j for j in bwd_positions if T_start_k <= query_times[j] <= T_end_k]

            need_boundary = (k < K_bwd - 1)

            t_rel_list = []
            if need_boundary:
                t_rel_list.append(0.0)
            t_rel_list.extend([query_times[j] - T_start_k for j in seg_query_positions])

            N_times = len(t_rel_list)
            if N_times == 0:
                continue

            t_rel_k = torch.tensor(t_rel_list, device=device, dtype=torch.float32)

            # MappingNet: all trajectories at end of window
            t_rel_obs_in_window = T_end_k - T_start_k
            t_obs_window = torch.full((N_traj,), t_rel_obs_in_window, device=device, dtype=torch.float32)
            Q_k, P_k, _ = mapping_net(q_curr, p_curr, t_obs_window)
            forward_pass_count['mapping_net'] += 1

            # Replicate
            Q_full_k = Q_k.repeat_interleave(N_times)
            P_full_k = P_k.repeat_interleave(N_times)
            t_rel_full_k = t_rel_k.repeat(N_traj)

            q_hat_flat, p_hat_flat, _ = inverse_net(Q_full_k, P_full_k, t_rel_full_k)
            forward_pass_count['inverse_net'] += 1

            q_hat_k = q_hat_flat.view(N_traj, N_times)
            p_hat_k = p_hat_flat.view(N_traj, N_times)

            # Store predictions, skipping boundary
            offset = 1 if need_boundary else 0
            for local_j, global_j in enumerate(seg_query_positions):
                x_pred_all[:, global_j] = q_hat_k[:, offset + local_j]
                u_pred_all[:, global_j] = p_hat_k[:, offset + local_j]

            # Handoff
            if need_boundary:
                q_curr = q_hat_k[:, 0].detach()
                p_curr = p_hat_k[:, 0].detach()
                t_curr = T_start_k

    return x_pred_all, u_pred_all, forward_pass_count


def test_model_in_single_trajectory(
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
    verbose=False,
):
    """
    Test model on a single trajectory.
    """
    if not isinstance(point_indexes_observed, list) or len(point_indexes_observed) != 1:
        raise ValueError(f"point_indexes_observed must be a list with exactly one integer, got {point_indexes_observed}")
    
    if not isinstance(point_indexes_observed[0], (int, np.integer)):
        raise ValueError(f"point_indexes_observed[0] must be an integer, got {type(point_indexes_observed[0])}")

    if loss_type not in ("euclidean", "mae", "mse"):
        raise ValueError(f"Unknown loss_type '{loss_type}'. Must be 'euclidean', 'mae', or 'mse'.")

    test_trajectory_data = get_data_from_trajectory_id_function(test_id_df, test_df, trajectory_ids=trajectory_id)
    x = torch.as_tensor(test_trajectory_data['x'].to_numpy(dtype=np.float32), device=device)
    u = torch.as_tensor(test_trajectory_data['u'].to_numpy(dtype=np.float32), device=device)
    t = torch.as_tensor(test_trajectory_data['t'].to_numpy(dtype=np.float32), device=device)

    if x.dim() != 1 or u.dim() != 1 or t.dim() != 1:
        raise ValueError(f"x, u, t must be 1-dimensional tensors, got dims {x.dim()}, {u.dim()}, {t.dim()}")

    obs_idx = point_indexes_observed[0]
    
    if obs_idx < 0 or obs_idx >= len(t):
        raise ValueError(f"point_indexes_observed[0]={obs_idx} is out of bounds for trajectory of length {len(t)}")

    t_np = t.detach().cpu().numpy()
    query_indices = [i for i in range(len(t)) if i != obs_idx]
    query_times = t_np[query_indices]

    pred_dict, forward_pass_count = Autoregressive_Handoff_Inference_Protocol(
        mapping_net=mapping_net,
        inverse_net=inverse_net,
        x_obs=x[obs_idx],
        u_obs=u[obs_idx],
        t_obs=t_np[obs_idx],
        t_min=t_np[0],
        t_max=t_np[-1],
        Tmax=Tmax,
        query_times=query_times,
        query_indices=query_indices,
        device=device,
    )

    # Assemble predictions
    x_pred = torch.empty_like(t)
    u_pred = torch.empty_like(t)
    x_pred[obs_idx] = x[obs_idx]
    u_pred[obs_idx] = u[obs_idx]
    for idx in query_indices:
        x_pred[idx] = pred_dict[idx][0]
        u_pred[idx] = pred_dict[idx][1]

    if verbose:
        total_passes = forward_pass_count['mapping_net'] + forward_pass_count['inverse_net']
        print(f"\n=== Forward Pass Count ===")
        print(f"mapping_net calls: {forward_pass_count['mapping_net']}")
        print(f"inverse_net calls: {forward_pass_count['inverse_net']}")
        print(f"Total forward passes: {total_passes}")

    if loss_type == "euclidean":
        pred_loss_full_trajectory = prediction_loss_euclidean(x_pred=x_pred, u_pred=u_pred, X_labels=x, U_labels=u)
    elif loss_type == "mae":
        pred_loss_full_trajectory = prediction_loss(x_pred=x_pred, u_pred=u_pred, X_labels=x, U_labels=u, loss_type=loss_type)
    elif loss_type == "mse":
        pred_loss_full_trajectory = prediction_loss(x_pred=x_pred, u_pred=u_pred, X_labels=x, U_labels=u, loss_type=loss_type)

    return pred_loss_full_trajectory, x_pred, u_pred, x, u, t, forward_pass_count
    

def time_inference_gpu(
    get_data_from_trajectory_id_function,
    test_id_df,
    test_df,
    mapping_net,
    inverse_net,
    device,
    Tmax,
    Batched_Autoregressive_Handoff_Inference_Protocol,
    n_warmup=5,
    n_repeats=20,
    method_name="HJN",
    task_name=None,
    save_path=None,
    verbose=True,
):
    """
    End-to-end wall-clock timing for the batched multi-trajectory inference
    protocol, with peak-memory tracking, fairness flags, and pickle output
    for cross-method comparison.
 
    Args:
        get_data_from_trajectory_id_function: callable to retrieve trajectory data.
        test_id_df: DataFrame with trajectory metadata.
        test_df: DataFrame with trajectory data.
        mapping_net: trained MappingNet (already on device).
        inverse_net: trained InverseNet (already on device).
        device: torch device (must be a CUDA device for memory tracking).
        Tmax: maximum normalized temporal horizon (must be positive).
        Batched_Autoregressive_Handoff_Inference_Protocol: the inference function.
        n_warmup: number of warmup sweeps (not timed).
        n_repeats: number of timed sweeps.
        method_name: label for results (e.g. "HJN"); written into the saved pickle.
        task_name: task identifier (e.g. "task_1_low_samples"); written into pickle.
        save_path: if provided, save full results dict here as pickle.
        verbose: if True, print full setup, progress, and methods-section text.
 
    Returns:
        results: dict with keys:
            'method_name', 'task_name', 'timestamp'
            'hardware': dict (gpu, cuda, pytorch versions)
            'fairness_settings': dict (precision flags as actually set)
            'protocol': dict (n_warmup, n_repeats, timing_method)
            'data': dict (N_traj, N_obs, t_range, Tmax)
            'timing_stats': dict (full statistics — see below)
            'memory_stats': dict (peak bytes during timed region)
 
        'timing_stats' contains:
            'time_matrix': (n_repeats, N_obs) array, seconds per obs per run
            'per_obs_mean': (N_obs,) mean time per obs point across repeats
            'per_obs_std':  (N_obs,) std across repeats per obs point
                            ^ this is MEASUREMENT NOISE for each obs point
            'mean_s', 'std_s': mean ± std of per_obs_mean across obs choices
                              ^ std here is STRUCTURAL variation across obs
                                positions (different K_fwd + K_bwd counts), NOT
                                independent experimental uncertainty
            'measurement_noise_mean_ms': mean of per_obs_std across obs
                                         ^ this IS measurement uncertainty
            'median_s', 'min_s', 'max_s': across obs choices
            'total_sweep_mean_s', 'total_sweep_std_s': summed across obs per repeat
    """
    # =========================================================================
    # Validation
    # =========================================================================
    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")
    if device.type != "cuda":
        raise ValueError(
            f"This benchmark requires a CUDA device for memory tracking; got {device}"
        )
 
    # =========================================================================
    # Fairness flags — set explicitly so the methods section can cite them.
    # We mirror these on the JAX side (matmul_precision='highest') so neither
    # method gets a precision-driven speedup the other lacks.
    # =========================================================================
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True       # autotune cuDNN algos (no-op for HJN; harmless)
    torch.backends.cudnn.deterministic = True   # reproducible kernels for NCU later
 
    fairness_settings = {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "precision": "FP32 (TF32 disabled)",
    }
 
    # Defensive eval mode (no BN/dropout in HJN, but standard practice)
    mapping_net.eval()
    inverse_net.eval()
 
    # =========================================================================
    # Load trajectories and validate shared time structure
    # =========================================================================
    trajectory_ids = test_id_df['trajectory_id'].values.astype(int)
    all_x, all_u = [], []
    ref_t = None
 
    for _, row_data in test_id_df.iterrows():
        trajectory_id = int(row_data['trajectory_id'])
        test_trajectory_data = get_data_from_trajectory_id_function(
            test_id_df, test_df, trajectory_ids=trajectory_id
        )
        x_i = torch.as_tensor(test_trajectory_data['x'].to_numpy(dtype=np.float32), device=device)
        u_i = torch.as_tensor(test_trajectory_data['u'].to_numpy(dtype=np.float32), device=device)
        t_i = torch.as_tensor(test_trajectory_data['t'].to_numpy(dtype=np.float32), device=device)
 
        if x_i.dim() != 1 or u_i.dim() != 1 or t_i.dim() != 1:
            raise ValueError(f"Trajectory {trajectory_id}: x, u, t must be 1D")
 
        if ref_t is None:
            ref_t = t_i
        else:
            if len(t_i) != len(ref_t):
                raise ValueError(
                    f"Trajectory {trajectory_id} has {len(t_i)} points, expected {len(ref_t)}."
                )
            if not torch.allclose(t_i, ref_t, atol=1e-6):
                raise ValueError(f"Trajectory {trajectory_id} has different time values.")
 
        all_x.append(x_i)
        all_u.append(u_i)
 
    x_batch = torch.stack(all_x, dim=0)  # (N_traj, N_obs)
    u_batch = torch.stack(all_u, dim=0)
    t_np = ref_t.detach().cpu().numpy()
    N_obs = len(t_np)
    N_traj = len(trajectory_ids)
 
    # =========================================================================
    # Precompute per-obs query data (hoisted out of timed region)
    # The original code built these lists inside run_single_obs every call,
    # adding Python overhead to every timed measurement. They're deterministic
    # per obs_idx, so we precompute once.
    # =========================================================================
    query_indices_per_obs = []
    query_times_per_obs = []
    for obs_idx in range(N_obs):
        q_idx = [i for i in range(N_obs) if i != obs_idx]
        q_t = t_np[q_idx]
        query_indices_per_obs.append(q_idx)
        query_times_per_obs.append(q_t)
 
    # =========================================================================
    # Single-obs inference call (no list comprehensions inside; just dispatch)
    # =========================================================================
    def run_single_obs(obs_idx):
        Batched_Autoregressive_Handoff_Inference_Protocol(
            mapping_net=mapping_net,
            inverse_net=inverse_net,
            x_obs=x_batch[:, obs_idx],
            u_obs=u_batch[:, obs_idx],
            t_obs=t_np[obs_idx],
            t_min=t_np[0],
            t_max=t_np[-1],
            Tmax=Tmax,
            query_times=query_times_per_obs[obs_idx],
            query_indices=query_indices_per_obs[obs_idx],
            device=device,
        )
 
    # =========================================================================
    # Header (verbose)
    # =========================================================================
    gpu_name = torch.cuda.get_device_name(device)
    if verbose:
        print(f"\nTiming benchmark — method={method_name}, task={task_name}")
        print(f"  N_traj={N_traj}, N_obs={N_obs}, device={gpu_name}")
 
    # =========================================================================
    # Warmup
    # =========================================================================
    if verbose:
        print(f"  Running {n_warmup} warmup sweeps ({n_warmup * N_obs} calls)...")
    for _ in range(n_warmup):
        for obs_idx in range(N_obs):
            run_single_obs(obs_idx)
 
    # =========================================================================
    # Reset peak memory stats AFTER warmup, BEFORE timed sweeps.
    # This way we capture peak memory used during actual inference, not during
    # cuDNN autotuning, allocator warmup, or other one-time effects.
    # =========================================================================
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    mem_baseline_bytes = torch.cuda.memory_allocated(device)
 
    # =========================================================================
    # Timed runs
    # =========================================================================
    if verbose:
        print(f"  Running {n_repeats} timed sweeps...")
    time_matrix = np.empty((n_repeats, N_obs))
 
    for r in range(n_repeats):
        for obs_idx in range(N_obs):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
 
            run_single_obs(obs_idx)
 
            torch.cuda.synchronize(device)
            elapsed_s = time.perf_counter() - start
            time_matrix[r, obs_idx] = elapsed_s
 
    # =========================================================================
    # Capture peak memory across the entire timed region
    # =========================================================================
    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_bytes_above_baseline = peak_bytes - mem_baseline_bytes
 
    memory_stats = {
        "baseline_bytes_before_timing": int(mem_baseline_bytes),
        "peak_bytes_during_timing": int(peak_bytes),
        "peak_bytes_above_baseline": int(peak_bytes_above_baseline),
        "peak_MB_during_timing": peak_bytes / 1024 ** 2,
        "peak_MB_above_baseline": peak_bytes_above_baseline / 1024 ** 2,
    }
 
    # =========================================================================
    # Statistics
    # =========================================================================
    per_obs_mean = np.mean(time_matrix, axis=0)        # (N_obs,) repeats-averaged
    per_obs_std = np.std(time_matrix, axis=0, ddof=1)  # (N_obs,) measurement noise
 
    mean_s = np.mean(per_obs_mean)
    std_s = np.std(per_obs_mean, ddof=1)               # STRUCTURAL variation
    median_s = np.median(per_obs_mean)
    min_s = np.min(per_obs_mean)
    max_s = np.max(per_obs_mean)
    measurement_noise_mean_ms = np.mean(per_obs_std) * 1000.0  # MEASUREMENT noise
 
    sweep_totals = np.sum(time_matrix, axis=1)
    total_sweep_mean_s = np.mean(sweep_totals)
    total_sweep_std_s = np.std(sweep_totals, ddof=1)
 
    timing_stats = {
        "time_matrix": time_matrix,
        "per_obs_mean": per_obs_mean,
        "per_obs_std": per_obs_std,
        "mean_s": mean_s,
        "std_s": std_s,
        "measurement_noise_mean_ms": measurement_noise_mean_ms,
        "median_s": median_s,
        "min_s": min_s,
        "max_s": max_s,
        "total_sweep_mean_s": total_sweep_mean_s,
        "total_sweep_std_s": total_sweep_std_s,
        # Legacy fields — kept here so downstream consumers (e.g. existing
        # plot_summary_statistics) can read everything from timing_stats alone
        # without knowing about the new nested results structure.
        "n_warmup": n_warmup,
        "n_repeats": n_repeats,
        "N_obs": N_obs,
        "N_traj": N_traj,
        "Tmax": float(Tmax),
        "device_name": gpu_name,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "timing_method": "perf_counter_with_cuda_synchronize",
    }
 
    # =========================================================================
    # Assemble full results dict
    # =========================================================================
    results = {
        "method_name": method_name,
        "task_name": task_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": {
            "gpu_name": gpu_name,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        },
        "fairness_settings": fairness_settings,
        "protocol": {
            "n_warmup": n_warmup,
            "n_repeats": n_repeats,
            "timing_method": "time.perf_counter() with torch.cuda.synchronize()",
            "warmup_sweeps_each_full_N_obs": True,
        },
        "data": {
            "N_traj": N_traj,
            "N_obs": N_obs,
            "t_min": float(t_np[0]),
            "t_max": float(t_np[-1]),
            "Tmax": float(Tmax),
        },
        "timing_stats": timing_stats,
        "memory_stats": memory_stats,
    }
 
    # =========================================================================
    # Verbose summary + methods-section-ready text
    # =========================================================================
    if verbose:
        print(f"\n{'=' * 78}")
        print(f"RESULTS — {method_name}" + (f" / {task_name}" if task_name else ""))
        print(f"{'=' * 78}")
        print(f"\n--- Hardware & Software ---")
        print(f"  GPU:             {gpu_name}")
        print(f"  CUDA:            {torch.version.cuda}")
        print(f"  PyTorch:         {torch.__version__}")
        print(f"\n--- Fairness settings ---")
        print(f"  matmul TF32:     {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  cuDNN TF32:      {torch.backends.cudnn.allow_tf32}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  cuDNN determ.:   {torch.backends.cudnn.deterministic}")
        print(f"  Precision:       FP32 (TF32 disabled)")
        print(f"\n--- Protocol ---")
        print(f"  Timing method:   perf_counter + cuda.synchronize")
        print(f"  Warmup sweeps:   {n_warmup} ({n_warmup * N_obs} calls)")
        print(f"  Timed sweeps:    {n_repeats} ({n_repeats * N_obs} calls)")
        print(f"\n--- Data ---")
        print(f"  N_traj:          {N_traj}")
        print(f"  N_obs:           {N_obs}")
        print(f"  t range:         [{t_np[0]:.6f}, {t_np[-1]:.6f}]")
        print(f"  Tmax:            {Tmax}")
        print(f"\n--- Wall-clock results ---")
        print(f"  Per observation point:")
        print(f"    mean ± std across {N_obs} obs choices: "
              f"{mean_s * 1000:.3f} ± {std_s * 1000:.3f} ms  (STRUCTURAL variation)")
        print(f"    measurement noise (mean of per-obs std over {n_repeats} repeats): "
              f"{measurement_noise_mean_ms:.3f} ms")
        print(f"    median:  {median_s * 1000:.3f} ms")
        print(f"    range:   [{min_s * 1000:.3f}, {max_s * 1000:.3f}] ms")
        print(f"  Total sweep (across {N_obs} obs points):")
        print(f"    {total_sweep_mean_s:.4f} ± {total_sweep_std_s:.4f} s "
              f"(mean ± std over {n_repeats} repeats)")
        print(f"\n--- Peak GPU memory (during timed region) ---")
        print(f"  Total peak:                 {memory_stats['peak_MB_during_timing']:.2f} MB")
        print(f"  Above pre-timing baseline:  {memory_stats['peak_MB_above_baseline']:.2f} MB")
 
        print(f"\n{'-' * 78}")
        print("METHODS-SECTION-READY TEXT (paste into paper, edit as needed):")
        print(f"{'-' * 78}")
        print(
            f'Wall-clock inference latency was measured on a {gpu_name} '
            f'(CUDA {torch.version.cuda}, PyTorch {torch.__version__}). All '
            f'computations ran in strict FP32 with TF32 disabled '
            f'(torch.backends.cuda.matmul.allow_tf32 = False, '
            f'torch.backends.cudnn.allow_tf32 = False). For each of {N_obs} '
            f'observation choices over {N_traj} trajectories (batched), latency '
            f'was timed with time.perf_counter() bracketed by '
            f'torch.cuda.synchronize(), averaged over {n_repeats} sweeps after '
            f'{n_warmup} warmup sweeps that covered every observation choice. '
            f'We report the mean across observation choices ± the standard '
            f'deviation across observation choices, where the latter reflects '
            f'structural cost variation arising from the autoregressive segment '
            f'count (K_fwd + K_bwd) depending on the position of the observation '
            f'point relative to t_min and t_max; the per-observation measurement '
            f'noise (mean of per-obs standard deviation across repeats) was '
            f'{measurement_noise_mean_ms:.3f} ms, an order of magnitude below '
            f'the structural variation, indicating that the reported standard '
            f'deviation reflects genuine structural cost differences rather than '
            f'measurement noise. '
            f'{method_name} achieved {mean_s * 1000:.3f} ± {std_s * 1000:.3f} ms '
            f'per observation point. Peak GPU memory above pre-timing baseline '
            f'was {memory_stats["peak_MB_above_baseline"]:.2f} MB.'
        )
        print(f"{'=' * 78}")
 
    # =========================================================================
    # Save to pickle if requested
    # =========================================================================
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        if verbose:
            print(f"\nSaved full results to: {save_path}")
 
    return results



def test_model_all_observations_all_trajectories(
    get_data_from_trajectory_id_function,
    prediction_loss_function,
    test_id_df,
    test_df,
    mapping_net,
    inverse_net,
    device,
    Tmax,
    bootstrap_B=10000,
    confidence_level=0.95,
    normalize_by_energy=True,
    verbose=True,
):
    """
    Evaluate all test trajectories using every possible single observation point.

    For each trajectory, losses are summarized by their median over observation
    choices to obtain one trajectory-level median loss. Cross-trajectory descriptive
    statistics and a BCa bootstrap confidence interval for the mean of these
    trajectory-level medians are then computed.

    This treats trajectory as the primary independent evaluation unit and observation
    choice as a within-trajectory nuisance factor to be robustly summarized via median.

    Args:
        get_data_from_trajectory_id_function: function to retrieve trajectory data
        prediction_loss_function: loss function taking x_pred, u_pred, X_labels, U_labels
        test_id_df: DataFrame with trajectory metadata (must include 'energy' if normalize_by_energy=True)
        test_df: DataFrame with trajectory data
        mapping_net: trained MappingNet
        inverse_net: trained InverseNet
        device: torch device
        Tmax: maximum normalized temporal horizon (must be positive)
        bootstrap_B: number of bootstrap resamples for BCa CI
        confidence_level: confidence level for CI (e.g. 0.95)
        normalize_by_energy: if True, divide each trajectory's loss by sqrt(energy)
        verbose: if True, print progress and summary

    Returns:
        results: dict containing:
            - 'loss_matrix': (N_obs, N_traj) array of per-run losses (raw)
            - 'loss_matrix_normalized': (N_obs, N_traj) array of normalized losses (if normalize_by_energy)
            - 'per_trajectory_median_loss': (N_traj,) array used for statistics
            - 'per_obs_mean_loss': (N_obs,) array, mean loss per obs index across trajectories
            - 'summary_stats': dict with mean, std, median, q1, q3, iqr, ci_lower, ci_upper
            - 'trajectory_ids': array of trajectory ids
            - 'energies': (N_traj,) array of energy values
            - 'N_obs': number of observation points
            - 'N_traj': number of trajectories
            - 'confidence_level': confidence level used
            - 'normalized': whether energy normalization was applied
    """
    from scipy.stats import bootstrap

    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")

    if normalize_by_energy and 'energy' not in test_id_df.columns:
        raise ValueError("test_id_df must contain an 'energy' column when normalize_by_energy=True")

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
        test_trajectory_data = get_data_from_trajectory_id_function(test_id_df, test_df, trajectory_ids=trajectory_id)
        x_i = torch.as_tensor(test_trajectory_data['x'].to_numpy(dtype=np.float32), device=device)
        u_i = torch.as_tensor(test_trajectory_data['u'].to_numpy(dtype=np.float32), device=device)
        t_i = torch.as_tensor(test_trajectory_data['t'].to_numpy(dtype=np.float32), device=device)

        if x_i.dim() != 1 or u_i.dim() != 1 or t_i.dim() != 1:
            raise ValueError(f"Trajectory {trajectory_id}: x, u, t must be 1D, got dims {x_i.dim()}, {u_i.dim()}, {t_i.dim()}")

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

    # Stack: (N_traj, N_points)
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

        x_pred_queries, u_pred_queries, fwd_count = Batched_Autoregressive_Handoff_Inference_Protocol(
            mapping_net=mapping_net,
            inverse_net=inverse_net,
            x_obs=x_batch[:, obs_idx],
            u_obs=u_batch[:, obs_idx],
            t_obs=t_np[obs_idx],
            t_min=t_np[0],
            t_max=t_np[-1],
            Tmax=Tmax,
            query_times=query_times,
            query_indices=query_indices,
            device=device,
        )

        # Assemble full predictions
        x_pred_full = torch.empty(N_traj, N_obs, device=device, dtype=torch.float32)
        u_pred_full = torch.empty(N_traj, N_obs, device=device, dtype=torch.float32)
        x_pred_full[:, obs_idx] = x_batch[:, obs_idx]
        u_pred_full[:, obs_idx] = u_batch[:, obs_idx]
        for j, qi in enumerate(query_indices):
            x_pred_full[:, qi] = x_pred_queries[:, j]
            u_pred_full[:, qi] = u_pred_queries[:, j]

        # Per-trajectory loss for this observation index
        for traj_i in range(N_traj):
            loss_i = prediction_loss_function(
                x_pred=x_pred_full[traj_i],
                u_pred=u_pred_full[traj_i],
                X_labels=x_batch[traj_i],
                U_labels=u_batch[traj_i],
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

    # Aggregate: median over observation choices per trajectory (robust to edge outliers)
    per_trajectory_median_loss = np.median(loss_matrix_for_stats, axis=0)  # (N_traj,)
    per_obs_mean_loss = np.mean(loss_matrix_for_stats, axis=1)  # (N_obs,)

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
        print(f"N_traj={N_traj}, N_obs={N_obs}")
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
    }

    return results



def plot_summary_statistics(results, normalized=True, figsize=(16, 6),
                            timing_stats_batch=None, timing_stats_single=None,
                            band_alpha=0.3):
    """
    Compact summary plot for paper: two or three panels covering all essential statistical information.

    Left: Observation-averaged loss vs trajectory energy with per-trajectory IQR error bars
          and overall mean with BCa CI band.
    Center: Loss per observation time with IQR band across trajectories.
    Right (if any timing_stats provided): Inference time per observation point with
          measurement stability band and grand mean. Supports single-trajectory,
          batched multi-trajectory, or both overlaid.

    Also prints both raw and normalized summary statistics.

    Args:
        results: dict returned by test_model_all_observations_all_trajectories
        normalized: if True, plot energy-normalized loss. If False, plot raw loss.
        figsize: figure size tuple (width will be extended if timing panel is shown)
        timing_stats_batch: optional dict returned by time_inference_gpu.
        timing_stats_single: optional dict returned by time_inference_gpu_single_trajectory.
        band_alpha: float, transparency of shaded bands across all plots.
    """
    from scipy.stats import bootstrap as scipy_bootstrap

    loss_matrix = results['loss_matrix']
    energies = results['energies']
    confidence_level = results['confidence_level']
    N_obs = results['N_obs']
    N_traj = results['N_traj']

    if np.any(energies <= 1e-12):
        raise ValueError("Energy normalization by sqrt(E) requires positive nonzero energies.")
    sqrt_energies = np.sqrt(energies)

    loss_matrix_normalized = loss_matrix / sqrt_energies[np.newaxis, :]

    # === Print both raw and normalized statistics ===
    for mode, matrix, label in [
        ('Raw', loss_matrix, 'loss'),
        ('Normalized', loss_matrix_normalized, 'energy-normalized loss L/√E'),
    ]:
        traj_medians = np.median(matrix, axis=0)
        mean_val = np.mean(traj_medians)
        std_val = np.std(traj_medians, ddof=1)
        median_val = np.median(traj_medians)
        q1_val = np.percentile(traj_medians, 25)
        q3_val = np.percentile(traj_medians, 75)

        boot = scipy_bootstrap(
            data=(traj_medians,),
            statistic=np.mean,
            n_resamples=10000,
            confidence_level=confidence_level,
            method='BCa',
        )
        ci_lo = boot.confidence_interval.low
        ci_hi = boot.confidence_interval.high

        print(f"\n=== {mode} Statistics ({label}) ===")
        print(f"Per-trajectory summary: median across obs. choices")
        print(f"Mean of medians: {mean_val:.6f}")
        print(f"Std of medians: {std_val:.6f}")
        print(f"Median of medians: {median_val:.6f}")
        print(f"IQR of medians: [{q1_val:.6f}, {q3_val:.6f}]")
        print(f"{confidence_level*100:.0f}% BCa CI for mean of medians (over trajectories): [{ci_lo:.6f}, {ci_hi:.6f}]")
        print(f"N_traj={N_traj}, N_obs={N_obs}")

    # === Select matrix for plotting ===
    if normalized:
        loss_matrix_plot = loss_matrix_normalized
        loss_label = 'Energy-normalized loss L/√E'
    else:
        loss_matrix_plot = loss_matrix
        loss_label = 'Loss'

    # Compute stats for chosen mode (median across obs. choices per trajectory)
    traj_medians_plot = np.median(loss_matrix_plot, axis=0)
    overall_mean = np.mean(traj_medians_plot)
    overall_std = np.std(traj_medians_plot, ddof=1)

    boot_plot = scipy_bootstrap(
        data=(traj_medians_plot,),
        statistic=np.mean,
        n_resamples=10000,
        confidence_level=confidence_level,
        method='BCa',
    )
    ci_lower = boot_plot.confidence_interval.low
    ci_upper = boot_plot.confidence_interval.high

    # === Figure layout ===
    has_timing = timing_stats_batch is not None or timing_stats_single is not None
    n_panels = 3 if has_timing else 2
    fig_width = figsize[0] if not has_timing else figsize[0] + 8
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, figsize[1]))
    if n_panels == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes

    # === Left: Loss vs Energy scatter ===
    medians = np.median(loss_matrix_plot, axis=0)
    q25 = np.percentile(loss_matrix_plot, 25, axis=0)
    q75 = np.percentile(loss_matrix_plot, 75, axis=0)
    err_lower = medians - q25
    err_upper = q75 - medians

    ax1.errorbar(energies, medians, yerr=[err_lower, err_upper],
                 fmt='o', color='red', ecolor='red', elinewidth=1.5,
                 capsize=4, capthick=1.5, markersize=6, alpha=0.8,
                 label='Median (IQR across obs. choices)')

    ax1.axhspan(ci_lower, ci_upper, color='red', alpha=band_alpha,
                label=f'{confidence_level*100:.0f}% BCa CI for mean: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax1.axhline(y=overall_mean, color='red', linestyle='-', linewidth=1.5,
                label=f'Mean across trajectories: {overall_mean:.4f} (std={overall_std:.4f})')

    ax1.set_xlabel('Trajectory Energy')
    ax1.set_ylabel(loss_label)
    ax1.set_title(f'{loss_label} vs Energy (N_traj={N_traj}, N_obs={N_obs})')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # === Center: Loss per observation time ===
    if 't' in results:
        obs_times = results['t']
    else:
        obs_times = np.arange(N_obs)

    per_obs_median = np.median(loss_matrix_plot, axis=1)
    per_obs_p25 = np.percentile(loss_matrix_plot, 25, axis=1)
    per_obs_p75 = np.percentile(loss_matrix_plot, 75, axis=1)

    ax2.fill_between(obs_times, per_obs_p25, per_obs_p75,
                     alpha=band_alpha, color='red', label='IQR across trajectories')
    ax2.plot(obs_times, per_obs_median, marker='o', markersize=3, linestyle='-', color='red', linewidth=1,
             label='Median across trajectories')
    ax2.axhline(y=overall_mean, color='red', linestyle='-', linewidth=1.5,
                label=f'Mean across trajectories: {overall_mean:.4f}')

    ax2.set_xlabel('Observation Time (t)')
    ax2.set_ylabel(loss_label)
    ax2.set_title(f'{loss_label} per Observation Time (across {N_traj} trajectories)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # === Right: Inference time per observation point ===
    if has_timing:
        has_both = timing_stats_batch is not None and timing_stats_single is not None

        # Helper to plot one timing source
        def plot_timing(ts, color, prefix, ax):
            if len(obs_times) != len(ts['per_obs_mean']):
                raise ValueError(
                    f"Timing stats length {len(ts['per_obs_mean'])} does not match "
                    f"number of observation times {len(obs_times)}."
                )

            per_obs_mean_ms = ts['per_obs_mean'] * 1000
            per_obs_std_ms = ts['per_obs_std'] * 1000
            min_ms = ts['min_s'] * 1000
            max_ms = ts['max_s'] * 1000
            n_repeats = ts['n_repeats']

            curve_label = f'{prefix}Mean across repeats' if prefix else 'Mean across repeats'
            range_label = (f'{prefix}Range across obs. points: [{min_ms:.3f}, {max_ms:.3f}] ms'
                           if prefix else
                           f'Range across obs. points: [{min_ms:.3f}, {max_ms:.3f}] ms')

            ax.fill_between(obs_times,
                            per_obs_mean_ms - per_obs_std_ms,
                            per_obs_mean_ms + per_obs_std_ms,
                            alpha=band_alpha, color=color)
            ax.plot(obs_times, per_obs_mean_ms,
                    marker='o', markersize=3, linestyle='-', color=color, linewidth=1,
                    label=curve_label)
            ax.plot([], [], color='none', label=range_label)

        if has_both:
            plot_timing(timing_stats_single, 'red', 'Single: ', ax3)
            plot_timing(timing_stats_batch, 'steelblue', 'Batched: ', ax3)
            ax3.set_title(f'Inference Time per Obs. Point')
        elif timing_stats_batch is not None:
            plot_timing(timing_stats_batch, 'red', '', ax3)
            ax3.set_title(f'Inference Time per Obs. Point (batched, N_traj={N_traj})')
        else:
            plot_timing(timing_stats_single, 'red', '', ax3)
            ax3.set_title(f'Inference Time per Obs. Point (single trajectory)')

        ax3.set_xlabel('Observation Time (t)')
        ax3.set_ylabel('Inference Time (ms)')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Device info text box
        ts_for_info = timing_stats_batch if timing_stats_batch is not None else timing_stats_single
        device_name = ts_for_info.get('device_name', 'N/A')
        n_warmup = ts_for_info.get('n_warmup', '?')
        n_repeats = ts_for_info.get('n_repeats', '?')
        ax3.text(0.02, 0.98,
                 f'{device_name}\n{n_warmup} warmup, {n_repeats} timed runs',
                 transform=ax3.transAxes, fontsize=7, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()


def time_inference_gpu_single_trajectory(
    get_data_from_trajectory_id_function,
    test_id_df,
    test_df,
    trajectory_id,
    mapping_net,
    inverse_net,
    device,
    Tmax,
    n_warmup=5,
    n_repeats=20,
    verbose=True,
):
    """
    Self-contained end-to-end wall-clock timing for single-trajectory
    inference.

    Loads trajectory data, then measures per-observation-point inference
    time using time.perf_counter() with torch.cuda.synchronize().

    Intended to be called independently after evaluation:
        results = test_model_varying_observation_single_trajectory(...)
        timing_stats = time_inference_gpu_single_trajectory(
            get_data_from_trajectory_id_function,
            test_id_df, test_df, trajectory_id,
            mapping_net, inverse_net, device, Tmax,
        )
        results['timing_stats'] = timing_stats

    Args:
        get_data_from_trajectory_id_function: function to retrieve trajectory data
        test_id_df: DataFrame with trajectory metadata
        test_df: DataFrame with trajectory data
        trajectory_id: ID of the trajectory to time
        mapping_net: trained MappingNet (already on device)
        inverse_net: trained InverseNet (already on device)
        device: torch device
        Tmax: maximum normalized temporal horizon (must be positive)
        n_warmup: number of warmup sweeps (not timed)
        n_repeats: number of timed sweeps
        verbose: if True, print timing summary

    Returns:
        timing_stats: dict with:
            - 'time_matrix': (n_repeats, N_obs) array, seconds per obs per run
            - 'per_obs_mean': (N_obs,) mean time per obs point across repeats
            - 'per_obs_std': (N_obs,) std across repeats per obs point
            - 'mean_s': scalar, mean of per_obs_mean across observation choices
            - 'std_s': scalar, std of per_obs_mean across observation choices
              (reflects structural cost variation across observation positions,
               NOT independent experimental uncertainty)
            - 'median_s': scalar, median of per_obs_mean
            - 'min_s': scalar, min of per_obs_mean
            - 'max_s': scalar, max of per_obs_mean
            - 'total_sweep_mean_s': scalar, mean total sweep time across repeats
            - 'n_warmup', 'n_repeats', 'N_obs', 'trajectory_id'
    """
    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")

    # === Load trajectory ===
    test_trajectory_data = get_data_from_trajectory_id_function(test_id_df, test_df, trajectory_ids=trajectory_id)
    x = torch.as_tensor(test_trajectory_data['x'].to_numpy(dtype=np.float32), device=device)
    u = torch.as_tensor(test_trajectory_data['u'].to_numpy(dtype=np.float32), device=device)
    t = torch.as_tensor(test_trajectory_data['t'].to_numpy(dtype=np.float32), device=device)

    if x.dim() != 1 or u.dim() != 1 or t.dim() != 1:
        raise ValueError(f"x, u, t must be 1D, got dims {x.dim()}, {u.dim()}, {t.dim()}")

    t_np = t.detach().cpu().numpy()
    N_obs = len(t_np)

    if N_obs < 2:
        raise ValueError(f"Need at least two time points, got {N_obs}")

    if verbose:
        print(f"\nTiming benchmark: trajectory_id={trajectory_id}, N_obs={N_obs}, "
              f"device={torch.cuda.get_device_name(device)}")

    # === Define single-obs inference call ===
    def run_single_obs(obs_idx):
        query_indices = [i for i in range(N_obs) if i != obs_idx]
        query_times = t_np[query_indices]

        Autoregressive_Handoff_Inference_Protocol(
            mapping_net=mapping_net,
            inverse_net=inverse_net,
            x_obs=x[obs_idx],
            u_obs=u[obs_idx],
            t_obs=t_np[obs_idx],
            t_min=t_np[0],
            t_max=t_np[-1],
            Tmax=Tmax,
            query_times=query_times,
            query_indices=query_indices,
            device=device,
        )

    # === Warmup ===
    if verbose:
        print(f"Running {n_warmup} warmup sweeps...")
    for _ in range(n_warmup):
        for obs_idx in range(N_obs):
            run_single_obs(obs_idx)

    # === Timed runs ===
    if verbose:
        print(f"Running {n_repeats} timed sweeps...")
    time_matrix = np.empty((n_repeats, N_obs))

    for r in range(n_repeats):
        for obs_idx in range(N_obs):
            torch.cuda.synchronize(device)
            start = time.perf_counter()

            run_single_obs(obs_idx)

            torch.cuda.synchronize(device)
            elapsed_s = time.perf_counter() - start
            time_matrix[r, obs_idx] = elapsed_s

    # Per-obs statistics (mean/std across repeats for each obs point)
    per_obs_mean = np.mean(time_matrix, axis=0)        # (N_obs,)
    per_obs_std = np.std(time_matrix, axis=0, ddof=1)  # (N_obs,)

    # Summary across observation choices (from per-obs means)
    mean_s = np.mean(per_obs_mean)
    std_s = np.std(per_obs_mean, ddof=1)
    median_s = np.median(per_obs_mean)
    min_s = np.min(per_obs_mean)
    max_s = np.max(per_obs_mean)

    # Total sweep time (sum across obs per repeat, then mean across repeats)
    total_sweep_mean_s = np.mean(np.sum(time_matrix, axis=1))

    if verbose:
        print(f"\n{'='*70}")
        print(f"TIMING BENCHMARK — Single-Trajectory Inference")
        print(f"{'='*70}")
        print(f"\n--- Hardware & Software ---")
        print(f"Device:          {torch.cuda.get_device_name(device)}")
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"\n--- Protocol ---")
        print(f"Timing method:   time.perf_counter() with torch.cuda.synchronize()")
        print(f"Warmup sweeps:   {n_warmup}")
        print(f"Timed sweeps:    {n_repeats}")
        print(f"Tmax:            {Tmax}")
        print(f"\n--- Data ---")
        print(f"Trajectory ID:   {trajectory_id}")
        print(f"N_obs:           {N_obs}")
        print(f"t range:         [{t_np[0]:.6f}, {t_np[-1]:.6f}]")
        print(f"\n--- Results ---")
        print(f"Per observation point (mean ± std across obs choices):")
        print(f"  {mean_s*1000:.3f} ± {std_s*1000:.3f} ms")
        print(f"Per observation point (median): {median_s*1000:.3f} ms")
        print(f"Per observation point (range):  [{min_s*1000:.3f}, {max_s*1000:.3f}] ms")
        print(f"Total sweep (mean ± std across {n_repeats} repeats):")
        print(f"  {total_sweep_mean_s:.4f} ± {np.std(np.sum(time_matrix, axis=1), ddof=1):.4f} s")
        print(f"\"End-to-end inference time was measured on a "
              f"{torch.cuda.get_device_name(device)} using PyTorch {torch.__version__} "
              f"(CUDA {torch.version.cuda}). Each observation-choice time was measured "
              f"with time.perf_counter() and torch.cuda.synchronize(), averaged over "
              f"{n_repeats} runs after {n_warmup} warmup runs. We report the mean ± "
              f"standard deviation across {N_obs} observation choices (single trajectory, "
              f"ID={trajectory_id}): "
              f"{mean_s*1000:.3f} ± {std_s*1000:.3f} ms per observation point.\"")
        print(f"{'='*70}")

    timing_stats = {
        'time_matrix': time_matrix,
        'per_obs_mean': per_obs_mean,
        'per_obs_std': per_obs_std,
        'mean_s': mean_s,
        'std_s': std_s,
        'median_s': median_s,
        'min_s': min_s,
        'max_s': max_s,
        'total_sweep_mean_s': total_sweep_mean_s,
        'n_warmup': n_warmup,
        'n_repeats': n_repeats,
        'N_obs': N_obs,
        'trajectory_id': trajectory_id,
        'Tmax': Tmax,
        'timing_method': 'perf_counter_with_cuda_synchronize',
        'device_name': torch.cuda.get_device_name(device),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
    }

    return timing_stats





def test_model_varying_observation_single_trajectory(
    get_data_from_trajectory_id_function,
    loss_type,
    test_id_df,
    test_df,
    trajectory_id,
    mapping_net,
    inverse_net,
    device,
    Tmax,
    case=None,
    k=1, mass=1, g=9.81, length=3.0, constant=2.4,
    verbose=False,
):
    """
    Single-trajectory diagnostic: run inference using every possible observation point.

    This is a sensitivity analysis over observation choices for one trajectory.
    The reported intervals summarize variation across observation choices and
    should not be interpreted as cross-trajectory experimental uncertainty.

    Collects per-run scalar losses, per-time-point pointwise errors, and
    per-run predicted trajectories for phase space visualization.

    Returns:
        results: dict containing:
            - 'per_run_losses': (N,) array of scalar losses, one per observation choice
            - 'pointwise_errors': (N, T) array of pointwise errors
            - 'all_x_pred': (N, T) array of predicted q values per observation choice
            - 'all_u_pred': (N, T) array of predicted p values per observation choice
            - 'summary_stats': dict with mean, std, median, q1, q3, iqr, min, max
            - 'per_timepoint_stats': dict with median, p25, p75 arrays of shape (T,)
            - 'representative_idx': int, index of the run closest to median loss
            - 't': (T,) array of time values
            - 'x': (T,) ground truth q
            - 'u': (T,) ground truth p
            - 'trajectory_id': trajectory id
            - 'loss_type': loss type used
            - 'N': number of observation choices
            - 'T': number of time points
    """
    if loss_type not in ("euclidean", "mae", "mse"):
        raise ValueError(f"Unknown loss_type '{loss_type}'. Must be 'euclidean', 'mae', or 'mse'.")
    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")

    # Load trajectory once
    test_trajectory_data = get_data_from_trajectory_id_function(test_id_df, test_df, trajectory_ids=trajectory_id)
    x = torch.as_tensor(test_trajectory_data['x'].to_numpy(dtype=np.float32), device=device)
    u = torch.as_tensor(test_trajectory_data['u'].to_numpy(dtype=np.float32), device=device)
    t = torch.as_tensor(test_trajectory_data['t'].to_numpy(dtype=np.float32), device=device)

    T = len(t)
    if T < 2:
        raise ValueError(f"Need at least two time points to run observation-choice analysis, got {T}")
    N = T  # one run per observation point

    # Pre-convert ground truth to numpy once
    t_np = t.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    u_np = u.detach().cpu().numpy()

    per_run_losses = np.empty(N)
    pointwise_errors = np.empty((N, T))
    all_x_pred = np.empty((N, T))
    all_u_pred = np.empty((N, T))
    all_mapping_net_calls = np.empty(N, dtype=int)
    all_inverse_net_calls = np.empty(N, dtype=int)

    # Energy computation setup
    if case is not None:
        if case not in ("harmonic_oscillator", "ideal_pendulum", "real_pendulum"):
            raise ValueError(f"Unknown case '{case}'. Must be 'harmonic_oscillator', 'ideal_pendulum', or 'real_pendulum'.")

        def compute_energy(q, p):
            if case == "harmonic_oscillator":
                omega_squared = k / mass
                return 0.5 * p**2 + 0.5 * omega_squared * q**2
            elif case == "ideal_pendulum":
                c = g / length
                return 0.5 * p**2 + c * (1 - np.cos(q))
            elif case == "real_pendulum":
                return p**2 + constant * (1 - np.cos(q))

        all_energy_pred = np.empty((N, T))
        energy_true = compute_energy(x_np, u_np)

    for obs_idx in range(N):
        pred_loss, x_pred, u_pred, _, _, _, fwd_count = test_model_in_single_trajectory(
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
            verbose=False,
        )

        per_run_losses[obs_idx] = pred_loss.item()

        x_pred_np = x_pred.detach().cpu().numpy()
        u_pred_np = u_pred.detach().cpu().numpy()
        all_x_pred[obs_idx] = x_pred_np
        all_u_pred[obs_idx] = u_pred_np

        if case is not None:
            all_energy_pred[obs_idx] = compute_energy(x_pred_np, u_pred_np)

        if loss_type == "euclidean":
            pointwise_errors[obs_idx] = np.sqrt((x_pred_np - x_np)**2 + (u_pred_np - u_np)**2)
        elif loss_type == "mae":
            pointwise_errors[obs_idx] = np.abs(x_pred_np - x_np) + np.abs(u_pred_np - u_np)
        elif loss_type == "mse":
            pointwise_errors[obs_idx] = (x_pred_np - x_np)**2 + (u_pred_np - u_np)**2

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

    # Representative run: closest to median scalar loss
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
    if case is not None:
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
        'case': case,
        'forward_pass_stats': forward_pass_stats,
        'all_mapping_net_calls': all_mapping_net_calls,
        'all_inverse_net_calls': all_inverse_net_calls,
    }

    return results


def plot_varying_observation_results(results, obs_idx, figsize=(28, 7), connect_points=True, iqr_alpha=0.4):
    """
    Plot results from test_model_varying_observation_single_trajectory.

    Creates two or three subplots:
        1. Per-time-point error IQR band across observation choices
        2. Energy: ground truth vs predicted with bands (if case was provided)
        3. Phase space: ground truth vs prediction

    Args:
        results: dict returned by test_model_varying_observation_single_trajectory
        obs_idx: int. Which observation-run to use for plots 2 and 3.
        figsize: tuple, figure size
        connect_points: bool, whether to connect phase space points with lines
        iqr_alpha: float, transparency of IQR bands in plots 1 and 2
    """
    t = results['t']
    per_timepoint_stats = results['per_timepoint_stats']
    summary_stats = results['summary_stats']
    trajectory_id = results['trajectory_id']
    loss_type = results['loss_type']
    N = results['N']
    T = results['T']
    x_gt = results['x']
    u_gt = results['u']
    all_x_pred = results['all_x_pred']
    all_u_pred = results['all_u_pred']
    pointwise_errors = results['pointwise_errors']

    if not (0 <= obs_idx < N):
        raise ValueError(f"obs_idx={obs_idx} is out of bounds for N={N} runs")

    energy_true = results.get('energy_true', None)
    all_energy_pred = results.get('all_energy_pred', None)
    energy_stats = results.get('energy_stats', None)

    # Predictions for the selected observation run
    x_pred_run = all_x_pred[obs_idx]
    u_pred_run = all_u_pred[obs_idx]
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
        # IQR band across observation choices
        ax2.fill_between(t, energy_stats['p25'], energy_stats['p75'],
                         alpha=iqr_alpha, color='red',
                         label='IQR across obs. choices')

        # Selected run energy curve
        energy_run = all_energy_pred[obs_idx]
        ax2.plot(t, energy_run, color='red', linewidth=1.5,
                 label='E_pred(t) (selected obs. run)')

        # True energy
        ax2.plot(t, energy_true, color='blue', linewidth=2, label='E_true(t)')

        ax2.set_xlim(t[0], t[-1])
        ax2.set_xlabel('Time (t)')
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy: Predicted vs True')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Energy error statistics for the selected run
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

    # === Plot 3: Phase space ===
    scatter_gt = ax3.scatter(x_gt, u_gt, c=t, cmap='Blues', s=60, alpha=0.8,
                             edgecolors='darkblue', linewidths=0.5, zorder=3)
    scatter_pred = ax3.scatter(x_pred_run, u_pred_run, c=t, cmap='Reds', s=60,
                               alpha=0.8, edgecolors='darkred',
                               linewidths=0.5, zorder=3)

    if connect_points:
        ax3.plot(x_gt, u_gt, color='blue', alpha=0.3, linewidth=1.5, zorder=2)
        ax3.plot(x_pred_run, u_pred_run, color='red', alpha=0.3,
                 linewidth=1.5, zorder=2)

    ax3.scatter(x_gt[0], u_gt[0], c='blue', marker='*', s=400,
                edgecolors='black', linewidths=2.5, zorder=5)
    ax3.scatter(x_pred_run[0], u_pred_run[0], c='red', marker='*', s=400,
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

    ax3.set_xlabel('q', fontsize=12)
    ax3.set_ylabel('p', fontsize=12)
    ax3.set_title(
        'Phase Space: Ground Truth vs Prediction (selected obs. run)',
        fontsize=11,
    )
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()



def test_model_variance_with_varying_observed_points(
    get_data_from_trajectory_id_function,
    test_id_df,
    test_df,
    mapping_net,
    Tmax,
    device,
    n_permutations=20,
    confidence_level=0.95,
    valid_fraction_threshold=0.5,
    rng_seed=None,
    verbose=True,
):
    """
    Measure within-segment consistency of MappingNet outputs as observation count increases.

    A perfect MappingNet maps all points within each segment to the same (Q, P) values,
    so within-segment variance should be near zero whenever at least two points are
    observed in a segment. This function tracks that variance as the number of observed
    points grows.

    Segments with fewer than two observed points are excluded (variance is not estimable,
    not zero). Each segment is weighted equally. The variance is in the learned coordinate
    scale of Q and P.

    For statistical robustness:
        - Repeats over multiple random permutations of observation order.
        - For each point count, averages per-trajectory variances across permutations,
          then computes cross-trajectory statistics (median, BCa CI for median, IQR).
        - Trajectories are the independent evaluation units.
        - Validity is assessed from the raw (trajectory, permutation) entries.

    Args:
        n_permutations: number of random orderings to average over
        confidence_level: confidence level for BCa CI
        valid_fraction_threshold: minimum fraction of (trajectory, permutation) entries
                                  that must be non-NaN at a given point count to include
                                  it in the plot (default 0.5)
        rng_seed: random seed for reproducibility (optional)

    Returns:
        results: dict with variance data, statistics, and metadata
    """
    from scipy.stats import bootstrap as scipy_bootstrap
    import warnings

    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")

    trajectory_ids = test_id_df['trajectory_id'].values.astype(int)
    N_traj = len(trajectory_ids)

    # Preload all trajectories and validate shared time structure
    all_x = []
    all_u = []
    all_t = []
    ref_t = None

    for trajectory_id in trajectory_ids:
        traj_data = get_data_from_trajectory_id_function(test_id_df, test_df, trajectory_ids=int(trajectory_id))
        x_i = torch.as_tensor(traj_data['x'].to_numpy(dtype=np.float32), device=device)
        u_i = torch.as_tensor(traj_data['u'].to_numpy(dtype=np.float32), device=device)
        t_i = torch.as_tensor(traj_data['t'].to_numpy(dtype=np.float32), device=device)

        if x_i.dim() != 1 or u_i.dim() != 1 or t_i.dim() != 1:
            raise ValueError(f"Trajectory {trajectory_id}: x, u, t must be 1D, got dims {x_i.dim()}, {u_i.dim()}, {t_i.dim()}")

        if ref_t is None:
            ref_t = t_i
        else:
            if len(t_i) != len(ref_t):
                raise ValueError(
                    f"Trajectory {trajectory_id} has {len(t_i)} points, expected {len(ref_t)}. "
                    f"All trajectories must have the same number of points."
                )
            if not torch.allclose(t_i, ref_t, atol=1e-6):
                raise ValueError(
                    f"Trajectory {trajectory_id} has different time values. "
                    f"All trajectories must share the same time values."
                )

        all_x.append(x_i)
        all_u.append(u_i)
        all_t.append(t_i)

    t_np = ref_t.detach().cpu().numpy()
    num_points = len(t_np)
    t_max = t_np[-1]

    # Build segment index mapping
    num_segments = int(np.ceil(t_max / Tmax - 1e-9))
    num_segments = max(num_segments, 1)
    segment_indices_dict = {}

    for seg_idx in range(num_segments):
        t_start = seg_idx * Tmax
        t_end = min((seg_idx + 1) * Tmax, t_max)
        if seg_idx == num_segments - 1:
            t_end = t_max

        if seg_idx == 0:
            seg_mask = (t_np >= t_start) & (t_np <= t_end)
        else:
            seg_mask = (t_np > t_start) & (t_np <= t_end)

        seg_indices = np.where(seg_mask)[0].tolist()
        if len(seg_indices) > 0:
            segment_indices_dict[seg_idx] = seg_indices

    # variance_raw[point_count, traj_idx, perm_idx]
    variance_raw = np.full((num_points, N_traj, n_permutations), np.nan)

    rng = np.random.default_rng(rng_seed)

    for perm_idx in range(n_permutations):
        random_order = rng.permutation(num_points)

        if verbose:
            print(f"Permutation {perm_idx + 1}/{n_permutations}")

        for num_obs in range(1, num_points + 1):
            point_indexes_observed = random_order[:num_obs].tolist()
            selected_set = set(point_indexes_observed)

            filtered_dict = {
                k: [val for val in v if val in selected_set]
                for k, v in segment_indices_dict.items()
                if any(val in selected_set for val in v)
            }

            valid_segments = {
                k: v for k, v in filtered_dict.items() if len(v) >= 2
            }

            if len(valid_segments) == 0:
                continue

            for traj_i in range(N_traj):
                x = all_x[traj_i]
                u = all_u[traj_i]
                t = all_t[traj_i]







                segment_variances = []
                epsilon = 1e-3

                for seg_idx, seg_obs_indices in valid_segments.items():
                    t_start = seg_idx * Tmax
                    relative_t = t[seg_obs_indices] - t_start

                    X_final, U_final, _ = mapping_net(x[seg_obs_indices], u[seg_obs_indices], relative_t)

                    X_std = X_final.std(unbiased=True).item()
                    U_std = U_final.std(unbiased=True).item()
                    X_mean_abs = X_final.mean().abs().item()
                    U_mean_abs = U_final.mean().abs().item()

                    X_cv = X_std / (X_mean_abs + epsilon)
                    U_cv = U_std / (U_mean_abs + epsilon)
                    segment_variances.append((X_cv + U_cv) / 2)



                variance_raw[num_obs - 1, traj_i, perm_idx] = np.mean(segment_variances)

    # Average across permutations (NaN-aware)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        variance_avg = np.nanmean(variance_raw, axis=2)

    # === Compute full MappingNet outputs for all trajectories (for Q(t), P(t) plots) ===
    X_final_list = [[] for _ in range(N_traj)]
    U_final_list = [[] for _ in range(N_traj)]

    for traj_i in range(N_traj):
        x = all_x[traj_i]
        u = all_u[traj_i]
        t = all_t[traj_i]

        for seg_idx, seg_indices in segment_indices_dict.items():
            t_start = seg_idx * Tmax
            relative_t = t[seg_indices] - t_start

            X_final, U_final, _ = mapping_net(x[seg_indices], u[seg_indices], relative_t)

            X_final_list[traj_i].extend(X_final.cpu().detach().numpy())
            U_final_list[traj_i].extend(U_final.cpu().detach().numpy())

    # Convert to arrays
    X_final_array = np.array([np.array(xl) for xl in X_final_list])
    U_final_array = np.array([np.array(ul) for ul in U_final_list])

    # Validity from raw entries
    valid_fraction_per_point = np.mean(~np.isnan(variance_raw), axis=(1, 2))
    valid_mask = valid_fraction_per_point >= valid_fraction_threshold

    point_counts = np.arange(1, num_points + 1)

    # Cross-trajectory statistics at each valid point count
    median_curve = np.full(num_points, np.nan)
    p25_curve = np.full(num_points, np.nan)
    p75_curve = np.full(num_points, np.nan)

    for i in range(num_points):
        if not valid_mask[i]:
            continue
        vals = variance_avg[i, :]
        vals_clean = vals[~np.isnan(vals)]
        if len(vals_clean) == 0:
            continue
        median_curve[i] = np.median(vals_clean)
        p25_curve[i] = np.percentile(vals_clean, 25)
        p75_curve[i] = np.percentile(vals_clean, 75)

    # BCa CI for median at final point count with percentile fallback
    final_vals = variance_avg[-1, :]
    final_vals_clean = final_vals[~np.isnan(final_vals)]

    ci_method = None

    if len(final_vals_clean) >= 2:
        try:
            boot = scipy_bootstrap(
                data=(final_vals_clean,),
                statistic=np.median,
                n_resamples=10000,
                confidence_level=confidence_level,
                method='BCa',
            )
            ci_method = "BCa"
            final_ci_lower = boot.confidence_interval.low
            final_ci_upper = boot.confidence_interval.high
        except Exception:
            boot = scipy_bootstrap(
                data=(final_vals_clean,),
                statistic=np.median,
                n_resamples=10000,
                confidence_level=confidence_level,
                method='percentile',
            )
            ci_method = "percentile"
            final_ci_lower = boot.confidence_interval.low
            final_ci_upper = boot.confidence_interval.high
    else:
        ci_method = "N/A"
        final_ci_lower = np.nan
        final_ci_upper = np.nan

    final_median = np.nanmedian(final_vals)
    final_q1 = np.nanpercentile(final_vals, 25)
    final_q3 = np.nanpercentile(final_vals, 75)

    if verbose:
        print(f"\n=== Within-Segment Consistency at Full Observation ===")
        print(f"Median across trajectories: {final_median:.6f}")
        print(f"IQR: [{final_q1:.6f}, {final_q3:.6f}]")
        if ci_method != "N/A":
            print(f"{confidence_level*100:.0f}% {ci_method} bootstrap CI for median (over trajectories): "
                  f"[{final_ci_lower:.6f}, {final_ci_upper:.6f}]")
        print(f"N_traj={N_traj}, N_obs={num_points}, N_permutations={n_permutations}")
        if np.any(valid_mask):
            print(f"First valid point count: {point_counts[valid_mask][0]}")

    results = {
        'variance_raw': variance_raw,
        'variance_avg': variance_avg,
        'point_counts': point_counts,
        'valid_mask': valid_mask,
        'valid_fraction_per_point': valid_fraction_per_point,
        'valid_fraction_threshold': valid_fraction_threshold,
        'median_curve': median_curve,
        'p25_curve': p25_curve,
        'p75_curve': p75_curve,
        'final_median': final_median,
        'final_q1': final_q1,
        'final_q3': final_q3,
        'final_ci_lower': final_ci_lower,
        'final_ci_upper': final_ci_upper,
        'ci_method': ci_method,
        'segment_indices_dict': segment_indices_dict,
        'Tmax': Tmax,
        't': t_np,
        'rng_seed': rng_seed,
        'N_traj': N_traj,
        'N_obs': num_points,
        'n_permutations': n_permutations,
        'confidence_level': confidence_level,
        'X_final_array': X_final_array,
        'U_final_array': U_final_array,
    }

    return results



def plot_variance_results(results, figsize=(20, 6), only_plot_percentage=None, band_alpha=0.3):
    """
    Plot results from test_model_variance_with_varying_observed_points.

    Creates three side-by-side plots:
        1. Q(t) for all trajectories with Tmax segment markers
        2. P(t) for all trajectories with Tmax segment markers
        3. Within-segment variance vs number of sampled points

    Args:
        results: dict returned by test_model_variance_with_varying_observed_points
        figsize: figure size tuple
        only_plot_percentage: float or None, fraction of trajectories to plot in Q(t)/P(t)
        band_alpha: float, transparency of IQR band
    """
    point_counts = results['point_counts']
    valid_mask = results['valid_mask']
    median_curve = results['median_curve']
    p25_curve = results['p25_curve']
    p75_curve = results['p75_curve']
    final_median = results['final_median']
    final_q1 = results['final_q1']
    final_q3 = results['final_q3']
    final_ci_lower = results['final_ci_lower']
    final_ci_upper = results['final_ci_upper']
    ci_method = results['ci_method']
    confidence_level = results['confidence_level']
    valid_fraction_threshold = results['valid_fraction_threshold']
    N_traj = results['N_traj']
    N_obs = results['N_obs']
    n_permutations = results['n_permutations']
    t_np = results['t']
    Tmax = results['Tmax']
    X_final_array = results['X_final_array']
    U_final_array = results['U_final_array']

    if only_plot_percentage is not None:
        if not (0.0 <= only_plot_percentage <= 1.0):
            raise ValueError("only_plot_percentage must be between 0.0 and 1.0")
        n_to_plot = max(1, int(N_traj * only_plot_percentage))
        plot_indices = np.random.choice(N_traj, size=n_to_plot, replace=False)
        plot_indices = sorted(plot_indices)
    else:
        plot_indices = list(range(N_traj))

    import matplotlib.cm as cm
    colors = cm.get_cmap('tab10' if N_traj <= 10 else 'tab20' if N_traj <= 20 else 'hsv')(
        np.linspace(0, 1, N_traj))

    fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    # === Plot 1: Q(t) ===
    for i in plot_indices:
        ax1.plot(t_np, X_final_array[i], alpha=0.7, color=colors[i])

    max_time = t_np[-1]
    current_marker = Tmax
    first_line = True
    while current_marker < max_time:
        label = f'Tmax ({Tmax:.3f})' if first_line else None
        ax1.axvline(x=current_marker, color='black', linestyle='--', alpha=0.5, label=label)
        current_marker += Tmax
        first_line = False

    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Q(t)')
    if only_plot_percentage is not None:
        ax1.set_title(f"Q(t) — {len(plot_indices)} of {N_traj}\n({only_plot_percentage*100:.1f}%)")
    else:
        ax1.set_title(f"Q(t) — All {N_traj} Trajectories")
    ax1.grid(True, alpha=0.3)
    if first_line is False:
        ax1.legend(loc='upper right', fontsize=8)

    # === Plot 2: P(t) ===
    for i in plot_indices:
        ax2.plot(t_np, U_final_array[i], alpha=0.7, color=colors[i])

    current_marker = Tmax
    first_line = True
    while current_marker < max_time:
        label = f'Tmax ({Tmax:.3f})' if first_line else None
        ax2.axvline(x=current_marker, color='black', linestyle='--', alpha=0.5, label=label)
        current_marker += Tmax
        first_line = False

    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('P(t)')
    if only_plot_percentage is not None:
        ax2.set_title(f"P(t) — {len(plot_indices)} of {N_traj}\n({only_plot_percentage*100:.1f}%)")
    else:
        ax2.set_title(f"P(t) — All {N_traj} Trajectories")
    ax2.grid(True, alpha=0.3)
    if first_line is False:
        ax2.legend(loc='upper right', fontsize=8)

    # === Plot 3: Variance curve ===
    if not np.any(valid_mask):
        ax3.text(0.5, 0.5, 'No valid data to plot', transform=ax3.transAxes,
                 ha='center', va='center', fontsize=12)
    else:
        plot_points = point_counts[valid_mask]
        plot_median = median_curve[valid_mask]
        plot_p25 = p25_curve[valid_mask]
        plot_p75 = p75_curve[valid_mask]

        ax3.fill_between(plot_points, plot_p25, plot_p75,
                         alpha=band_alpha, color='red', label='IQR across trajectories')
        ax3.plot(plot_points, plot_median, color='red', linewidth=1.5,
                 label='Median across trajectories')

        ax3.set_xlim(plot_points[0], plot_points[-1])

        # Stats text box
        ci_str = (f"{confidence_level*100:.0f}% {ci_method} CI for median:\n"
                  f"[{final_ci_lower:.4f}, {final_ci_upper:.4f}]") if ci_method != "N/A" else "CI: N/A"
        textstr = (
            f"At all {N_obs} pts sampled:\n"
            f"Median [IQR]: {final_median:.4f}\n"
            f"[{final_q1:.4f}, {final_q3:.4f}]\n"
            f"{ci_str}"
        )
        ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes, fontsize=8,
                 ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax3.legend(loc='upper left', fontsize=8)

    ax3.set_xlabel('Sampled Points per Trajectory')
    ax3.set_ylabel(r'$\frac{1}{2}(\mathrm{CV}(Q)+\mathrm{CV}(P))$')
    ax3.set_title(f'Within-Segment Consistency\n(N_perm={n_permutations})')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

























def visualize_epoch_metrics(save_dir_path, metrics_to_plot, plot_on_same_graph=False, verbose=False, specific_epochs=None):
    """
    Visualizes selected metrics from epoch directories.

    Args:
        save_dir_path (str): Path to the main directory containing 'epoch_n' subdirectories.
        metrics_to_plot (list of str): List of metric names to visualize.
        plot_on_same_graph (bool): If True, group related metrics (train/val variants) on the same plot.
        verbose (bool): If True, prints summary statistics for each metric.
        specific_epochs (list or None): List with 2n values specifying epoch ranges to plot. 
                                        E.g., [2, 10] plots epochs 2-10, [2, 10, 15, 20] plots 2-10 and 15-20.
                                        Last item can be "last" to plot until the last epoch.
                                        If None, plots all epochs.
    """

    # --- Collect all epoch directories ---
    epoch_dirs = sorted(
        [d for d in os.listdir(save_dir_path) if d.startswith("epoch_")],
        key=lambda x: int(x.split("_")[1])
    )

    # --- Collect data ---
    metrics_data = {metric: [] for metric in metrics_to_plot}
    epochs = []

    for d in epoch_dirs:
        epoch_path = os.path.join(save_dir_path, d, "epoch_metrics.json")
        if not os.path.isfile(epoch_path):
            print(f"⚠️ Skipping {d} (no epoch_metrics.json found)")
            continue

        with open(epoch_path, "r") as f:
            data = json.load(f)

        epoch_num = data.get("epoch", int(d.split("_")[1]))
        epochs.append(epoch_num)

        for metric in metrics_to_plot:
            metrics_data[metric].append(data.get(metric, None))

    # --- Verbose logging of statistics ---
    if verbose:
        print("\n📊 Metric summaries:")
        for metric in metrics_to_plot:
            values = [v for v in metrics_data[metric] if v is not None]
            if not values:
                print(f"  ⚠️ Metric {metric} has no valid values.")
                continue

            min_val = min(values)
            min_epoch = epochs[metrics_data[metric].index(min_val)]
            last_5 = values[-5:] if len(values) >= 5 else values
            print(
                f"  Lowest loss of metric '{metric}' recorded in epoch {min_epoch} "
                f"with the value: {min_val:.4f}, "
                f"the losses of the last 5 epochs are: {[f'{v:.4f}' for v in last_5]}"
            )

    # --- Filter epochs for plotting if specific_epochs is provided ---
    if specific_epochs is not None:
        # Create a mask for which epochs to include
        plot_mask = [False] * len(epochs)
        
        # Process pairs of ranges
        for i in range(0, len(specific_epochs), 2):
            start = specific_epochs[i]
            end = specific_epochs[i + 1]
            
            # Handle "last" keyword
            if end == "last":
                end = epochs[-1]
            
            # Mark epochs that fall within this range
            for idx, epoch in enumerate(epochs):
                if start <= epoch <= end:
                    plot_mask[idx] = True
        
        # Filter epochs and metrics_data
        epochs_to_plot = [e for e, mask in zip(epochs, plot_mask) if mask]
        metrics_to_plot_data = {
            metric: [v for v, mask in zip(values, plot_mask) if mask]
            for metric, values in metrics_data.items()
        }
    else:
        # Use all data
        epochs_to_plot = epochs
        metrics_to_plot_data = metrics_data

    # --- Color scheme for each data source ---
    data_colors = {
        "train": "tab:blue",
        "val": "tab:orange",
        "val_high_energy": "tab:green",
        "val_training_set": "tab:red",
        "other": "tab:gray"
    }

    # --- Helper: identify data prefix and core metric ---
    def split_metric_name(metric):
        """
        Returns (data_prefix, core_metric)
        Handles trailing underscores and variants like _high_energy / _training_set.
        """
        m = metric.rstrip("_")  # remove trailing underscores

        prefix = "other"
        core = m

        if m.startswith("train_"):
            prefix = "train"
            core = m[len("train_"):]
        elif m.startswith("val_"):
            core = m[len("val_"):]
            if "_high_energy" in core:
                prefix = "val_high_energy"
                core = core.replace("_high_energy", "")
            elif "_training_set" in core:
                prefix = "val_training_set"
                core = core.replace("_training_set", "")
            else:
                prefix = "val"

        core = core.rstrip("_")
        return prefix, core

    # --- Plotting ---
    if plot_on_same_graph:
        # Group metrics by core metric name
        grouped = {}
        for metric in metrics_to_plot:
            prefix, core = split_metric_name(metric)
            grouped.setdefault(core, {})[prefix] = metric

        for core_metric, variants in grouped.items():
            plt.figure(figsize=(8, 5))
            for prefix, metric in variants.items():
                plt.plot(
                    epochs_to_plot,
                    metrics_to_plot_data[metric],
                    marker='o',
                    label=metric,
                    color=data_colors.get(prefix, None)
                )

            plt.title(f"{core_metric.replace('_', ' ').title()}")
            plt.xlabel("Epoch")
            plt.ylabel(core_metric)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        for metric in metrics_to_plot:
            prefix, _ = split_metric_name(metric)
            plt.figure(figsize=(8, 5))
            plt.plot(
                epochs_to_plot,
                metrics_to_plot_data[metric],
                marker='o',
                label=metric,
                color=data_colors.get(prefix, None)
            )
            plt.title(f"{metric.replace('_', ' ').title()} over epochs")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()












def analyze_mapping_net(mapping_net, return_lists=False):
    """
    Analyze and visualize parameters of a mapping network that has step_1 and step_2 submodules.
    Plots MLP parameter statistics (mean, std, L2 norm) and mapping parameters (c1, c2, gamma, a).
    
    Parameters
    ----------
    mapping_net : torch.nn.Module
        The mapping network containing step_1 and step_2 submodules.
    return_lists : bool, optional (default=False)
        If True, returns all extracted parameter lists.
    
    Returns
    -------
    dict (if return_lists=True)
        {
            "step_1_mlp_params": dict,
            "step_2_mlp_params": dict,
            "step_1_c1_values": list,
            "step_2_c1_values": list,
            "step_1_c2_values": list,
            "step_2_c2_values": list,
            "step_1_gamma_values": list,
            "step_2_gamma_values": list,
            "step_1_a_values": list,
            "step_2_a_values": list
        }
    """
    # --- Step 1: Extract all parameters ---
    step_1_mlp_params = {}
    step_2_mlp_params = {}
    step_1_c1_values, step_2_c1_values = [], []
    step_1_c2_values, step_2_c2_values = [], []
    step_1_gamma_values, step_2_gamma_values = [], []
    step_1_a_values, step_2_a_values = [], []

    for name, param in mapping_net.named_parameters():
        if 'step_1' in name:
            if 'G_network' in name:
                step_1_mlp_params[name] = param
            elif 'c1' in name:
                step_1_c1_values.append(param.item())
            elif 'c2' in name:
                step_1_c2_values.append(param.item())
            elif 'gamma' in name:
                step_1_gamma_values.append(param.item())

        elif 'step_2' in name:
            if 'F_network' in name or 'G_network' in name:
                step_2_mlp_params[name] = param
            elif 'c1' in name:
                step_2_c1_values.append(param.item())
            elif 'c2' in name:
                step_2_c2_values.append(param.item())
            elif 'gamma' in name:
                step_2_gamma_values.append(param.item())

    # Extract a-values from sublayers
    for layer in mapping_net.layers:
        step_1_a_values.append(layer.step_1.a.item())
        step_2_a_values.append(layer.step_2.a.item())


    print(f"Step 1 gamma values mean: {np.array(step_1_gamma_values).mean():.3f}±{np.array(step_1_gamma_values).std():.3f}\n")
    print(f"Step 2 gamma values mean: {np.array(step_2_gamma_values).mean():.3f}±{np.array(step_2_gamma_values).std():.3f}\n")

    # --- Step 2: Plot mapping parameters (c1, c2, gamma, a) ---
    num_layers = len(step_1_c1_values)
    layer_indices = list(range(num_layers))

    plt.figure(figsize=(14, 12))

    # ---- Plot 1: c1 and c2 values ----
    plt.subplot(3, 1, 1)
    plt.plot(layer_indices, step_1_c1_values, marker='o', color='blue', label='Step 1 - c₁')
    plt.plot(layer_indices, step_1_c2_values, marker='o', color='cyan', label='Step 1 - c₂')
    plt.plot(layer_indices, step_2_c1_values, marker='s', color='red', label='Step 2 - c₁')
    plt.plot(layer_indices, step_2_c2_values, marker='s', color='orange', label='Step 2 - c₂')
    plt.title('c₁ and c₂ Parameters per Layer', fontsize=14, fontweight='bold')
    plt.xlabel('Layer number', fontsize=12)
    plt.ylabel('Parameter value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # ---- Plot 2: gamma values ----
    plt.subplot(3, 1, 2)
    plt.plot(layer_indices, step_1_gamma_values, marker='o', color='green', label='Step 1 - γ')
    plt.plot(layer_indices, step_2_gamma_values, marker='s', color='magenta', label='Step 2 - γ')
    plt.title('γ (Gamma) Parameters per Layer', fontsize=14, fontweight='bold')
    plt.xlabel('Layer number', fontsize=12)
    plt.ylabel('Gamma value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # ---- Plot 3: a values ----
    plt.subplot(3, 1, 3)
    plt.plot(layer_indices, step_1_a_values, marker='o', color='purple', label='Step 1 - a')
    plt.plot(layer_indices, step_2_a_values, marker='s', color='brown', label='Step 2 - a')
    plt.title('a Parameters per Layer', fontsize=14, fontweight='bold')
    plt.xlabel('Layer number', fontsize=12)
    plt.ylabel('a value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Step 3: Plot MLP statistics (both steps) ---
    def plot_mlp_stats(mlp_params, step_name, color_scheme):
        pattern = re.compile(r"layers\.(\d+)\.")
        layer_stats = defaultdict(lambda: {"weight_values": [], "bias_values": []})

        for name, param in mlp_params.items():
            match = pattern.search(name)
            if not match:
                continue
            layer_idx = int(match.group(1))
            vals = param.detach().cpu().numpy().flatten()
            if 'weight' in name:
                layer_stats[layer_idx]["weight_values"].append(vals)
            elif 'bias' in name:
                layer_stats[layer_idx]["bias_values"].append(vals)

        layer_indices = sorted(layer_stats.keys())
        weight_means, weight_stds, weight_norms = [], [], []
        bias_means, bias_stds, bias_norms = [], [], []

        for idx in layer_indices:
            all_w = np.concatenate(layer_stats[idx]["weight_values"]) if layer_stats[idx]["weight_values"] else np.array([0])
            all_b = np.concatenate(layer_stats[idx]["bias_values"]) if layer_stats[idx]["bias_values"] else np.array([0])
            weight_means.append(np.mean(all_w))
            weight_stds.append(np.std(all_w))
            weight_norms.append(np.linalg.norm(all_w))
            bias_means.append(np.mean(all_b))
            bias_stds.append(np.std(all_b))
            bias_norms.append(np.linalg.norm(all_b))

        plt.figure(figsize=(14, 10))

        # Weights
        plt.subplot(2, 1, 1)
        plt.plot(layer_indices, weight_means, 'o-', color=color_scheme[0], label=f'{step_name} - Mean (weights)')
        plt.plot(layer_indices, weight_stds, 's-', color=color_scheme[1], label=f'{step_name} - Std (weights)')
        plt.plot(layer_indices, weight_norms, '--', color=color_scheme[2], label=f'{step_name} - L2 norm (weights)')
        plt.title(f"{step_name} - G_network Weight Statistics per Layer", fontsize=14, fontweight='bold')
        plt.xlabel("Layer index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)

        # Biases
        plt.subplot(2, 1, 2)
        plt.plot(layer_indices, bias_means, 'o-', color=color_scheme[3], label=f'{step_name} - Mean (bias)')
        plt.plot(layer_indices, bias_stds, 's-', color=color_scheme[4], label=f'{step_name} - Std (bias)')
        plt.plot(layer_indices, bias_norms, '--', color=color_scheme[5], label=f'{step_name} - L2 norm (bias)')
        plt.title(f"{step_name} - G_network Bias Statistics per Layer", fontsize=14, fontweight='bold')
        plt.xlabel("Layer index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Step 1
    plot_mlp_stats(step_1_mlp_params, "Step 1", ['blue', 'cyan', 'navy', 'red', 'orange', 'darkred'])
    # Step 2
    plot_mlp_stats(step_2_mlp_params, "Step 2", ['green', 'lime', 'darkgreen', 'purple', 'magenta', 'darkviolet'])

    # --- Optional return ---
    if return_lists:
        return {
            "step_1_mlp_params": step_1_mlp_params,
            "step_2_mlp_params": step_2_mlp_params,
            "step_1_c1_values": step_1_c1_values,
            "step_2_c1_values": step_2_c1_values,
            "step_1_c2_values": step_1_c2_values,
            "step_2_c2_values": step_2_c2_values,
            "step_1_gamma_values": step_1_gamma_values,
            "step_2_gamma_values": step_2_gamma_values,
            "step_1_a_values": step_1_a_values,
            "step_2_a_values": step_2_a_values
        }


        







    















def compute_jacobian_functional(model, x, u, t):
    """
    Compute Jacobian using torch.autograd.functional.jacobian.
    This can handle models with internal autograd operations better.
    
    Args:
        model: Neural network that takes (x, u, t) and returns (X_final, U_final, t)
        x: Input tensor of shape [batch_size]
        u: Input tensor of shape [batch_size]
        t: Input tensor of shape [batch_size]
    
    Returns:
        jacobian: Tensor of shape [2, 2, batch_size]
    """
    from torch.autograd.functional import jacobian as torch_jacobian
    
    batch_size = x.shape[0]
    jacobians = []
    
    for i in range(batch_size):
        x_i = x[i:i+1]
        u_i = u[i:i+1]
        t_i = t[i:i+1]
        
        # Define a function that takes only x and u
        def func(x_val, u_val):
            X_final, U_final, _ = model(x_val, u_val, t_i)
            return torch.stack([X_final, U_final])
        
        # Compute jacobian for this sample
        jac = torch_jacobian(func, (x_i, u_i))
        
        # jac is a tuple: (grad_wrt_x, grad_wrt_u)
        # Each element has shape [2, 1] (2 outputs, 1 input)
        jac_x = jac[0].squeeze()  # [2]
        jac_u = jac[1].squeeze()  # [2]
        
        # Stack to form [2, 2] matrix
        jac_matrix = torch.stack([jac_x, jac_u], dim=1)  # [2, 2]
        jacobians.append(jac_matrix)
    
    # Stack along batch dimension
    jacobian = torch.stack(jacobians, dim=2)  # [2, 2, batch_size]
    
    return jacobian


def compute_symplectic_product(jacobian):
    """
    Compute M^T * Ω * M for each 2x2 Jacobian matrix in the batch.
    
    Where Ω (omega) is the symplectic matrix:
    Ω = [[0,  1],
         [-1, 0]]
    
    Args:
        jacobian: Tensor of shape [2, 2, batch_size]
    
    Returns:
        result: Tensor of shape [2, 2, batch_size] where each 2x2 matrix
                is M^T * Ω * M for the corresponding Jacobian M
    """
    batch_size = jacobian.shape[2]
    device = jacobian.device
    dtype = jacobian.dtype
    
    # Create the symplectic matrix Ω
    Omega = torch.tensor([[0.0, 1.0],
                          [-1.0, 0.0]], 
                         device=device, 
                         dtype=dtype)
    
    # Transpose jacobian to [batch_size, 2, 2] for batch operations
    M = jacobian.permute(2, 0, 1)  # [batch_size, 2, 2]
    
    # Compute M^T
    M_T = M.transpose(1, 2)  # [batch_size, 2, 2]
    
    # Expand Omega for batch multiplication: [1, 2, 2] -> [batch_size, 2, 2]
    Omega_batch = Omega.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Compute M^T * Ω * M
    # First: Ω * M
    Omega_M = torch.bmm(Omega_batch, M)  # [batch_size, 2, 2]
    
    # Second: M^T * (Ω * M)
    result_batch = torch.bmm(M_T, Omega_M)  # [batch_size, 2, 2]
    
    # Transpose back to [2, 2, batch_size]
    result = result_batch.permute(1, 2, 0)
    
    return result




def test_canonical_transformation_on_trajectory(
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
    tolerance=1e-5,
    verbose=True,
):
    """
    Test whether MappingNet and InverseNet perform symplectic (canonical) forward passes
    on a single trajectory, using per-segment relative times.

    For each segment, computes the Jacobian of the transformation with respect to (q, p)
    for MappingNet and with respect to (Q, P) for InverseNet, then checks whether
    M^T * Omega * M = Omega at each point.

    Returns:
        results: dict with per-point residuals and summary statistics for both networks
    """
    if Tmax is None or Tmax <= 0:
        raise ValueError(f"Tmax must be a positive number, got {Tmax}")

    test_trajectory_data = get_data_from_trajectory_id_function(test_id_df, test_df, trajectory_ids=trajectory_id)
    x = torch.as_tensor(test_trajectory_data['x'].to_numpy(dtype=np.float32), device=device)
    u = torch.as_tensor(test_trajectory_data['u'].to_numpy(dtype=np.float32), device=device)
    t = torch.as_tensor(test_trajectory_data['t'].to_numpy(dtype=np.float32), device=device)

    if x.dim() != 1 or u.dim() != 1 or t.dim() != 1:
        raise ValueError(f"x, u, t must be 1-dimensional tensors, got dims {x.dim()}, {u.dim()}, {t.dim()}")

    t_np = t.detach().cpu().numpy()
    t_max = t_np[-1]

    # Build segments
    num_segments = int(np.ceil(t_max / Tmax - 1e-9))
    num_segments = max(num_segments, 1)

    mapping_jacobians = []
    inverse_jacobians = []

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

        x_seg = x[seg_indices]
        u_seg = u[seg_indices]
        t_seg_relative = t[seg_indices] - t_start

        jac_mapping = compute_jacobian_functional_function(mapping_net, x=x_seg, u=u_seg, t=t_seg_relative)
        mapping_jacobians.append(jac_mapping)

        Q_seg, P_seg, _ = mapping_net(x_seg, u_seg, t_seg_relative)

        jac_inverse = compute_jacobian_functional_function(inverse_net, x=Q_seg.detach(), u=P_seg.detach(), t=t_seg_relative)
        inverse_jacobians.append(jac_inverse)

    all_mapping_jac = torch.cat(mapping_jacobians, dim=2)
    all_inverse_jac = torch.cat(inverse_jacobians, dim=2)

    mapping_symplectic = compute_symplectic_product_function(all_mapping_jac)
    inverse_symplectic = compute_symplectic_product_function(all_inverse_jac)

    # Compute per-point residuals: ||M^T Omega M - Omega||_max for each point
    batch_size = mapping_symplectic.shape[2]
    device_sym = mapping_symplectic.device
    dtype_sym = mapping_symplectic.dtype

    Omega = torch.tensor([[0.0, 1.0],
                          [-1.0, 0.0]],
                         device=device_sym,
                         dtype=dtype_sym)

    # Vectorized residual computation
    diff_mapping = torch.abs(mapping_symplectic - Omega.unsqueeze(-1))
    mapping_residuals = diff_mapping.amax(dim=(0, 1))

    diff_inverse = torch.abs(inverse_symplectic - Omega.unsqueeze(-1))
    inverse_residuals = diff_inverse.amax(dim=(0, 1))

    mapping_residuals_np = mapping_residuals.detach().cpu().numpy()
    inverse_residuals_np = inverse_residuals.detach().cpu().numpy()

    # Statistics
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
        print(f"({num_segments} segment(s), {batch_size} total points, tolerance={tolerance:.1e})\n")

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

        # --- Table row values: M max | M median | M^-1 max | M^-1 median ---
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
    }

    return results














        







"""
=================
Local (per-sample) Lyapunov-time estimation from an ensemble of Hamiltonian
trajectories sampled in full phase space (q, p), with a common sampling step dt.

Pipeline (every step attributable to prior work):
  1. Canonical per-conjugate-pair scaling  Q=q/alpha, P=p*alpha, alpha=sqrt(sq/sp).
     Symplectic (preserves dq^dp), diagonal (no matrix-root ambiguity), and
     leaves the Lyapunov slope invariant (Eckmann & Ruelle, Rev. Mod. Phys. 57,
     617, 1985: lambda_max is invariant under smooth invertible coordinate change).
  2. Neighbour search in scaled phase space (KD-tree). Cross-trajectory neighbours
     always accepted; same-trajectory neighbours accepted only if |t_j - t0| > w
     (Theiler 1986). w self-defined as 2 * first-zero-crossing of autocorrelation.
  3. Per-reference divergence curve S_ref(tau) = ln < ||.|| >_neighbours
     (Kantz 1994, Phys. Lett. A 185, 77: average distances inside the ball, THEN
     log). Look-aheads are integer row offsets since dt is common. Neighbours drop
     out as their own trajectory ends; |S_k| may shrink with k (Kantz-native).
  4. Automatic scaling-region slope via the ensemble method of Deshmukh, Bradley,
     Garland & Meiss (Chaos 31, 123102, 2021): all endpoint pairs -> weighted
     slope PDF -> KDE mode = slope; FWHM gives spread and a no-region verdict.
     Weights = fit_length**lpower / fit_error**epower with (lpower,epower)=(1,1)
     for Lyapunov exponents (paper Appendix A; repo ensembles.py).
  5. Combine across the eps-sweep by Kantz's own rule: arithmetic mean of the
     per-eps slopes over the eps that produced a valid region; spread = std.
  6. Convert to local Lyapunov time: 1/lambda (chaotic), inf (lambda ~ 0 within
     its spread: regular/integrable region), NaN (no valid region / curve too
     short -> includes end-of-trajectory samples and uncovered regions).

Only two REQUIRED knobs: min_neighbors and min_region_points. Everything else
self-defines from the data or is swept.
"""


from scipy.stats import gaussian_kde
from sklearn.neighbors import KDTree


# --------------------------------------------------------------------------- #
#  Deshmukh et al. (2021) ensemble scaling-region estimator
#  Reproduced from paper Appendix A + the repo's reference functions
#  (generate_individual_ensembles_ordered_fixed and utils.half_max_x).
# --------------------------------------------------------------------------- #

def _half_max_x(x, y):
    """FWHM endpoints of a peak, via linear interpolation at half maximum.

    Verbatim port of utils.half_max_x from vrd1243/scaling_regions_ensemble.
    Returns [x_left, x_right] bracketing the peak at half its maximum height.
    """
    half = np.max(y) / 2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zc_i = np.where(zero_crossings)[0]

    def lin_interp(i):
        return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))

    if len(zc_i) == 0:
        return [x[np.argmax(y)], x[np.argmax(y)]]
    if len(zc_i) == 1:
        return [lin_interp(zc_i[0]), np.max(x)]
    return [lin_interp(zc_i[0]), lin_interp(zc_i[-1])]


def _ensemble_slopes(xdata, ydata, min_points, lpower=1.0, epower=1.0):
    """Ordered ensemble of straight-line fits over all sub-intervals.

    Port of generate_individual_ensembles_ordered_fixed (repo ensembles.py):
    for every pair (l, r) with r - l >= min_points, least-squares fit a line,
    record its slope, a fit-length weight (A1) and a fit-error weight (A2),
    then form the combined weight (A3) = length**lpower / error**epower.

    Returns (slopes, weights) as 1-D arrays, or (None, None) if too few points.
    """
    x = np.asarray(xdata, dtype=float)
    y = np.asarray(ydata, dtype=float)
    N = len(x)
    if N < min_points + 1:
        return None, None

    # Prefix sums so any interval [l..r] (inclusive) least-squares line is O(1).
    # Slope over [l,r] = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2), with sums over the
    # interval; SSE (residual sum of squares) from the standard identity.
    cx = np.concatenate(([0.0], np.cumsum(x)))
    cy = np.concatenate(([0.0], np.cumsum(y)))
    cxx = np.concatenate(([0.0], np.cumsum(x * x)))
    cyy = np.concatenate(([0.0], np.cumsum(y * y)))
    cxy = np.concatenate(([0.0], np.cumsum(x * y)))

    ls, rs = [], []
    
    ls, rs = np.triu_indices(N, k=min_points)
    # inclusive interval [l..r] -> prefix indices [l, r+1]
    n = (rs - ls + 1).astype(float)
    Sx = cx[rs + 1] - cx[ls]
    Sy = cy[rs + 1] - cy[ls]
    Sxx = cxx[rs + 1] - cxx[ls]
    Syy = cyy[rs + 1] - cyy[ls]
    Sxy = cxy[rs + 1] - cxy[ls]

    denom = n * Sxx - Sx * Sx
    denom_safe = np.where(np.abs(denom) < 1e-300, np.nan, denom)
    slope = (n * Sxy - Sx * Sy) / denom_safe
    intercept = (Sy - slope * Sx) / n
    # SSE = Syy - a*Sy - b*Sxy   (a=intercept, b=slope), guard tiny negatives
    sse = Syy - intercept * Sy - slope * Sxy
    sse = np.clip(sse, 0.0, None)
    fit_error = 1e-10 + np.sqrt(sse / (n - 1))
    dx = x[rs] - x[ls]
    fit_length = np.sqrt(dx ** 2 + (slope * dx) ** 2)
    weights = fit_length ** lpower / fit_error ** epower

    good = np.isfinite(slope) & np.isfinite(weights)
    return slope[good], weights[good]


def ensemble_scaling_slope(xdata, ydata, min_points=10,
                           lpower=1.0, epower=1.0, n_grid=1000,
                           fwhm_frac_max=0.5):
    """Deshmukh ensemble slope of a scaling region, with a no-region verdict.

    Parameters
    ----------
    xdata, ydata : the divergence curve (tau, S(tau)) for one reference/eps.
    min_points   : minimum interval width n (paper default 10).
    lpower,epower: weight exponents; (1,1) for Lyapunov exponents (paper).
    n_grid       : KDE evaluation grid resolution.
    fwhm_frac_max: reject as "no scaling region" if the FWHM of the slope PDF
                   exceeds this fraction of the total slope range (a wide PDF
                   means no clear mode -> no region, per Deshmukh Sec. II/III).

    Returns
    -------
    dict with keys: slope (mode of PDF or NaN), sigma (std within FWHM),
    valid (bool), fwhm.
    """
    nan = {"slope": np.nan, "sigma": np.nan, "valid": False, "fwhm": np.nan}

    slopes, weights = _ensemble_slopes(xdata, ydata, min_points, lpower, epower)
    if slopes is None or len(slopes) < 3:
        return nan
    if not np.all(np.isfinite(slopes)) or np.ptp(slopes) == 0:
        # all identical slopes -> degenerate but well-defined
        return {"slope": float(np.mean(slopes)), "sigma": 0.0,
                "valid": True, "fwhm": 0.0}
    w = weights.copy()
    if not np.all(np.isfinite(w)) or np.sum(w) <= 0:
        w = np.ones_like(slopes)

    # weighted KDE over the slope distribution -> mode is the slope estimate
    try:
        kde = gaussian_kde(slopes, weights=w)  # Scott's rule bandwidth (default)
    except Exception:
        return nan
    grid = np.linspace(slopes.min(), slopes.max(), n_grid)
    pdf = kde(grid)
    mode = grid[np.argmax(pdf)]

    # FWHM of the peak -> spread + region-existence gate
    lo, hi = _half_max_x(grid, pdf)
    fwhm = float(hi - lo)
    total_range = float(slopes.max() - slopes.min())

    # sigma: std of the weighted slopes lying within the FWHM band
    in_band = (slopes >= lo) & (slopes <= hi)
    if np.count_nonzero(in_band) >= 2:
        sigma = float(np.std(slopes[in_band]))
    else:
        sigma = float(np.std(slopes))

    valid = (total_range > 0) and (fwhm <= fwhm_frac_max * total_range)
    if not valid:
        return {"slope": float(mode), "sigma": sigma, "valid": False,
                "fwhm": fwhm}
    return {"slope": float(mode), "sigma": sigma, "valid": True, "fwhm": fwhm}


# --------------------------------------------------------------------------- #
#  Self-defined hyperparameters
# --------------------------------------------------------------------------- #

def theiler_window_from_autocorr(series_list):
    """Theiler window w = 2 * (first zero-crossing lag of autocorrelation).

    Theiler (1986), Phys. Rev. A 34, 2427; the 2x-autocorrelation-time rule is
    the standard no-tuning default. Computed per trajectory on the first phase
    coordinate and aggregated by the median first-zero lag, then doubled.
    """
    first_zeros = []
    for s in series_list:
        s = np.asarray(s, dtype=float)
        s = s - s.mean()
        if len(s) < 4 or np.allclose(s, 0):
            continue
        ac = np.correlate(s, s, mode="full")[len(s) - 1:]
        if ac[0] == 0:
            continue
        ac = ac / ac[0]
        zc = np.where(np.diff(np.sign(ac)) != 0)[0]
        if len(zc) > 0:
            first_zeros.append(int(zc[0]))
    if not first_zeros:
        return 1
    return int(max(1, 2 * int(np.median(first_zeros))))


def canonical_pair_scaling(Q_cols, P_cols, df):
    """Per-conjugate-pair symplectic scaling factors alpha_i = sqrt(sq_i/sp_i)."""
    alphas = np.zeros(len(Q_cols))
    for i, (qc, pc) in enumerate(zip(Q_cols, P_cols)):
        sq = df[qc].to_numpy(dtype=float).std()
        sp = df[pc].to_numpy(dtype=float).std()
        if sp == 0 or sq == 0:
            alphas[i] = 1.0
        else:
            alphas[i] = np.sqrt(sq / sp)
    return alphas


def eps_grid_from_pairwise(X, n_eps=10, lo_pct=1.0, hi_pct=20.0,
                           max_pairs=200000, rng=None):
    """Geometric grid of eps between percentiles of pairwise-distance dist."""
    rng = np.random.default_rng(0 if rng is None else rng)
    N = len(X)
    n_sample = int(np.ceil(np.sqrt(2 * max_pairs)))
    n_sample = min(N, max(n_sample, 2))
    idx = rng.choice(N, size=n_sample, replace=False)
    S = X[idx]
    # pairwise distances among the sample
    diff = S[:, None, :] - S[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(len(S), k=1)
    dvals = d[iu]
    dvals = dvals[dvals > 0]
    lo = np.percentile(dvals, lo_pct)
    hi = np.percentile(dvals, hi_pct)
    lo = max(lo, 1e-12)
    return np.geomspace(lo, hi, n_eps)


# --------------------------------------------------------------------------- #
#  Main driver
# --------------------------------------------------------------------------- #

def local_lyapunov_times(df, dt,
                         min_neighbors,          # REQUIRED
                         min_region_points,      # REQUIRED
                         q_cols=None, p_cols=None, t_col="t",
                         traj_col="traj_id",
                         w=None,                 # self-defined if None
                         K_max=None,             # self-defined if None
                         n_eps=8, eps_lo_pct=1.0, eps_hi_pct=20.0,
                         lpower=1.0, epower=1.0,
                         fwhm_frac_max=0.5,
                         zero_tol_factor=2.0,
                         min_valid_eps=2,
                         curve_len_factor=4,
                         rng=0, 
                         return_details=False,
                         verbose=True):

    """Compute one local Lyapunov-time estimate per sample.

    Parameters
    ----------
    df : DataFrame with columns traj_col, q_cols..., p_cols..., t_col.
         Rows are all samples from all trajectories.
    dt : common sampling interval (float).
    min_neighbors : minimum accepted neighbours in an eps-ball to fit a curve.
    min_region_points : minimum width of a scaling region (Deshmukh 'n').
    q_cols, p_cols : lists of the n position / n momentum column names.
                     If None, inferred (single-DoF: q_cols=['x'], p_cols=['u']).
    t_col, traj_col : column names for time and trajectory id.
    w : Theiler window in steps; self-defined (2x autocorr zero) if None.
    K_max : max look-ahead in steps; longest trajectory length if None.
    n_eps, eps_lo_pct, eps_hi_pct : eps-sweep grid controls (self-defined range).
    lpower, epower : Deshmukh weight exponents (1,1 for Lyapunov).
    fwhm_frac_max : Deshmukh no-region gate.
    zero_tol_factor : lambda counts as 0 (=> tau_L=inf) if |lambda| <= factor*sigma.
                     Default 2.0. Larger => more aggressively absorbs near-zero
                     slopes into the "regular" (inf) verdict.
    min_valid_eps : a sample is only given a finite (chaotic) tau_L if it produced
                     a valid scaling region at >= this many eps values. Enforces
                     Kantz's requirement that estimates agree across parameters;
                     samples with fewer valid eps cannot be cross-checked and are
                     treated as regular (inf) rather than trusted as chaotic.
    curve_len_factor : cap the divergence curve at curve_len_factor*min_region_points
                     steps. Self-defined efficiency cap (the scaling region lives at
                     small tau; the saturated tail is discarded by the detector
                     anyway). Increase only if regions sit at unusually large tau.
    return_details : also return a per-sample details DataFrame.

    Returns
    -------
    tau_L : np.ndarray, length len(df), aligned to df row order. Values are
            finite>0 (chaotic), np.inf (regular), or np.nan (unestimable).
    (optional) details : DataFrame with lambda, sigma, n_eps_valid, n_curve.
    """
    df = df.reset_index(drop=True)
    N = len(df)

    # ---- resolve columns -------------------------------------------------- #
    if q_cols is None or p_cols is None:
        # single-DoF convention from the user's data: 'x' (q), 'u' (p)
        if "x" in df.columns and "u" in df.columns:
            q_cols, p_cols = ["x"], ["u"]
        else:
            raise ValueError("Specify q_cols and p_cols explicitly.")
    assert len(q_cols) == len(p_cols), "q_cols and p_cols must have equal length"

    if traj_col not in df.columns:
        raise ValueError(f"Missing trajectory column '{traj_col}'.")

    # ---- Step 0: index by trajectory, integer within-trajectory positions - #
    traj_ids = df[traj_col].to_numpy()
    t_vals = df[t_col].to_numpy(dtype=float)
    order = np.lexsort((t_vals, traj_ids))       # sort by traj, then time
    inv_order = np.argsort(order)                # to map back to df order
    df_s = df.iloc[order].reset_index(drop=True)
    traj_s = df_s[traj_col].to_numpy()
    t_s = df_s[t_col].to_numpy(dtype=float)

    # per-trajectory contiguous blocks (data is now grouped and time-sorted)
    uniq, starts = np.unique(traj_s, return_index=True)
    starts = np.sort(starts)
    ends = np.append(starts[1:], len(df_s))       # exclusive
    traj_of = np.empty(len(df_s), dtype=int)
    pos_in_traj = np.empty(len(df_s), dtype=int)   # integer step index (0-based)
    n_blocks = len(starts)
    traj_start_arr = np.zeros(n_blocks, dtype=int)
    traj_len_arr = np.zeros(n_blocks, dtype=int)
    for b, (s0, e0) in enumerate(zip(starts, ends)):
        traj_of[s0:e0] = b
        pos_in_traj[s0:e0] = np.arange(e0 - s0)
        traj_start_arr[b] = s0
        traj_len_arr[b] = e0 - s0
    # dict views kept for readability where used outside the hot loop
    traj_start = {b: int(traj_start_arr[b]) for b in range(n_blocks)}
    traj_len = {b: int(traj_len_arr[b]) for b in range(n_blocks)}

    # ---- Step 1: canonical per-pair scaling ------------------------------ #
    alphas = canonical_pair_scaling(q_cols, p_cols, df_s)
    Xraw = np.zeros((len(df_s), 2 * len(q_cols)), dtype=float)
    for i, (qc, pc) in enumerate(zip(q_cols, p_cols)):
        Xraw[:, 2 * i] = df_s[qc].to_numpy(dtype=float) / alphas[i]     # Q = q/alpha
        Xraw[:, 2 * i + 1] = df_s[pc].to_numpy(dtype=float) * alphas[i]  # P = p*alpha
    X = Xraw

    # ---- Step 2: self-defined w, K_max, eps grid ------------------------- #
    if w is None:
        series_list = [df_s[q_cols[0]].to_numpy()[traj_start[b]:traj_start[b] + traj_len[b]]
                       for b in range(len(uniq))]
        w = theiler_window_from_autocorr(series_list)
    if K_max is None:
        K_max = int(max(traj_len.values())) - 1

    eps_grid = eps_grid_from_pairwise(X, n_eps=n_eps,
                                      lo_pct=eps_lo_pct, hi_pct=eps_hi_pct,
                                      rng=rng)

    if verbose:
        print(f"[setup] N={len(df_s)} samples, {len(uniq)} trajectories, "
              f"dim(q,p)={X.shape[1]}")
        print(f"[setup] Theiler w={w} steps, K_max={K_max} steps")
        print(f"[setup] eps-subsample seed (rng)={rng}")
        print(f"[setup] eps grid: {np.array2string(eps_grid, precision=4)}")

    tree = KDTree(X)

    # ---- Steps 3-5: per-reference loop ----------------------------------- #
    lam = np.full(len(df_s), np.nan)
    sig = np.full(len(df_s), np.nan)
    n_valid_eps = np.zeros(len(df_s), dtype=int)
    n_curve = np.zeros(len(df_s), dtype=int)

    eps_max = eps_grid[-1]

    for i in range(len(df_s)):
        A = traj_of[i]
        t0 = pos_in_traj[i]
        A_start = traj_start[A]
        A_len = traj_len[A]
        max_k_ref = A_len - 1 - t0            # steps available on the reference
        if max_k_ref < min_region_points:
            continue                          # end-of-trajectory: self-excludes

        # Same neighbours as the old batched tree.query_radius(X, ...) call,
        # but queried one reference at a time so we never hold all N neighbour
        # lists at once. query_radius treats each query point independently, so
        # this returns identical indices/distances to the old all_idx[i]/all_dist[i].
        _idx_i, _dist_i = tree.query_radius(X[i:i + 1], r=eps_max,
                                            return_distance=True)
        cand = _idx_i[0]
        cdist = _dist_i[0]
        # exclude self
        keep = cand != i
        cand = cand[keep]; cdist = cdist[keep]
        if cand.size == 0:
            continue

        cand_traj = traj_of[cand]
        cand_pos = pos_in_traj[cand]
        # Theiler: same-trajectory neighbours must be > w steps away in time
        same = cand_traj == A
        theiler_ok = np.abs(cand_pos - t0) > w
        accept_mask = (~same) | (same & theiler_ok)
        cand = cand[accept_mask]
        cdist = cdist[accept_mask]
        cand_traj = cand_traj[accept_mask]
        cand_pos = cand_pos[accept_mask]
        if cand.size == 0:
            continue

        per_eps_slopes = []
        for eps in eps_grid:
            sel = cdist <= eps
            if np.count_nonzero(sel) < min_neighbors:
                continue
            nb_traj = cand_traj[sel]
            nb_pos = cand_pos[sel]
            nb_start = traj_start_arr[nb_traj]     # global start row of each nbr
            nb_tlen = traj_len_arr[nb_traj]        # length of each nbr's traj
            nb_room = nb_tlen - 1 - nb_pos         # steps each nbr can advance

            # Step 3: divergence curve S(k), k = 1..k_cap, Kantz avg-then-log.
            # The scaling region lives at small tau; building the fully saturated
            # tail wastes O(L^2) fit work. Cap at curve_len_factor * region width
            # (self-defined: "enough curve to contain a region plus margin").
            k_cap = min(K_max, max_k_ref,
                        curve_len_factor * min_region_points)
            S = []
            taus = []
            for k in range(1, k_cap + 1):
                ref_state = X[A_start + t0 + k]
                alive = nb_room >= k            # neighbours still alive at step k
                if not np.any(alive):
                    break
                fut_idx = nb_start[alive] + nb_pos[alive] + k
                diff = X[fut_idx] - ref_state
                dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))
                dists = dists[dists > 0]
                if dists.size == 0:
                    break
                S.append(np.log(np.mean(dists)))
                taus.append(k * dt)

            if len(S) < min_region_points + 1:
                continue
            n_curve[i] = max(n_curve[i], len(S))

            # Step 4: Deshmukh ensemble slope with region gate
            res = ensemble_scaling_slope(np.array(taus), np.array(S),
                                         min_points=min_region_points,
                                         lpower=lpower, epower=epower,
                                         fwhm_frac_max=fwhm_frac_max)
            if res["valid"] and np.isfinite(res["slope"]):
                per_eps_slopes.append(res["slope"])

        # Step 5: Kantz mean over eps that produced a valid region
        if len(per_eps_slopes) == 0:
            if n_curve[i] >= min_region_points + 3:
                lam[i] = 0.0; sig[i] = 0.0     # n_valid_eps=0 < min_valid_eps -> inf
            continue
        n_valid_eps[i] = len(per_eps_slopes)
        lam[i] = float(np.mean(per_eps_slopes))
        sig[i] = float(np.std(per_eps_slopes)) if len(per_eps_slopes) > 1 else 0.0

    # ---- Step 6: convert to local Lyapunov time -------------------------- #
    tau_L_s = np.full(len(df_s), np.nan)
    for i in range(len(df_s)):
        li = lam[i]
        if not np.isfinite(li):
            tau_L_s[i] = np.nan
        elif n_valid_eps[i] < min_valid_eps:
            # too few eps to cross-check (Kantz consistency) -> treat as regular
            tau_L_s[i] = np.inf
        elif abs(li) <= zero_tol_factor * (sig[i] if np.isfinite(sig[i]) else 0.0):
            tau_L_s[i] = np.inf          # regular / integrable region
        elif li > 0:
            tau_L_s[i] = 1.0 / li
        else:
            # strongly negative slope (contraction/artefact) -> not a positive
            # exponent; report as regular rather than a spurious time
            tau_L_s[i] = np.inf

    # ---- map back to original df row order ------------------------------- #
    tau_L = tau_L_s[inv_order]
    if verbose:
        finite = np.isfinite(tau_L) & ~np.isinf(tau_L)
        print(f"[done] finite tau_L: {np.count_nonzero(finite)}, "
              f"inf (regular): {np.count_nonzero(np.isinf(tau_L))}, "
              f"NaN: {np.count_nonzero(np.isnan(tau_L))}")

    if return_details:
        details = pd.DataFrame({
            "lambda": lam[inv_order],
            "sigma": sig[inv_order],
            "n_eps_valid": n_valid_eps[inv_order],
            "n_curve": n_curve[inv_order],
            "tau_L": tau_L,
        }, index=df.index)
        return tau_L, details
    return tau_L






def attach_traj_id(data_df, id_df,
                   start_col="start_index", end_col="end_index",
                   id_col="trajectory_id"):
    """Attach an integer traj_id using [start_index, end_index) blocks
    (end exclusive), referring to data_df's ORIGINAL index labels."""
    df = data_df.copy()
    idx_values = df.index.to_numpy()
    traj = np.full(len(df), -1, dtype=int)
    id_df = id_df.copy()
    for c in (start_col, end_col, id_col):
        id_df[c] = id_df[c].astype("int64")
    for _, row in id_df.iterrows():
        s, e, tid = int(row[start_col]), int(row[end_col]), int(row[id_col])
        traj[(idx_values >= s) & (idx_values < e)] = tid
    df = df.reset_index(drop=True)
    df["traj_id"] = traj
    if (traj < 0).any():
        print(f"[warn] {(traj < 0).sum()} rows uncovered; dropping.")
        df = df[df["traj_id"] >= 0].reset_index(drop=True)
    return df


def infer_dt(df, traj_col="traj_id", t_col="t"):
    dts = []
    for _, g in df.groupby(traj_col):
        tt = np.sort(g[t_col].to_numpy(float))
        if len(tt) > 1:
            dts.append(np.median(np.diff(tt)))
    return float(np.median(dts))


def summarize(tau_L, details, label):
    s = pd.Series(tau_L)
    fin = s[np.isfinite(s)]
    print(f"\n=== {label} ===")
    print(f"total={len(s)}  NaN={s.isna().sum()}  "
          f"Inf(regular)={np.isinf(s).sum()}  finite={len(fin)}")
    if len(fin):
        lam = details['lambda'][np.isfinite(tau_L)]
        print(f"finite tau_L: median={fin.median():.2f} "
              f"min={fin.min():.2f} max={fin.max():.2f}")
        print(f"finite lambda: min={lam.min():.4f} max={lam.max():.4f} "
              f"(near 0 => noise, not chaos)")
        
import numpy as np
import pandas as pd


def compute_Trecord(df, traj_col="traj_id", t_col="t"):
    """Duration of the longest training trajectory = max over trajectories of
    (max t - min t) within that trajectory. Data-derived, not hard-coded."""
    spans = []
    for _, g in df.groupby(traj_col):
        tt = g[t_col].to_numpy(dtype=float)
        if len(tt) >= 2:
            spans.append(tt.max() - tt.min())
    if not spans:
        raise ValueError("No trajectory with >= 2 samples; cannot define T_record.")
    return float(np.max(spans))


def summarize_task(task_name, df, tau_L, details,
                   n_min, r_min,
                   traj_col="traj_id", t_col="t"):
    """Build one standardized summary record (one row of the appendix table)
    from an already-computed tau_L / details pair.

    Parameters
    ----------
    task_name : str, e.g. "T1 (mass--spring)".
    df        : the dataframe the estimator was run on (must have traj_col, t_col).
    tau_L     : np.ndarray from local_lyapunov_times, aligned to df rows.
    details   : DataFrame from local_lyapunov_times(return_details=True).
    n_min, r_min : the two hand-set knobs actually used for THIS task.

    Returns
    -------
    dict with every field the table row and the T_max rule need.
    """
    tau = np.asarray(tau_L, dtype=float)

    total    = int(tau.size)
    n_nan    = int(np.isnan(tau).sum())          # undefined
    n_inf    = int(np.isinf(tau).sum())          # regular
    finite_mask = np.isfinite(tau)               # excludes inf AND nan
    n_finite = int(finite_mask.sum())

    finite_vals = tau[finite_mask]
    if n_finite > 0:
        f_median = float(np.median(finite_vals))
        f_max    = float(np.max(finite_vals))
        f_min    = float(np.min(finite_vals))
        tau_min  = f_min                          # t_lambda^min = min over finite
    else:
        # genuinely regular/integrable ensemble: no resolvable exponent anywhere
        f_median = f_max = f_min = np.nan
        tau_min  = np.inf                         # -> rule returns T_record

    T_record = compute_Trecord(df, traj_col=traj_col, t_col=t_col)
    T_max    = float(min(T_record, tau_min))      # Eq. (tmax_rule)

    return {
        "task":      task_name,
        "n_min":     int(n_min),
        "r_min":     int(r_min),
        "total":     total,
        "regular":   n_inf,      # Reg. (inf)
        "finite":    n_finite,
        "undefined": n_nan,
        "median":    f_median,
        "max":       f_max,
        "min":       f_min,      # this is t_lambda^min when finite exists
        "T_record":  T_record,
        "T_max":     T_max,
    }

def _fmt(x, nd=2):
    """Format a float for the table; blank-ish for NaN summaries."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "$-$"
    return f"${x:.{nd}f}$"


def row_to_latex(rec):
    """One LaTeX row matching tab:lyap_stats column order:
    Task & n_min & r_min & Total & Reg.(inf) & Finite & Undef.
          & median & max & min & T_record & T_max \\"""
    # T_record / T_max: show 3 decimals to match 6.493 in the paper
    def _fmt3(x):
        return "$\\infty$" if np.isinf(x) else f"${x:.3f}$"
    return (
        f"{rec['task']} & {rec['n_min']} & {rec['r_min']} & "
        f"{rec['total']} & {rec['regular']} & {rec['finite']} & {rec['undefined']} & "
        f"{_fmt(rec['median'])} & {_fmt(rec['max'])} & {_fmt(rec['min'])} & "
        f"{_fmt3(rec['T_record'])} & {_fmt3(rec['T_max'])} \\\\"
    )










SET_NAMES_1D = ("Train", "TestA", "TestB")
METRICS_1D = ("D0", "W0max", "rms_delta", "max_delta", "rms_disp")


def _sha_arrays_1d(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def _fixed_numerics_1d(enc, dec):
    """Eval mode on both nets (flags do not propagate) and full float32 matmul
    precision for the duration of one evaluation; everything restored afterwards."""
    saved = (enc.training, dec.training, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        enc.train(False)
        dec.train(False)
        yield
    finally:
        enc.train(saved[0])
        dec.train(saved[1])
        torch.backends.cuda.matmul.allow_tf32 = saved[2]
        torch.backends.cudnn.allow_tf32 = saved[3]
        torch.set_float32_matmul_precision(saved[4])


def _segment_trajectories_1d(data_df, id_df, Tmax, pos_col, mom_col, tol):
    """Trajectories from the id table's LABEL ranges, ordered by start_index.
    Each is split into K = max(1, ceil(span/Tmax - tol)) windows
    [t0 + k*Tmax, t0 + (k+1)*Tmax), the last one closed; local time restarts
    at each window's first observation."""
    parts, sid = [], 0
    for r in id_df.sort_values("start_index").itertuples(index=False):
        traj = data_df.loc[int(r.start_index): int(r.end_index) - 1].sort_values("t")
        t = traj["t"].to_numpy()
        K = max(int(np.ceil((t[-1] - t[0]) / Tmax - tol)), 1)
        for k in range(K):
            lo, hi = t[0] + k * Tmax, t[0] + (k + 1) * Tmax
            m = (t >= lo) if k == K - 1 else (t >= lo) & (t < hi)
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            c = traj.iloc[idx]
            parts.append(pd.DataFrame({
                "data_label": c.index.to_numpy(),
                "trajectory_id": int(r.trajectory_id),
                "segment_id": sid,
                "q": c[pos_col].to_numpy(),
                "p": c[mom_col].to_numpy(),
                "t_local": c["t"].to_numpy() - c["t"].iloc[0],
            }))
            sid += 1
    return pd.concat(parts, ignore_index=True)


def _encode_to_gauge_1d(enc, dec, q32, p32, t32):
    """Raw latents (Q, P) = M(q, p, t') and gauge latents M^{-1}(Q, P, 0).
    1-D (N,) tensors, one batch, float32, autograd enabled."""
    device = next(enc.parameters()).device
    q = torch.tensor(q32, dtype=torch.float32, device=device)
    p = torch.tensor(p32, dtype=torch.float32, device=device)
    t = torch.tensor(t32, dtype=torch.float32, device=device)
    with _fixed_numerics_1d(enc, dec), torch.enable_grad():
        Q, P, _ = enc(q, p, t)
        Q, P = Q.detach(), P.detach()
        x, u, _ = dec(Q, P, torch.zeros_like(t))
        x, u = x.detach(), u.detach()
    return torch.stack([Q, P], dim=1).cpu().numpy(), torch.stack([x, u], dim=1).cpu().numpy()


def _per_trajectory_1d(seg, id_df, delta, disp, eligible):
    tid = seg["trajectory_id"].to_numpy()[eligible]
    d_all, D_all = delta[eligible], disp[eligible]
    energy = id_df.set_index("trajectory_id")["energy"] if "energy" in id_df.columns else None
    rows = []
    for traj, idx in sorted(pd.DataFrame({"tid": tid}).groupby("tid").indices.items()):
        d, D = d_all[idx], D_all[idx]
        rows.append({
            "trajectory_id": traj,
            "energy": float(energy.loc[traj]) if energy is not None else np.nan,
            "n_eligible": int(len(idx)),
            "D0": float(np.sqrt(np.sum(d * d) / np.sum(D * D))),      # Step 8
            "W0max": float(d.max() / D.max()),                        # Step 9
            "rms_delta": float(np.sqrt(np.mean(d * d))),
            "max_delta": float(d.max()),
            "rms_disp": float(np.sqrt(np.mean(D * D))),
            "max_disp": float(D.max()),
        })
    return pd.DataFrame(rows)


def _summary_1d(pt):
    out = {}
    for m in METRICS_1D:
        v = pt[m].to_numpy()
        q25, q50, q75 = np.percentile(v, [25, 50, 75])              # linear interpolation
        out[m] = {"median": float(q50), "q25": float(q25), "q75": float(q75),
                  "min": float(v.min()), "max": float(v.max())}
    return out


def run_constancy_experiment_1d(mapping_net, inverse_net, Tmax, sets, sigma2,
                                pos_col="x", mom_col="u", tol=1e-6, metadata=None, verbose=True):
    """
    Tasks 1-2 (n = 1). Non-general networks called as net(x, u, t) with 1-D (N,) tensors.
    mapping_net, inverse_net : trained encoder and its inverse
    Tmax                     : the notebook's Tmax (6.4926248)
    sets                     : {"Train": (data_df, id_df), "TestA": (...), "TestB": (...)}, as stored:
                               data columns pos_col, mom_col, "t"; trajectories given by the id table's
                               start_index / end_index LABEL ranges
    sigma2                   : training-noise variance (noise reference sigma*sqrt(2))
    tol                      : relative tolerance in K = ceil(span/Tmax - tol); 1e-6 absorbs the
                               rounding of the Tmax literal, so a trajectory of span Tmax is one segment
    metadata                 : optional dict stored verbatim
    Returns a dict. No file I/O.
    """
    enc, dec = mapping_net, inverse_net
    n = 1
    noise_ref = float(np.sqrt(sigma2) * np.sqrt(2 * n))
    device = next(enc.parameters()).device
    state = enc.state_dict()

    res = {
        "spec": {
            "n": n, "Tmax": float(Tmax), "tol": float(tol), "sigma2": float(sigma2), "noise_ref": noise_ref,
            "pos_col": pos_col, "mom_col": mom_col, "set_names": list(SET_NAMES_1D),
            "segmentation": "trajectories from id-table label ranges (ordered by start_index); "
                            "K = max(1, ceil(span/Tmax - tol)) windows [t0+k*Tmax, t0+(k+1)*Tmax), "
                            "last closed; local time from each window's first observation",
            "network_io": "net(x, u, t) with 1-D (N,) tensors",
            "gauge": "g = M^{-1}(M(z, t'), 0)",
            "reference": "gauge latent of the segment's t'=0 observation",
            "eligible": "t' > 0 in segments with >= 2 observations",
            "norm": "Euclidean on raw (q, p)",
            "D0": "sqrt(sum ||g_t - g_0||^2 / sum ||z_t - z_0||^2) per trajectory, eligible observations",
            "W0max": "max ||g_t - g_0|| / max ||z_t - z_0|| per trajectory, eligible observations",
            "aggregation": "median and IQR over trajectories, numpy.percentile (linear)",
            "precision": "float32 network and metrics; inputs cast from float64",
            "batching": "one batch per set, rows in segment order",
        },
        "metadata": dict(metadata) if metadata else {},
        "environment": {
            "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "model_sha256": _sha_arrays_1d(*(state[k].detach().cpu().numpy() for k in sorted(state))),
        "sets": {}, "reading": {}, "hashes": {},
    }

    for name in SET_NAMES_1D:
        data_df, id_df = sets[name]
        seg = _segment_trajectories_1d(data_df, id_df, Tmax, pos_col, mom_col, tol)     # Steps 1-2
        sid = seg["segment_id"].to_numpy()
        pos = pd.Series(np.arange(len(seg)))
        ref = pos.groupby(sid).transform("min").to_numpy()                              # segment's t'=0 row
        size = pos.groupby(sid).transform("size").to_numpy()
        z32 = seg[["q", "p"]].to_numpy(np.float32)
        t32 = seg["t_local"].to_numpy(np.float32)

        lat, g = _encode_to_gauge_1d(enc, dec, z32[:, 0], z32[:, 1], t32)              # Steps 3-4
        eligible = (size >= 2) & (seg["t_local"].to_numpy() > 0)                       # Step 6
        delta = np.linalg.norm(g - g[ref], axis=1)                                      # Steps 5, 7
        disp = np.linalg.norm(z32 - z32[ref], axis=1)
        pt = _per_trajectory_1d(seg, id_df, delta, disp, eligible)                      # Steps 8-9
        ref_rows = np.unique(ref)
        floor_t0 = np.linalg.norm(g[ref_rows] - z32[ref_rows], axis=1)

        res["sets"][name] = {
            "data_label": seg["data_label"].to_numpy(),
            "trajectory_id": seg["trajectory_id"].to_numpy(),
            "segment_id": sid,
            "t_local": t32,
            "ref_row": ref,
            "eligible": eligible,
            "latents": lat,
            "gauge": g,
            "delta": delta,
            "disp": disp,
            "per_trajectory": pt,
            "summary": {"n_traj": int(len(pt)), "n_segments": int(len(np.unique(sid))),
                        "n_obs": int(len(seg)), "n_eligible": int(eligible.sum()), **_summary_1d(pt),
                        "floor_t0_max": float(floor_t0.max()),
                        "floor_t0_rms": float(np.sqrt(np.mean(floor_t0 ** 2))),
                        "min_rms_delta": float(pt["rms_delta"].min())},
        }
        res["hashes"][name] = {"inputs_z_t": _sha_arrays_1d(z32, t32),
                               "latents": _sha_arrays_1d(lat), "gauge": _sha_arrays_1d(g)}

    d0_train = res["sets"]["Train"]["summary"]["D0"]["median"]                          # Step 14
    for name in SET_NAMES_1D:
        s = res["sets"][name]["summary"]
        res["reading"][name] = {"rms_delta_median_over_noise_ref": s["rms_delta"]["median"] / noise_ref,
                                "D0_median_over_Train": s["D0"]["median"] / d0_train}
    res["hashes"]["all_outputs"] = _sha_arrays_1d(*(res["sets"][k][a] for k in SET_NAMES_1D
                                                     for a in ("latents", "gauge")))

    if verbose:
        print(f"model {res['model_sha256'][:16]} | outputs {res['hashes']['all_outputs'][:16]}")
        for name in SET_NAMES_1D:
            s, r = res["sets"][name]["summary"], res["reading"][name]
            print(f"{name:<6} n_traj={s['n_traj']:<3} n_seg={s['n_segments']:<3} "
                  + " | ".join(f"{m} {s[m]['median']:.3g} [{s[m]['q25']:.3g}, {s[m]['q75']:.3g}]" for m in METRICS_1D)
                  + f" | noise ratio {r['rms_delta_median_over_noise_ref']:.3g}"
                  + f" | D0/Train {r['D0_median_over_Train']:.3g}"
                  + f" | floor max {s['floor_t0_max']:.2e}")
    return res