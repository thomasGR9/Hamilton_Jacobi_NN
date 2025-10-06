import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.model_selection import train_test_split
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
        
        # Initialization parameters
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        
        # Architectural choices
        use_bias: bool = True,
        
        # Input/Output parameters
        input_dim: int = 2,  # x and t
        output_dim: int = 1,  # scalar G
        
        # Trainable parameter constraints
        a_eps_min: float = 0.1,  # Minimum value for a
        a_eps_max: float = 10.0,  # Maximum value for a  
        a_k: float = 0.1,  # Sigmoid steepness
    ):
        """
        Initialize the velocity transformation layer with trainable parameters.
        """
        super(Step_1, self).__init__()
        
        # Initialize trainable parameters
        self.a_raw = nn.Parameter(torch.tensor(0.0))  
        self.c1 = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.c2 = nn.Parameter(torch.tensor(0.0))
        
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
        final_activation, use_bias
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
        
        # Initialization parameters
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        
        # Architectural choices
        use_bias: bool = True,
        
        # Input/Output parameters
        input_dim: int = 2,  # u and t
        output_dim: int = 1,  # scalar F
        
        # Trainable parameter constraints
        a_eps_min: float = 0.1,
        a_eps_max: float = 10.0,
        a_k: float = 0.1,
    ):
        super(Step_2, self).__init__()
        
        # Store hyperparameters for a
        self.a_eps_min = a_eps_min
        self.a_eps_max = a_eps_max
        self.a_k = a_k
        
        # Initialize trainable parameters
        self.a_raw = nn.Parameter(torch.tensor(0.0))
        self.c1 = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.c2 = nn.Parameter(torch.tensor(0.0))
        
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
        final_activation, use_bias
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

        
        # Step 2 parameters (F network: takes U,t → scalar)
        step2_hidden_dims: Union[int, List[int]] = 32,
        step2_n_hidden_layers: int = 2,
        step2_activation: str = 'tanh',
        step2_activation_params: Optional[dict] = None,
        step2_final_activation: Optional[str] = None,

        
        # Shared initialization parameters (apply to both steps)
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        


        use_bias: bool = True,

        # Input/Output parameters
        input_dim: int = 2,  # x and t
        output_dim: int = 1,  # scalar G
        
        # Trainable parameter constraints
        a_eps_min: float = 0.1,  # Minimum value for a
        a_eps_max: float = 10.0,  # Maximum value for a  
        a_k: float = 0.1,  # Sigmoid steepness

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
            weight_init=weight_init,
            weight_init_params=weight_init_params,
            bias_init=bias_init,
            bias_init_value=bias_init_value,
            use_bias=use_bias,
            input_dim= input_dim,  # x and t
            output_dim= output_dim,  # scalar G
            
            # Trainable parameter constraints
            a_eps_min= a_eps_min,  # Minimum value for a
            a_eps_max= a_eps_max,  # Maximum value for a  
            a_k= a_k,  # Sigmoid steepness

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
            weight_init=weight_init,
            weight_init_params=weight_init_params,
            bias_init=bias_init,
            bias_init_value=bias_init_value,
            use_bias=use_bias,
            input_dim= input_dim,  # x and t
            output_dim= output_dim,  # scalar G
            
            # Trainable parameter constraints
            a_eps_min= a_eps_min,  # Minimum value for a
            a_eps_max= a_eps_max,  # Maximum value for a  
            a_k= a_k,  # Sigmoid steepness

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
        weight_init: str = 'xavier_uniform',
        weight_init_params: Optional[dict] = None,
        bias_init: str = 'zeros',
        bias_init_value: float = 0.0,
        use_bias: bool = True,


        # Input/Output parameters
        input_dim: int = 2,  # x and t
        output_dim: int = 1,  # scalar G
        
        # Trainable parameter constraints
        a_eps_min: float = 0.1,  # Minimum value for a
        a_eps_max: float = 10.0,  # Maximum value for a  
        a_k: float = 0.1,  # Sigmoid steepness
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
        

        
        for i in range(n_layers):
            layer = CombinedHamiltonianLayer(
                # Step 1 parameters
                step1_hidden_dims=hidden_dims,
                step1_n_hidden_layers=n_hidden_layers,
                step1_activation=activation,
                step1_activation_params=activation_params,
                step1_final_activation=final_activation,
                # Step 2 parameters (same as step 1 for simplicity)
                step2_hidden_dims=hidden_dims,
                step2_n_hidden_layers=n_hidden_layers,
                step2_activation=activation,
                step2_activation_params=activation_params,
                step2_final_activation=final_activation,
                # Shared parameters
                weight_init=weight_init,
                weight_init_params=weight_init_params,
                bias_init=bias_init,
                bias_init_value=bias_init_value,
                use_bias=use_bias,
                input_dim= input_dim,  # x and t
                output_dim= output_dim,  # scalar G
                
                # Trainable parameter constraints
                a_eps_min= a_eps_min,  # Minimum value for a
                a_eps_max= a_eps_max,  # Maximum value for a  
                a_k= a_k,  # Sigmoid steepness
                **kwargs  # Pass any remaining kwargs
            )
            self.layers.append(layer)
    
    def forward(self, x, u, t):
        """Forward pass through all layers."""
        X, U, T = x, u, t
        for layer in self.layers:
            X, U, T = layer(X, U, T)
        return X, U, T


class IntraTrajectoryVarianceLossEfficient(nn.Module):
    """
    Mean Variance loss with guaranteed gradient flow - exact functionality preserved
    """
    
    def __init__(self):
        super(IntraTrajectoryVarianceLossEfficient, self).__init__()
            
    def forward(
        self, 
        X_final: torch.Tensor, 
        U_final: torch.Tensor, 
        trajectory_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficient computation using scatter operations with smooth gradient flow.
        """
        # Get unique trajectories and their counts
        unique_ids, inverse_indices = torch.unique(trajectory_ids, return_inverse=True)
        n_trajectories = len(unique_ids)
        
        # Count samples per trajectory
        counts = torch.zeros(n_trajectories, device=trajectory_ids.device)
        counts = counts.scatter_add(0, inverse_indices, 
                                   torch.ones_like(inverse_indices, dtype=torch.float))
        
        # Compute means per trajectory using scatter
        X_sums = torch.zeros(n_trajectories, device=X_final.device, dtype=X_final.dtype)
        U_sums = torch.zeros(n_trajectories, device=U_final.device, dtype=U_final.dtype)
        X_sums = X_sums.scatter_add(0, inverse_indices, X_final)
        U_sums = U_sums.scatter_add(0, inverse_indices, U_final)
        
        X_means = X_sums / counts
        U_means = U_sums / counts
        
        # Compute squared differences from means
        X_centered = X_final - X_means[inverse_indices]
        U_centered = U_final - U_means[inverse_indices]
        
        # Sum squared differences per trajectory
        X_sq_sums = torch.zeros(n_trajectories, device=X_final.device, dtype=X_final.dtype)
        U_sq_sums = torch.zeros(n_trajectories, device=U_final.device, dtype=U_final.dtype)
        X_sq_sums = X_sq_sums.scatter_add(0, inverse_indices, X_centered ** 2)
        U_sq_sums = U_sq_sums.scatter_add(0, inverse_indices, U_centered ** 2)
        
        # Compute variances - GRADIENT SAFE VERSION
        # Create mask as float (this allows gradient to flow, just with zero values)
        valid_mask_float = (counts > 1).float()
        
        # Use torch.maximum to prevent division by zero/negative
        # This is differentiable and safe
        safe_denominator = torch.maximum(
            counts - 1, 
            torch.ones_like(counts)  # Minimum value of 1
        )
        
        # Compute variances for all trajectories
        X_variances_all = X_sq_sums / safe_denominator
        U_variances_all = U_sq_sums / safe_denominator
        
        # Apply mask through multiplication (preserves gradient flow)
        # For count <= 1: mask = 0, so variance = 0
        # For count > 1: mask = 1, so variance = computed value
        X_variances = X_variances_all * valid_mask_float
        U_variances = U_variances_all * valid_mask_float
        
        # Weighted sum of variances
        total_variance = X_variances.sum() + U_variances.sum()
        
        # Mean variance
        total_mean_variance = total_variance / max(n_trajectories, 1)
        
        return total_mean_variance
    


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
    

        

def reconstruction_loss(
    x_recon, u_recon, t_recon,
    x_orig, u_orig, t_orig,
    loss_type='mse'
):
    """
    Compute reconstruction loss between reconstructed and original values.
    
    Args:
        x_recon, u_recon, t_recon: Reconstructed values from inverse network
        x_orig, u_orig, t_orig: Original batch values
        x_weight, u_weight, t_weight: Weights for each component (t_weight=0 by default since t shouldn't change)
        loss_type: 'mse' for mean squared error or 'mae' for mean absolute error
        
    Returns:
        loss: Scalar loss value
    """
    if loss_type == 'mse':
        x_loss = torch.mean((x_recon - x_orig) ** 2)
        u_loss = torch.mean((u_recon - u_orig) ** 2)
        t_loss = torch.mean((t_recon - t_orig) ** 2)
    elif loss_type == 'mae':
        x_loss = torch.mean(torch.abs(x_recon - x_orig))
        u_loss = torch.mean(torch.abs(u_recon - u_orig))
        t_loss = torch.mean(torch.abs(t_recon - t_orig))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Weighted sum
    total_loss = x_loss + u_loss + t_loss
    
    return total_loss

def prepare_prediction_inputs(
    X_final: torch.Tensor,
    U_final: torch.Tensor, 
    t_batch: torch.Tensor,
    trajectory_ids: torch.Tensor,
    possible_t_values: List[float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare inputs for the prediction network with guaranteed gradient flow.
    Functionality is EXACTLY identical to original.
    
    Args:
        X_final: Final positions from mapping network, shape (batch_size,)
        U_final: Final velocities from mapping network, shape (batch_size,)
        t_batch: Time values from batch, shape (batch_size,)
        trajectory_ids: Trajectory IDs for each sample, shape (batch_size,)
        possible_t_values: List of all possible time values for any trajectory
        
    Returns:
        X_final_mean: Each element replaced with its trajectory's mean X_final
        U_final_mean: Each element replaced with its trajectory's mean U_final  
        t_for_pred: New time values not in the original batch for each trajectory
    """
    device = X_final.device
    batch_size = X_final.shape[0]
    
    # Get unique trajectories
    unique_ids, inverse_indices = torch.unique(trajectory_ids, return_inverse=True)
    n_trajectories = len(unique_ids)
    
    # ===== Compute trajectory means (gradient-safe operations) =====
    # Count samples per trajectory
    counts = torch.zeros(n_trajectories, device=device)
    counts = counts.scatter_add(
        0, 
        inverse_indices, 
        torch.ones_like(inverse_indices, dtype=torch.float)
    )
    
    # Sum X and U per trajectory (preserve input dtype)
    X_sums = torch.zeros(n_trajectories, device=device, dtype=X_final.dtype)
    U_sums = torch.zeros(n_trajectories, device=device, dtype=U_final.dtype)
    X_sums = X_sums.scatter_add(0, inverse_indices, X_final)
    U_sums = U_sums.scatter_add(0, inverse_indices, U_final)
    
    # Compute means - exactly as original
    # Note: counts should never be 0 since each trajectory has at least one sample
    # If worried about edge cases, can use: torch.clamp(counts, min=1e-8)
    X_means = X_sums / counts
    U_means = U_sums / counts
    
    # Broadcast means back to original shape
    # This indexing operation maintains gradient flow
    X_final_mean = X_means[inverse_indices]
    U_final_mean = U_means[inverse_indices]
    
    # ===== Generate new time values =====
    # Note: t_for_pred must remain part of computation graph for use in next network
    # Even though gradients won't flow through it back to t_batch, it needs to be
    # part of the graph for computing gradients w.r.t. the next network's parameters
    t_for_pred = torch.zeros_like(t_batch)
    possible_t_tensor = torch.tensor(possible_t_values, device=device, dtype=t_batch.dtype)
    
    # Process each trajectory separately - exactly as original
    for traj_idx, traj_id in enumerate(unique_ids):
        # Get mask for this trajectory
        mask = (trajectory_ids == traj_id)
        n_samples = mask.sum().item()
        
        # Get sampled times for this trajectory
        sampled_times = t_batch[mask]
        
        # Find which times from possible_t_values are NOT in sampled_times
        is_sampled = torch.isclose(
            possible_t_tensor.unsqueeze(1),
            sampled_times.unsqueeze(0),
            rtol=1e-5,
            atol=1e-8
        ).any(dim=1)
        available_times = possible_t_tensor[~is_sampled]
        
        if len(available_times) < n_samples:
            # Not enough unsampled times - handle gracefully
            if len(available_times) > 0:
                # Repeat available times to have enough samples
                repeated_times = available_times.repeat((n_samples // len(available_times)) + 1)
                selected_times = repeated_times[:n_samples]
                # Shuffle to randomize
                perm = torch.randperm(n_samples, device=device)
                selected_times = selected_times[perm]
            else:
                # No available times at all - use random times from possible_t_values
                indices = torch.randint(0, len(possible_t_values), (n_samples,), device=device)
                selected_times = possible_t_tensor[indices]
        else:
            # Randomly select n_samples from available times without replacement
            perm = torch.randperm(len(available_times), device=device)[:n_samples]
            selected_times = available_times[perm]
        
        # Assign to output
        t_for_pred[mask] = selected_times

    return X_final_mean, U_final_mean, t_for_pred


def generate_prediction_labels(
    train_df: pd.DataFrame,
    train_id_df: pd.DataFrame,
    trajectory_ids: torch.Tensor,
    t_for_pred: torch.Tensor,
    get_data_from_trajectory_id: callable
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate ground truth labels for prediction task.
    
    Args:
        train_df: DataFrame with columns ['x', 'u', 't']
        train_id_df: DataFrame with trajectory metadata
        trajectory_ids: Trajectory IDs for each sample in batch, shape (batch_size,)
        t_for_pred: New time values to predict at, shape (batch_size,)
        get_data_from_trajectory_id: Function to get trajectory data
        
    Returns:
        X_labels: Ground truth x values at t_for_pred times, shape (batch_size,)
        U_labels: Ground truth u values at t_for_pred times, shape (batch_size,)
        t_labels: Same as t_for_pred (for consistency), shape (batch_size,)
    """
    batch_size = trajectory_ids.shape[0]
    device = trajectory_ids.device
    
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


def prediction_loss(
    x_pred: torch.Tensor,
    u_pred: torch.Tensor,
    X_labels: torch.Tensor,
    U_labels: torch.Tensor,

) -> torch.Tensor:
    """
    Compute prediction loss using Mean Square Error.
    
    Args:
        x_pred: Predicted positions from inverse network, shape (batch_size,)
        u_pred: Predicted velocities from inverse network, shape (batch_size,)
        X_labels: Ground truth positions at t_for_pred times, shape (batch_size,)
        U_labels: Ground truth velocities at t_for_pred times, shape (batch_size,)

        
    Returns:
        loss: Scalar MSE loss
    """
    # Mean Square Error for each component
    x_mse = torch.mean((x_pred - X_labels) ** 2)
    u_mse = torch.mean((u_pred - U_labels) ** 2)
    
    # Combination
    total_loss =  x_mse + u_mse
    
    return total_loss


def compute_total_loss(mapping_loss, reconstruction_loss, prediction_loss,
                      mapping_coefficient, prediction_coefficient,
                      mapping_loss_scale, prediction_loss_scale,
                      reconstruction_threshold, reconstruction_loss_multiplier):
    """
    Compute final total loss with scale normalization and gradient-safe safety switch.
    
    Args:
        mapping_loss: Raw mapping loss tensor (with gradients)
        reconstruction_loss: Raw reconstruction loss tensor (with gradients)
        prediction_loss: Raw prediction loss tensor (with gradients)
        mapping_coefficient: Fixed weight for mapping loss (float, no gradients)
        prediction_coefficient: Fixed weight for prediction loss (float, no gradients)
        mapping_loss_scale: Fixed scale for mapping loss normalization (float, no gradients)
        prediction_loss_scale: Fixed scale for prediction loss normalization (float, no gradients)
        reconstruction_threshold: Threshold for reconstruction safety switch (float, no gradients)
        reconstruction_loss_multiplier: Target multiplier when threshold exceeded (float, no gradients)
        
    Returns:
        total_loss: Final weighted and scaled loss tensor (gradients flow back to input losses)
    """
    
    # Scale mapping and prediction losses by fixed scales (gradients preserved)
    mapping_loss_scaled = mapping_loss / mapping_loss_scale
    prediction_loss_scaled = prediction_loss / prediction_loss_scale
    
    # ===== GRADIENT-SAFE SAFETY SWITCH =====
    # Two options provided - choose based on your needs
    
    # OPTION 1: Very sharp transition (almost like original but differentiable)
    # This creates an almost step-like function but remains differentiable
    sharpness = 100.0  # Higher = sharper transition (closer to original if/else)
    
    smooth_factor = torch.sigmoid(
        sharpness * (reconstruction_loss.detach() - reconstruction_threshold)
    )
    
    # Compute the high-loss multiplier
    # Use detach() to get values without gradient flow for the multiplier calculation
    avg_other_losses = (mapping_loss_scaled.detach() + prediction_loss_scaled.detach()) / 2
    target_loss = reconstruction_loss_multiplier * avg_other_losses
    
    # Safe division with epsilon
    eps = 1e-8
    high_multiplier = target_loss / (reconstruction_loss.detach() + eps)
    
    # Smooth interpolation between multipliers
    # Below threshold: multiplier ≈ 1.0
    # Above threshold: multiplier ≈ high_multiplier
    multiplier = 1.0 * (1 - smooth_factor) + high_multiplier * smooth_factor
    
    reconstruction_loss_scaled = reconstruction_loss * multiplier
    
    # Combine all losses with coefficients
    total_loss = (
        mapping_coefficient * mapping_loss_scaled + 
        reconstruction_loss_scaled +
        prediction_coefficient * prediction_loss_scaled
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


def generate_validation_labels_single_trajectory(
    val_df: pd.DataFrame,
    val_id_df: pd.DataFrame,
    trajectory_id: int,  # Single trajectory ID (np.int64)
    t_rearranged: torch.Tensor,  # Rearranged times to predict at
    get_data_from_trajectory_id: callable
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate ground truth labels for validation with single trajectory.
    
    Args:
        val_df: DataFrame with columns ['x', 'u', 't']
        val_id_df: DataFrame with trajectory metadata
        trajectory_id: Single trajectory ID (int or np.int64)
        t_rearranged: Rearranged time values to predict at
        get_data_from_trajectory_id: Function to get trajectory data
        
    Returns:
        X_val_labels: Ground truth x values at t_rearranged times
        U_val_labels: Ground truth u values at t_rearranged times
        t_val_labels: Same as t_rearranged (for consistency)
    """
    n_points = len(t_rearranged)
    device = t_rearranged.device
    
    # Initialize output tensors
    X_val_labels = torch.zeros(n_points, device=device)
    U_val_labels = torch.zeros(n_points, device=device)
    
    # Get trajectory data once
    traj_df = get_data_from_trajectory_id(
        ids_df=val_id_df,
        data_df=val_df,
        trajectory_ids=trajectory_id
    )
    
    # Get trajectory values
    t_values = traj_df['t'].values
    x_values = traj_df['x'].values
    u_values = traj_df['u'].values
    
    # Convert rearranged times to numpy
    t_rearranged_np = t_rearranged.cpu().numpy()
    
    # For each rearranged time, find corresponding x, u
    for i in range(n_points):
        t_target = t_rearranged_np[i]
        
        # Find exact match (should always exist for validation)
        exact_match = np.isclose(t_values, t_target, atol=1e-8)
        if exact_match.any():
            idx = np.where(exact_match)[0][0]
            X_val_labels[i] = x_values[idx]
            U_val_labels[i] = u_values[idx]
        else:
            print(f"Warning: Time {t_target} not found in trajectory {trajectory_id}")
    
    # t_labels is just t_rearranged for consistency
    t_val_labels = t_rearranged.clone()
    
    return X_val_labels, U_val_labels, t_val_labels

def calculate_losses_scale_on_untrained(train_loader, mapping_net, inverse_net, var_loss_class, get_data_from_trajectory_id, possible_t_values, train_df, train_id_df, save_returned_values, save_dir, noise_threshold_mean_divided_by_std = 2, device="cuda"):
    mapping_net.to(device)
    mapping_net.eval()

    def forward_pass_for_rescaling(batch):
        X_final, U_final, t_final = mapping_net(batch['x'], batch['u'], batch['t'])
        variance_loss_ = var_loss_class(X_final, U_final, batch['trajectory_ids'])

        X_final_mean, U_final_mean, t_for_pred = prepare_prediction_inputs(
            X_final, U_final, 
            t_batch=batch['t'], 
            trajectory_ids=batch['trajectory_ids'], 
            possible_t_values=possible_t_values)
        x_pred, u_pred, _ = inverse_net(X_final_mean, U_final_mean, t_for_pred)
        X_labels, U_labels, _ = generate_prediction_labels(
                train_df=train_df, 
                train_id_df=train_id_df, 
                trajectory_ids=batch['trajectory_ids'], 
                t_for_pred=t_for_pred, 
                get_data_from_trajectory_id=get_data_from_trajectory_id
            )
        prediction_loss_ = prediction_loss(
                x_pred=x_pred, 
                u_pred=u_pred, 
                X_labels=X_labels, 
                U_labels=U_labels
            )

        return variance_loss_.item(), prediction_loss_.item()

    variance_losses_list = []
    prediction_losses_list = []
    for batch_idx, batch in enumerate(train_loader):
        variance_loss_, prediction_loss_ = forward_pass_for_rescaling(batch)
        variance_losses_list.append(variance_loss_)
        prediction_losses_list.append(prediction_loss_)
    variance_loss_epoch_mean = np.mean(variance_losses_list)
    prediction_loss_epoch_mean = np.mean(prediction_losses_list)

    variance_loss_epoch_std = np.std(variance_losses_list)
    prediction_loss_epoch_std = np.std(prediction_losses_list)

    if (np.abs(variance_loss_epoch_mean/variance_loss_epoch_std)<noise_threshold_mean_divided_by_std) or (np.abs(prediction_loss_epoch_mean/prediction_loss_epoch_std)<noise_threshold_mean_divided_by_std):
        variance_loss_epoch_median = np.median(variance_losses_list)
        prediction_loss_epoch_median = np.median(prediction_losses_list)
        print(f"Calculated epoch's variance loss: {variance_loss_epoch_mean:.4f}±{variance_loss_epoch_std:.4f} and prediction loss: {prediction_loss_epoch_mean:.4f}±{prediction_loss_epoch_std:.4f}\nWhich is too noisy so using median which is for variance:{variance_loss_epoch_median:.4f} and for prediction:{prediction_loss_epoch_median:.4f}")
        if save_returned_values:
            loss_scales = {"saved_mapping_loss_scale": variance_loss_epoch_median, "saved_prediction_loss_scale":prediction_loss_epoch_median}
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "loss_scales.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(loss_scales, f)
            print(f"Saved values at {save_path}")
        return variance_loss_epoch_median, prediction_loss_epoch_median
    else:
        print(f"Calculated epoch's variance loss: {variance_loss_epoch_mean:.4f}±{variance_loss_epoch_std:.4f} and prediction loss: {prediction_loss_epoch_mean:.4f}±{prediction_loss_epoch_std:.4f}, returning means")
        if save_returned_values:
            loss_scales = {"saved_mapping_loss_scale": variance_loss_epoch_mean, "saved_prediction_loss_scale":prediction_loss_epoch_mean}
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "loss_scales.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(loss_scales, f)
            print(f"Saved values at {save_path}")
        return variance_loss_epoch_mean, prediction_loss_epoch_mean



def train_model(
    # Dataloaders
    train_loader,
    val_loader, 
    val_loader_high_energy,
    val_loader_training_set,
    
    #Network
    mapping_net, 
    inverse_net,
     
    #Needed objects
    var_loss_class,
    get_data_from_trajectory_id,
    possible_t_values,
    
    #Needed pandas dataframes
    train_df,
    train_id_df,
    val_df,
    val_id_df,
    val_df_high_energy,
    val_id_df_high_energy,
    

    #Loss calculation hyperparameters
    mapping_loss_scale,
    prediction_loss_scale,
    mapping_coefficient,
    prediction_coefficient,
    reconstruction_threshold,
    reconstruction_loss_multiplier,
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
    val_batches_high_energy = len(val_loader_high_energy)
    val_batches_training_set = len(val_loader_training_set)
    
    
    # Create the original all_params list for gradient clipping
    all_params = mapping_net.parameters()
    


    if optimizer==None:
        if optimizer_type == 'Adam':
            print(f"Initializing new Adam optimizer with learning rate: {learning_rate}")
            optimizer = Adam(all_params, lr=learning_rate)
        elif optimizer_type == 'AdamW':
            print(f"Initializing new AdamW optimizer with learning rate: {learning_rate}, and weight decay {weight_decay}")
            optimizer = AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)  
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
    def forward_pass(batch):
        X_final, U_final, t_final = mapping_net(batch['x'], batch['u'], batch['t'])
        variance_loss_ = var_loss_class(X_final, U_final, batch['trajectory_ids'])


        x_recon, u_recon, t_recon = inverse_net(X_final, U_final, t_final)
        reconstruction_loss_ = reconstruction_loss(
                x_recon=x_recon, u_recon=u_recon, t_recon=t_recon,
                x_orig=batch['x'], u_orig=batch['u'], t_orig=batch['t'],
                loss_type='mse'
            )

        X_final_mean, U_final_mean, t_for_pred = prepare_prediction_inputs(
            X_final, U_final, 
            t_batch=batch['t'], 
            trajectory_ids=batch['trajectory_ids'], 
            possible_t_values=possible_t_values)

        x_pred, u_pred, _ = inverse_net(X_final_mean, U_final_mean, t_for_pred)

        X_labels, U_labels, _ = generate_prediction_labels(
                train_df=train_df, 
                train_id_df=train_id_df, 
                trajectory_ids=batch['trajectory_ids'], 
                t_for_pred=t_for_pred, 
                get_data_from_trajectory_id=get_data_from_trajectory_id
            )

        prediction_loss_ = prediction_loss(
                x_pred=x_pred, 
                u_pred=u_pred, 
                X_labels=X_labels, 
                U_labels=U_labels
            )

        total_loss_ = compute_total_loss(
                mapping_loss=variance_loss_, 
                reconstruction_loss=reconstruction_loss_, 
                prediction_loss=prediction_loss_,
                mapping_coefficient=mapping_coefficient, 
                prediction_coefficient=prediction_coefficient,
                mapping_loss_scale=mapping_loss_scale, 
                prediction_loss_scale=prediction_loss_scale,
                reconstruction_threshold=reconstruction_threshold, 
                reconstruction_loss_multiplier=reconstruction_loss_multiplier
            )
        return total_loss_, variance_loss_, reconstruction_loss_, prediction_loss_
    
    
    
    def validation_forward_pass(batch, val_df, val_id_df):
        X_final, U_final, t_final = mapping_net(batch['x'], batch['u'], batch['t'])
        variance_loss_, X_mean, U_mean, X_var, U_var, X_std, U_std = compute_single_trajectory_stats(X_final, U_final)


        x_recon, u_recon, t_recon = inverse_net(X_final, U_final, t_final)
        reconstruction_loss_ = reconstruction_loss(
                x_recon=x_recon, u_recon=u_recon, t_recon=t_recon,
                x_orig=batch['x'], u_orig=batch['u'], t_orig=batch['t'],
                loss_type='mse'
            )

        X_final_mean_full, U_final_mean_full, t_rearranged = prepare_validation_inputs(X_final, U_final, t_final)

        x_pred, u_pred, _ = inverse_net(X_final_mean_full, U_final_mean_full, t_rearranged)

        X_val_labels, U_val_labels, _ = generate_validation_labels_single_trajectory(
                val_df=val_df, 
                val_id_df=val_id_df, 
                trajectory_id=batch['trajectory_id'], 
                t_rearranged=t_rearranged, 
                get_data_from_trajectory_id=get_data_from_trajectory_id
            )

        prediction_loss_ = prediction_loss(
                x_pred=x_pred, 
                u_pred=u_pred, 
                X_labels=X_val_labels, 
                U_labels=U_val_labels
            )

        total_loss_ = compute_total_loss(
                mapping_loss=variance_loss_, 
                reconstruction_loss=reconstruction_loss_, 
                prediction_loss=prediction_loss_,
                mapping_coefficient=mapping_coefficient, 
                prediction_coefficient=prediction_coefficient,
                mapping_loss_scale=mapping_loss_scale, 
                prediction_loss_scale=prediction_loss_scale,
                reconstruction_threshold=reconstruction_threshold, 
                reconstruction_loss_multiplier=reconstruction_loss_multiplier
            )
        return total_loss_.item(), variance_loss_.item(), reconstruction_loss_.item(), prediction_loss_.item(), X_mean, U_mean, X_var, U_var, X_std, U_std
    # Training loop
    start_epoch=0
    if continue_from_epoch is not None:
        start_epoch=continue_from_epoch
        
    try:
        for epoch in range(start_epoch, start_epoch+num_epochs):
                # Initialize epoch_metrics
            epoch_metrics = {}
            epoch_start_time = time.time()

            # Set all models to training mode
            mapping_net.train()
            # Initialize metrics
            train_total_loss_ = 0.0
            train_variance_loss_ = 0.0
            train_reconstruction_loss_ = 0.0
            train_prediction_loss_ = 0.0
            batch_count = 0

            # Progress tracking
            if verbose > 0:
                print(f"\n{'='*20} Epoch {epoch}/{start_epoch+num_epochs} {'='*20}")

            # Training loop
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                # Run forward pass
                total_loss_, variance_loss_, reconstruction_loss_, prediction_loss_ = forward_pass(batch)


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
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 10.0:  # Threshold for reporting
                            print(f"High gradient norm in {name}: {grad_norm}")
                        if grad_norm < 0.00001:  
                            print(f"Low gradient norm in {name}: {grad_norm}")

                # Gradient clipping
                if grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(all_params, grad_clip_value)

                # Optimizer step
                optimizer.step()

                # Update metrics
                train_total_loss_ += total_loss_.item()
                train_variance_loss_ += variance_loss_.item()
                train_reconstruction_loss_ += reconstruction_loss_.item()
                train_prediction_loss_ += prediction_loss_.item()
                batch_count += 1

                # Log progress
                if verbose > 1 and batch_idx % log_freq_batches == 0:
                    print(f"Batch {batch_idx}/{train_batches} - Total Loss: {total_loss_.item():.4f} - Variance Loss: {variance_loss_.item():.4f} - Reconstruction Loss: {reconstruction_loss_.item():.4f}) - Prediction Loss: {prediction_loss_.item():.4f}")


                
            # Calculate epoch metrics
            train_total_loss_ /= max(1, batch_count)
            train_variance_loss_ /= max(1, batch_count)
            train_reconstruction_loss_ /= max(1, batch_count)
            train_prediction_loss_ /= max(1, batch_count)

            # Validation phase
            val_total_loss_ = 0.0
            val_variance_loss_ = 0.0
            val_reconstruction_loss_ = 0.0
            val_prediction_loss_ = 0.0
            val_batch_count = 0


            # Set model to evaluation mode
            mapping_net.eval()
            
            epoch_saving_path = os.path.join(save_dir, f"epoch_{epoch}")
            
            val_trajectories_dir = os.path.join(epoch_saving_path, f"val_trajectories_data")
            os.makedirs(val_trajectories_dir, exist_ok=True)

            val_high_energy_trajectories_dir = os.path.join(epoch_saving_path, f"val_high_energy_trajectories_data")
            os.makedirs(val_high_energy_trajectories_dir, exist_ok=True)

            val_train_set_trajectories_dir = os.path.join(epoch_saving_path, f"val_train_set_trajectories_data")
            os.makedirs(val_train_set_trajectories_dir, exist_ok=True)


            
            
            # Validation loop

            print("Begining validation phase")
            for batch_idx, batch in enumerate(val_loader):
                # Run forward pass
                total_loss_, variance_loss_, reconstruction_loss_, prediction_loss_, X_mean, U_mean, X_var, U_var, X_std, U_std = validation_forward_pass(batch, val_df=val_df, val_id_df=val_id_df)
                # Update metrics
                if total_loss_ is not None:
                    val_total_loss_ += total_loss_
                    val_variance_loss_ += variance_loss_
                    val_reconstruction_loss_ += reconstruction_loss_
                    val_prediction_loss_ += prediction_loss_
                    if verbose > 1 and batch_idx % log_freq_batches == 0:
                        print(f"Batch {batch_idx}/{val_batches} - Total Loss: {total_loss_:.4f}- Variance Loss: {variance_loss_:.4f} - Reconstruction Loss: {reconstruction_loss_:.4f}) - Prediction Loss: {prediction_loss_:.4f}")
                        
                    trajectory_data = {
                        'total_loss' : total_loss_,
                        'variance_loss' : variance_loss_,
                        'reconstruction_loss' : reconstruction_loss_,
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
            val_reconstruction_loss_ /= max(1, val_batch_count)
            val_prediction_loss_ /= max(1, val_batch_count)




            #High Energy validation loop
            
            val_total_loss_high_energy = 0.0
            val_variance_loss_high_energy = 0.0
            val_reconstruction_loss_high_energy = 0.0
            val_prediction_loss_high_energy = 0.0
            val_batch_count_high_energy = 0
            

            print("Begining high energy validation phase")
            for batch_idx, batch in enumerate(val_loader_high_energy):
                # Run forward pass
                total_loss_, variance_loss_, reconstruction_loss_, prediction_loss_, X_mean, U_mean, X_var, U_var, X_std, U_std = validation_forward_pass(batch, val_df=val_df_high_energy, val_id_df=val_id_df_high_energy)
                
                # Update metrics
                if total_loss_ is not None:
                    val_total_loss_high_energy += total_loss_
                    val_variance_loss_high_energy += variance_loss_
                    val_reconstruction_loss_high_energy += reconstruction_loss_
                    val_prediction_loss_high_energy += prediction_loss_
                    if verbose > 1 and batch_idx % log_freq_batches == 0:
                        print(f"Batch {batch_idx}/{val_batches_high_energy} - Total Loss: {total_loss_:.4f}- Variance Loss: {variance_loss_:.4f} - Reconstruction Loss: {reconstruction_loss_:.4f}) - Prediction Loss: {prediction_loss_:.4f}")
                        
                    trajectory_data = {
                        'total_loss' : total_loss_,
                        'variance_loss' : variance_loss_,
                        'reconstruction_loss' : reconstruction_loss_,
                        'prediction_loss' : prediction_loss_,
                        'X_mean': X_mean,
                        'U_mean': U_mean,
                        'X_var': X_var,
                        'U_var': U_var,
                        'X_std': X_std,
                        'U_std': U_std,
                    }           
                    with open(os.path.join(val_high_energy_trajectories_dir, f"trajectory_id_{batch['trajectory_id']}_data.json"), "w") as f:
                        # Convert any non-serializable objects
                        serializable_trajectory_data = make_json_serializable(trajectory_data)
                        json.dump(serializable_trajectory_data, f, indent=2)
                val_batch_count_high_energy += 1

                    
            # Calculate validation metrics
            val_total_loss_high_energy /= max(1, val_batch_count_high_energy)
            val_variance_loss_high_energy /= max(1, val_batch_count_high_energy)
            val_reconstruction_loss_high_energy /= max(1, val_batch_count_high_energy)
            val_prediction_loss_high_energy /= max(1, val_batch_count_high_energy)

            #Validation loop on training set data (per trajectory)
            
            val_total_loss_training_set = 0.0
            val_variance_loss_training_set = 0.0
            val_reconstruction_loss_training_set = 0.0
            val_prediction_loss_training_set = 0.0
            val_batch_count_training_set = 0
            

            print("Begining validation loop on training set data")
            for batch_idx, batch in enumerate(val_loader_training_set):
                # Run forward pass
                total_loss_, variance_loss_, reconstruction_loss_, prediction_loss_, X_mean, U_mean, X_var, U_var, X_std, U_std = validation_forward_pass(batch, val_df=train_df, val_id_df=train_id_df)
                
                # Update metrics
                if total_loss_ is not None:
                    val_total_loss_training_set += total_loss_
                    val_variance_loss_training_set += variance_loss_
                    val_reconstruction_loss_training_set += reconstruction_loss_
                    val_prediction_loss_training_set += prediction_loss_
                    if verbose > 1 and batch_idx % log_freq_batches == 0:
                        print(f"Batch {batch_idx}/{val_batches_training_set} - Total Loss: {total_loss_:.4f}- Variance Loss: {variance_loss_:.4f} - Reconstruction Loss: {reconstruction_loss_:.4f}) - Prediction Loss: {prediction_loss_:.4f}")
                    trajectory_data = {
                        'total_loss' : total_loss_,
                        'variance_loss' : variance_loss_,
                        'reconstruction_loss' : reconstruction_loss_,
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
            val_reconstruction_loss_training_set /= max(1, val_batch_count_training_set)
            val_prediction_loss_training_set /= max(1, val_batch_count_training_set)


            # Update epoch_metrics
            epoch_metrics['epoch'] = epoch
            
            epoch_metrics['train_total_loss_'] = train_total_loss_
            epoch_metrics['train_variance_loss_'] = train_variance_loss_
            epoch_metrics['train_reconstruction_loss_'] = train_reconstruction_loss_
            epoch_metrics['train_prediction_loss_'] = train_prediction_loss_

            epoch_metrics['val_total_loss_'] = val_total_loss_
            epoch_metrics['val_variance_loss_'] = val_variance_loss_
            epoch_metrics['val_reconstruction_loss_'] = val_reconstruction_loss_
            epoch_metrics['val_prediction_loss_'] = val_prediction_loss_
            
            epoch_metrics['val_total_loss_high_energy'] = val_total_loss_high_energy
            epoch_metrics['val_variance_loss_high_energy'] = val_variance_loss_high_energy
            epoch_metrics['val_reconstruction_loss_high_energy'] = val_reconstruction_loss_high_energy
            epoch_metrics['val_prediction_loss_high_energy'] = val_prediction_loss_high_energy
            
            epoch_metrics['val_total_loss_training_set'] = val_total_loss_training_set
            epoch_metrics['val_variance_loss_training_set'] = val_variance_loss_training_set
            epoch_metrics['val_reconstruction_loss_training_set'] = val_reconstruction_loss_training_set
            epoch_metrics['val_prediction_loss_training_set'] = val_prediction_loss_training_set
            

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            epoch_metrics['learning_rates'] = current_lr


            validation_criterio = on_distribution_val_criterio_weight*val_total_loss_ + (1.0-on_distribution_val_criterio_weight)*val_total_loss_high_energy
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
                print(f"Mean total train loss: {train_total_loss_:.4f}")
                print(f"Mean variance train loss: {train_variance_loss_:.4f}")
                print(f"Mean reconstruction train loss: {train_reconstruction_loss_:.4f}")
                print(f"Mean prediction train loss: {train_prediction_loss_:.4f}")
                
                
                print(f"Mean total val loss: {val_total_loss_:.4f}")
                print(f"Mean variance val loss: {val_variance_loss_:.4f}")
                print(f"Mean reconstruction val loss: {val_reconstruction_loss_:.4f}")
                print(f"Mean prediction val loss: {val_prediction_loss_:.4f}")
                
                print(f"Mean total val loss high energy: {val_total_loss_high_energy:.4f}")
                print(f"Mean variance val loss high energy: {val_variance_loss_high_energy:.4f}")
                print(f"Mean reconstruction val loss high energy: {val_reconstruction_loss_high_energy:.4f}")
                print(f"Mean prediction val loss high energy: {val_prediction_loss_high_energy:.4f}")               

                print(f"Mean total val loss training set: {val_total_loss_training_set:.4f}")
                print(f"Mean variance val loss training set: {val_variance_loss_training_set:.4f}")
                print(f"Mean reconstruction val loss training set: {val_reconstruction_loss_training_set:.4f}")
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
    val_loader,
    val_loader_high_energy,
    val_loader_training_set,
    
    # Network
    mapping_net,  # inverse_net is created automatically from mapping_net
    
    # Needed objects
    var_loss_class,
    get_data_from_trajectory_id,
    possible_t_values,
    
    # Needed pandas dataframes
    train_df,
    train_id_df,
    val_df,
    val_id_df,
    val_df_high_energy,
    val_id_df_high_energy,
    
    # Loss calculation hyperparameters
    mapping_loss_scale,
    prediction_loss_scale,
    mapping_coefficient,
    prediction_coefficient,
    reconstruction_threshold,
    reconstruction_loss_multiplier,
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
    scheduler_params={'mode': 'min', 'factor': 0.1, 'patience': 5, 'verbose': True},  # MODE A: can differ. Functionality would depend on reset_scheduler_patience | MODE B: can be different
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
        all_params = mapping_net.parameters()
        if optimizer_type == 'Adam':
            temp_optimizer = Adam(all_params)
        elif optimizer_type == 'AdamW':
            temp_optimizer = AdamW(all_params)
        temp_scheduler_params = scheduler_params 
        temp_scheduler = ReduceLROnPlateau(temp_optimizer, **temp_scheduler_params)
        # Load checkpoint
        epoch, extra_info, best_val_loss = load_checkpoint(
            checkpoint_path, mapping_net, device, temp_optimizer, temp_scheduler
        )
        # Reset scheduler patience counter if requested
        if reset_scheduler_patience:
            temp_scheduler.num_bad_epochs = 0
            temp_scheduler.cooldown_counter = 0
            print(f"Reset scheduler patience counter. Will wait full {temp_scheduler.patience} epochs before reducing LR.")
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
            val_loader=val_loader,
            val_loader_high_energy=val_loader_high_energy,
            val_loader_training_set=val_loader_training_set,
            
            # Network
            mapping_net=mapping_net, 
            inverse_net=inverse_net,
            
            # Needed objects
            var_loss_class=var_loss_class,
            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values=possible_t_values,
            
            # Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,
            val_df=val_df,
            val_id_df=val_id_df,
            val_df_high_energy=val_df_high_energy,
            val_id_df_high_energy=val_id_df_high_energy,
            
            # Loss calculation hyperparameters
            mapping_loss_scale=mapping_loss_scale,
            prediction_loss_scale=prediction_loss_scale,
            mapping_coefficient=mapping_coefficient,
            prediction_coefficient=prediction_coefficient,
            reconstruction_threshold=reconstruction_threshold,
            reconstruction_loss_multiplier=reconstruction_loss_multiplier,
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
            val_loader=val_loader,
            val_loader_high_energy=val_loader_high_energy,
            val_loader_training_set=val_loader_training_set,
            
            # Network
            mapping_net=mapping_net, 
            inverse_net=inverse_net,
            
            # Needed objects
            var_loss_class=var_loss_class,
            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values=possible_t_values,
            
            # Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,
            val_df=val_df,
            val_id_df=val_id_df,
            val_df_high_energy=val_df_high_energy,
            val_id_df_high_energy=val_id_df_high_energy,
            
            # Loss calculation hyperparameters
            mapping_loss_scale=mapping_loss_scale,
            prediction_loss_scale=prediction_loss_scale,
            mapping_coefficient=mapping_coefficient,
            prediction_coefficient=prediction_coefficient,
            reconstruction_threshold=reconstruction_threshold,
            reconstruction_loss_multiplier=reconstruction_loss_multiplier,
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
        if hasattr(scheduler, 'patience'):  # ReduceLROnPlateau
            print(f"  patience: {scheduler.patience}")
            print(f"  factor: {scheduler.factor}")
            print(f"  mode: {scheduler.mode}")
            print(f"  threshold: {scheduler.threshold}")
            print(f"  cooldown: {scheduler.cooldown}")
            print(f"  best: {scheduler.best}")
            print(f"  num_bad_epochs: {scheduler.num_bad_epochs}")
    
    # Return epoch 
    return (
        checkpoint.get('epoch', -1), 
        checkpoint.get('extra_info', None),
        checkpoint.get('best_val_loss', None)
    )



