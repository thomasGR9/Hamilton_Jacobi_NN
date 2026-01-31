import numpy as np
import random
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Any, Callable, Union
import warnings
from collections import defaultdict
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

from modules import (
    get_data_from_trajectory_id,
    get_trajectory_ids_by_energies,
    find_valid_combinations,
    SimpleHarmonicDataLoader,
    create_simple_dataloader,
    Step_1,
    Step_2,
    CombinedHamiltonianLayer,
    SimpleStackedHamiltonianNetwork,
    ReverseStep2,
    ReverseStep1,
    ReverseCombinedHamiltonianLayer,
    InverseStackedHamiltonianNetwork,
    prepare_prediction_inputs_for_real_pendulum_2,
    generate_prediction_labels_for_real_pendulum_2,
    prediction_loss,
    train_model_real_pendulum,
    resume_training_from_checkpoint_real_pendulum,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    init_step_params,
    solve_a_raw,


)


with open("all_dataframes_real_pendulum_retry.pkl", "rb") as f:
    loaded_dfs = pickle.load(f)

# Access them like before:
train_df = loaded_dfs['train_df_real_pendulum_retry']
val_df = loaded_dfs['test_df_real_pendulum_retry']

train_id_df = loaded_dfs['train_id_df_real_pendulum_retry']
val_id_df = loaded_dfs['test_id_df_real_pendulum_retry']



add_noise = False


def main():
    resume_training = True

    device = "cuda"

    if resume_training == False:
        training_seed = 42
    else:
        resume_training_counter = 2
        training_seed = 42 + resume_training_counter
    
    train_dataloader = create_simple_dataloader(
    train_df=train_df,
    train_id_df=train_id_df,
    ratio=0.083333,
    batch_size=12,
    segment_length=1,
    get_data_func=get_data_from_trajectory_id,
    device=device,
    seed=42
    )

    val_dataloader = create_simple_dataloader(
    train_df=val_df,
    train_id_df=val_id_df,
    ratio=1.0,
    batch_size=1,
    segment_length=1,
    get_data_func=get_data_from_trajectory_id,
    device=device,
    seed=training_seed
    )



    mapping_net = SimpleStackedHamiltonianNetwork(
        #Hpw many Step_1 + Step_2 layers to stack
        n_layers=10,
        # MLP Architecture parameters
        hidden_dims= [33, 60, 32],
        n_hidden_layers = None,   #Leave None if you provide list on hidden_dims
        
        # Activation parameters
        activation = 'gelu',
        activation_params = None,
        final_activation = None,   #Final layer activation function
        final_activation_only_on_final_layer = True,
        tanh_wrapper = False,
        
        # Initialization parameters
        weight_init = 'orthogonal',
        weight_init_params = None,
        bias_init = 'zeros',
        bias_init_value = 0.0,
        
        
        # Architectural choices
        use_bias = True,
        use_layer_norm = False,
        
        # Input/Output parameters
        input_dim = 2,  # x or u and t
        output_dim = 1,  # scalar G or F
        a_eps_min= 0.5,  # Minimum value for a
        a_eps_max= 2,  # Maximum value for a  
        a_k= 0.1,

        step_1_a_mean_innit= 1.6,
        step_2_a_mean_innit= 1.6,
        std_to_mean_ratio_a_mean_init= 0.1,

        step_1_gamma_mean_innit= 2.0,
        step_2_gamma_mean_innit= 2.0,
        std_to_mean_ratio_gamma_mean_init= 0.1,

        step_1_c1_mean_innit= 0.1,
        step_2_c1_mean_innit= -0.1,
        std_to_mean_ratio_c1_mean_init= 1.0,

        step_1_c2_mean_innit= -0.1,
        step_2_c2_mean_innit= 0.1,
        std_to_mean_ratio_c2_mean_init= 1.0,

        bound_innit=0.0,
    ).to(device)







    n_train_trajectories = len(train_id_df)  # 12
    n_times = train_id_df['generated_points'].iloc[0]  # 37

    train_possible_t_values = np.zeros((n_train_trajectories, n_times), dtype=np.float32)

    for traj_id in train_id_df['trajectory_id'].values:
        traj_id = int(traj_id)
        t_values = get_data_from_trajectory_id(
            ids_df=train_id_df, 
            data_df=train_df, 
            trajectory_ids=traj_id
        )['t'].values
        train_possible_t_values[traj_id] = t_values

    possible_t_values_per_trajectory = torch.tensor(train_possible_t_values, device=device)



    n_val_trajectories = len(val_id_df)  # 3
    n_times_val = val_id_df['generated_points'].iloc[0]  # 37

    val_possible_t_values = np.zeros((n_val_trajectories, n_times_val), dtype=np.float32)

    for traj_id in val_id_df['trajectory_id'].values:
        traj_id = int(traj_id)
        t_values = get_data_from_trajectory_id(
            ids_df=val_id_df, 
            data_df=val_df, 
            trajectory_ids=traj_id
        )['t'].values
        val_possible_t_values[traj_id] = t_values

    possible_t_values_per_trajectory_val = torch.tensor(val_possible_t_values, device=device)



    loss_type = "mae" #Either 'mae' or 'mse'
    predict_full_trajectory = True


    save_dir_1 = "save_directory_noisy_full_pred_real_pendulum_1"

    

    if not resume_training:

        inverse_net = InverseStackedHamiltonianNetwork(forward_network=mapping_net)


        train_model_real_pendulum(
            # Dataloaders
            train_loader=train_dataloader,
            randomize_each_epoch_plan=True,
            val_loader=val_dataloader, 


            #Network
            mapping_net=mapping_net, 
            inverse_net=inverse_net,

            #Needed objects
            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values_per_trajectory=possible_t_values_per_trajectory,
            possible_t_values_per_trajectory_val=possible_t_values_per_trajectory_val,

            #Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,

            val_df=val_df,
            val_id_df=val_id_df,


            add_noise = add_noise,


            #Loss calculation hyperparameters
            loss_type=loss_type,
            predict_full_trajectory=predict_full_trajectory,




            # Optimizer parameters
            optimizer_type = 'AdamW',

            learning_rate=1e-4,
            weight_decay=1e-4,

            scheduler_type='plateau', 
            scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 600, 'verbose': True},    # Dict of params specific to the scheduler. Set if it is a fresh training session.


            # Training parameters
            num_epochs=600,
            grad_clip_value=20.0,


            # Early stopping parameters
            early_stopping=True,
            patience=600,
            min_delta=0.001,

            # Checkpointing
            save_dir=save_dir_1,
            save_best_only=False,
            save_freq_epochs=10,
            auto_rescue=True,

            # Logging
            log_freq_batches=10,
            verbose=1,

            # Device
            device=device,

            #Used for continuing training from checkpoint, leave all None if it is a fresh training session.
            optimizer=None, 
            scheduler=None,  
            continue_from_epoch=None,
            best_validation_criterio_loss_till_now=None,
        )
    
    else:

        checkpoint_path = os.path.join(save_dir_1, "best_model.pt")



        resume_training_from_checkpoint_real_pendulum(
            # Checkpoint to resume from
            checkpoint_path=checkpoint_path,
            
            # Dataloaders
            train_loader=train_dataloader,
            randomize_each_epoch_plan=True,
            val_loader=val_dataloader,

            
            # Network
            mapping_net=mapping_net,  # Must be same architecture as original. inverse_net is created automatically from mapping_net
            
            # Needed objects
            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values_per_trajectory=possible_t_values_per_trajectory,
            possible_t_values_per_trajectory_val=possible_t_values_per_trajectory_val,
            
            # Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,

            val_df=val_df,
            val_id_df=val_id_df,


            add_noise = add_noise,
            
            # Loss calculation hyperparameters
            loss_type=loss_type,
            predict_full_trajectory=predict_full_trajectory,

            
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
            load_scheduler_and_optimizer=True,
            
            # Optimizer parameters
            learning_rate=0.00001,  # MODE A: None=use checkpoint LR, or specify to override | MODE B: REQUIRED, must specify
            weight_decay=1e-4,   # MODE A: None=use checkpoint weight_decay, or specify to override | MODE B: REQUIRED, must specify
            optimizer_type='AdamW',  # MODE A: must match original | MODE B: can be different
            
            # Scheduler parameters  
            scheduler_type='plateau',  # MODE A: must match original | MODE B: can be different
            scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 100, 'verbose': True},  # MODE A: can differ. Functionality would depend on reset_scheduler_patience | MODE B: can be different
            reset_scheduler_patience = True, #Only relevant on MODE A. Set True to reset num_bad_epochs. Use True if you want the learning rate to be lowered after the full patience amount. Use False if you want continuity, already waited N epochs, just need M-N more where M: loaded num_bad_epochs from previous training
            
            # Training parameters
            num_epochs=400,  # Number of ADDITIONAL epochs to train 
            grad_clip_value=20.0, #Use None if you dont want gradient clipping, specify to use torch.nn.utils.clip_grad_norm_ in MLP parameters
            
            # Early stopping parameters
            early_stopping=True,
            patience=100,
            min_delta=0.001,
            
            # Checkpointing
            save_dir=save_dir_1,  # Should typically match the directory where checkpoint_path is located
            save_best_only=False,
            save_freq_epochs=10,
            auto_rescue=True,
            
            # Logging
            log_freq_batches=10,
            verbose=1,
            
            # Device
            device=device
        )




if __name__ == "__main__":
    main()



    
