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
    add_gaussian_noise,
    SimpleHarmonicDataLoader,
    create_simple_dataloader,
    Step_1,
    Step_2,
    CombinedHamiltonianLayer,
    SimpleStackedHamiltonianNetwork,
    IntraTrajectoryVarianceLossEfficient,
    AdaptiveSoftRepulsionLoss,
    ReverseStep2,
    ReverseStep1,
    ReverseCombinedHamiltonianLayer,
    InverseStackedHamiltonianNetwork,
    reconstruction_loss,
    prepare_prediction_inputs,
    generate_prediction_labels,
    prediction_loss,
    trajectory_normalized_mse,
    compute_total_loss,
    TrajectoryDataset,
    trajectory_collate_fn,
    create_val_dataloader_full_trajectory,
    prepare_validation_inputs,
    compute_single_trajectory_stats,
    single_trajectory_normalized_mse,
    generate_validation_labels_single_trajectory,
    calculate_losses_scale_on_untrained,
    train_model,
    resume_training_from_checkpoint,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    init_step_params,
    solve_a_raw,


)


with open("all_dataframes_low_samples.pkl", "rb") as f:
    loaded_dfs = pickle.load(f)

# Access them like before:
train_df = loaded_dfs['train_df_low_samples']
val_df = loaded_dfs['val_df_low_samples']

train_id_df = loaded_dfs['train_id_df_low_samples']
val_id_df = loaded_dfs['val_id_df_low_samples']



add_noise = True



'''
def main():
    resume_training = True

    device = "cuda"

    if resume_training == False:
        training_seed = 42
    else:
        resume_training_counter = 1
        training_seed = 42 + resume_training_counter
    
    train_dataloader = create_simple_dataloader(
    train_df=train_df,
    train_id_df=train_id_df,
    ratio=0.041667,
    batch_size=24,
    segment_length=1,
    get_data_func=get_data_from_trajectory_id,
    device=device,
    seed=training_seed
    )

    val_dataloader = create_val_dataloader_full_trajectory(
    val_df=val_df,
    val_id_df=val_id_df,
    get_data_func=get_data_from_trajectory_id,
    device=device,
    seed=42
    )

    val_dataloader_high_energy = create_val_dataloader_full_trajectory(
    val_df=val_df_high_energy,
    val_id_df=val_id_df_high_energy,
    get_data_func=get_data_from_trajectory_id,
    device=device, 
    seed=42
    )


    val_dataloader_training_set = create_val_dataloader_full_trajectory(
    val_df=train_df,
    val_id_df=train_id_df,
    get_data_func=get_data_from_trajectory_id,
    device=device,
    seed=42
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


    var_loss_class = IntraTrajectoryVarianceLossEfficient()
    repulsion_loss_class = AdaptiveSoftRepulsionLoss(epsilon=0.01, k=0.5)
    possible_t_values = get_data_from_trajectory_id(ids_df=train_id_df, data_df=train_df, trajectory_ids=30)['t'].values.tolist() #Same possible t values for every trajectory
    normalized_mse = False
    loss_type = "mae" #Either 'mae' or 'mse'
    predict_full_trajectory = True

    save_dir_1 = "./save_directory_low_samples_1" 
    save_dir_2 = "./save_directory_noisy_low_samples_2" 
    
    save_dir_3 = "./save_directory_noisy_full_pred_low_samples_3" 

    save_dir_4 = "./save_directory_noisy_full_pred_low_samples_4" 




    

    if not resume_training:

        inverse_net = InverseStackedHamiltonianNetwork(forward_network=mapping_net)

        derived_mapping_loss_scale, derived_prediction_loss_scale = calculate_losses_scale_on_untrained(train_loader=train_dataloader, mapping_net=mapping_net, inverse_net=inverse_net, var_loss_class=var_loss_class, get_data_from_trajectory_id=get_data_from_trajectory_id, possible_t_values=possible_t_values, train_df=train_df, train_id_df=train_id_df, loss_type=loss_type, predict_full_trajectory=predict_full_trajectory, add_noise=add_noise, normalized_mse=normalized_mse, save_returned_values=True, save_dir=save_dir_4, noise_threshold_mean_divided_by_std = 2, device=device)

        train_model(
            # Dataloaders
            train_loader=train_dataloader,
            randomize_each_epoch_plan=True,
            val_loader=val_dataloader, 
            val_loader_high_energy=val_dataloader_high_energy,
            val_loader_training_set=val_dataloader_training_set,

            #Network
            mapping_net=mapping_net, 
            inverse_net=inverse_net,

            #Needed objects
            var_loss_class=var_loss_class,
            repulsion_loss_class=repulsion_loss_class,
            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values=possible_t_values,

            #Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,

            val_df=val_df,
            val_id_df=val_id_df,

            val_df_high_energy=val_df_high_energy,
            val_id_df_high_energy=val_id_df_high_energy,

            add_noise = add_noise,


            #Loss calculation hyperparameters
            mapping_loss_scale=1.0,
            prediction_loss_scale=derived_prediction_loss_scale,
            normalized_mse=normalized_mse,
            loss_type=loss_type,
            predict_full_trajectory=predict_full_trajectory,

            mapping_coefficient=0.0,
            repulsion_coefficient=0.0,
            prediction_coefficient=2.0,
            hsic_loss_max_want=0.2,

            reconstruction_threshold=10**-6,
            reconstruction_loss_multiplier=5,

            on_distribution_val_criterio_weight = 1.0,


            # Optimizer parameters
            optimizer_type = 'AdamW',

            learning_rate=3e-3,
            weight_decay=1e-4,

            scheduler_type='plateau', 
            scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 600, 'verbose': True},    # Dict of params specific to the scheduler. Set if it is a fresh training session.


            # Training parameters
            num_epochs=600,
            grad_clip_value=60.0,


            # Early stopping parameters
            early_stopping=True,
            patience=600,
            min_delta=0.001,

            # Checkpointing
            save_dir=save_dir_4,
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

        checkpoint_path = os.path.join(save_dir_4, "best_model.pt")

        loss_scales_save_path = os.path.join(save_dir_4, "loss_scales.pkl")

        with open(loss_scales_save_path, "rb") as f:
            loss_scales = pickle.load(f)
        
        saved_mapping_loss_scale = loss_scales['saved_mapping_loss_scale']
        saved_prediction_loss_scale = loss_scales['saved_prediction_loss_scale']


        resume_training_from_checkpoint(
            # Checkpoint to resume from
            checkpoint_path=checkpoint_path,
            
            # Dataloaders
            train_loader=train_dataloader,
            randomize_each_epoch_plan=True,
            val_loader=val_dataloader,
            val_loader_high_energy=val_dataloader_high_energy,
            val_loader_training_set=val_dataloader_training_set,
            
            # Network
            mapping_net=mapping_net,  # Must be same architecture as original. inverse_net is created automatically from mapping_net
            
            # Needed objects
            var_loss_class= var_loss_class,
            repulsion_loss_class=repulsion_loss_class,
            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values=possible_t_values,
            
            # Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,

            val_df=val_df,
            val_id_df=val_id_df,

            val_df_high_energy=val_df_high_energy,
            val_id_df_high_energy=val_id_df_high_energy,

            add_noise = add_noise,
            
            # Loss calculation hyperparameters
            mapping_loss_scale=1.0,
            prediction_loss_scale=saved_prediction_loss_scale,
            normalized_mse=normalized_mse,
            loss_type=loss_type,
            predict_full_trajectory=predict_full_trajectory,

            mapping_coefficient=0.0,
            repulsion_coefficient=0.0,
            prediction_coefficient=2.0,
            hsic_loss_max_want=0.2,

            reconstruction_threshold=10**-6,
            reconstruction_loss_multiplier=5,

            on_distribution_val_criterio_weight=0.75,
            
            # === RESUME MODE SELECTION ===
            # MODE A (load_scheduler_and_optimizer=True): Resume with exact optimizer/scheduler state
            
            #   - learning_rate: Set to None to use checkpoint LR, or specify to override
            #   - optimizer_type: MUST match the type used in original training
            #   - scheduler_type: MUST match the type used in original training
            #  
            #
            # MODE B (load_scheduler_and_optimizer=False): Resume with fresh optimizer/scheduler

            #   - learning_rate: REQUIRED - must specify, will error if None
            #   - optimizer_type: Can be different from original
            #   - scheduler_type: Can be different from original
            #   - scheduler_params: Can be different from original
            load_scheduler_and_optimizer=True,
            
            # Optimizer parameters
            learning_rate=1e-3,  # MODE A: None=use checkpoint LR, or specify to override | MODE B: REQUIRED, must specify
            weight_decay=1e-4,   # MODE A: None=use checkpoint weight_decay, or specify to override | MODE B: REQUIRED, must specify
            optimizer_type='AdamW',  # MODE A: must match original | MODE B: can be different
            
            # Scheduler parameters  
            scheduler_type='plateau',  # MODE A: must match original | MODE B: can be different
            scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 100, 'verbose': True},  # MODE A: can differ. Functionality would depend on reset_scheduler_patience | MODE B: can be different
            reset_scheduler_patience = True, #Only relevant on MODE A. Set True to reset num_bad_epochs. Use True if you want the learning rate to be lowered after the full patience amount. Use False if you want continuity, already waited N epochs, just need M-N more where M: loaded num_bad_epochs from previous training
            
            # Training parameters
            num_epochs=400,  # Number of ADDITIONAL epochs to train 
            grad_clip_value=60.0, #Use None if you dont want gradient clipping, specify to use torch.nn.utils.clip_grad_norm_ in MLP parameters
            
            # Early stopping parameters
            early_stopping=True,
            patience=200,
            min_delta=0.001,
            
            # Checkpointing
            save_dir=save_dir_4,  # Should typically match the directory where checkpoint_path is located
            save_best_only=False,
            save_freq_epochs=10,
            auto_rescue=True,
            
            # Logging
            log_freq_batches=10,
            verbose=1,
            
            # Device
            device=device
        )

'''
def main():
    resume_training = True

    device = "cuda"

    if resume_training == False:
        training_seed = 42
    else:
        resume_training_counter = 1
        training_seed = 42 + resume_training_counter
    
    train_dataloader = create_simple_dataloader(
    train_df=train_df,
    train_id_df=train_id_df,
    ratio=0.041667,
    batch_size=24,
    segment_length=1,
    get_data_func=get_data_from_trajectory_id,
    device=device,
    seed=training_seed
    )

    val_dataloader = create_val_dataloader_full_trajectory(
    val_df=val_df,
    val_id_df=val_id_df,
    get_data_func=get_data_from_trajectory_id,
    device=device,
    seed=42
    )




    val_dataloader_training_set = create_val_dataloader_full_trajectory(
    val_df=train_df,
    val_id_df=train_id_df,
    get_data_func=get_data_from_trajectory_id,
    device=device,
    seed=42
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



    possible_t_values = get_data_from_trajectory_id(ids_df=train_id_df, data_df=train_df, trajectory_ids=30)['t'].values.tolist() #Same possible t values for every trajectory
    normalized_mse = False
    loss_type = "mae" #Either 'mae' or 'mse'
    predict_full_trajectory = True

    save_dir_1 = "./save_directory_low_samples_1" 
    save_dir_2 = "./save_directory_noisy_low_samples_2" 
    
    save_dir_3 = "./save_directory_noisy_full_pred_low_samples_3" 

    save_dir_4 = "./save_directory_noisy_full_pred_low_samples_4" 




    

    if not resume_training:

        inverse_net = InverseStackedHamiltonianNetwork(forward_network=mapping_net)

        derived_prediction_loss_scale = calculate_losses_scale_on_untrained(train_loader=train_dataloader, mapping_net=mapping_net, inverse_net=inverse_net, get_data_from_trajectory_id=get_data_from_trajectory_id, possible_t_values=possible_t_values, train_df=train_df, train_id_df=train_id_df, loss_type=loss_type, predict_full_trajectory=predict_full_trajectory, add_noise=add_noise, normalized_mse=normalized_mse, save_returned_values=True, save_dir=save_dir_4, noise_threshold_mean_divided_by_std = 2, device=device)

        train_model(
            # Dataloaders
            train_loader=train_dataloader,
            randomize_each_epoch_plan=True,
            val_loader=val_dataloader, 

            val_loader_training_set=val_dataloader_training_set,

            #Network
            mapping_net=mapping_net, 
            inverse_net=inverse_net,

            #Needed objects
            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values=possible_t_values,

            #Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,

            val_df=val_df,
            val_id_df=val_id_df,



            add_noise = add_noise,


            #Loss calculation hyperparameters
            prediction_loss_scale=derived_prediction_loss_scale,
            normalized_mse=normalized_mse,
            loss_type=loss_type,
            predict_full_trajectory=predict_full_trajectory,


            prediction_coefficient=2.0,
            hsic_loss_max_want=0.1,



            on_distribution_val_criterio_weight = 1.0,


            # Optimizer parameters
            optimizer_type = 'AdamW',

            learning_rate=3e-3,
            weight_decay=1e-4,

            scheduler_type='plateau', 
            scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 600, 'verbose': True},    # Dict of params specific to the scheduler. Set if it is a fresh training session.


            # Training parameters
            num_epochs=600,
            grad_clip_value=60.0,


            # Early stopping parameters
            early_stopping=True,
            patience=600,
            min_delta=0.001,

            # Checkpointing
            save_dir=save_dir_4,
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

        checkpoint_path = os.path.join(save_dir_4, "best_model.pt")

        loss_scales_save_path = os.path.join(save_dir_4, "loss_scales.pkl")

        with open(loss_scales_save_path, "rb") as f:
            loss_scales = pickle.load(f)
        
        saved_prediction_loss_scale = loss_scales['saved_prediction_loss_scale']


        resume_training_from_checkpoint(
            # Checkpoint to resume from
            checkpoint_path=checkpoint_path,
            
            # Dataloaders
            train_loader=train_dataloader,
            randomize_each_epoch_plan=True,
            val_loader=val_dataloader,

            val_loader_training_set=val_dataloader_training_set,
            
            # Network
            mapping_net=mapping_net,  # Must be same architecture as original. inverse_net is created automatically from mapping_net
            
            # Needed objects
            get_data_from_trajectory_id=get_data_from_trajectory_id,
            possible_t_values=possible_t_values,
            
            # Needed pandas dataframes
            train_df=train_df,
            train_id_df=train_id_df,

            val_df=val_df,
            val_id_df=val_id_df,



            add_noise = add_noise,
            
            # Loss calculation hyperparameters
            prediction_loss_scale=saved_prediction_loss_scale,
            normalized_mse=normalized_mse,
            loss_type=loss_type,
            predict_full_trajectory=predict_full_trajectory,


            prediction_coefficient=2.0,
            hsic_loss_max_want=0.1,


            on_distribution_val_criterio_weight=1.0,
            
            # === RESUME MODE SELECTION ===
            # MODE A (load_scheduler_and_optimizer=True): Resume with exact optimizer/scheduler state
            
            #   - learning_rate: Set to None to use checkpoint LR, or specify to override
            #   - optimizer_type: MUST match the type used in original training
            #   - scheduler_type: MUST match the type used in original training
            #  
            #
            # MODE B (load_scheduler_and_optimizer=False): Resume with fresh optimizer/scheduler

            #   - learning_rate: REQUIRED - must specify, will error if None
            #   - optimizer_type: Can be different from original
            #   - scheduler_type: Can be different from original
            #   - scheduler_params: Can be different from original
            load_scheduler_and_optimizer=True,
            
            # Optimizer parameters
            learning_rate=1e-5,  # MODE A: None=use checkpoint LR, or specify to override | MODE B: REQUIRED, must specify
            weight_decay=1e-4,   # MODE A: None=use checkpoint weight_decay, or specify to override | MODE B: REQUIRED, must specify
            optimizer_type='AdamW',  # MODE A: must match original | MODE B: can be different
            
            # Scheduler parameters  
            scheduler_type='plateau',  # MODE A: must match original | MODE B: can be different
            scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 100, 'verbose': True},  # MODE A: can differ. Functionality would depend on reset_scheduler_patience | MODE B: can be different
            reset_scheduler_patience = True, #Only relevant on MODE A. Set True to reset num_bad_epochs. Use True if you want the learning rate to be lowered after the full patience amount. Use False if you want continuity, already waited N epochs, just need M-N more where M: loaded num_bad_epochs from previous training
            
            # Training parameters
            num_epochs=300,  # Number of ADDITIONAL epochs to train 
            grad_clip_value=60.0, #Use None if you dont want gradient clipping, specify to use torch.nn.utils.clip_grad_norm_ in MLP parameters
            
            # Early stopping parameters
            early_stopping=True,
            patience=100,
            min_delta=0.001,
            
            # Checkpointing
            save_dir=save_dir_4,  # Should typically match the directory where checkpoint_path is located
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



    
