"""
HJN config file template — fill in with your project's imports and parameters.

This file is consumed by time_one_sweep_pytorch.py via --config-py.

It must define exactly three things:

  1. build_networks(device, checkpoint_path) -> (mapping_net, inverse_net)
       Returns models on `device` with checkpoint weights loaded.

  2. get_data_from_trajectory_id(ids_df, data_df, trajectory_ids)
       Your existing data accessor function.

  3. Batched_Autoregressive_Handoff_Inference_Protocol(...)
       Your existing inference function.

You can either copy-paste your functions/classes into this file, or
re-export them from your project modules (the second is cleaner).
"""

# ============================================================
# Imports from your project (adjust paths as needed)
# ============================================================
import sys
sys.path.append("/absolute/path/to/your/project/root")  # so the imports below work

from modules import (
    SimpleStackedHamiltonianNetwork,
    InverseStackedHamiltonianNetwork,
    load_checkpoint,
    get_data_from_trajectory_id,
    Batched_Autoregressive_Handoff_Inference_Protocol
)




# ============================================================
# build_networks: must match the signature the script expects
# ============================================================
def build_networks(device, checkpoint_path):
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
    #final_activation_only_on_final_layer = True,
    tanh_wrapper = False,
        
        # Initialization parameters
    weight_init = 'orthogonal',
    weight_init_params = {'gain': 1},
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

    step_1_a_mean_innit= 1.2,
    step_2_a_mean_innit= 1.2,
    std_to_mean_ratio_a_mean_init= 0.0,

    step_1_gamma_mean_innit= 0.5,
    step_2_gamma_mean_innit= 0.5,
    std_to_mean_ratio_gamma_mean_init= 0.0,

    step_1_c1_mean_innit= 0.0,
    step_2_c1_mean_innit= 0.0,
    std_to_mean_ratio_c1_mean_init= 1.0,

    step_1_c2_mean_innit= 0.0,
    step_2_c2_mean_innit= 0.0,
    std_to_mean_ratio_c2_mean_init= 1.0,

    bound_innit=0.0,
    ).to(device)

    load_checkpoint(
        path=checkpoint_path,
        mapping_net=mapping_net,
        device=device,
        optimizer=None,
        scheduler=None,
    )

    inverse_net = InverseStackedHamiltonianNetwork(forward_network=mapping_net)
    return mapping_net, inverse_net