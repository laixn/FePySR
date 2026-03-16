from omegaconf import OmegaConf
import fepysr.functions as fun

def init_fmn_config(cfg, data_dim):
    """
    Initialize the structure configuration of the Feature Mapping Network (FMN) and mount it to cfg.
    
    Dynamically build the pool of activation/operation functions required for each layer of the network,
    based on the input data dimension (data_dim) and whether the user provided a custom function list.

    Args:
        cfg: The composed Hydra configuration object.
        data_dim (int): The dimension of the input features.

    Returns:
        cfg: The updated configuration object with the FMN network structure.
    """
    # Check if the user has provided a custom function list in the configuration
    if cfg.FMN.net.fun_list:
        if data_dim == 1:
            # When the dimension is 1, use special layer building logic
            network_layers = _build_1d_network_layers(cfg)
        else:
            # For multi-dimensional data, use the full set of custom functions for each layer, repeating each function 4 times
            # Use a more Pythonic list comprehension, iterating through elements directly rather than indices
            layer_funcs = [
                fun.func_mapping[func_name] 
                for func_name in cfg.FMN.net.fun_list 
                for _ in range(4)
            ]
            network_layers = [layer_funcs] * cfg.FMN.net.net_depth
    else:
        # The user did not provide a custom list; use the default function pool strategy
        if data_dim == 1:
            # Assume default_func1 is for the first layer, and default_func2 is for subsequent layers
            network_layers = [fun.default_func1] + [fun.default_func2] * 2
        else:
            network_layers = [fun.default_func2] * 4
            
        # When using the default strategy, automatically update the network's depth parameter based on the generated list length
        cfg.FMN.net.net_depth = len(network_layers)

    # Forcibly mount the generated network layer list back into the configuration as a Python object
    cfg.FMN.net.full_net = OmegaConf.create(network_layers, flags={"allow_objects": True})
    
    return cfg
    
def _build_1d_network_layers(cfg):
    """
    Build a custom network function list for input data with a dimension of 1.
    
    Building strategy:
    - Layer 1: Only filter functions where dim=1.
    - Subsequent layers: Use the full set of functions.
    - All functions will be repeated 4 times in their corresponding layers.

    Args:
        cfg: Configuration object, must contain FMN.net.fun_list and FMN.net.net_depth.

    Returns:
        list: Nested function list, where the outer length is net_depth, and the inner layer is the function pool for that layer.
    """
    # 1. Part One: Filter functions with dim=1, repeat each function 4 times, and generate a single-layer nested list
    first_layer_funcs = [
        [
            func_mapping[cfg.FMN.net.fun_list[i]]
            for i in range(len(cfg.FMN.net.fun_list))
            for _ in range(4)  # Repeat each function 4 times
            if func_mapping[cfg.FMN.net.fun_list[i]].dim == 1  # Filter functions with dim=1
        ]
    ]
    
    # 2. Part Two: Full function list, repeat each function 4 times, and repeat for subsequent (net_depth-1) layers
    full_funcs = [
        [
            func_mapping[cfg.FMN.net.fun_list[i]]
            for i in range(len(cfg.FMN.net.fun_list))
            for _ in range(4)  # Repeat each function 4 times
        ]
    ]
    repeated_full_funcs = full_funcs * (cfg.FMN.net.net_depth - 1)  # Repeat concatenation (net_depth-1) times
    
    # 3. Concatenate the two parts to generate the final default list (use parentheses to clarify the concatenation logic and avoid syntax ambiguity)
    default = first_layer_funcs + repeated_full_funcs
    
    return default