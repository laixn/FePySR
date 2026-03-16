#训练Feature代码
import torch
import torch.nn as nn
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
from . import feature_mapping_network
from . import optimization
from .feature_maker import extract_layer_expressions,merge_experiment_results

def _train_single_fmn(features,labels,feature_names,cfg):
    """
    Executes a single training experiment of the Feature Mapping Network (FMN) and extracts symbolic expressions.
    
    Args:
        features (Tensor): Input feature tensor.
        labels (Tensor): Target label tensor.
        feature_names (list): List of feature names (i.e., symbols of the original variables).
        cfg (DictConfig): Hydra configuration object containing hyperparameters.
        
    Returns:
        list: A nested list containing the candidate symbolic expressions extracted from each layer of the network.
    """
    # 1. Parse hyperparameters
    batch_size = cfg.FMN.batch_size
    lr = cfg.FMN.lr
    num_epochs = cfg.FMN.num_epochs
    loss_str =cfg.FMN.loss
    lambda1=cfg.FMN.loss_parameter.lambda1
    lambda2=cfg.FMN.loss_parameter.lambda2


    # 2. Dynamically load the loss function
    try:
        module_name, func_name = loss_str.rsplit('.', 1)
        my_module = importlib.import_module(module_name)
        criterion = getattr(my_module, func_name)
    except (ValueError, AttributeError, ModuleNotFoundError) as e:
        raise ImportError(f"Cannot load loss function '{loss_str}', please check the spelling in the Hydra configuration file. Details: {e}")

    feature_names_all=[[] for _ in range(cfg.FMN.net.net_depth)]
    full_net = feature_mapping_network.Symnet_all(
        features.shape[1], 
        cfg.FMN.net.net_depth, 
        cfg.FMN.net.full_net
    )    
    full_net.double()  

    for epoch in range(num_epochs):
        for X, y_true in optimization.data_iter(batch_size, features, labels):
            y_pred=full_net(X)
            total_loss = (
                criterion(y_pred, y_true)
                + lambda1*full_net.loss_cor()
                + lambda2*full_net.sparsification_net()
            )

            total_loss.sum().backward()

            nn.utils.clip_grad_norm_(full_net.parameters(), max_norm=1.0)

            optimization.max_norm_sgd(full_net.parameters(), lr, batch_size)  

    # 5. After training, extract specific mathematical symbolic expressions from the converged network weights
    extract_layer_expressions(feature_names_all,cfg.FMN.net.net_depth,feature_names,full_net)
    return feature_names_all

def run_experiments(data_analyzer,cfg):
    """
    Run the complete feature extraction experiment.
    Supports running the FMN network multiple times in parallel via multiprocessing to eliminate the random variance of neural network training,
    and finally merges the high-quality features from all run results.

    Args:
        data_analyzer (DataAnalyzer): Preprocessed data object containing features, labels, and dimensional information.
        cfg (DictConfig): Global configuration object containing hyperparameters for parallel computation.

    Returns:
        list: List of all candidate feature expressions merged after multiple experiments.
    """

    num_experiments=cfg.Parallel.num_experiments
    num_workers=cfg.Parallel.num_workers

    print(f"--- Starting Feature Mapping Network ---")
    print(f"Set number of runs: {num_experiments}, Allocated number of workers: {num_workers}")
    # Use partial to freeze fixed parameters for convenient passing into the process pool
    prepared_task = partial(
        _train_single_fmn, 
        features=data_analyzer.features, 
        labels=data_analyzer.labels, 
        feature_names=data_analyzer.feature_names, 
        cfg=cfg
    )

    all_results = []

    # Determine whether to use serial or parallel computing based on the number of worker processes
    if num_workers > 1:
        print("Mode: Parallel computation")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(prepared_task) for _ in range(num_experiments)]
            for future in as_completed(futures):
                all_results.append(future.result())
    else:
        print("Mode: Serial computation")
        for i in range(num_experiments):
            result = prepared_task( )
            all_results.append(result)
    
    merged_results=merge_experiment_results(all_results)
    return merged_results



# def _train_single_fmn(features,labels,feature_names,cfg):
#     """
#     Execute a single training experiment of the Feature Mapping Network (FMN) and extract symbolic expressions.
    
#     Args:
#         features (Tensor): Input feature tensor.
#         labels (Tensor): Target label tensor.
#         feature_names (list): List of feature names (i.e., symbols of the original variables).
#         cfg (DictConfig): Hydra configuration object containing hyperparameters.
        
#     Returns:
#         list: A nested list containing candidate symbolic expressions extracted from each layer of the network.
#     """
#     # 1. Parse hyperparameters
#     batch_size = cfg.FMN.batch_size
#     lr = cfg.FMN.lr
#     num_epochs = cfg.FMN.num_epochs
#     loss_str =cfg.FMN.loss
#     lambda1=cfg.FMN.loss_parameter.lambda1
#     lambda2=cfg.FMN.loss_parameter.lambda2

#     # Use the get method to prevent errors if the device field is missing in older config files, defaulting to "auto"
#     device_preference = str(cfg.FMN.get("device", "auto")).lower()
    
#     if device_preference == "cpu":
#         device = torch.device("cpu")
#     elif device_preference == "cuda" and torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if device_preference == "cuda" and device.type == "cpu":
#             print("⚠️ Warning: 'cuda' was specified in the config, but no available GPU was detected in the current environment. Automatically falling back to 'cpu' safely.")

#     # 2. Dynamically load the loss function
#     try:
#         module_name, func_name = loss_str.rsplit('.', 1)
#         my_module = importlib.import_module(module_name)
#         criterion = getattr(my_module, func_name)
#     except (ValueError, AttributeError, ModuleNotFoundError) as e:
#         raise ImportError(f"Failed to load the loss function '{loss_str}', please check the spelling in the Hydra configuration file. Details: {e}")

#     feature_names_all=[[] for _ in range(cfg.FMN.net.net_depth)]
#     full_net = feature_mapping_network.Symnet_all(
#         features.shape[1], 
#         cfg.FMN.net.net_depth, 
#         cfg.FMN.net.full_net
#     )    
#     full_net.double()  
    
#     # Push the model to the selected device
#     full_net = full_net.to(device)

#     for epoch in range(num_epochs):
#         for X, y_true in optimization.data_iter(batch_size, features, labels):
#             X = X.to(device)
#             y_true = y_true.to(device)
            
#             y_pred=full_net(X)
#             total_loss = (
#                 criterion(y_pred, y_true)
#                 + lambda1*full_net.loss_cor()
#                 + lambda2*full_net.sparsification_net()
#             )

#             total_loss.sum().backward()

#             nn.utils.clip_grad_norm_(full_net.parameters(), max_norm=1.0)

#             optimization.max_norm_sgd(full_net.parameters(), lr, batch_size)  

#     # 5. After training, extract specific mathematical symbolic expressions from the converged network weights
#     extract_layer_expressions(feature_names_all,cfg.FMN.net.net_depth,feature_names,full_net)
    
#     return feature_names_all

# def run_experiments(data_analyzer,cfg):
    """
    Run the complete feature extraction experiment.
    Supports running the FMN network multiple times in parallel via multiprocessing to eliminate random variance in neural network training, and ultimately merges the high-quality features from all run results.

    Args:
        data_analyzer (DataAnalyzer): Preprocessed data object containing features, labels, and dimensional information.
        cfg (DictConfig): Global configuration object containing hyperparameters for parallel computing.

    Returns:
        list: List of all candidate feature expressions merged after multiple experiments.
    """

    num_experiments=cfg.Parallel.num_experiments
    num_workers=cfg.Parallel.num_workers

    print(f"--- Starting Feature Mapping Network ---")
    print(f"Target runs: {num_experiments}, Allocated workers: {num_workers}")

    # Use partial to freeze fixed parameters for easy passing into the process pool
    prepared_task = partial(
        _train_single_fmn, 
        features=data_analyzer.features, 
        labels=data_analyzer.labels, 
        feature_names=data_analyzer.feature_names, 
        cfg=cfg
    )

    all_results = []

    # Read config to determine if GPU-level multiprocessing is needed
    device_preference = str(cfg.FMN.get("device", "auto")).lower()
    use_gpu_multiprocessing = False
    if (device_preference == "cuda" or device_preference == "auto") and torch.cuda.is_available():
        use_gpu_multiprocessing = True

    # Determine whether to run serially or in parallel based on the number of worker processes
    if num_workers > 1:
        # Dynamically select multiprocessing engine based on device
        if use_gpu_multiprocessing:
            print("Mode: Parallel Computing (GPU Safe Mode - Spawn)")
            mp_context = mp.get_context('spawn')
            executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context)
        else:
            print("Mode: Parallel Computing (Pure CPU Lightweight Mode)")
            # In CPU mode, use native process pool directly for lower overhead and faster speed
            executor = ProcessPoolExecutor(max_workers=num_workers)
            
        with executor:
            futures = [executor.submit(prepared_task) for _ in range(num_experiments)]
            for future in as_completed(futures):
                all_results.append(future.result())
    else:
        print("Mode: Serial Computing (using for-loop)")
        for i in range(num_experiments):
            result = prepared_task()
            all_results.append(result)
    
    merged_results=merge_experiment_results(all_results)
    return merged_results