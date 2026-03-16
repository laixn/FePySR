from pysr import PySRRegressor
import time
import numpy as np
from omegaconf import OmegaConf

model_default = PySRRegressor(
    populations=8,
    population_size=50,
    ncycles_per_iteration=500,
    niterations=2000,  # Run forever
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-9 && complexity < 40"
    ),
    timeout_in_seconds=60 * 60 * 24,
    maxsize=25,
    maxdepth=10,
    binary_operators=["*", "+", "-"],
    unary_operators=[
        "cos",
        "sqrt",
        # "log",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # constraints={
    #     "/": (-1, 9),
    #     "square": 9,
    #     "cube": 9,
    #     "exp": 9,
    # },
    nested_constraints={
        "exp": {"exp": 0},  # Prohibit nesting like exp(exp(x))
        # "inv": {"inv": 0},  # Prohibit nesting like inv(inv(x))
    },
    # nested_constraints={
    #     "square": {"square": 1, "cube": 1, "exp": 0},
    #     "cube": {"square": 1, "cube": 1, "exp": 0},
    #     "exp": {"square": 1, "cube": 1, "exp": 0},
    # },
    # complexity_of_operators={"/": 2, "exp": 3},
    # complexity_of_constants=2,
    # select_k_features=4,
    # progress=True,
    # weight_randomize=0.1,
    # cluster_manager=None,
    # precision=64,
    # warm_start=True,
    # turbo=True
)

def R_scor(y_true,y_pred):
    """
    Calculate the standard R^2 coefficient of determination.
    
    Args:
        y_true (np.ndarray): Ground truth label values.
        y_pred (np.ndarray): Model predicted values.
        
    Returns:
        float: R^2 score. The closer to 1, the better the fit.
    """
    # Step 1: Calculate the total sum of squares (SStot)
    y_pred = y_pred.reshape(y_true.shape)
    epsilon = 1e-8
    y_mean = np.mean(y_true)
# The numerator is the squared residual of each point
    numerators = (y_true - y_pred)**2
# The denominator is the squared difference between each point and the mean
    denominators = (y_true - y_mean)**2

    # Step 3: Calculate R^2
    r2 = 1 - np.sum((numerators /  (denominators+epsilon)))
    return r2

def pysr_train(data_analyzer,cfg,model=None):
    """
    Run PySR symbolic regression training.
    Supports dynamically loading parameters from the Hydra configuration, or passing in an instantiated model.

    Args:
        data_analyzer (DataAnalyzer): Object containing preprocessed data.
        num_features (int): The top K number of features to intercept for training.
        cfg (DictConfig, optional): Hydra configuration object, must contain the pysr_params node.
        model (PySRRegressor, optional): An already initialized model passed directly from the outside.

    Returns:
        tuple: (Optimal PySR model equation object, running time (seconds), R^2 evaluation score)
    """

    num_features = cfg.data_symbol.pysr_num

    if not model:
        if cfg and "pysr_params" in cfg:
            # Core: Convert the Hydra configuration into a native dictionary, and unpack it to pass as arguments to PySR
            # resolve=True ensures interpolations in the configuration are correctly parsed
            pysr_kwargs = OmegaConf.to_container(cfg.pysr_params, resolve=True)
            
            # Handle special parameters in the configuration that cannot be directly expressed in yaml (e.g., lambda functions)
            if "inv(x) = 1/x" in pysr_kwargs.get("unary_operators", []):
                pysr_kwargs["extra_sympy_mappings"] = {"inv": lambda x: 1 / x}
                
            model = PySRRegressor(**pysr_kwargs)
        else:
            raise ValueError("必须提供实例化的 model,或者在 cfg 中包含 pysr_params 配置！")

    # 2. Prepare data
    X_train = data_analyzer.stacked_numpy_features[:, :num_features]

    if hasattr(data_analyzer.labels, "numpy"):
        y_train = data_analyzer.labels.detach().cpu().numpy()
    else:
        y_train = data_analyzer.labels

    # 3. Execute training and time it
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    running_time = end_time - start_time

    y_pred=model.predict(data_analyzer.stacked_numpy_features[:,:num_features])
    reward=R_scor(data_analyzer.labels.numpy(),y_pred)
    best_model_equation = model.get_best()
            

    return best_model_equation,running_time,reward,model