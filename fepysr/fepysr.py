import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

import torch
from fepysr.fmn_train import run_experiments
import fepysr.feature_maker as fm
from fepysr.data_analyzer import DataAnalyzer
from fepysr.pysr_train import pysr_train
import hydra
from fepysr.fmn_builder import init_fmn_config


class FePySR:
    """
    FePySR (Feature Engineering + PySR) symbolic regression estimator.
    
    Adopts a two-stage algorithm architecture:
    1. Neural network feature extraction and construction (Feature Making)
    2. Genetic algorithm-based symbolic regression (Symbolic Regression)
    
    Designed following the scikit-learn API specifications.
    """

    def __init__(self, overrides=None, custom_pysr_model=None):
        """
        Initialize the FePySR framework.
        
        Args:
            overrides (list, optional): Hydra parameter override list, e.g., ["FMN.lr=0.01"]. Defaults to None.
            custom_pysr_model (PySRRegressor, optional): User-defined PySR model instance.
                If provided, the PySR parameters in the configuration will be ignored. Defaults to None.
        """
        # 1. Record the passed hyperparameters (sklearn specification: only assignments are made in __init__)
        self.overrides = overrides or []
        self.custom_pysr_model = custom_pysr_model
        
        # 2. Pre-declare attributes that will be generated after training (sklearn specification: ending with _)
        self.cfg_ = None
        self.best_model_ = None
        self.best_equation_ = None
        self.feature_names_ = None
        self.all_feature_names_ = None
        self.reward_ = None
        self.running_time_ = None
        self.solved_pysr_model = None
        self.data_analyzer=None
        self.enriched_data_=None

    def fit(self, X, y):
        """
        Train the model: extract features and perform symbolic regression.
        
        Args:
            X (Array-like): Training input samples, shape (n_samples, n_features).
            y (Array-like): Target labels, shape (n_samples,).
            
        Returns:
            self: Returns the model instance itself (conforming to sklearn chaining conventions).
        """
        # ==========================================
        # Step 0: Safely initialize Hydra configuration
        # ==========================================
        # Add foolproof design: allow users to call fit() multiple times in the same code/Notebook without throwing errors
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
            
        with hydra.initialize(config_path="config", version_base="1.1"):
            self.cfg_ = hydra.compose(
                config_name="config_regression", 
                overrides=self.overrides
            )

        # Call core training logic
        self._core_train(X, y)
        
        # Return self to support chaining operations, e.g., eq = FePySR().fit(X, y).best_equation_
        return self

    def _core_train(self, X, y):
        """
        Internal core training pipeline, called by the fit() method.
        """
        # 1. Data initialization and preprocessing
        self.data_analyzer = DataAnalyzer(X, y, self.cfg_.data_symbol.output_symbols)

        # 2. Dynamically build network configuration
        self.cfg_ = init_fmn_config(self.cfg_, self.data_analyzer.dim)

        # 3. Stage 1: Neural network feature extraction
        candidate_features = run_experiments(self.data_analyzer, self.cfg_)
        enriched_data, f_names, all_f_names = fm.feature_maker(
            self.cfg_, candidate_features, self.data_analyzer
        )

        self.feature_names_ = f_names
        self.all_feature_names_ = all_f_names
        self.data_analyzer = enriched_data

        if self.cfg_.FMN.FMN_only:
            print(f"[FePySR] FMN_only mode activated. Extracted {len(f_names)} top features. Skipping PySR training.")
            return

        # 4. Stage 2: PySR symbolic regression
        best_model, run_time, reward , pysr_model = pysr_train(
            self.data_analyzer, self.cfg_, self.custom_pysr_model
        )
        
        # 5. Result post-processing (replace variable names)
        final_eq = fm.replace_pysr_variables(best_model.equation, f_names)
        
        # 6. Save training results as instance attributes
        self.best_model_ = best_model
        self.running_time_ = run_time
        self.reward_ = reward
        self.best_equation_ = final_eq
        self.solved_pysr_model=pysr_model

    def predict(self, X=None):
        """
        Directly evaluate data tensors using the discovered symbolic formula.
        
        Args:
            X (Tensor or Array-like, optional): New test data.
                If None, directly use the training data (self.data_analyzer.feature_dict) for calculation.
                
        Returns:
            torch.Tensor: Prediction results.
        """
        if self.best_equation_ is None:
            raise ValueError("The model has not been trained yet! No mathematical formula available.")

        # 1. Syntax adaptation: If ** was replaced by ^ during simplification, it needs to be replaced back here, as Python only recognizes **
        expr_str = self.best_equation_.replace('^', '**')

        # 2. Prepare the execution environment (inject PyTorch mathematical operators)
        # This way, when eval reads "sin(X0)" in the string, it will automatically call torch.sin
        eval_env = {
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': torch.exp,
            'log': torch.log,
            'abs': torch.abs,
            'sqrt': torch.sqrt,
        }

        # 3. Prepare data dictionary
        if X is None:
            # [Your idea]: Directly reuse the data stored in the dictionary during training
            data_dict = self.data_analyzer.feature_dict
        else:
            # If the user passes new data, temporarily slice it to construct a dictionary of the same format
            if not torch.is_tensor(X):
                X = torch.tensor(X, dtype=torch.float64)
            dim = X.shape[1]
            data_dict = {f"X{i}": X[:, [i]] for i in range(dim)}

        # Stuff the data dictionary into the execution environment
        eval_env.update(data_dict)

        # 4. Core magic: Use eval to directly execute tensor operations
        try:
            # {"__builtins__": None} restricts eval from accessing system built-in functions to ensure code safety
            y_pred = eval(expr_str, {"__builtins__": None}, eval_env)
        except Exception as e:
            raise RuntimeError(f"Formula evaluation failed, please check if the formula contains unsupported functions. Error: {e}")

        # 5. Foolproof: If the formula is a pure constant (e.g., "2.5"), expand it to a Tensor of the corresponding length
        if isinstance(y_pred, (int, float)):
            sample_length = list(data_dict.values())[0].shape[0]
            y_pred = torch.full((sample_length, 1), y_pred, dtype=torch.float64)

        return y_pred