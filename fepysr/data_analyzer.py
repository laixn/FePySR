import torch
import numpy as np

class DataAnalyzer:
    """
    Core data analysis and preprocessing class for FePySR.
    Used to manage the input feature matrix, labels, and the corresponding symbol (variable name) mappings.
    """
    
    def __init__(self, features, labels, feature_names=None):
        # self.features = features
        # self.labels = labels
        self.features = self._standardize_data(features, is_feature=True)
        self.labels = self._standardize_data(labels, is_feature=True)
        # Use shape instead of size() to ensure compatibility with both PyTorch Tensors and NumPy Arrays
        self.dim = self._detect_dim()
        
        # Pre-declare class attributes to avoid structural confusion caused by dynamic creation in external methods
        self.feature_names = None
        self.feature_dict = None
        self.current_feature_names = []
        self.stacked_numpy_features = None

        # Initialize the symbol list and feature dictionary
        self.feature_names, self.feature_dict = self._create_feature_dict(feature_names)
 
    def _standardize_data(self, data, is_feature=True):
        """
        数据安检口：
        1. 统一类型：将 NumPy array, List 等统一转为 PyTorch Tensor，并强制设为 float64 (double)。
        2. 统一维度：将 1D 数据 (N,) 强制转化为 2D 矩阵 (N, 1)，防止后续矩阵运算报错。
        """
        # --- 第 1 步：类型转换 ---
        if not isinstance(data, torch.Tensor):
            # 如果是 numpy，会自动转换为 tensor
            data = torch.tensor(data, dtype=torch.float64)
        else:
            # 如果已经是 tensor，确保它的类型是 float64
            data = data.to(torch.float64)

        # --- 第 2 步：维度对齐 ---
        if data.dim() == 1:
            # 解决 torch.randn(10) 的问题：把 (10,) 变成 (10, 1)
            data = data.view(-1, 1)
        elif data.dim() == 0:
            raise ValueError("FePySR: Input data cannot be a scalar (0D tensor).")

        # 针对标签 (y) 的特殊处理：确保它是 (N, 1) 的列向量，对 PyTorch 的 loss 计算最友好
        if not is_feature and data.shape[1] > 1:
            # 防止用户误传了行向量 shape (1, N)
            if data.shape[0] == 1:
                data = data.view(-1, 1)
            else:
                raise ValueError("FePySR: The labels (y) must be a 1D array or a 2D column vector.")

        return data

    def _detect_dim(self):
        """
        Get the number of features (dimension).
        """
        return self.features.shape[-1]

    def _create_feature_dict(self, feature_names):
        """
        Split the feature matrix column-wise and map them to variable names in a dictionary.
        """
        # Validation check: Ensure the number of provided symbols matches the feature dimension
        if feature_names is not None:
            if len(feature_names) != self.dim:
                raise ValueError(f"The number of provided feature names ({len(feature_names)}) does not match the feature dimension ({self.dim})!")
            sym = feature_names
        else:
            # Default generation: X0, X1, X2...
            sym = [f"X{i}" for i in range(self.dim)]
            
        # Split the tensor column-wise and pack it into a dict, preserving the 2D slice shape, e.g., (N, 1)
        feature_dict = {sym[i]: self.features[:, [i]] for i in range(self.dim)}
        
        return sym, feature_dict
    
    def get_results(self):
        """Return the core state dictionary of the current data analyzer."""
        return {
            'dim': self.dim,
            'features': self.features,
            'labels': self.labels,
            'feature_names': self.feature_names,
            'feature_dict': self.feature_dict,
        }

    def stack_features_to_numpy(self):
        """
        Extract all tensors from the feature dictionary, stack them column-wise, and convert them to a NumPy array.
        
        This method is typically called before feeding data into PySR training. It re-aggregates 
        the scattered feature dictionary into a 2D NumPy feature matrix.
        """
        self.current_feature_names = []
        tensor_list = []
        
        # Extract variable names and corresponding tensor data from the dictionary
        for sym_str, sym_data in self.feature_dict.items():    
            self.current_feature_names.append(sym_str)
            tensor_list.append(sym_data)
        
        # Stack tensors column-wise
        stacked_tensor = torch.stack(tensor_list, dim=1)
        
        # Add .detach().cpu() to safely convert to NumPy regardless of device (CPU/GPU) or gradient requirements
        stacked_np_array = stacked_tensor.detach().cpu().numpy()
        
        # Remove the extra dimension introduced by slicing, e.g., from (n_samples, n_features, 1) to (n_samples, n_features)
        self.stacked_numpy_features = np.squeeze(stacked_np_array, axis=-1) 
        
        return self.current_feature_names, self.stacked_numpy_features