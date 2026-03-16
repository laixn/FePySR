## FePySR: Feature Mapping Network-PySR
FePySR: A Two-Stage Symbolic Regression Framework via Feature Engineering

## Why FePySR?
Feature Mapping Network-PySR is a method that addresses the core challenges of the NP-hard problem in symbolic regression — the combinatorial explosion of variable selection and the inherent difficulties in function structure search — by proposing an innovative feature space reconstruction strategy, thus providing an efficient solution for automated scientific discovery. This method compensates for the shortcomings of existing methods in strongly nonlinear problems, can deeply empower various advanced search algorithms, and offers a brand-new breakthrough for the development of the symbolic regression field.


While traditional Symbolic Regression (SR) struggles with high-dimensional inputs and vast search spaces, FePySR bridges the gap between deep learning and symbolic logic:

1. **Neural-Guided Library Augmentation**: Instead of blindly searching, Stage 1 trains a sparse FMN to learn a "library" of critical basis functions and composite features directly from the data.
2. **Exponentially Reduced Search Space**: By feeding these pre-constructed features into PySR (Stage 2), FePySR solves complex physics and biological kinetics problems that traditional SR cannot reach.
3. **Scikit-Learn API**: Built for practitioners. Just call `.fit()` and `.predict()`.
4. **Declarative Configuration**: Powered by `hydra`, you can customize neural network depth, operator pools, and parallel workers without modifying a single line of source code.
5. **CPU-Optimized Concurrency**: Safe, fast, and lightweight native multi-processing. No CUDA/GPU headaches required.

---

## How It Works 
FePySR operates in a pipeline:
* **Stage 1: Feature Mapping Network (FMN)**. A PyTorch-based symbolic neural network uses sparsity-inducing regularization to learn meaningful mathematical mappings. It extracts top-tier candidate expressions (e.g., $\sin(x_1) \cdot x_2^2$).
* **Stage 2: PySR Backend**. The extracted features act as new basis variables for PySR's high-performance Julia backend, accelerating the evolutionary search for the final, elegant governing equation.

---

## The current version is to provide reviewers with reproducible experimental results
```bash
/home/yioily/SymbolisRegression/PySR/github/
├─fun_rec_data
├─fun_rec_feature_num_test_data
│  ├─loss_pic
│  └─test_data
├─fun_unrec_data
│  ├─best_model
│  ├─data
│  ├─equation
│  └─loss
├─FePySR
│  └─utils
│     ├─__pycache__
│     ├─config
│     │  └─config_regression.yaml
│     ├─__init__.py
│     ├─data_analyzer.py
│     ├─feature_maker.py
│     ├─feature_mapping_network.py
│     ├─fepysr.py
│     ├─fmn_builder.py
│     ├─fmn_train.py
│     ├─functions.py
│     ├─optimization.py
│     └─pysr_train.py
```

## Installation
FePySR requires Python (for deep learning) and Julia (for symbolic search). For optimal environment management, we strongly recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) for your Python environment, and [Scoop](https://scoop.sh/) (for Windows users) to install and manage Julia.

### 1. Install Julia

If you are on Windows, you can easily install Julia using Scoop. Scoop will automatically add Julia to your system `PATH`, which prevents most backend connection issues.
(Windows 用户可以使用 Scoop 一键安装 Julia，它会自动配置好系统环境变量，能避免绝大多数的后端连接报错。)

```bash
scoop install julia
```
> *For macOS/Linux users, please download Julia from the [official website](https://julialang.org/downloads/) or use your system's package manager.*

### 2. Create Python Environment

Create and activate a clean Conda environment:
```bash
conda create -n fepysr_env python=3.10 -y
conda activate fepysr_env
```
### 3. Install Dependencies 

FePySR is highly optimized for CPU multiprocessing. We recommend installing the lightweight CPU-only version of PyTorch for maximum stability and speed.

**Using Pip (Recommended):**
```bash
pip install pysr torch hydra-core numpy
```

**Using Conda:**
```bash
conda install -c conda-forge pysr hydra-core numpy
conda install pytorch cpuonly -c pytorch
```

### 4. Configure PySR Backend

Finally, run the following command to let PySR link with your installed Julia environment. Since Scoop already added Julia to your PATH, this step should be blazing fast.
```bash
python -c "import pysr; pysr.install()"
```

Note: Since the feature network is relatively shallow, FePySR defaults to CPU parallelism. The commands above will install the lightweight CPU-only version of PyTorch.



## Quickstart
You may wish to quickly reproduce the experiments in the article, and take the first example of Experiment 2 as an illustration
```python
import numpy as np
from pysr import PySRRegressor
import torch
from utils.fepysr import FePySR

# 1. Load the benchmark data (加载测试数据)
data = np.load("fun_rec_data/fun_rec_1.npy")
X = torch.from_numpy(data[:, :1])
y = torch.from_numpy(data[:, 1:])

# 2. Initialize and train the model (初始化并训练模型)
model = FePySR(overrides=["Parallel.num_workers=8", "Parallel.num_experiments=32"])
model.fit(X, y)

# 3. Print the most intuitive results (打印最直观的核心结果)
print("Best Equation: ", model.best_equation_)
print("Extracted Features: ", model.feature_names_)
```
**run Output**
```console
Best Equation:  'X0*(X0 + 1)*(X0^4 + X0^2 + 1)'
Extracted Features:  ['sin(X1)''X1**2','X0*X1','X0 + X1','X0**2','cos(cos(X1))', 'cos(X1)' ,'exp(X0)' ,'exp(X0*X1)','sin(X0)']
```

**Core Attributes**

To avoid cluttering the terminal, FePySR stores detailed background data as model attributes. You can access them directly after `.fit()`:

| Attribute| Type | Description  |
| :--- | :--- | :--- |
| `best_equation_` | `str` | The final, most accurate symbolic expression.  |
| `feature_names_` | `list` | Deduplicated list of high-quality features extracted by FMN. |
| `all_feature_names_` | `dict` | Complete pool of features with their extraction frequencies.  |
| `best_model_` | `pd.Series` | PySR's underlying model state for the best equation.  |
| `solved_pysr_model` | `PySRRegressor` | The complete trained PySR estimator object.  |

### Detailed Example
The following code makes use of as many PySR features as possible.
```python
model = FePySR(
    overrides=[
        "FMN.batch_size=50",
        "FMN.lr=0.5",
        "FMN.num_epochs=100",
        "FMN.loss=utils.optimization.squared_loss",
        "FMN.loss_parameter.lambda1=0.08",
        "FMN.loss_parameter.lambda2=0.001",
        "FMN.net.fun_list=null",
        "FMN.net.net_depth=4",
        "FMN.net.fun_re=4",
        "FMN.net.full_net=null",
        "FMN.device=auto",
        "FMN.FMN_only=null",
		
        "data_symbol.raw_data=null",
        "data_symbol.output_symbols=null",
        "data_symbol.fea_num=10",
        "data_symbol.pysr_num=6",
		
        "Parallel.num_experiments=16",
        "Parallel.num_workers=8",
		
        "pysr_params.populations=8",
        "pysr_params.population_size=50",
        "pysr_params.ncycles_per_iteration=500",
        "pysr_params.niterations=2000",
        "pysr_params.timeout_in_seconds=86400",
        "pysr_params.maxsize=25",
        "pysr_params.maxdepth=10",
        "pysr_params.early_stop_condition='stop_if(loss, complexity) = loss < 1e-9 && complexity < 40'",
        
        "pysr_params.binary_operators=['*','+','-']",
        "pysr_params.unary_operators=['cos','sqrt','exp','sin','inv(x) = 1/x']",
        "pysr_params.elementwise_loss='loss(prediction, target) = (prediction - target)^2'",
        "pysr_params.nested_constraints.exp.exp=0"
    ]
)

```
All hyperparameters are comprehensively documented in [`config_regression.yaml`](./config_regression.yaml). We specifically introduced the `FMN_only` toggle, which allows you to execute only the neural network feature extraction stage (Stage 1). This empowers you to use FePySR as a standalone feature engineering tool, extracting high-quality mathematical features to feed into other symbolic regression solvers or traditional machine learning algorithms.
```python
model = FePySR(overrides=['FMN.FMN_only=True'],custom_pysr_model=model_default)
```
Please note that the `pysr_params` section in `config_regression.yaml` exposes only the most frequently used PySR configurations. To unlock the full potential and advanced features of PySR, you can pre-configure a native `PySRRegressor` instance and inject it directly into FePySR using the `custom_pysr_model` parameter.
```python
from pysr import PySRRegressor
from utils.fepysr import FePySR

model_pysr = PySRRegressor(
    populations=8,
    population_size=50,
    ncycles_per_iteration=500,
    maxsize=25,
    maxdepth=10,
    binary_operators=["*", "+", "-"],
    unary_operators=[
        "cos",
        "sqrt",
        "exp",
        "sin",
        "inv(x) = 1/x",
    ],
)

model_FePySR = FePySR(custom_pysr_model=model_pysr)
```


## Citation 

If you use FePySR in your research (e.g., for symbolic regression in biokinetics or physics), please cite our paper:

```bibtex
@article{yu2026fepysr,
  title={FePySR: A Two-Stage Symbolic Regression Framework via Feature Engineering},
  author={Yu, Zhiming and others},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.




